"""
Standalone variant runner for MS4 fusion experiments on Carya.

Runs the same 10-fold CV as Model_training_bootstrap.py (same Dataset,
Build_Dataset, evaluate, hyperparams) but lets you choose a fusion variant
via --variant without touching the bootstrap.

Usage:
    python scripts/train_ms4plus_carya.py --bp SBP --variant film
    python scripts/train_ms4plus_carya.py --bp DBP --variant gate
    python scripts/train_ms4plus_carya.py --bp SBP --variant filmgate

Submit via:
    sbatch jobs/train_ms4plus_SBP_film.sbatch
    sbatch jobs/train_ms4plus_DBP_film.sbatch
    (etc.)
"""
import os
import sys
import csv
import random
import argparse
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from mat73 import loadmat
from sklearn.model_selection import KFold

# ── Import the real S4 from Carya's existing Model_Def ───────────────────────
_CARYA_MODEL_TRAINING = "/project/rhu/PulseBP/Pulse/PulseDB_multi_full/Model_Training"
sys.path.insert(0, _CARYA_MODEL_TRAINING)
from Model_Def.s42 import S4 as S42

# =============================================================================
# S4Model — verbatim copy from original MS4.py (do not modify)
# =============================================================================

class S4Model(nn.Module):
    def __init__(self, d_input, d_output, d_state=64, d_model=512, n_layers=4,
                 dropout=0.2, prenorm=False, l_max=1024, transposed_input=True,
                 bidirectional=True, layer_norm=True, pooling=True):
        super().__init__()
        self.prenorm = prenorm
        self.transposed_input = transposed_input
        self.layer_norm = layer_norm
        if d_input is None:
            self.encoder = nn.Identity()
        else:
            self.encoder = nn.Conv1d(d_input, d_model, 1) if transposed_input else nn.Linear(d_input, d_model)
        self.s4_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.s4_layers.append(S42(
                d_state=d_state, l_max=l_max, d_model=d_model,
                bidirectional=bidirectional, postact='glu',
                dropout=dropout, transposed=True,
            ))
            self.norms.append(nn.LayerNorm(d_model) if layer_norm else nn.BatchNorm1d(d_model))
            self.dropouts.append(nn.Dropout2d(dropout))
        self.pooling = pooling
        self.decoder = None if d_output is None else nn.Linear(d_model, d_output)

    def forward(self, x, rate=1.0):
        x = self.encoder(x)
        if not self.transposed_input:
            x = x.transpose(-1, -2)
        for layer, norm, dropout in zip(self.s4_layers, self.norms, self.dropouts):
            z = x
            if self.prenorm:
                z = norm(z.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
            z, _ = layer(z, rate=rate)
            z = dropout(z)
            x = z + x
            if not self.prenorm:
                x = norm(x.transpose(-1, -2)).transpose(-1, -2) if self.layer_norm else norm(z)
        x = x.transpose(-1, -2)
        if self.pooling:
            x = x.mean(dim=1)
        if self.decoder is not None:
            x = self.decoder(x)
        if not self.pooling and self.transposed_input:
            x = x.transpose(-1, -2)
        return x

# =============================================================================
# Building blocks — matches deploy/MS4.py exactly (MS4_2-based architecture)
# =============================================================================

class S4Block(nn.Module):
    """Pre-norm residual block: S4 + pointwise FFN."""
    def __init__(self, d_model, d_state=64, l_max=2048, dropout=0.2, bidirectional=True):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.s4 = S42(d_state=d_state, l_max=l_max, d_model=d_model,
                      bidirectional=bidirectional, postact='glu',
                      dropout=dropout, transposed=True)
        self._bidir = bidirectional
        self.drop = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Conv1d(d_model, 4 * d_model, kernel_size=1), nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(4 * d_model, d_model, kernel_size=1),
        )

    def forward(self, x, rate=1.0):  # x: (B, C, L)
        y = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        y, _ = self.s4(y, rate=rate)
        x = x + self.drop(y)
        y = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        x = x + self.drop(self.ffn(y))
        return x


class AttnPool1D(nn.Module):
    """Learned attention pooling over time."""
    def __init__(self, d_model, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d_model, hidden), nn.Tanh(), nn.Linear(hidden, 1))

    def forward(self, x):  # x: (B, L, C)
        w = torch.softmax(self.scorer(x).squeeze(-1), dim=1)
        return torch.bmm(w.unsqueeze(1), x).squeeze(1)


class _FiLM(nn.Module):
    """FiLM on sequence (B, L, C) given static embedding (B, D)."""
    def __init__(self, static_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(static_dim, 2 * d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, s):
        gamma, beta = self.proj(s).chunk(2, dim=-1)
        return x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)


class _Gate(nn.Module):
    """Sigmoid gate on sequence (B, L, C) given static embedding (B, D)."""
    def __init__(self, static_dim, d_model):
        super().__init__()
        self.proj = nn.Linear(static_dim, d_model)
        nn.init.zeros_(self.proj.weight)
        nn.init.constant_(self.proj.bias, 1.0)

    def forward(self, x, s):
        g = torch.sigmoid(self.proj(s)).unsqueeze(1)
        return x * g


# =============================================================================
# Variant models — conv stem + S4Blocks + sequence fusion + AttnPool
# (same architecture as deploy/MS4.py; only fusion mechanism differs)
# =============================================================================

class MS4_FiLM(nn.Module):
    """Conv stem → S4Blocks → sequence FiLM → AttnPool → head."""
    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2, bidirectional=True):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout,
                    bidirectional=bidirectional)
            for _ in range(s4_n_layers)
        ])
        self.static_norm = nn.BatchNorm1d(num_static_features)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden), nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden), nn.ReLU(inplace=True),
        )
        self.film = _FiLM(static_hidden, D)
        self.pool = AttnPool1D(D, hidden=128)
        self.head = nn.Sequential(
            nn.Linear(D + static_hidden, fusion_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fusion_hidden, num_BP),
        )

    def forward(self, ppg, static_feat):
        x = self.stem(ppg)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(-1, -2)
        s = self.static_mlp(self.static_norm(static_feat))
        x = self.film(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


class MS4_Gate(nn.Module):
    """Conv stem → S4Blocks → sequence gate → AttnPool → head."""
    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2, bidirectional=True):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout,
                    bidirectional=bidirectional)
            for _ in range(s4_n_layers)
        ])
        self.static_norm = nn.BatchNorm1d(num_static_features)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden), nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden), nn.ReLU(inplace=True),
        )
        self.gate = _Gate(static_hidden, D)
        self.pool = AttnPool1D(D, hidden=128)
        self.head = nn.Sequential(
            nn.Linear(D + static_hidden, fusion_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fusion_hidden, num_BP),
        )

    def forward(self, ppg, static_feat):
        x = self.stem(ppg)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(-1, -2)
        s = self.static_mlp(self.static_norm(static_feat))
        x = self.gate(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


class MS4_FiLMGate(nn.Module):
    """Conv stem → S4Blocks → sequence FiLM then gate → AttnPool → head."""
    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2, bidirectional=True):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout,
                    bidirectional=bidirectional)
            for _ in range(s4_n_layers)
        ])
        self.static_norm = nn.BatchNorm1d(num_static_features)
        self.static_mlp = nn.Sequential(
            nn.Linear(num_static_features, static_hidden), nn.ReLU(inplace=True),
            nn.Linear(static_hidden, static_hidden), nn.ReLU(inplace=True),
        )
        self.film = _FiLM(static_hidden, D)
        self.gate = _Gate(static_hidden, D)
        self.pool = AttnPool1D(D, hidden=128)
        self.head = nn.Sequential(
            nn.Linear(D + static_hidden, fusion_hidden), nn.ReLU(inplace=True),
            nn.Dropout(dropout), nn.Linear(fusion_hidden, num_BP),
        )

    def forward(self, ppg, static_feat):
        x = self.stem(ppg)
        for blk in self.blocks:
            x = blk(x)
        x = x.transpose(-1, -2)
        s = self.static_mlp(self.static_norm(static_feat))
        x = self.film(x, s)
        x = self.gate(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


class MS4_FiLMOnly(nn.Module):
    """
    CLEAN ABLATION — original MS4 baseline with ONLY the fusion swapped.

    Identical to the paper's MS4 baseline (S4Model d_model=512, mean-pool,
    no conv stem, no FFN, no attention pool) EXCEPT demographics are injected
    via FiLM on the pooled vector instead of plain concatenation.

    Purpose: isolate the contribution of demographic conditioning alone, so a
    win can be attributed to the fusion mechanism rather than to the conv
    stem / FFN / attention-pool additions in the other variants.
    """
    def __init__(self, num_static_features=3, num_BP=1, D=512, demo_hidden=16, dropout=0.1):
        super().__init__()
        self.s4model = S4Model(d_input=1, d_output=None, d_state=64,
                               d_model=D, n_layers=4, l_max=2048, pooling=True)
        self.demo_mlp = nn.Sequential(
            nn.Linear(num_static_features, demo_hidden), nn.ReLU(inplace=True),
            nn.Linear(demo_hidden, demo_hidden), nn.ReLU(inplace=True),
        )
        # FiLM params from demographics; zero-init -> starts as identity (= baseline)
        self.film = nn.Linear(demo_hidden, 2 * D)
        nn.init.zeros_(self.film.weight); nn.init.zeros_(self.film.bias)
        self.ppg_mlp = nn.Sequential(
            nn.Linear(D, 256), nn.ReLU(inplace=True), nn.Linear(256, 128), nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.Linear(128, 32), nn.ReLU(inplace=True), nn.Dropout(dropout), nn.Linear(32, num_BP),
        )

    def forward(self, ppg, static_feat):
        feat = self.s4model(ppg)                       # (B, 512) mean-pooled
        d = self.demo_mlp(static_feat)                 # (B, 16)
        gamma, beta = self.film(d).chunk(2, dim=-1)    # (B, 512) each
        feat = feat * (1.0 + gamma) + beta             # FiLM on pooled vector
        return self.head(self.ppg_mlp(feat)).squeeze(-1)


VARIANTS = {
    "film": MS4_FiLM,
    "gate": MS4_Gate,
    "filmgate": MS4_FiLMGate,
    "film_baseline": MS4_FiLMOnly,   # clean ablation: baseline + FiLM only
}

# =============================================================================
# Settings
# =============================================================================

DATA_FOLDER         = "/project/rhu/PulseBP/Pulse/pulsedb/PulseDB/Subset_Files"
_train_filtered     = os.path.join(DATA_FOLDER, "Train_Subset_filtered.mat")
_train_fallback     = os.path.join(DATA_FOLDER, "AAMI_Cal_Subset.mat")
TRAIN_FILE          = _train_filtered if os.path.exists(_train_filtered) else _train_fallback
TEST_CALBASED_FILE  = os.path.join(DATA_FOLDER, "CalBased_Test_Subset_filtered.mat")
TEST_CALFREE_FILE   = os.path.join(DATA_FOLDER, "CalFree_Test_Subset_filtered.mat")

SEED         = 6
N_SPLITS     = 10
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
LR           = 2e-5
WEIGHT_DECAY = 1e-8
BETAS        = (0.9, 0.999)

# =============================================================================
# Data utilities — verbatim from Model_training_bootstrap.py
# =============================================================================

def encode_gender(gender_arr):
    return np.array([1 if str(g).upper().startswith('M') else 0 for g in gender_arr], dtype=np.float32)

def Seed(seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    np.random.seed(seed); random.seed(seed)
    torch.backends.cudnn.benchmark = False; torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, Input, Age, BMI, Gender, Label):
        self.Input = Input; self.Age = Age; self.BMI = BMI
        self.Gender = Gender; self.Label = Label
    def __len__(self): return len(self.Input)
    def __getitem__(self, idx):
        sf = np.array([self.Age[idx], self.BMI[idx], self.Gender[idx]], dtype=np.float32)
        return (self.Input[idx].astype(np.float32), sf), self.Label[idx]

def Build_Dataset(Path, Label):
    D       = loadmat(Path)
    signals = np.expand_dims(D['Subset']['Signals'][:, 1, :], axis=1)
    age     = np.array(D['Subset']['Age'],    dtype=np.float32).flatten()
    bmi     = np.array(D['Subset']['BMI'],    dtype=np.float32).flatten()
    gender  = encode_gender(D['Subset']['Gender'])
    label   = np.array(D['Subset'][Label],    dtype=np.float32).flatten()
    N, mask = len(age), np.ones(len(age), dtype=bool)
    for i in range(N):
        if (np.any(np.isnan([age[i], bmi[i], gender[i]])) or
                np.any(np.isnan(signals[i])) or np.any(np.isinf(signals[i]))):
            mask[i] = False
    signals, age, bmi, gender, label = signals[mask], age[mask], bmi[mask], gender[mask], label[mask]
    print(f"{os.path.basename(Path)}: {N} -> {mask.sum()} samples")
    return Dataset(signals, age, bmi, gender, label)

def subset_dataset(ds, indices):
    return Dataset(ds.Input[indices], ds.Age[indices], ds.BMI[indices], ds.Gender[indices], ds.Label[indices])

def to_device(batch, device):
    (sig, sf), y = batch
    if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
    if isinstance(sf,  np.ndarray): sf  = torch.from_numpy(sf)
    if not isinstance(y, torch.Tensor): y = torch.from_numpy(np.array(y, dtype=np.float32))
    return (sig.to(device), sf.to(device)), y.to(device).view(-1)

def mae(a, b):      return float(np.mean(np.abs(a - b)))
def rmse(a, b):     return float(np.sqrt(np.mean((a - b) ** 2)))
def r2(a, b):
    ss = np.sum((a - b) ** 2); st = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")
def mean_err(a, b): return float(np.mean(b - a))          # signed: pred - true
def std_err(a, b):  return float(np.std(b - a, ddof=1)) if len(a) > 1 else 0.0  # AAMI std

@torch.no_grad()
def evaluate(model, ds, device, batch_size=256) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)
    for batch in loader:
        (sig, sf), y = to_device(batch, device)
        out = model(sig, sf)
        if hasattr(model, "module"):   # DataParallel wrapper
            out = out
        ps.append(out.view(-1).cpu().numpy())
        ys.append(y.view(-1).cpu().numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    return {
        "MAE":      mae(yt, yp),
        "RMSE":     rmse(yt, yp),
        "R2":       r2(yt, yp),
        "MEAN_ERR": mean_err(yt, yp),   # signed mean error (AAMI numerator)
        "STD_ERR":  std_err(yt, yp),    # std of signed errors (AAMI denominator)
    }

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train(); running = n = 0
    for batch in loader:
        (sig, sf), y = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(sig, sf), y)
        loss.backward(); optimizer.step()
        running += loss.item() * y.size(0); n += y.size(0)
    return running / max(n, 1)

# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp",      choices=["SBP", "DBP"], default="SBP")
    parser.add_argument("--variant", choices=list(VARIANTS),  default="film",
                        help="Fusion variant: film | gate | filmgate | film_baseline")
    parser.add_argument("--no-bidir", dest="bidir", action="store_false",
                        help="Use unidirectional S4 (default: bidirectional)")
    parser.set_defaults(bidir=True)
    args = parser.parse_args()

    BP, variant, bidir = args.bp, args.variant, args.bidir
    bidir_tag = "bidir" if bidir else "uni"
    OUT_DIR = f"./ms4_{variant}_{bidir_tag}_cv_{BP.lower()}"
    os.makedirs(OUT_DIR, exist_ok=True)

    Seed(SEED)
    print(f"BP={BP}  variant={variant}  bidirectional={bidir}  out={OUT_DIR}")

    train_data = Build_Dataset(TRAIN_FILE,         BP)
    cb_data    = Build_Dataset(TEST_CALBASED_FILE, BP)
    cf_data    = Build_Dataset(TEST_CALFREE_FILE,  BP)

    torch.cuda.empty_cache()
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  GPUs available: {n_gpus}")
    for i in range(n_gpus):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")

    per_epoch_csv = os.path.join(OUT_DIR, f"{BP}_per_epoch_metrics.csv")
    with open(per_epoch_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold","epoch","train_loss",
            "cb_MAE","cb_RMSE","cb_R2","cb_MEAN_ERR","cb_STD_ERR",
            "cf_MAE","cf_RMSE","cf_R2","cf_MEAN_ERR","cf_STD_ERR",
        ])

    per_fold_csv = os.path.join(OUT_DIR, f"{BP}_per_fold_best.csv")
    with open(per_fold_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold",
            "best_cb_MAE","best_cb_epoch","best_cb_MEAN_ERR","best_cb_STD_ERR","best_cb_R2",
            "best_cf_MAE","best_cf_epoch","best_cf_MEAN_ERR","best_cf_STD_ERR","best_cf_R2",
        ])

    N        = len(train_data)
    kf       = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    ModelCls = VARIANTS[variant]
    best_cb_list, best_cf_list = [], []

    for fold_id, (train_idx, _) in enumerate(kf.split(np.arange(N)), start=1):
        print(f"\n===== Fold {fold_id}/{N_SPLITS} =====")
        ds_train     = subset_dataset(train_data, train_idx)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        Seed(SEED + fold_id)
        _model_kwargs = {"num_static_features": 3, "num_BP": 1}
        if variant != "film_baseline":   # film_baseline uses S4Model which has its own bidir
            _model_kwargs["bidirectional"] = bidir
        _model    = ModelCls(**_model_kwargs).to(device)
        model     = nn.DataParallel(_model) if n_gpus > 1 else _model
        optimizer = torch.optim.Adam(_model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        best_cb = {"MAE": float("inf"), "epoch": -1, "MEAN_ERR": None, "STD_ERR": None, "R2": None}
        best_cf = {"MAE": float("inf"), "epoch": -1, "MEAN_ERR": None, "STD_ERR": None, "R2": None}
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        def save_ckpt(path):
            torch.save(_model.state_dict(), path)

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            cb   = evaluate(model, cb_data, device)
            cf   = evaluate(model, cf_data, device)

            with open(per_epoch_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    fold_id, epoch, f"{loss:.6f}",
                    f"{cb['MAE']:.6f}", f"{cb['RMSE']:.6f}", f"{cb['R2']:.6f}",
                    f"{cb['MEAN_ERR']:.6f}", f"{cb['STD_ERR']:.6f}",
                    f"{cf['MAE']:.6f}", f"{cf['RMSE']:.6f}", f"{cf['R2']:.6f}",
                    f"{cf['MEAN_ERR']:.6f}", f"{cf['STD_ERR']:.6f}",
                ])

            if cb["MAE"] < best_cb["MAE"]:
                best_cb = {"MAE": cb["MAE"], "epoch": epoch,
                           "MEAN_ERR": cb["MEAN_ERR"], "STD_ERR": cb["STD_ERR"], "R2": cb["R2"]}
                save_ckpt(os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_cb.pth"))

            if cf["MAE"] < best_cf["MAE"]:
                best_cf = {"MAE": cf["MAE"], "epoch": epoch,
                           "MEAN_ERR": cf["MEAN_ERR"], "STD_ERR": cf["STD_ERR"], "R2": cf["R2"]}
                save_ckpt(os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_cf.pth"))

        save_ckpt(os.path.join(fold_dir, f"{BP}_fold{fold_id}_final.pth"))

        with open(per_fold_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                fold_id,
                f"{best_cb['MAE']:.6f}", best_cb["epoch"],
                f"{best_cb['MEAN_ERR']:.6f}", f"{best_cb['STD_ERR']:.6f}", f"{best_cb['R2']:.6f}",
                f"{best_cf['MAE']:.6f}", best_cf["epoch"],
                f"{best_cf['MEAN_ERR']:.6f}", f"{best_cf['STD_ERR']:.6f}", f"{best_cf['R2']:.6f}",
            ])
        best_cb_list.append(best_cb)
        best_cf_list.append(best_cf)

        print(f"[Fold {fold_id}] CalBased MAE={best_cb['MAE']:.4f} @ ep{best_cb['epoch']}  "
              f"MeanErr={best_cb['MEAN_ERR']:.4f}  STD={best_cb['STD_ERR']:.4f}  R2={best_cb['R2']:.4f}")
        print(f"[Fold {fold_id}] CalFree  MAE={best_cf['MAE']:.4f} @ ep{best_cf['epoch']}  "
              f"MeanErr={best_cf['MEAN_ERR']:.4f}  STD={best_cf['STD_ERR']:.4f}  R2={best_cf['R2']:.4f}")

    # ── Final summary ────────────────────────────────────────────────────────
    cb_mae  = np.array([r["MAE"]      for r in best_cb_list])
    cf_mae  = np.array([r["MAE"]      for r in best_cf_list])
    cb_me   = np.array([r["MEAN_ERR"] for r in best_cb_list])
    cf_me   = np.array([r["MEAN_ERR"] for r in best_cf_list])
    cb_se   = np.array([r["STD_ERR"]  for r in best_cb_list])
    cf_se   = np.array([r["STD_ERR"]  for r in best_cf_list])

    # AAMI/ISO 81060-2: |mean(signed error)| <= 5 mmHg AND std(signed error) <= 8 mmHg
    cb_aami = abs(cb_me.mean()) <= 5.0 and cb_se.mean() <= 8.0
    cf_aami = abs(cf_me.mean()) <= 5.0 and cf_se.mean() <= 8.0

    lines = [
        f"{'='*65}",
        f"  {N_SPLITS}-Fold CV Results  |  BP={BP}  |  variant={variant}  |  S4={'bidir' if bidir else 'uni'}",
        f"{'='*65}",
        f"  {'Metric':<28} {'Cal-Based':>12} {'Cal-Free':>12}",
        f"  {'-'*52}",
        f"  {'MAE mean (mmHg)':<28} {cb_mae.mean():>12.4f} {cf_mae.mean():>12.4f}",
        f"  {'MAE std across folds':<28} {cb_mae.std(ddof=1):>12.4f} {cf_mae.std(ddof=1):>12.4f}",
        f"  {'Mean signed error (mmHg)':<28} {cb_me.mean():>12.4f} {cf_me.mean():>12.4f}",
        f"  {'Std signed error (mmHg)':<28} {cb_se.mean():>12.4f} {cf_se.mean():>12.4f}",
        f"  {'-'*52}",
        f"  AAMI/ISO 81060-2: |mean_err| <= 5 AND std_err <= 8",
        f"  Cal-Based : {'PASS' if cb_aami else 'FAIL'}  "
        f"(mean_err={cb_me.mean():.4f}, std_err={cb_se.mean():.4f})",
        f"  Cal-Free  : {'PASS' if cf_aami else 'FAIL'}  "
        f"(mean_err={cf_me.mean():.4f}, std_err={cf_se.mean():.4f})",
        f"{'='*65}",
    ]
    report = "\n".join(lines)

    summary = os.path.join(OUT_DIR, f"{BP}_cv_summary.txt")
    with open(summary, "w") as f:
        f.write(report + "\n")

    print("\n" + report)


if __name__ == "__main__":
    main()
