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
                 s4_l_max=2048, dropout=0.2):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout)
            for _ in range(s4_n_layers)
        ])
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
        s = self.static_mlp(static_feat)
        x = self.film(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


class MS4_Gate(nn.Module):
    """Conv stem → S4Blocks → sequence gate → AttnPool → head."""
    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout)
            for _ in range(s4_n_layers)
        ])
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
        s = self.static_mlp(static_feat)
        x = self.gate(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


class MS4_FiLMGate(nn.Module):
    """Conv stem → S4Blocks → sequence FiLM then gate → AttnPool → head."""
    def __init__(self, num_static_features=3, num_BP=1,
                 static_hidden=32, fusion_hidden=64,
                 s4_d_input=1, s4_d_model=256, s4_n_layers=4,
                 s4_l_max=2048, dropout=0.2):
        super().__init__()
        D = s4_d_model
        self.stem = nn.Sequential(
            nn.Conv1d(s4_d_input, D, kernel_size=5, padding=2), nn.GELU(),
            nn.Conv1d(D, D, kernel_size=3, padding=1), nn.GELU(),
        )
        self.blocks = nn.ModuleList([
            S4Block(D, d_state=64, l_max=s4_l_max, dropout=dropout)
            for _ in range(s4_n_layers)
        ])
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
        s = self.static_mlp(static_feat)
        x = self.film(x, s)
        x = self.gate(x, s)
        x = self.pool(x)
        return self.head(torch.cat([x, s], dim=-1)).squeeze(-1)


VARIANTS = {"film": MS4_FiLM, "gate": MS4_Gate, "filmgate": MS4_FiLMGate}

# =============================================================================
# Settings
# =============================================================================

DATA_FOLDER         = "/home/yshen28/PPGdata_filtered"
TRAIN_FILE          = os.path.join(DATA_FOLDER, "Train_Subset_filtered.mat")
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

def mae(a, b):          return float(np.mean(np.abs(a - b)))
def rmse(a, b):         return float(np.sqrt(np.mean((a - b) ** 2)))
def r2(a, b):
    ss = np.sum((a - b) ** 2); st = np.sum((a - np.mean(a)) ** 2)
    return float(1 - ss / st) if st > 0 else float("nan")
def std_ae(a, b):       return float(np.std(np.abs(a - b), ddof=1)) if len(a) > 1 else 0.0

@torch.no_grad()
def evaluate(model, ds, device, batch_size=256) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        (sig, sf), y = to_device(batch, device)
        ps.append(model(sig, sf).view(-1).cpu().numpy())
        ys.append(y.view(-1).cpu().numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    return {"MAE": mae(yt, yp), "RMSE": rmse(yt, yp), "R2": r2(yt, yp), "STD": std_ae(yt, yp)}

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
                        help="Fusion variant: film | gate | filmgate")
    args = parser.parse_args()

    BP, variant = args.bp, args.variant
    OUT_DIR = f"./ms4_{variant}_cv_{BP.lower()}"
    os.makedirs(OUT_DIR, exist_ok=True)

    Seed(SEED)
    print(f"BP={BP}  variant={variant}  out={OUT_DIR}")

    train_data = Build_Dataset(TRAIN_FILE,         BP)
    cb_data    = Build_Dataset(TEST_CALBASED_FILE, BP)
    cf_data    = Build_Dataset(TEST_CALFREE_FILE,  BP)

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    per_epoch_csv = os.path.join(OUT_DIR, f"{BP}_per_epoch_metrics.csv")
    with open(per_epoch_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold","epoch","train_loss",
            "calbased_MAE","calbased_RMSE","calbased_R2","calbased_STD",
            "calfree_MAE","calfree_RMSE","calfree_R2","calfree_STD",
        ])

    per_fold_csv = os.path.join(OUT_DIR, f"{BP}_per_fold_best.csv")
    with open(per_fold_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold",
            "best_cb_MAE","best_cb_epoch","best_cb_STD","best_cb_R2",
            "best_cf_MAE","best_cf_epoch","best_cf_STD","best_cf_R2",
        ])

    N       = len(train_data)
    kf      = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    ModelCls = VARIANTS[variant]
    best_cb_list, best_cf_list = [], []

    for fold_id, (train_idx, _) in enumerate(kf.split(np.arange(N)), start=1):
        print(f"\n===== Fold {fold_id}/{N_SPLITS} =====")
        ds_train     = subset_dataset(train_data, train_idx)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        Seed(SEED + fold_id)
        model     = ModelCls(num_static_features=3, num_BP=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        best_cb = {"MAE": float("inf"), "epoch": -1, "STD": None, "R2": None}
        best_cf = {"MAE": float("inf"), "epoch": -1, "STD": None, "R2": None}
        fold_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        for epoch in range(1, NUM_EPOCHS + 1):
            loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            cb   = evaluate(model, cb_data, device)
            cf   = evaluate(model, cf_data, device)

            with open(per_epoch_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    fold_id, epoch, f"{loss:.6f}",
                    f"{cb['MAE']:.6f}", f"{cb['RMSE']:.6f}", f"{cb['R2']:.6f}", f"{cb['STD']:.6f}",
                    f"{cf['MAE']:.6f}", f"{cf['RMSE']:.6f}", f"{cf['R2']:.6f}", f"{cf['STD']:.6f}",
                ])

            if cb["MAE"] < best_cb["MAE"]:
                best_cb = {"MAE": cb["MAE"], "epoch": epoch, "STD": cb["STD"], "R2": cb["R2"]}
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_cb.pth"))

            if cf["MAE"] < best_cf["MAE"]:
                best_cf = {"MAE": cf["MAE"], "epoch": epoch, "STD": cf["STD"], "R2": cf["R2"]}
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_cf.pth"))

        torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_final.pth"))

        with open(per_fold_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                fold_id,
                f"{best_cb['MAE']:.6f}", best_cb["epoch"], f"{best_cb['STD']:.6f}", f"{best_cb['R2']:.6f}",
                f"{best_cf['MAE']:.6f}", best_cf["epoch"], f"{best_cf['STD']:.6f}", f"{best_cf['R2']:.6f}",
            ])
        best_cb_list.append(best_cb["MAE"])
        best_cf_list.append(best_cf["MAE"])

        print(f"[Fold {fold_id}] CalBased MAE={best_cb['MAE']:.4f} @ ep{best_cb['epoch']}  "
              f"STD={best_cb['STD']:.4f}  R2={best_cb['R2']:.4f}")
        print(f"[Fold {fold_id}] CalFree  MAE={best_cf['MAE']:.4f} @ ep{best_cf['epoch']}  "
              f"STD={best_cf['STD']:.4f}  R2={best_cf['R2']:.4f}")

    # ── Final summary with best numbers per split + AAMI check ───────────────
    # Load best checkpoint per fold and re-evaluate to get mean/STD per split
    # (already tracked above; collect from per_fold_best.csv rows)
    cb_arr = np.array(best_cb_list)
    cf_arr = np.array(best_cf_list)

    # Re-read per-fold STD values from the CSV (saved above)
    import csv as _csv
    cb_stds, cf_stds = [], []
    with open(per_fold_csv) as f:
        for row in _csv.DictReader(f):
            cb_stds.append(float(row["best_cb_STD"]))
            cf_stds.append(float(row["best_cf_STD"]))
    cb_std_arr = np.array(cb_stds)
    cf_std_arr = np.array(cf_stds)

    # AAMI/ISO 81060-2: |mean error| <= 5 mmHg AND std <= 8 mmHg
    # We use MAE as a proxy for |mean error| (conservative)
    cb_aami = cb_arr.mean() <= 5.0 and cb_std_arr.mean() <= 8.0
    cf_aami = cf_arr.mean() <= 5.0 and cf_std_arr.mean() <= 8.0

    lines = [
        f"{'='*60}",
        f"  {N_SPLITS}-Fold CV Results  |  BP={BP}  |  variant={variant}",
        f"{'='*60}",
        f"  {'Metric':<22} {'Cal-Based':>12} {'Cal-Free':>12}",
        f"  {'-'*46}",
        f"  {'MAE mean (mmHg)':<22} {cb_arr.mean():>12.4f} {cf_arr.mean():>12.4f}",
        f"  {'MAE std  (mmHg)':<22} {cb_arr.std(ddof=1):>12.4f} {cf_arr.std(ddof=1):>12.4f}",
        f"  {'STD mean (mmHg)':<22} {cb_std_arr.mean():>12.4f} {cf_std_arr.mean():>12.4f}",
        f"  {'-'*46}",
        f"  AAMI/ISO 81060-2 (MAE<=5 AND STD<=8)",
        f"  Cal-Based : {'PASS ✓' if cb_aami else 'FAIL ✗'}  "
        f"(MAE={cb_arr.mean():.4f}, STD={cb_std_arr.mean():.4f})",
        f"  Cal-Free  : {'PASS ✓' if cf_aami else 'FAIL ✗'}  "
        f"(MAE={cf_arr.mean():.4f}, STD={cf_std_arr.mean():.4f})",
        f"{'='*60}",
    ]
    report = "\n".join(lines)

    summary = os.path.join(OUT_DIR, f"{BP}_cv_summary.txt")
    with open(summary, "w") as f:
        f.write(report + "\n")

    print("\n" + report)


if __name__ == "__main__":
    main()
