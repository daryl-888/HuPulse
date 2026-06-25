"""
Standalone ViT1D (PPG-only baseline) training script on Carya.

Runs the same 10-fold CV as Model_training_bootstrap.py.
Model: ViT1D from Model_Def/transformer.py — patch-based 1D Vision Transformer,
PPG signal only, no demographic conditioning.

Usage:
    python scripts/train_vit1d_carya.py --bp SBP
    python scripts/train_vit1d_carya.py --bp DBP

Submit via:
    sbatch jobs/train_vit1d_SBP.sbatch
    sbatch jobs/train_vit1d_DBP.sbatch
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

# ── Import ViT1D from Carya's Model_Def (PulseDB_trans) ─────────────────────
_CARYA_TRANS = "/project/rhu/PulseBP/Pulse/PulseDB_trans/Model_Training"
sys.path.insert(0, _CARYA_TRANS)

_ViT1DBase = None
_CARYA_API = False
try:
    from Model_Def.transformer import ViT1D as _ViT1DBase
    print("Loaded ViT1D from Carya Model_Def.transformer")
    _CARYA_API = True
except (ImportError, AttributeError):
    try:
        from Model_Def.transformer import ViT_1D as _ViT1DBase
        print("Loaded ViT_1D from Carya Model_Def.transformer")
        _CARYA_API = True
    except ImportError:
        print("WARNING: Could not import from Carya — using inline ViT1D fallback")


# ── Inline fallback — exact copy of transformer.py ───────────────────────────

class _PatchEmbedding1D(nn.Module):
    def __init__(self, in_channels, patch_size, emb_dim, seq_len):
        super().__init__()
        self.patch_size = patch_size
        self.n_patches = seq_len // patch_size
        self.proj = nn.Conv1d(in_channels, emb_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).transpose(1, 2)


class _TransformerEncoder1D(nn.Module):
    def __init__(self, emb_dim, nhead, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(emb_dim)
        self.attn  = nn.MultiheadAttention(embed_dim=emb_dim, num_heads=nhead, batch_first=True)
        self.norm2 = nn.LayerNorm(emb_dim)
        self.mlp   = nn.Sequential(
            nn.Linear(emb_dim, int(emb_dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(emb_dim * mlp_ratio), emb_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class _ViT1DInline(nn.Module):
    def __init__(self, in_channels=1, seq_len=1250, patch_size=10,
                 emb_dim=128, depth=6, nhead=8, mlp_ratio=4.0,
                 num_BP=1, use_cls_token=False, dropout=0.1):
        super().__init__()
        self.patch_embed  = _PatchEmbedding1D(in_channels, patch_size, emb_dim, seq_len)
        n_patches = seq_len // patch_size
        self.use_cls_token = use_cls_token
        if use_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, emb_dim))
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, emb_dim))
        else:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, emb_dim))
        self.pos_drop = nn.Dropout(p=dropout)
        self.encoder  = nn.ModuleList([
            _TransformerEncoder1D(emb_dim, nhead, mlp_ratio, dropout) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(emb_dim)
        self.head = nn.Linear(emb_dim, num_BP)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if use_cls_token:
            nn.init.trunc_normal_(self.cls_token, std=0.02)

    def forward(self, x):
        x = self.patch_embed(x)
        if self.use_cls_token:
            B = x.shape[0]
            x = torch.cat((self.cls_token.expand(B, -1, -1), x), dim=1)
        x = x + self.pos_embed[:, :x.shape[1], :]
        x = self.pos_drop(x)
        for blk in self.encoder:
            x = blk(x)
        x = self.norm(x)
        out = x[:, 0] if self.use_cls_token else x.mean(dim=1)
        return self.head(out)


# Wrapper so training loop always calls model(ppg, static_feat)
class _ViT1DWrapper(nn.Module):
    """Wraps ViT1D to accept (ppg, static_feat) — static_feat is unused."""
    def __init__(self, base):
        super().__init__()
        self.base = base
    def forward(self, ppg, static_feat):
        return self.base(ppg).squeeze(-1)


def _make_model(device) -> nn.Module:
    if _CARYA_API and _ViT1DBase is not None:
        try:
            base = _ViT1DBase(in_channels=1, seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
                              emb_dim=EMB_DIM, depth=DEPTH, nhead=NHEAD,
                              mlp_ratio=MLP_RATIO, num_BP=1, dropout=DROPOUT)
            print(f"Using Carya ViT1D: {_ViT1DBase.__name__}")
            return _ViT1DWrapper(base).to(device)
        except Exception as e:
            print(f"Carya import failed ({e}), using inline fallback")
    base = _ViT1DInline(in_channels=1, seq_len=SEQ_LEN, patch_size=PATCH_SIZE,
                        emb_dim=EMB_DIM, depth=DEPTH, nhead=NHEAD,
                        mlp_ratio=MLP_RATIO, num_BP=1, dropout=DROPOUT)
    print("Using inline ViT1D fallback")
    return _ViT1DWrapper(base).to(device)


# =============================================================================
# Model hyperparameters
# =============================================================================

SEQ_LEN    = 1250   # PPG signal length (125 Hz × 10 s)
PATCH_SIZE = 10     # 125 patches per signal
EMB_DIM    = 128
DEPTH      = 6
NHEAD      = 8
MLP_RATIO  = 4.0
DROPOUT    = 0.1

# =============================================================================
# Training settings — identical to paper bootstrap
# =============================================================================

DATA_FOLDER        = "/home/yshen28/PPGdata_filtered"
TRAIN_FILE         = os.path.join(DATA_FOLDER, "Train_Subset_filtered.mat")
TEST_CALBASED_FILE = os.path.join(DATA_FOLDER, "CalBased_Test_Subset_filtered.mat")
TEST_CALFREE_FILE  = os.path.join(DATA_FOLDER, "CalFree_Test_Subset_filtered.mat")

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
    return np.array([1 if str(g).upper().startswith('M') else 0
                     for g in gender_arr], dtype=np.float32)

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
    signals, age, bmi, gender, label = (signals[mask], age[mask], bmi[mask],
                                        gender[mask], label[mask])
    print(f"{os.path.basename(Path)}: {N} -> {mask.sum()} samples  "
          f"signal_len={signals.shape[-1]}")
    return Dataset(signals, age, bmi, gender, label)

def subset_dataset(ds, indices):
    return Dataset(ds.Input[indices], ds.Age[indices], ds.BMI[indices],
                   ds.Gender[indices], ds.Label[indices])

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
def mean_err(a, b): return float(np.mean(b - a))
def std_err(a, b):  return float(np.std(b - a, ddof=1)) if len(a) > 1 else 0.0

@torch.no_grad()
def evaluate(model, ds, device, batch_size=256) -> Dict[str, float]:
    model.eval()
    ys, ps = [], []
    for batch in DataLoader(ds, batch_size=batch_size, shuffle=False):
        (sig, sf), y = to_device(batch, device)
        ps.append(model(sig, sf).view(-1).cpu().numpy())
        ys.append(y.view(-1).cpu().numpy())
    yt, yp = np.concatenate(ys), np.concatenate(ps)
    return {
        "MAE":      mae(yt, yp),
        "RMSE":     rmse(yt, yp),
        "R2":       r2(yt, yp),
        "MEAN_ERR": mean_err(yt, yp),
        "STD_ERR":  std_err(yt, yp),
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
    parser.add_argument("--bp", choices=["SBP", "DBP"], default="SBP")
    args = parser.parse_args()
    BP = args.bp

    OUT_DIR = f"./vit1d_cv_{BP.lower()}"
    os.makedirs(OUT_DIR, exist_ok=True)

    Seed(SEED)
    print(f"BP={BP}  model=ViT1D (PPG-only)  out={OUT_DIR}")
    print(f"  seq_len={SEQ_LEN}  patch_size={PATCH_SIZE}  "
          f"emb_dim={EMB_DIM}  depth={DEPTH}  nhead={NHEAD}")

    train_data = Build_Dataset(TRAIN_FILE,         BP)
    cb_data    = Build_Dataset(TEST_CALBASED_FILE, BP)
    cf_data    = Build_Dataset(TEST_CALFREE_FILE,  BP)

    torch.cuda.empty_cache()
    n_gpus = torch.cuda.device_count()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  GPUs: {n_gpus}")
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

    N  = len(train_data)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    best_cb_list, best_cf_list = [], []

    for fold_id, (train_idx, _) in enumerate(kf.split(np.arange(N)), start=1):
        print(f"\n===== Fold {fold_id}/{N_SPLITS} =====")
        ds_train     = subset_dataset(train_data, train_idx)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE,
                                  shuffle=True, drop_last=False)

        Seed(SEED + fold_id)
        _model    = _make_model(device)
        model     = nn.DataParallel(_model) if n_gpus > 1 else _model
        optimizer = torch.optim.Adam(_model.parameters(), lr=LR,
                                     betas=BETAS, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        best_cb = {"MAE": float("inf"), "epoch": -1,
                   "MEAN_ERR": None, "STD_ERR": None, "R2": None}
        best_cf = {"MAE": float("inf"), "epoch": -1,
                   "MEAN_ERR": None, "STD_ERR": None, "R2": None}
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
                           "MEAN_ERR": cb["MEAN_ERR"], "STD_ERR": cb["STD_ERR"],
                           "R2": cb["R2"]}
                save_ckpt(os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_cb.pth"))

            if cf["MAE"] < best_cf["MAE"]:
                best_cf = {"MAE": cf["MAE"], "epoch": epoch,
                           "MEAN_ERR": cf["MEAN_ERR"], "STD_ERR": cf["STD_ERR"],
                           "R2": cf["R2"]}
                save_ckpt(os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_cf.pth"))

        save_ckpt(os.path.join(fold_dir, f"{BP}_fold{fold_id}_final.pth"))

        with open(per_fold_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                fold_id,
                f"{best_cb['MAE']:.6f}", best_cb["epoch"],
                f"{best_cb['MEAN_ERR']:.6f}", f"{best_cb['STD_ERR']:.6f}",
                f"{best_cb['R2']:.6f}",
                f"{best_cf['MAE']:.6f}", best_cf["epoch"],
                f"{best_cf['MEAN_ERR']:.6f}", f"{best_cf['STD_ERR']:.6f}",
                f"{best_cf['R2']:.6f}",
            ])
        best_cb_list.append(best_cb)
        best_cf_list.append(best_cf)

        print(f"[Fold {fold_id}] CalBased MAE={best_cb['MAE']:.4f} @ ep{best_cb['epoch']}  "
              f"MeanErr={best_cb['MEAN_ERR']:.4f}  STD={best_cb['STD_ERR']:.4f}  "
              f"R2={best_cb['R2']:.4f}")
        print(f"[Fold {fold_id}] CalFree  MAE={best_cf['MAE']:.4f} @ ep{best_cf['epoch']}  "
              f"MeanErr={best_cf['MEAN_ERR']:.4f}  STD={best_cf['STD_ERR']:.4f}  "
              f"R2={best_cf['R2']:.4f}")

    # ── Final summary ────────────────────────────────────────────────────────
    cb_mae = np.array([r["MAE"]      for r in best_cb_list])
    cf_mae = np.array([r["MAE"]      for r in best_cf_list])
    cb_me  = np.array([r["MEAN_ERR"] for r in best_cb_list])
    cf_me  = np.array([r["MEAN_ERR"] for r in best_cf_list])
    cb_se  = np.array([r["STD_ERR"]  for r in best_cb_list])
    cf_se  = np.array([r["STD_ERR"]  for r in best_cf_list])

    cb_aami = abs(cb_me.mean()) <= 5.0 and cb_se.mean() <= 8.0
    cf_aami = abs(cf_me.mean()) <= 5.0 and cf_se.mean() <= 8.0

    lines = [
        f"{'='*65}",
        f"  {N_SPLITS}-Fold CV Results  |  BP={BP}  |  model=ViT1D (PPG-only)",
        f"  Note: no demographic conditioning — pure signal baseline",
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
