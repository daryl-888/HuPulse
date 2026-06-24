"""
Train MS4Plus on Carya — standalone script.

Reproduces the existing model_training_bootstrap.py training loop and
data loading EXACTLY (same hyperparameters, same Dataset class, same
evaluate() function, same CSV output format) but swaps in MS4Plus_1D
instead of MS4_1D.

No existing file on Carya is modified.

Usage (change BP at the top):
    BP = "SBP"   # or "DBP"

Submit via:
    sbatch jobs/train_ms4plus_SBP.sbatch
    sbatch jobs/train_ms4plus_DBP.sbatch
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

# ── Add repo root to path so we can import the improved model ────────────────
REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)
from models.ms4_improved import MS4Plus_1D

# =============================================================================
# SETTINGS  — only change BP here; everything else matches the paper exactly
# =============================================================================
SUBSET_BASE = "/project/rhu/PulseBP/Pulse/pulsedb/PulseDB/Subset_Files"
# Training data lives in the zouridakis project space (same as existing bootstrap)
TRAIN_FILE         = "/project/zouridakis/gzlab2/PulseDB/Subset_Files/Train_Subset_filtered.mat"
TEST_CALBASED_FILE = os.path.join(SUBSET_BASE, "CalBased_Test_Subset_filtered.mat")
TEST_CALFREE_FILE  = os.path.join(SUBSET_BASE, "CalFree_Test_Subset_filtered.mat")

BP           = "SBP"       # default; overridden by --bp CLI arg
SEED         = 6
N_SPLITS     = 10
BATCH_SIZE   = 32
NUM_EPOCHS   = 100
LR           = 2e-5        # same as paper
WEIGHT_DECAY = 1e-8
BETAS        = (0.9, 0.999)

OUT_DIR = f"./ms4plus_cv_results_{BP.lower()}"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================================================
# Data utilities — copied verbatim from model_training_bootstrap.py
# =============================================================================

def encode_gender(gender_arr):
    return np.array(
        [1 if str(g).upper().startswith('M') else 0 for g in gender_arr],
        dtype=np.float32
    )

def Seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class Dataset(data.Dataset):
    def __init__(self, Input, Age, BMI, Gender, Label):
        self.Input  = Input    # (N, 1, seq_len)
        self.Age    = Age
        self.BMI    = BMI
        self.Gender = Gender
        self.Label  = Label

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        static_feat = np.array(
            [self.Age[idx], self.BMI[idx], self.Gender[idx]], dtype=np.float32
        )
        return (self.Input[idx, :].astype(np.float32), static_feat), self.Label[idx]

def Build_Dataset(Path, Label, show_info=True):
    Data    = loadmat(Path)
    signals = np.expand_dims(Data['Subset']['Signals'][:, 1, :], axis=1)
    age     = np.array(Data['Subset']['Age'],    dtype=np.float32).flatten()
    bmi     = np.array(Data['Subset']['BMI'],    dtype=np.float32).flatten()
    gender  = encode_gender(Data['Subset']['Gender'])
    label   = np.array(Data['Subset'][Label],    dtype=np.float32).flatten()

    N, mask = len(age), np.ones(len(age), dtype=bool)
    for i in range(N):
        if (np.any(np.isnan([age[i], bmi[i], gender[i]])) or
                np.any(np.isnan(signals[i])) or
                np.any(np.isinf(signals[i]))):
            mask[i] = False
    signals, age, bmi, gender, label = (
        signals[mask], age[mask], bmi[mask], gender[mask], label[mask]
    )
    if show_info:
        print(f"{os.path.basename(Path)}: {N} -> {mask.sum()} samples after NaN/Inf filter")
    return Dataset(signals, age, bmi, gender, label)

def subset_dataset(ds: Dataset, indices: np.ndarray) -> Dataset:
    return Dataset(
        ds.Input[indices], ds.Age[indices],
        ds.BMI[indices],   ds.Gender[indices], ds.Label[indices]
    )

def to_device(batch, device):
    (sig, sf), y = batch
    sig = torch.from_numpy(sig) if isinstance(sig, np.ndarray) else sig
    sf  = torch.from_numpy(sf)  if isinstance(sf,  np.ndarray) else sf
    if not isinstance(y, torch.Tensor):
        y = torch.from_numpy(np.array(y, dtype=np.float32))
    return (sig.to(device), sf.to(device)), y.to(device).view(-1)

# =============================================================================
# Metrics — same as bootstrap
# =============================================================================

def mae(y_true, y_pred):      return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred):     return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def abs_err_std(y_true, y_pred):
    return float(np.std(np.abs(y_true - y_pred), ddof=1)) if len(y_true) > 1 else 0.0
def r2_score(y_true, y_pred):
    ybar = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - ybar)   ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")

@torch.no_grad()
def evaluate(model, ds: Dataset, device, batch_size=256) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    ys, ps = [], []
    for batch in loader:
        (sig, sf), y = to_device(batch, device)
        out = model(sig, sf).view(-1).detach().cpu().numpy()
        ys.append(y.view(-1).detach().cpu().numpy())
        ps.append(out)
    y_true = np.concatenate(ys)
    y_pred = np.concatenate(ps)
    return {
        "MAE":  mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2":   r2_score(y_true, y_pred),
        "STD":  abs_err_std(y_true, y_pred),
    }

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running, n = 0.0, 0
    for batch in loader:
        (sig, sf), y = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out  = model(sig, sf)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        bs       = y.size(0)
        running += loss.item() * bs
        n       += bs
    return running / max(n, 1)

# =============================================================================
# Main — K-fold CV identical to existing bootstrap
# =============================================================================

def main():
    global BP, OUT_DIR
    parser = argparse.ArgumentParser()
    parser.add_argument("--bp", choices=["SBP", "DBP"], default=BP,
                        help="Blood pressure target (default: SBP)")
    args = parser.parse_args()
    BP = args.bp
    OUT_DIR = f"./ms4plus_cv_results_{BP.lower()}"
    os.makedirs(OUT_DIR, exist_ok=True)

    Seed(SEED)
    print(f"Target: {BP}")
    print(f"Train : {TRAIN_FILE}")
    print(f"OutDir: {OUT_DIR}")

    Train_Data          = Build_Dataset(TRAIN_FILE,         BP)
    Test_CalBased_Data  = Build_Dataset(TEST_CALBASED_FILE, BP)
    Test_CalFree_Data   = Build_Dataset(TEST_CALFREE_FILE,  BP)

    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    per_epoch_csv = os.path.join(OUT_DIR, f"{BP}_per_epoch_metrics.csv")
    with open(per_epoch_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold", "epoch", "train_loss",
            "calbased_MAE", "calbased_RMSE", "calbased_R2", "calbased_STD",
            "calfree_MAE",  "calfree_RMSE",  "calfree_R2",  "calfree_STD",
        ])

    per_fold_best_csv = os.path.join(OUT_DIR, f"{BP}_per_fold_best.csv")
    with open(per_fold_best_csv, "w", newline="") as f:
        csv.writer(f).writerow([
            "fold",
            "best_calbased_MAE", "best_calbased_epoch", "best_calbased_STD", "best_calbased_R2",
            "best_calfree_MAE",  "best_calfree_epoch",  "best_calfree_STD",  "best_calfree_R2",
        ])

    N       = len(Train_Data)
    indices = np.arange(N)
    kf      = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    best_cb_list, best_cf_list = [], []

    for fold_id, (train_idx, _) in enumerate(kf.split(indices), start=1):
        print(f"\n===== Fold {fold_id}/{N_SPLITS} =====")
        ds_train     = subset_dataset(Train_Data, train_idx)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        Seed(SEED + fold_id)
        model     = MS4Plus_1D(num_static_features=3, num_BP=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()

        best_calbased = {"MAE": float("inf"), "epoch": -1, "STD": None, "R2": None}
        best_calfree  = {"MAE": float("inf"), "epoch": -1, "STD": None, "R2": None}

        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
            cb = evaluate(model, Test_CalBased_Data, device)
            cf = evaluate(model, Test_CalFree_Data,  device)

            with open(per_epoch_csv, "a", newline="") as f:
                csv.writer(f).writerow([
                    fold_id, epoch, f"{train_loss:.6f}",
                    f"{cb['MAE']:.6f}", f"{cb['RMSE']:.6f}", f"{cb['R2']:.6f}", f"{cb['STD']:.6f}",
                    f"{cf['MAE']:.6f}", f"{cf['RMSE']:.6f}", f"{cf['R2']:.6f}", f"{cf['STD']:.6f}",
                ])

            fold_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
            os.makedirs(fold_dir, exist_ok=True)

            if cb["MAE"] < best_calbased["MAE"]:
                best_calbased = {"MAE": cb["MAE"], "epoch": epoch, "STD": cb["STD"], "R2": cb["R2"]}
                torch.save(model.state_dict(),
                           os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_calbased.pth"))

            if cf["MAE"] < best_calfree["MAE"]:
                best_calfree = {"MAE": cf["MAE"], "epoch": epoch, "STD": cf["STD"], "R2": cf["R2"]}
                torch.save(model.state_dict(),
                           os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_calfree.pth"))

        torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_final.pth"))

        with open(per_fold_best_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                fold_id,
                f"{best_calbased['MAE']:.6f}", best_calbased["epoch"],
                f"{best_calbased['STD']:.6f}", f"{best_calbased['R2']:.6f}",
                f"{best_calfree['MAE']:.6f}",  best_calfree["epoch"],
                f"{best_calfree['STD']:.6f}",  f"{best_calfree['R2']:.6f}",
            ])
        best_cb_list.append(best_calbased["MAE"])
        best_cf_list.append(best_calfree["MAE"])

        print(f"[Fold {fold_id}] CalBased best MAE={best_calbased['MAE']:.4f} @ ep {best_calbased['epoch']}  "
              f"STD={best_calbased['STD']:.4f}  R2={best_calbased['R2']:.4f}")
        print(f"[Fold {fold_id}] CalFree  best MAE={best_calfree['MAE']:.4f} @ ep {best_calfree['epoch']}  "
              f"STD={best_calfree['STD']:.4f}  R2={best_calfree['R2']:.4f}")

    summary_path = os.path.join(OUT_DIR, f"{BP}_cv_summary.txt")
    cb_arr = np.array(best_cb_list, dtype=np.float32)
    cf_arr = np.array(best_cf_list, dtype=np.float32)
    with open(summary_path, "w") as f:
        f.write(f"{N_SPLITS}-Fold CV  target={BP}  model=MS4Plus\n")
        f.write(f"CalBased MAE: mean={cb_arr.mean():.4f}  std={cb_arr.std(ddof=1):.4f}\n")
        f.write(f"CalFree  MAE: mean={cf_arr.mean():.4f}  std={cf_arr.std(ddof=1):.4f}\n")
    print("\n=== CV Summary ===")
    print(open(summary_path).read())


if __name__ == "__main__":
    main()
