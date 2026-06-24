"""
Standalone MResNet50 validation runner on Carya.

Runs the same 10-fold CV as Model_training_bootstrap.py so results are
directly comparable to the paper's reported numbers (Cal-Based MAE 5.35/3.24
mmHg SBP/DBP).

Imports MResNet50_1D from Carya's existing Model_Def directory — the same
model that produced the paper numbers — so this is a true apples-to-apples
comparison against the paper.

Usage:
    python scripts/train_mresnet50_carya.py --bp SBP
    python scripts/train_mresnet50_carya.py --bp DBP

Submit via:
    sbatch jobs/train_mresnet50_SBP.sbatch
    sbatch jobs/train_mresnet50_DBP.sbatch
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

# ── Import MResNet50 from Carya's Model_Def ──────────────────────────────────
_CARYA_MODEL_TRAINING = "/project/rhu/PulseBP/Pulse/PulseDB_multi_full/Model_Training"
sys.path.insert(0, _CARYA_MODEL_TRAINING)

try:
    from Model_Def.MResNet50 import MResNet50_1D as _MResNetCls
    print("Loaded MResNet50_1D from Carya Model_Def")
    _CARYA_API = True
except (ImportError, AttributeError):
    try:
        from Model_Def.MResNet50 import MResNet50 as _MResNetCls
        print("Loaded MResNet50 from Carya Model_Def")
        _CARYA_API = True
    except ImportError:
        _MResNetCls = None
        _CARYA_API  = False
        print("WARNING: Could not import from Carya Model_Def — using inline fallback")


# ── Inline fallback (matches paper spec exactly) ──────────────────────────────

class _Bottleneck1D(nn.Module):
    expansion = 4
    def __init__(self, in_ch, mid_ch, stride=1, downsample=None):
        super().__init__()
        out_ch = mid_ch * self.expansion
        self.conv1 = nn.Conv1d(in_ch, mid_ch, 1, bias=False)
        self.bn1   = nn.BatchNorm1d(mid_ch)
        self.conv2 = nn.Conv1d(mid_ch, mid_ch, 3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(mid_ch)
        self.conv3 = nn.Conv1d(mid_ch, out_ch, 1, bias=False)
        self.bn3   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.downsample = downsample
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)

class _XResNet1D50(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 32, 3, padding=1, bias=False),
            nn.BatchNorm1d(32), nn.ReLU(inplace=True),
            nn.Conv1d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm1d(64), nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = self._make_layer(64,   64,  3, stride=1)
        self.layer2 = self._make_layer(256,  128, 4, stride=2)
        self.layer3 = self._make_layer(512,  256, 6, stride=2)
        self.layer4 = self._make_layer(1024, 512, 3, stride=2)
        self.pool   = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 2048
    def _make_layer(self, in_ch, mid_ch, n_blocks, stride):
        out_ch = mid_ch * _Bottleneck1D.expansion
        ds = None
        if stride != 1 or in_ch != out_ch:
            ds = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                               nn.BatchNorm1d(out_ch))
        layers = [_Bottleneck1D(in_ch, mid_ch, stride=stride, downsample=ds)]
        for _ in range(1, n_blocks):
            layers.append(_Bottleneck1D(out_ch, mid_ch))
        return nn.Sequential(*layers)
    def forward(self, x):
        return self.pool(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))).squeeze(-1)

class _MResNet50Fallback(nn.Module):
    """
    MResNet50: XResNet-50 1D backbone + late demographic fusion.
    Paper: Demo(3) -> Linear(16) -> ReLU -> Linear(16) -> ReLU
           Concat(2064) -> Linear(32) -> ReLU -> Linear(1)
    Expects ppg: (B, 1, L), demo: (B, 3)
    """
    def __init__(self, num_static_features=3, num_BP=1):
        super().__init__()
        self.backbone = _XResNet1D50()
        self.demo_mlp = nn.Sequential(
            nn.Linear(num_static_features, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 16), nn.ReLU(inplace=True),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.backbone.out_dim + 16, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_BP),
        )
    def forward(self, ppg, demo):
        # ppg: (B, 1, L) — channel dim already present from data pipeline
        feat = torch.cat([self.backbone(ppg), self.demo_mlp(demo)], dim=-1)
        return self.fusion(feat).squeeze(-1)


def _make_model(bp: str, device) -> nn.Module:
    """Instantiate MResNet50, trying Carya's Model_Def first."""
    if _CARYA_API and _MResNetCls is not None:
        try:
            m = _MResNetCls(num_static_features=3, num_BP=1)
            print(f"Using Carya MResNet50 class: {_MResNetCls.__name__}")
            return m.to(device)
        except TypeError:
            pass
        try:
            m = _MResNetCls(target=bp, demo_dim=3)
            print(f"Using Carya MResNet50 (target={bp}): {_MResNetCls.__name__}")
            return m.to(device)
        except Exception as e:
            print(f"Carya import failed ({e}), falling back to inline implementation")
    print("Using inline MResNet50 fallback")
    return _MResNet50Fallback(num_static_features=3, num_BP=1).to(device)


# =============================================================================
# Settings — identical to train_ms4plus_carya.py
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
    signals = np.expand_dims(D['Subset']['Signals'][:, 1, :], axis=1)  # (N, 1, L)
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

    OUT_DIR = f"./mresnet50_cv_{BP.lower()}"
    os.makedirs(OUT_DIR, exist_ok=True)

    Seed(SEED)
    print(f"BP={BP}  model=MResNet50  out={OUT_DIR}")

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

    N = len(train_data)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    best_cb_list, best_cf_list = [], []

    for fold_id, (train_idx, _) in enumerate(kf.split(np.arange(N)), start=1):
        print(f"\n===== Fold {fold_id}/{N_SPLITS} =====")
        ds_train     = subset_dataset(train_data, train_idx)
        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        Seed(SEED + fold_id)
        _model    = _make_model(BP, device)
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
        f"  {N_SPLITS}-Fold CV Results  |  BP={BP}  |  model=MResNet50",
        f"  Paper targets: Cal-Based MAE {'5.35' if BP=='SBP' else '3.24'} mmHg",
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
