# File: Model_training_bootstrap.py
import os
import csv
import random
import numpy as np
from typing import Dict

import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from mat73 import loadmat
from sklearn.model_selection import KFold

# ??????
from Model_Def import MResNet18, MResNet50, Mtransformer, Minception, MS4, MLenet

# =========================
# ?????
# =========================
DATA_FOLDER = "/home/yshen28/PPGdata_filtered/"
TRAIN_FILE = os.path.join(DATA_FOLDER, "Train_Subset_filtered.mat")
TEST_CALBASED_FILE = os.path.join(DATA_FOLDER, "CalBased_Test_Subset_filtered.mat")
TEST_CALFREE_FILE = os.path.join(DATA_FOLDER, "CalFree_Test_Subset_filtered.mat")

BP = "SBP"          # "SBP" ? "DBP"
SEED = 6
N_SPLITS = 10
BATCH_SIZE = 32
NUM_EPOCHS = 100
LR = 2e-5
WEIGHT_DECAY = 1e-8
BETAS = (0.9, 0.999)

MODEL_NAME = "ms4"  # ??: "ResNet18","ResNet50","Transformer","Inception","S4","LeNet"

OUT_DIR = "./bootstrap_cv_results"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# ????
# =========================
def encode_gender(gender_arr):
    return np.array([1 if str(g).upper().startswith('M') else 0 for g in gender_arr], dtype=np.float32)

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
        self.Input = Input   # (N, 1, seq_len)
        self.Age = Age       # (N,)
        self.BMI = BMI       # (N,)
        self.Gender = Gender # (N,)
        self.Label = Label   # (N,)

    def __len__(self):
        return len(self.Input)

    def __getitem__(self, idx):
        static_feat = np.array([self.Age[idx], self.BMI[idx], self.Gender[idx]], dtype=np.float32)
        return (self.Input[idx, :].astype(np.float32), static_feat), self.Label[idx]

def Build_Dataset(Path, Label, show_info=True):
    Data = loadmat(Path)
    signals = np.expand_dims(Data['Subset']['Signals'][:, 1, :], axis=1)    # (N, 1, seq_len)
    age = np.array(Data['Subset']['Age'], dtype=np.float32).flatten()
    bmi = np.array(Data['Subset']['BMI'], dtype=np.float32).flatten()
    gender_raw = Data['Subset']['Gender']
    gender = encode_gender(gender_raw)
    label = np.array(Data['Subset'][Label], dtype=np.float32).flatten()

    N = len(age)
    before_num = N
    mask = np.ones(N, dtype=bool)
    for i in range(N):
        features = [age[i], bmi[i], gender[i]]
        if (
            np.any(np.isnan(features)) or np.any(np.isinf(features)) or
            np.any(np.isnan(signals[i])) or np.any(np.isinf(signals[i]))
        ):
            mask[i] = False

    signals = signals[mask]
    age = age[mask]
    bmi = bmi[mask]
    gender = gender[mask]
    label = label[mask]
    after_num = len(age)

    if show_info:
        print(f"{Path}: Before removing nan/inf: {before_num}, After: {after_num}")

    return Dataset(signals, age, bmi, gender, label)

def subset_dataset(ds: Dataset, indices: np.ndarray) -> Dataset:
    return Dataset(ds.Input[indices], ds.Age[indices], ds.BMI[indices], ds.Gender[indices], ds.Label[indices])

def to_device(batch, device):
    (sig, sf), y = batch
    if isinstance(sig, np.ndarray): sig = torch.from_numpy(sig)
    if isinstance(sf, np.ndarray):  sf = torch.from_numpy(sf)
    if isinstance(y, (np.ndarray, list, float)): y = torch.from_numpy(np.array(y, dtype=np.float32))
    return (sig.to(device), sf.to(device)), y.to(device).view(-1)

def accepts_two_inputs(model) -> bool:
    # ??????? forward(signal, static_feat)????? signal ???,?????? model.expects_two_inputs=False
    return getattr(model, "expects_two_inputs", True)

def get_model_instance(name: str, num_static_features=3, num_BP=1):
    n = name.lower()
    if n in ["lenet","mlenet","lenet1d","m-lenet"]:
        return MLenet.MLeNet1d(num_static_features=num_static_features, num_BP=num_BP)
    if n in ["s4","ms4","s4_1d"]:
        return MS4.MS4_1D(num_static_features=num_static_features, num_BP=num_BP)
    if n in ["inception","minception","inception1d"]:
        return Minception.MInception1d(num_static_features=num_static_features, num_BP=num_BP)
    if n in ["resnet18","mresnet18","resnet18_1d"]:
        return MResNet18.MResnet18_1D(num_static_features=num_static_features, num_BP=num_BP)
    if n in ["resnet50","mresnet50","resnet50_1d"]:
        return MResNet50.MResnet50_1D(num_static_features=num_static_features, num_BP=num_BP)
    if n in ["transformer","vit","vit1d"]:
        return Mtransformer.ViT1D_Fusion_Model(
            in_channels=1, seq_len=1250, patch_size=10, emb_dim=128,
            depth=6, nhead=8, mlp_ratio=4, num_BP=num_BP, dropout=0.1, num_static_features=num_static_features
        )
    raise ValueError(f"Unknown model: {name}")

# ===== ?? =====
def mae(y_true, y_pred): return float(np.mean(np.abs(y_true - y_pred)))
def rmse(y_true, y_pred): return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
def r2_score(y_true, y_pred):
    ybar = np.mean(y_true)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - ybar) ** 2)
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
def abs_err_std(y_true, y_pred): return float(np.std(np.abs(y_true - y_pred), ddof=1)) if len(y_true) > 1 else 0.0

@torch.no_grad()
def evaluate(model, ds: Dataset, device, batch_size=256) -> Dict[str, float]:
    model.eval()
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    ys, ps = [], []
    for batch in loader:
        (sig, sf), y = to_device(batch, device)
        out = model(sig, sf) if accepts_two_inputs(model) else model(sig)
        out = out.view(-1).detach().cpu().numpy()
        y = y.view(-1).detach().cpu().numpy()
        ys.append(y); ps.append(out)
    y_true = np.concatenate(ys); y_pred = np.concatenate(ps)
    return {
        "MAE": mae(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "STD": abs_err_std(y_true, y_pred)
    }

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    running = 0.0
    n = 0
    for batch in loader:
        (sig, sf), y = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        out = model(sig, sf) if accepts_two_inputs(model) else model(sig)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        bs = y.size(0)
        running += loss.item() * bs
        n += bs
    return running / max(n, 1)

# =========================
# ???:K ? + ?? epoch ?? Cal-Based/Cal-Free ?????
# =========================
def main():
    # ??
    Seed(SEED)
    Train_Data = Build_Dataset(TRAIN_FILE, BP)
    Test_CalBased_Data = Build_Dataset(TEST_CALBASED_FILE, BP)
    Test_CalFree_Data = Build_Dataset(TEST_CALFREE_FILE, BP)

    # ??
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    if torch.cuda.is_available():
        print(torch.cuda.get_device_name(0))

    # ?? CSV
    per_epoch_csv = os.path.join(OUT_DIR, f"{BP}_per_epoch_metrics.csv")
    with open(per_epoch_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold","epoch",
            "train_loss",
            "calbased_MAE","calbased_RMSE","calbased_R2","calbased_STD",
            "calfree_MAE","calfree_RMSE","calfree_R2","calfree_STD"
        ])

    per_fold_best_csv = os.path.join(OUT_DIR, f"{BP}_per_fold_best.csv")
    with open(per_fold_best_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold",
            "best_calbased_MAE","best_calbased_epoch","best_calbased_STD","best_calbased_R2",
            "best_calfree_MAE","best_calfree_epoch","best_calfree_STD","best_calfree_R2"
        ])

    # K ?
    N = len(Train_Data)
    indices = np.arange(N)
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)

    # ??(??)
    best_cb_list = []
    best_cf_list = []

    for fold_id, (train_idx, _val_idx) in enumerate(kf.split(indices), start=1):
        print(f"\n===== Fold {fold_id}/{N_SPLITS} =====")
        ds_train = subset_dataset(Train_Data, train_idx)

        train_loader = DataLoader(ds_train, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

        # ?????????????
        Seed(SEED + fold_id)
        model = get_model_instance(MODEL_NAME, num_static_features=3, num_BP=1).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=LR, betas=BETAS, weight_decay=WEIGHT_DECAY)
        criterion = nn.MSELoss()
        
        best_calbased = {"MAE": float("inf"), "epoch": -1, "STD": None, "R2": None}
        best_calfree  = {"MAE": float("inf"), "epoch": -1, "STD": None, "R2": None}

        # ????
        for epoch in range(1, NUM_EPOCHS + 1):
            train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)

            # ?? epoch ? Cal-Based / Cal-Free ???
            cb_metrics = evaluate(model, Test_CalBased_Data, device)
            cf_metrics = evaluate(model, Test_CalFree_Data, device)

            # ?? per-epoch CSV
            with open(per_epoch_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    fold_id, epoch,
                    f"{train_loss:.6f}",
                    f"{cb_metrics['MAE']:.6f}", f"{cb_metrics['RMSE']:.6f}", f"{cb_metrics['R2']:.6f}", f"{cb_metrics['STD']:.6f}",
                    f"{cf_metrics['MAE']:.6f}", f"{cf_metrics['RMSE']:.6f}", f"{cf_metrics['R2']:.6f}", f"{cf_metrics['STD']:.6f}",
                ])

            if cb_metrics["MAE"] < best_calbased["MAE"]:
                best_calbased = {
                    "MAE": cb_metrics["MAE"],
                    "epoch": epoch,
                    "STD": cb_metrics["STD"],
                    "R2": cb_metrics["R2"]
                }
                fold_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
                os.makedirs(fold_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_calbased.pth"))

            if cf_metrics["MAE"] < best_calfree["MAE"]:
                best_calfree = {
                    "MAE": cf_metrics["MAE"],
                    "epoch": epoch,
                    "STD": cf_metrics["STD"],
                    "R2": cf_metrics["R2"]
                }
                fold_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
                os.makedirs(fold_dir, exist_ok=True)
                torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_best_calfree.pth"))


        with open(per_fold_best_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                fold_id,
                f"{best_calbased['MAE']:.6f}", best_calbased["epoch"], f"{best_calbased['STD']:.6f}", f"{best_calbased['R2']:.6f}",
                f"{best_calfree['MAE']:.6f}", best_calfree["epoch"], f"{best_calfree['STD']:.6f}", f"{best_calfree['R2']:.6f}",
            ])

        best_cb_list.append(best_calbased["MAE"])
        best_cf_list.append(best_calfree["MAE"])

        fold_dir = os.path.join(OUT_DIR, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(fold_dir, f"{BP}_fold{fold_id}_final.pth"))

        print(f"[Fold {fold_id}] Best Cal-Based MAE={best_calbased['MAE']:.4f} @ epoch {best_calbased['epoch']}, "
              f"STD={best_calbased['STD']:.4f}, R2={best_calbased['R2']:.4f}")
        print(f"[Fold {fold_id}] Best Cal-Free  MAE={best_calfree['MAE']:.4f} @ epoch {best_calfree['epoch']}, "
              f"STD={best_calfree['STD']:.4f}, R2={best_calfree['R2']:.4f}")

    summary_path = os.path.join(OUT_DIR, f"{BP}_cv_summary.txt")
    with open(summary_path, "w") as f:
        cb_arr = np.array(best_cb_list, dtype=np.float32)
        cf_arr = np.array(best_cf_list, dtype=np.float32)
        f.write(f"{N_SPLITS}-Fold CV (task={BP})\n")
        f.write(f"Best Cal-Based MAE over folds: mean={cb_arr.mean():.4f}, std={cb_arr.std(ddof=1):.4f}\n")
        f.write(f"Best Cal-Free  MAE over folds: mean={cf_arr.mean():.4f}, std={cf_arr.std(ddof=1):.4f}\n")

    print("\n=== CV Summary ===")
    print(open(summary_path, "r").read())


if __name__ == "__main__":
    main()
