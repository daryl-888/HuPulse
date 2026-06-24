"""
PulseDB dataset loader.

PulseDB (Wang et al., 2023) stores data as MATLAB HDF5 (.mat) files.
The `explore_data.py` script should be run first to confirm the exact
keys and layout for the version on Carya.

Expected per-file contents (based on PulseDB paper):
    PPG     : (N, 1250)  — 10-second segments at 125 Hz
    SBP     : (N,)       — systolic BP label (mmHg)
    DBP     : (N,)       — diastolic BP label (mmHg)
    AGE     : (N,)       — subject age (years)
    SEX     : (N,)       — sex (0=Female, 1=Male, or similar encoding)
    BMI     : (N,)       — body mass index (kg/m²)
    SubjectID: (N,)      — subject identifier (for cal-based split)

Demographics are normalised in __getitem__:
    age  / 100
    sex  (kept as 0/1)
    bmi  / 40

Augmentation (training only):
    - Random amplitude scaling ∈ [0.9, 1.1]
    - Additive Gaussian noise  (std ≈ 1% of signal std)
    - Random baseline wander   (low-freq sine added)
"""
import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset


# ── Candidate key names for each field ────────────────────────────────────────
_PPG_KEYS  = ['PPG', 'ppg', 'signal', 'waveform', 'x', 'data']
_SBP_KEYS  = ['SBP', 'sbp', 'sys', 'systolic']
_DBP_KEYS  = ['DBP', 'dbp', 'dia', 'diastolic']
_AGE_KEYS  = ['AGE', 'age', 'Age']
_SEX_KEYS  = ['SEX', 'sex', 'Sex', 'gender', 'Gender']
_BMI_KEYS  = ['BMI', 'bmi', 'Bmi']
_SUBJ_KEYS = ['SubjectID', 'subject_id', 'Subject', 'subjectID', 'ID']


def _find_key(d: dict, candidates: list):
    for k in candidates:
        if k in d:
            return np.asarray(d[k]).squeeze()
    return None


def _load_mat(path: str) -> dict:
    """Try mat73 (HDF5 .mat) then scipy.io (legacy .mat)."""
    try:
        import mat73
        return mat73.loadmat(path)
    except Exception:
        pass
    try:
        import scipy.io as sio
        return sio.loadmat(path, squeeze_me=True, struct_as_record=False)
    except Exception:
        pass
    try:
        import h5py
        out = {}
        with h5py.File(path, 'r') as f:
            def visitor(name, obj):
                if isinstance(obj, __import__('h5py').Dataset):
                    out[name.split('/')[-1]] = obj[()]
            f.visititems(visitor)
        return out
    except Exception as e:
        raise RuntimeError(f"Cannot load {path}: {e}")


def _load_split_files(data_dir: str, split: str) -> dict:
    """
    Search `data_dir` for files matching `split` and load them.

    Split aliases (case-insensitive):
        train      → train, Train, training
        cal_based  → CalBased, cal_based, calibration_based, test
        cal_free   → CalFree, cal_free, calibration_free, val
    """
    aliases = {
        'train':     ['train', 'Train', 'training', 'Train_data'],
        'cal_based': ['CalBased', 'cal_based', 'calibration_based', 'test', 'Test'],
        'cal_free':  ['CalFree', 'cal_free', 'calibration_free', 'val', 'Val'],
    }
    names = aliases.get(split, [split])

    # 1) Look for a sub-directory named after the split
    for name in names:
        sub = os.path.join(data_dir, name)
        if os.path.isdir(sub):
            files = sorted(glob.glob(os.path.join(sub, '*.mat')) +
                           glob.glob(os.path.join(sub, '*.h5')))
            if files:
                return _merge_files(files)

    # 2) Look for a single file named after the split
    for name in names:
        for ext in ('.mat', '.h5', '.hdf5'):
            fpath = os.path.join(data_dir, name + ext)
            if os.path.isfile(fpath):
                return _load_mat(fpath)

    # 3) Glob all .mat files at root and filter by split keyword
    root_mats = sorted(glob.glob(os.path.join(data_dir, '*.mat')) +
                       glob.glob(os.path.join(data_dir, '*.h5')))
    matched = [f for f in root_mats
               if any(n.lower() in os.path.basename(f).lower() for n in names)]
    if matched:
        return _merge_files(matched)

    raise FileNotFoundError(
        f"Cannot find '{split}' split in {data_dir}. "
        f"Looked for subdirs/files: {names}. "
        f"Run `scripts/explore_data.py` to inspect the directory layout."
    )


def _merge_files(paths: list) -> dict:
    """Load and concatenate a list of .mat files along axis 0."""
    parts = [_load_mat(p) for p in paths]
    merged = {}
    for key in parts[0]:
        arrays = []
        for p in parts:
            if key in p:
                arr = np.asarray(p[key])
                if arr.ndim == 0:
                    continue
                arrays.append(arr)
        if arrays:
            merged[key] = np.concatenate(arrays, axis=0)
    return merged


class PulseDBDataset(Dataset):
    """
    PyTorch Dataset for PulseDB.

    Args:
        data_dir: Path to a PulseDB split directory (e.g. PulseDB_multi_full/).
        split:    One of 'train', 'cal_based', 'cal_free'.
        target:   'SBP' or 'DBP'.
        augment:  If True, apply signal augmentation (use for training only).
        seg_len:  Expected PPG segment length (samples). If the loaded segments
                  are longer, a random crop of this length is taken.
    """

    def __init__(self,
                 data_dir: str,
                 split: str = 'train',
                 target: str = 'SBP',
                 augment: bool = False,
                 seg_len: int = 1250):
        super().__init__()
        assert target in ('SBP', 'DBP')
        self.target = target
        self.augment = augment
        self.seg_len = seg_len

        raw = _load_split_files(data_dir, split)

        self.ppg = self._extract_ppg(raw)   # (N, seg_len) float32
        self.sbp = self._extract_field(raw, _SBP_KEYS, 'SBP')  # (N,)
        self.dbp = self._extract_field(raw, _DBP_KEYS, 'DBP')  # (N,)
        self.age = self._extract_field(raw, _AGE_KEYS, 'AGE')  # (N,)
        self.sex = self._extract_field(raw, _SEX_KEYS, 'SEX')  # (N,)
        self.bmi = self._extract_field(raw, _BMI_KEYS, 'BMI')  # (N,)

        n = len(self.ppg)
        print(f"[PulseDBDataset] split={split} target={target} "
              f"N={n} ppg.shape={self.ppg.shape} "
              f"SBP {self.sbp.mean():.1f}±{self.sbp.std():.1f}  "
              f"DBP {self.dbp.mean():.1f}±{self.dbp.std():.1f}")

    # ── Helpers ────────────────────────────────────────────────────────────────

    def _extract_ppg(self, raw: dict) -> np.ndarray:
        arr = _find_key(raw, _PPG_KEYS)
        if arr is None:
            raise KeyError(f"PPG not found. Available keys: {list(raw.keys())}")
        arr = np.asarray(arr, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[-1] < arr.shape[0]:
            arr = arr.T  # ensure (N, L)
        # Crop or pad to seg_len
        L = arr.shape[1]
        if L > self.seg_len:
            # Random crop during __getitem__, store full length here
            pass
        elif L < self.seg_len:
            pad = self.seg_len - L
            arr = np.pad(arr, ((0, 0), (0, pad)), mode='edge')
        return arr

    def _extract_field(self, raw: dict, keys: list, name: str) -> np.ndarray:
        arr = _find_key(raw, keys)
        if arr is None:
            raise KeyError(f"Field '{name}' not found. Keys: {list(raw.keys())}")
        return np.asarray(arr, dtype=np.float32).ravel()

    # ── Dataset protocol ───────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self.ppg)

    def __getitem__(self, idx: int):
        ppg = self.ppg[idx].copy()  # (L,)
        L = ppg.shape[0]

        # Random crop if longer than seg_len
        if L > self.seg_len:
            start = np.random.randint(0, L - self.seg_len + 1)
            ppg = ppg[start: start + self.seg_len]

        if self.augment:
            ppg = _augment_ppg(ppg)

        # Normalised demographics: [age/100, sex, bmi/40]
        demo = np.array([
            self.age[idx] / 100.0,
            float(self.sex[idx]),
            self.bmi[idx] / 40.0,
        ], dtype=np.float32)

        label = self.sbp[idx] if self.target == 'SBP' else self.dbp[idx]

        return (
            torch.from_numpy(ppg),            # (seg_len,)
            torch.from_numpy(demo),            # (3,)
            torch.tensor(label, dtype=torch.float32),  # scalar
        )


# ── Augmentation ──────────────────────────────────────────────────────────────

def _augment_ppg(ppg: np.ndarray) -> np.ndarray:
    """Light augmentation to improve generalisation."""
    L = len(ppg)

    # Amplitude scaling ∈ [0.9, 1.1]
    scale = np.random.uniform(0.9, 1.1)
    ppg = ppg * scale

    # Additive Gaussian noise (SNR ~35 dB)
    noise_std = ppg.std() * 0.02
    ppg = ppg + np.random.randn(L).astype(np.float32) * noise_std

    # Baseline wander: low-frequency sine (< 0.5 Hz; 125 Hz sampling)
    freq = np.random.uniform(0.05, 0.5)
    phase = np.random.uniform(0, 2 * np.pi)
    t = np.arange(L) / 125.0
    amp = ppg.std() * np.random.uniform(0.0, 0.05)
    ppg = ppg + amp * np.sin(2 * np.pi * freq * t + phase).astype(np.float32)

    return ppg
