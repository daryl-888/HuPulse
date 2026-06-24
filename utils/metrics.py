"""
Evaluation metrics for blood pressure estimation.

Primary metric: MAE (mean absolute error) in mmHg.
AAMI/ISO 81060-2 standard: mean error ≤ 5 mmHg AND std ≤ 8 mmHg.
"""
import numpy as np
import torch


def compute_metrics(preds, targets):
    """
    Args:
        preds, targets: array-like of shape (N,), values in mmHg
    Returns:
        dict with keys: mae, std, mean_error, r2, n
    """
    preds = np.asarray(preds, dtype=float)
    targets = np.asarray(targets, dtype=float)
    errors = preds - targets
    mae = float(np.mean(np.abs(errors)))
    std = float(np.std(errors))
    mean_error = float(np.mean(errors))
    ss_res = np.sum(errors ** 2)
    ss_tot = np.sum((targets - targets.mean()) ** 2)
    r2 = float(1.0 - ss_res / (ss_tot + 1e-12))
    return {'mae': mae, 'std': std, 'mean_error': mean_error, 'r2': r2, 'n': len(preds)}


def aami_check(metrics: dict) -> bool:
    """Return True if results meet AAMI/ISO 81060-2 (|mean| ≤ 5, std ≤ 8)."""
    return abs(metrics['mean_error']) <= 5.0 and metrics['std'] <= 8.0


def print_metrics(metrics: dict, prefix: str = ''):
    tag = f"[{prefix}] " if prefix else ''
    aami = '✓ AAMI' if aami_check(metrics) else '✗ below AAMI'
    print(f"{tag}MAE={metrics['mae']:.3f}  std={metrics['std']:.3f}  "
          f"mean_err={metrics['mean_error']:.3f}  R²={metrics['r2']:.3f}  "
          f"N={metrics['n']}  {aami}")
