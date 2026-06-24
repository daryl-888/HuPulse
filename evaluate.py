"""
Evaluate a trained MS4Plus checkpoint on cal_based and cal_free splits.

Usage:
    python evaluate.py --ckpt results/best_ms4plus_SBP.pt
    python evaluate.py --ckpt results/best_ms4plus_DBP.pt --data_dir /path/to/data
"""
import os
import argparse
import numpy as np
import torch
from torch.utils.data import DataLoader

from models.ms4_improved import MS4Plus
from data.pulsedb import PulseDBDataset
from utils.metrics import compute_metrics, print_metrics, aami_check


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True)
    p.add_argument('--data_dir', default=None)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--num_workers', type=int, default=4)
    return p.parse_args()


def evaluate_split(model, data_dir, split, target, batch_size, num_workers, device):
    ds = PulseDBDataset(data_dir, split=split, target=target, augment=False)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True)
    preds, targets = [], []
    model.eval()
    with torch.no_grad():
        for ppg, demo, label in loader:
            pred = model(ppg.to(device), demo.to(device))
            preds.extend(pred.cpu().numpy())
            targets.extend(label.numpy())
    return compute_metrics(preds, targets)


def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt = torch.load(args.ckpt, map_location=device)
    cfg  = ckpt.get('cfg', {})
    target = cfg.get('target', 'SBP')
    data_dir = args.data_dir or cfg.get('data_dir')

    if not data_dir:
        raise ValueError('Provide --data_dir or ensure it is stored in the checkpoint cfg.')

    model = MS4Plus(
        target   = target,
        d_model  = cfg.get('d_model',  128),
        n_layers = cfg.get('n_layers', 6),
        d_state  = cfg.get('d_state',  32),
        dropout  = 0.0,
    ).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"\n{'='*60}")
    print(f"Checkpoint : {args.ckpt}")
    print(f"Target     : {target}")
    print(f"Data dir   : {data_dir}")
    print(f"Best epoch : {ckpt.get('epoch', '?')}")
    print(f"{'='*60}\n")

    for split in ('cal_based', 'cal_free'):
        try:
            m = evaluate_split(model, data_dir, split, target,
                               args.batch_size, args.num_workers, device)
            print_metrics(m, f'{split:10s} {target}')
        except FileNotFoundError as e:
            print(f"[{split}] Skipped: {e}")

    print()


if __name__ == '__main__':
    main()
