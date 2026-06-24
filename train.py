"""
Training script for MS4Plus (improved S4-based BP estimation model).

Each model predicts a single BP component (SBP or DBP).
Run separately for each:
    python train.py --target SBP --config configs/ms4_improved_SBP.yaml
    python train.py --target DBP --config configs/ms4_improved_DBP.yaml

Key differences from the baseline MS4 training:
    - FiLM demographic conditioning (replaces simple concat)
    - Bidirectional S4D backbone
    - Huber loss (delta=5 mmHg) instead of MSE
    - AdamW with cosine LR schedule + linear warmup
    - Data augmentation during training
    - Gradient clipping
"""
import os
import argparse
import time
import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.ms4_improved import MS4Plus_1D as MS4Plus
from data.pulsedb import PulseDBDataset
from utils.metrics import compute_metrics, print_metrics


# ── Argument parsing ──────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/ms4_improved_SBP.yaml')
    p.add_argument('--target', choices=['SBP', 'DBP'], default=None,
                   help='Override target in config')
    p.add_argument('--data_dir', default=None, help='Override data_dir in config')
    p.add_argument('--out_dir', default=None, help='Override output directory')
    p.add_argument('--resume', default=None, help='Path to checkpoint to resume from')
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


# ── LR schedule: linear warmup + cosine decay ────────────────────────────────

def build_lr_lambda(warmup_epochs: int, total_epochs: int):
    import math
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lr_lambda


# ── Training / validation loops ───────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, device,
              criterion, grad_clip, is_train):
    model.train(is_train)
    total_loss = 0.0
    all_preds, all_targets = [], []

    ctx = torch.enable_grad() if is_train else torch.no_grad()
    with ctx:
        for ppg, demo, label in loader:
            ppg    = ppg.to(device)
            demo   = demo.to(device)
            label  = label.to(device)

            pred = model(ppg, demo)
            loss = criterion(pred, label)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                if grad_clip > 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            total_loss += loss.item() * len(ppg)
            all_preds.extend(pred.detach().cpu().numpy())
            all_targets.extend(label.detach().cpu().numpy())

    if is_train and scheduler is not None:
        scheduler.step()

    metrics = compute_metrics(all_preds, all_targets)
    metrics['loss'] = total_loss / len(loader.dataset)
    return metrics


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    cfg = load_config(args.config)

    # CLI overrides
    if args.target:
        cfg['target'] = args.target
    if args.data_dir:
        cfg['data_dir'] = args.data_dir
    if args.out_dir:
        cfg['out_dir'] = args.out_dir

    target  = cfg['target']
    out_dir = cfg['out_dir']
    os.makedirs(out_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Target: {target}  |  Device: {device}")

    # ── Datasets & loaders ────────────────────────────────────────────────────
    data_dir = cfg['data_dir']
    train_ds = PulseDBDataset(data_dir, split='train',     target=target, augment=True)
    val_ds   = PulseDBDataset(data_dir, split='cal_based', target=target, augment=False)

    bs = cfg.get('batch_size', 64)
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True,
                              num_workers=cfg.get('num_workers', 4),
                              pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=bs * 2, shuffle=False,
                              num_workers=cfg.get('num_workers', 4),
                              pin_memory=True)

    # ── Model ─────────────────────────────────────────────────────────────────
    model = MS4Plus(
        target   = target,
        d_model  = cfg.get('d_model',  128),
        n_layers = cfg.get('n_layers', 6),
        d_state  = cfg.get('d_state',  32),
        dropout  = cfg.get('dropout',  0.1),
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {n_params:,}")

    # ── Loss ──────────────────────────────────────────────────────────────────
    # Huber loss with delta=5 mmHg: squared for |error|<5, linear beyond.
    # More robust than MSE for BP outliers.
    huber_delta = cfg.get('huber_delta', 5.0)
    criterion = nn.SmoothL1Loss(beta=huber_delta)

    # ── Optimiser & schedule ──────────────────────────────────────────────────
    lr           = cfg.get('lr', 3e-4)
    weight_decay = cfg.get('weight_decay', 0.01)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr,
                                   betas=(0.9, 0.999), weight_decay=weight_decay)

    epochs       = cfg.get('epochs', 150)
    warmup       = cfg.get('warmup_epochs', 10)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=build_lr_lambda(warmup, epochs))

    grad_clip = cfg.get('grad_clip', 1.0)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch = 0
    best_mae    = float('inf')

    if args.resume and os.path.isfile(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_mae    = ckpt.get('best_mae', best_mae)
        print(f"Resumed from epoch {start_epoch}, best MAE={best_mae:.4f}")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        t0 = time.time()

        train_m = run_epoch(model, train_loader, optimizer, scheduler,
                            device, criterion, grad_clip, is_train=True)
        val_m   = run_epoch(model, val_loader, optimizer, None,
                            device, criterion, grad_clip, is_train=False)

        elapsed = time.time() - t0
        lr_now  = scheduler.get_last_lr()[0]
        print(f"Epoch {epoch+1:3d}/{epochs}  lr={lr_now:.2e}  t={elapsed:.0f}s")
        print_metrics(train_m, f'Train {target}')
        print_metrics(val_m,   f'Val   {target}')

        # Save best model
        if val_m['mae'] < best_mae:
            best_mae = val_m['mae']
            ckpt_path = os.path.join(out_dir, f'best_ms4plus_{target}.pt')
            torch.save({
                'epoch': epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_mae': best_mae,
                'val_metrics': val_m,
                'cfg': cfg,
            }, ckpt_path)
            print(f"  ★ New best saved: MAE={best_mae:.4f} -> {ckpt_path}")

        # Periodic checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            ckpt_path = os.path.join(out_dir, f'ms4plus_{target}_ep{epoch+1:03d}.pt')
            torch.save({'epoch': epoch, 'model': model.state_dict(), 'cfg': cfg}, ckpt_path)

    print(f"\nTraining complete. Best val MAE ({target}): {best_mae:.4f} mmHg")


if __name__ == '__main__':
    main()
