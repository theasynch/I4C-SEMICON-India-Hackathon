"""
WaveSemiNet — Training Script

Main entry point for training the semiconductor image restoration model.

Usage:
    python train.py --config configs/train_unified.yaml
    python train.py --config configs/train_unified.yaml --resume weights/latest.pth
"""

import os
import sys
import time
import argparse
from pathlib import Path

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from models.waveseminet import WaveSemiNet, build_waveseminet
from losses.composite_loss import CompositeLoss
from data.dataset import SemiconductorDataset, create_dataloaders
from utils.metrics import compute_psnr, compute_ssim


class EMA:
    """Exponential Moving Average of model parameters."""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = (
                    self.decay * self.shadow[name] +
                    (1 - self.decay) * param.data
                )
    
    def apply_shadow(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]


class WarmupScheduler:
    """Linear warmup wrapper for any LR scheduler."""
    
    def __init__(self, optimizer, warmup_epochs: int, start_lr: float,
                 scheduler):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.start_lr = start_lr
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        self.scheduler = scheduler
        self.current_epoch = 0
    
    def step(self, epoch: int):
        self.current_epoch = epoch
        if epoch < self.warmup_epochs:
            # Linear warmup
            alpha = epoch / max(self.warmup_epochs, 1)
            for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                pg['lr'] = self.start_lr + alpha * (base_lr - self.start_lr)
        else:
            self.scheduler.step(epoch - self.warmup_epochs)
    
    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]['lr']


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: CompositeLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    device: torch.device,
    epoch: int,
    config: dict,
    ema: EMA | None = None,
) -> dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    epoch_losses = {k: 0.0 for k in ['total', 'pixel', 'freq', 'edge', 'ssim']}
    num_batches = 0
    
    use_amp = config.get('training', {}).get('mixed_precision', True) and device.type == 'cuda'
    grad_clip = config.get('training', {}).get('gradient_clip', 1.0)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False)
    
    for batch in pbar:
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        with autocast(device_type=device.type, enabled=use_amp):
            pred = model(noisy, task_id=0)
            
            # Handle size mismatch (super-resolution)
            if pred.shape != clean.shape:
                # Resize clean to match pred if needed
                clean = torch.nn.functional.interpolate(
                    clean, size=pred.shape[2:], mode='bicubic',
                    align_corners=False
                )
            
            loss, loss_dict = criterion(pred, clean)
        
        scaler.scale(loss).backward()
        
        if grad_clip > 0:
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        
        scaler.step(optimizer)
        scaler.update()
        
        if ema is not None:
            ema.update(model)
        
        for k, v in loss_dict.items():
            epoch_losses[k] += v
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{loss_dict['total']:.4f}",
            'psnr': f"{10 * np.log10(1.0 / max(loss_dict['pixel'], 1e-8)):.1f}dB"
        })
    
    # Average losses
    for k in epoch_losses:
        epoch_losses[k] /= max(num_batches, 1)
    
    return epoch_losses


@torch.no_grad()
def validate(
    model: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Validate the model and compute metrics."""
    model.eval()
    
    psnr_values = []
    ssim_values = []
    
    for batch in tqdm(val_loader, desc="Validating", leave=False):
        noisy = batch['noisy'].to(device)
        clean = batch['clean'].to(device)
        
        pred = model(noisy, task_id=0)
        
        # Handle size mismatch
        if pred.shape != clean.shape:
            clean = torch.nn.functional.interpolate(
                clean, size=pred.shape[2:], mode='bicubic',
                align_corners=False
            )
        
        # Compute per-image metrics
        pred_np = pred.squeeze().cpu().numpy()
        clean_np = clean.squeeze().cpu().numpy()
        
        pred_np = np.clip(pred_np, 0.0, 1.0)
        clean_np = np.clip(clean_np, 0.0, 1.0)
        
        psnr_values.append(compute_psnr(pred_np, clean_np))
        ssim_values.append(compute_ssim(pred_np, clean_np))
    
    return {
        'psnr': float(np.mean(psnr_values)),
        'ssim': float(np.mean(ssim_values)),
    }


def save_checkpoint(model: nn.Module, optimizer, scheduler, scaler,
                    epoch: int, best_psnr: float, save_path: str,
                    ema: EMA | None = None):
    """Save training checkpoint."""
    state = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'best_psnr': best_psnr,
    }
    if ema is not None:
        state['ema_shadow'] = ema.shadow
    
    torch.save(state, save_path)


def main():
    parser = argparse.ArgumentParser(description='WaveSemiNet Training')
    parser.add_argument('--config', type=str, default='configs/train_unified.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Device setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create model
    model = build_waveseminet(config)
    model = model.to(device)
    print(f"Model parameters: {model.count_parameters():,}")
    
    # Create dataloaders
    data_cfg = config.get('data', {})
    train_dir = data_cfg.get('train_dir', 'data/train')
    val_dir = data_cfg.get('val_dir', None)
    batch_size = data_cfg.get('batch_size', 8)
    num_workers = data_cfg.get('num_workers', 4)
    patch_size = data_cfg.get('patch_size', 128)
    
    if val_dir:
        # Explicit val directory
        train_loader, val_loader = create_dataloaders(
            train_noisy_dir=os.path.join(train_dir, 'NoisyLR'),
            train_clean_dir=os.path.join(train_dir, 'GT'),
            val_noisy_dir=os.path.join(val_dir, 'NoisyLR'),
            val_clean_dir=os.path.join(val_dir, 'GT'),
            patch_size=patch_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )
    else:
        # Split training data 90/10 for validation
        from torch.utils.data import random_split
        full_dataset = SemiconductorDataset(
            noisy_dir=os.path.join(train_dir, 'NoisyLR'),
            clean_dir=os.path.join(train_dir, 'GT'),
            patch_size=patch_size,
            augment=True,
            normalize=True,
        )
        val_size = max(1, len(full_dataset) // 10)
        train_size = len(full_dataset) - val_size
        train_subset, val_subset = random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        # Disable augmentation for validation subset by wrapping
        train_loader = DataLoader(
            train_subset, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        val_loader = DataLoader(
            val_subset, batch_size=1, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )
    
    print(f"Training samples: {len(train_loader.dataset)}")
    if val_loader:
        print(f"Validation samples: {len(val_loader.dataset)}")
    
    # Loss function
    loss_cfg = config.get('loss', {})
    criterion = CompositeLoss(
        pixel_weight=loss_cfg.get('pixel', {}).get('weight', 1.0),
        freq_weight=loss_cfg.get('frequency', {}).get('weight', 0.1),
        edge_weight=loss_cfg.get('edge', {}).get('weight', 0.05),
        ssim_weight=loss_cfg.get('ssim', {}).get('weight', 0.1),
    ).to(device)
    
    # Optimizer
    train_cfg = config.get('training', {})
    opt_cfg = train_cfg.get('optimizer', {})
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(opt_cfg.get('lr', 1e-3)),
        weight_decay=float(opt_cfg.get('weight_decay', 1e-4)),
        betas=tuple(opt_cfg.get('betas', [0.9, 0.999])),
    )
    
    # Scheduler
    sched_cfg = train_cfg.get('scheduler', {})
    base_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=sched_cfg.get('T_0', 50),
        T_mult=sched_cfg.get('T_mult', 2),
        eta_min=float(sched_cfg.get('eta_min', 1e-6)),
    )
    
    warmup_cfg = train_cfg.get('warmup', {})
    scheduler = WarmupScheduler(
        optimizer,
        warmup_epochs=warmup_cfg.get('epochs', 5),
        start_lr=float(warmup_cfg.get('start_lr', 1e-6)),
        scheduler=base_scheduler,
    )
    
    # Mixed precision scaler
    use_amp = train_cfg.get('mixed_precision', True) and device.type == 'cuda'
    scaler = GradScaler(device.type, enabled=use_amp)
    
    # EMA
    ema = None
    if train_cfg.get('ema', True):
        ema = EMA(model, decay=train_cfg.get('ema_decay', 0.999))
    
    # Resume from checkpoint
    start_epoch = 0
    best_psnr = 0.0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0.0)
        if ema and 'ema_shadow' in checkpoint:
            ema.shadow = checkpoint['ema_shadow']
        print(f"Resumed from epoch {start_epoch}, best PSNR: {best_psnr:.4f}")
    
    # Create save directory
    log_cfg = config.get('logging', {})
    save_dir = Path(log_cfg.get('save_dir', 'weights'))
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    epochs = train_cfg.get('epochs', 200)
    
    print(f"\nStarting training for {epochs} epochs...")
    print(f"Config: {args.config}")
    print("-" * 60)
    
    for epoch in range(start_epoch, epochs):
        scheduler.step(epoch)
        lr = scheduler.get_lr()
        
        # Train
        train_losses = train_one_epoch(
            model, train_loader, criterion, optimizer,
            scaler, device, epoch, config, ema
        )
        
        # Log
        print(f"Epoch {epoch:3d}/{epochs} | "
              f"LR: {lr:.2e} | "
              f"Loss: {train_losses['total']:.4f} | "
              f"Pixel: {train_losses['pixel']:.4f} | "
              f"Freq: {train_losses['freq']:.4f} | "
              f"Edge: {train_losses['edge']:.4f}")
        
        # Validate
        val_every = log_cfg.get('val_every', 1)
        if val_loader and (epoch + 1) % val_every == 0:
            # Use EMA weights for validation
            if ema:
                ema.apply_shadow(model)
            
            val_metrics = validate(model, val_loader, device)
            
            if ema:
                ema.restore(model)
            
            print(f"  Val PSNR: {val_metrics['psnr']:.4f} | "
                  f"Val SSIM: {val_metrics['ssim']:.4f}")
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                save_checkpoint(model, optimizer, scheduler, scaler,
                               epoch, best_psnr,
                               str(save_dir / 'best.pth'), ema)
                print(f"  ★ New best PSNR: {best_psnr:.4f}")
        
        # Save periodic checkpoint
        save_every = log_cfg.get('save_every', 10)
        if (epoch + 1) % save_every == 0:
            save_checkpoint(model, optimizer, scheduler, scaler,
                           epoch, best_psnr,
                           str(save_dir / 'latest.pth'), ema)
    
    # Save final model
    save_checkpoint(model, optimizer, scheduler, scaler,
                   epochs - 1, best_psnr,
                   str(save_dir / 'final.pth'), ema)
    
    print(f"\nTraining complete! Best PSNR: {best_psnr:.4f}")
    print(f"Weights saved to: {save_dir}")


if __name__ == "__main__":
    main()
