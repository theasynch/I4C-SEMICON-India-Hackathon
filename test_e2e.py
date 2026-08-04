"""
End-to-end smoke test — validates the entire pipeline works from
data loading through model forward pass, loss computation, and backward.
"""
import sys
import time
import numpy as np
import torch
import yaml

from models.waveseminet import WaveSemiNet, build_waveseminet
from data.dataset import SemiconductorDataset, InferenceDataset, create_dataloaders
from losses.composite_loss import CompositeLoss
from utils.metrics import compute_psnr, compute_ssim

def main():
    print("=" * 60)
    print("  WaveSemiNet — End-to-End Smoke Test")
    print("=" * 60)

    device = torch.device('cpu')

    # 1. Load config
    print("\n[1] Loading config...")
    with open('configs/train_unified.yaml', 'r') as f:
        config = yaml.safe_load(f)
    print("    Config loaded OK")

    # 2. Build model
    print("\n[2] Building model...")
    model = build_waveseminet(config)
    model = model.to(device)
    total_params = model.count_parameters()
    print(f"    Parameters: {total_params:,}")
    branch_params = model.get_branch_params()
    for name, count in branch_params.items():
        print(f"      {name}: {count:,}")

    # 3. Load real data sample
    print("\n[3] Loading dataset...")
    train_ds = SemiconductorDataset(
        noisy_dir='Data-public/train/train/NoisyLR',
        clean_dir='Data-public/train/train/GT',
        patch_size=None,
        augment=False,
        normalize=True,
    )
    print(f"    Train samples: {len(train_ds)}")

    sample = train_ds[0]
    noisy = sample['noisy'].unsqueeze(0).to(device)  # (1, 1, 128, 128)
    clean = sample['clean'].unsqueeze(0).to(device)  # (1, 1, 256, 256)
    print(f"    Noisy: {noisy.shape}, range [{noisy.min():.4f}, {noisy.max():.4f}]")
    print(f"    Clean: {clean.shape}, range [{clean.min():.4f}, {clean.max():.4f}]")
    print(f"    Scale: {sample['scale']}x")

    # 4. Forward pass
    print("\n[4] Forward pass...")
    model.eval()
    with torch.no_grad():
        t0 = time.time()
        pred = model(noisy, task_id=0)
        t1 = time.time()
    print(f"    Output: {pred.shape}, range [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"    Inference time: {(t1-t0)*1000:.1f} ms")

    # 5. Loss computation
    print("\n[5] Computing loss...")
    criterion = CompositeLoss(
        pixel_weight=1.0, freq_weight=0.1,
        edge_weight=0.05, ssim_weight=0.1,
    ).to(device)

    # Resize clean to match pred for loss if sizes differ
    if pred.shape != clean.shape:
        clean_resized = torch.nn.functional.interpolate(
            clean, size=pred.shape[2:], mode='bicubic', align_corners=False
        )
        print(f"    Resized clean: {clean.shape} -> {clean_resized.shape}")
    else:
        clean_resized = clean

    model.train()
    pred_train = model(noisy, task_id=0)
    if pred_train.shape != clean_resized.shape:
        clean_resized = torch.nn.functional.interpolate(
            clean, size=pred_train.shape[2:], mode='bicubic', align_corners=False
        )
    loss, loss_dict = criterion(pred_train, clean_resized)
    print(f"    Loss components:")
    for k, v in loss_dict.items():
        print(f"      {k}: {v:.6f}")

    # 6. Backward pass
    print("\n[6] Backward pass...")
    loss.backward()
    grad_norms = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grad_norms.append((name, p.grad.norm().item()))
    print(f"    Parameters with gradients: {len(grad_norms)}/{total_params}")
    # Show top 5 gradient norms
    grad_norms.sort(key=lambda x: x[1], reverse=True)
    for name, norm in grad_norms[:5]:
        print(f"      {name}: grad_norm={norm:.6f}")

    # 7. Metrics
    print("\n[7] Computing metrics...")
    pred_np = pred.squeeze().cpu().numpy()
    pred_np = np.clip(pred_np, 0.0, 1.0)
    clean_np = clean_resized.squeeze().detach().cpu().numpy()
    clean_np = np.clip(clean_np, 0.0, 1.0)

    psnr = compute_psnr(pred_np, clean_np)
    ssim = compute_ssim(pred_np, clean_np)
    print(f"    PSNR: {psnr:.4f} dB")
    print(f"    SSIM: {ssim:.4f}")

    # 8. Test inference dataset
    print("\n[8] Testing inference dataset...")
    test_ds = InferenceDataset('Data-public/test/NoisyLR')
    print(f"    Test samples: {len(test_ds)}")
    test_sample = test_ds[0]
    print(f"    Test shape: {test_sample['noisy'].shape}")
    print(f"    Test filename: {test_sample['filename']}")

    # 9. Test dataloader creation (with val split from train)
    print("\n[9] Testing dataloader creation...")
    train_loader, val_loader = create_dataloaders(
        train_noisy_dir='Data-public/train/train/NoisyLR',
        train_clean_dir='Data-public/train/train/GT',
        patch_size=128,
        batch_size=4,
        num_workers=0,
    )
    print(f"    Train batches: {len(train_loader)}")
    batch = next(iter(train_loader))
    print(f"    Batch noisy: {batch['noisy'].shape}")
    print(f"    Batch clean: {batch['clean'].shape}")

    print("\n" + "=" * 60)
    print("  ALL TESTS PASSED!")
    print("=" * 60)

if __name__ == "__main__":
    main()
