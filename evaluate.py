"""
WaveSemiNet — Evaluation Script

This is the script that hackathon judges will run.
It must work out-of-the-box with minimal setup.

Usage:
    python evaluate.py --weights weights/best.pth --data Data-public/test/NoisyLR --output results/

The script:
1. Loads the trained WaveSemiNet model
2. Processes all test images
3. Saves restored images as .npy files
4. Computes PSNR/SSIM/LPIPS if ground truth is available
5. Generates comparison visualizations
6. Prints a summary metrics table
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from models.waveseminet import WaveSemiNet, build_waveseminet
from data.dataset import InferenceDataset
from utils.metrics import MetricsCalculator, compute_psnr, compute_ssim


def load_model(weights_path: str, device: torch.device,
               config: dict | None = None) -> WaveSemiNet:
    """
    Load a trained WaveSemiNet model from checkpoint.
    
    Args:
        weights_path: Path to .pth checkpoint file
        device: Target device
        config: Model config (if None, uses defaults)
    
    Returns:
        Loaded model in eval mode
    """
    model = build_waveseminet(config)
    
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    # Handle different checkpoint formats
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Handle EMA weights if available
    if 'ema_shadow' in checkpoint:
        print("Using EMA weights for evaluation")
        # Reconstruct state dict from EMA shadow
        ema_shadow = checkpoint['ema_shadow']
        for name in list(state_dict.keys()):
            if name in ema_shadow:
                state_dict[name] = ema_shadow[name]
    
    model.load_state_dict(state_dict, strict=False)
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def restore_image(model: WaveSemiNet, noisy: torch.Tensor,
                  device: torch.device,
                  tile_size: int | None = None) -> torch.Tensor:
    """
    Restore a single image. Supports tiled inference for large images.
    
    Args:
        model: Trained WaveSemiNet model
        noisy: Degraded image tensor (1, 1, H, W)
        device: Computation device
        tile_size: If set, use tiled inference (for memory efficiency)
    
    Returns:
        Restored image tensor (1, 1, H_out, W_out)
    """
    noisy = noisy.to(device)
    
    if tile_size is None or (noisy.shape[2] <= tile_size and
                              noisy.shape[3] <= tile_size):
        # Direct inference
        restored = model(noisy, task_id=0)
    else:
        # Tiled inference with overlap
        overlap = 16
        restored = tiled_inference(model, noisy, tile_size, overlap)
    
    return restored.clamp(0, 1)


def tiled_inference(model: WaveSemiNet, noisy: torch.Tensor,
                    tile_size: int, overlap: int = 16) -> torch.Tensor:
    """
    Process large images in overlapping tiles to avoid OOM.
    Blends overlapping regions with linear weights.
    Handles super-resolution (output larger than input).
    """
    B, C, H, W = noisy.shape
    scale = getattr(model, 'scale_factor', 1)
    step = tile_size - overlap
    
    # Output may be larger than input (super-resolution)
    out_H, out_W = H * scale, W * scale
    output = torch.zeros(B, C, out_H, out_W, device=noisy.device)
    weight = torch.zeros(B, C, out_H, out_W, device=noisy.device)
    
    for y in range(0, H, step):
        for x in range(0, W, step):
            y_end = min(y + tile_size, H)
            x_end = min(x + tile_size, W)
            y_start = max(y_end - tile_size, 0)
            x_start = max(x_end - tile_size, 0)
            
            tile = noisy[:, :, y_start:y_end, x_start:x_end]
            restored_tile = model(tile, task_id=0)
            
            # Map input coordinates to output coordinates
            oy_start = y_start * scale
            oy_end = oy_start + restored_tile.shape[2]
            ox_start = x_start * scale
            ox_end = ox_start + restored_tile.shape[3]
            
            output[:, :, oy_start:oy_end, ox_start:ox_end] += restored_tile
            weight[:, :, oy_start:oy_end, ox_start:ox_end] += 1.0
    
    return output / weight.clamp(min=1.0)


def save_visualization(noisy: np.ndarray, restored: np.ndarray,
                       clean: np.ndarray | None, filename: str,
                       save_path: str):
    """Save side-by-side comparison visualization."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    n_cols = 3 if clean is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(6 * n_cols, 6))
    
    axes[0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Degraded Input', fontsize=12)
    axes[0].axis('off')
    
    axes[1].imshow(restored, cmap='gray', vmin=0, vmax=1)
    axes[1].set_title('Restored (WaveSemiNet)', fontsize=12)
    axes[1].axis('off')
    
    if clean is not None and n_cols == 3:
        axes[2].imshow(clean, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Ground Truth', fontsize=12)
        axes[2].axis('off')
    
    plt.suptitle(filename, fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='WaveSemiNet Evaluation — Semiconductor Image Restoration'
    )
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights (.pth)')
    parser.add_argument('--data', type=str, required=True,
                        help='Path to test images directory (NoisyLR folder)')
    parser.add_argument('--gt', type=str, default=None,
                        help='Path to ground truth directory (CleanHR folder)')
    parser.add_argument('--output', type=str, default='results/',
                        help='Output directory for restored images')
    parser.add_argument('--config', type=str, default=None,
                        help='Model config YAML (uses defaults if not specified)')
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Tile size for large image inference')
    parser.add_argument('--save_viz', action='store_true', default=True,
                        help='Save comparison visualizations')
    parser.add_argument('--num_viz', type=int, default=20,
                        help='Number of visualizations to save')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    args = parser.parse_args()
    
    # Setup
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load config
    config = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    
    # Load model
    print(f"Loading model from: {args.weights}")
    model = load_model(args.weights, device, config)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    
    # Load test dataset
    test_dataset = InferenceDataset(args.data)
    print(f"Test images: {len(test_dataset)}")
    
    # Load ground truth if available
    gt_files = None
    if args.gt:
        gt_dir = Path(args.gt)
        gt_files = sorted(gt_dir.glob("*.npy"))
        print(f"Ground truth images: {len(gt_files)}")
    
    # Create output directories
    output_dir = Path(args.output)
    restored_dir = output_dir / 'restored'
    viz_dir = output_dir / 'visualizations'
    restored_dir.mkdir(parents=True, exist_ok=True)
    if args.save_viz:
        viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics calculator
    metrics = MetricsCalculator(use_lpips=True) if gt_files else None
    
    # Process all images
    print("\nRestoring images...")
    total_time = 0.0
    
    for idx in tqdm(range(len(test_dataset)), desc="Processing"):
        sample = test_dataset[idx]
        noisy = sample['noisy'].unsqueeze(0)  # Add batch dim
        filename = sample['filename']
        
        # Restore
        t0 = time.time()
        restored = restore_image(model, noisy, device, args.tile_size)
        torch.cuda.synchronize() if device.type == 'cuda' else None
        elapsed = time.time() - t0
        total_time += elapsed
        
        # To numpy
        restored_np = restored.squeeze().cpu().numpy()
        restored_np = np.clip(restored_np, 0.0, 1.0)
        noisy_np = sample['noisy'].squeeze().numpy()
        noisy_np = np.clip(noisy_np, 0.0, 1.0)
        
        # Save restored image
        np.save(str(restored_dir / f'{filename}.npy'), restored_np)
        
        # Compute metrics if GT available
        clean_np = None
        if gt_files and idx < len(gt_files):
            clean_np = np.load(str(gt_files[idx])).astype(np.float32)
            clean_np = np.clip(clean_np, 0.0, 1.0)
            
            # Resize if needed (super-resolution case)
            if clean_np.shape != restored_np.shape:
                from skimage.transform import resize
                restored_for_metrics = resize(restored_np, clean_np.shape,
                                               anti_aliasing=True)
            else:
                restored_for_metrics = restored_np
            
            metrics.update(restored_for_metrics, clean_np)
        
        # Save visualization
        if args.save_viz and idx < args.num_viz:
            save_visualization(
                noisy_np, restored_np, clean_np,
                filename, str(viz_dir / f'{filename}_comparison.png')
            )
    
    # Summary
    avg_time = total_time / len(test_dataset)
    print(f"\n{'='*60}")
    print(f"  WaveSemiNet Evaluation Results")
    print(f"{'='*60}")
    print(f"  Images processed: {len(test_dataset)}")
    print(f"  Average inference time: {avg_time*1000:.1f} ms")
    print(f"  Throughput: {1.0/avg_time:.1f} images/sec")
    print(f"  Model parameters: {param_count:,}")
    print(f"  Restored images saved to: {restored_dir}")
    
    if metrics:
        metrics.print_summary()
    
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
