"""
WaveSemiNet — Dataset Analysis Script

Analyzes the hackathon dataset and generates statistics & visualizations.
Useful for understanding data distribution and calibrating the model.

Usage:
    python scripts/analyze_dataset.py --data Data-public/train/train
"""

import os
import argparse
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def analyze_directory(dir_path: str, name: str) -> dict:
    """Analyze all .npy files in a directory."""
    files = sorted(Path(dir_path).glob("*.npy"))
    
    if not files:
        print(f"  No .npy files found in {dir_path}")
        return {}
    
    shapes = []
    mins = []
    maxs = []
    means = []
    stds = []
    
    for f in files:
        img = np.load(str(f))
        shapes.append(img.shape)
        mins.append(img.min())
        maxs.append(img.max())
        means.append(img.mean())
        stds.append(img.std())
    
    stats = {
        'name': name,
        'count': len(files),
        'shape': shapes[0],
        'all_same_shape': len(set(shapes)) == 1,
        'min_range': (min(mins), max(mins)),
        'max_range': (min(maxs), max(maxs)),
        'mean_range': (min(means), max(means)),
        'std_range': (min(stds), max(stds)),
        'global_mean': float(np.mean(means)),
        'global_std': float(np.mean(stds)),
        'means': means,
        'stds': stds,
    }
    
    print(f"\n  [{name}]")
    print(f"    Files:       {stats['count']}")
    print(f"    Shape:       {stats['shape']}")
    print(f"    Same shape:  {stats['all_same_shape']}")
    print(f"    Min range:   [{stats['min_range'][0]:.4f}, {stats['min_range'][1]:.4f}]")
    print(f"    Max range:   [{stats['max_range'][0]:.4f}, {stats['max_range'][1]:.4f}]")
    print(f"    Mean range:  [{stats['mean_range'][0]:.4f}, {stats['mean_range'][1]:.4f}]")
    print(f"    Std range:   [{stats['std_range'][0]:.4f}, {stats['std_range'][1]:.4f}]")
    print(f"    Global mean: {stats['global_mean']:.4f}")
    print(f"    Global std:  {stats['global_std']:.4f}")
    
    return stats


def plot_distributions(noisy_stats: dict, gt_stats: dict, save_dir: str):
    """Plot distribution comparisons."""
    os.makedirs(save_dir, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Mean distribution
    axes[0].hist(noisy_stats['means'], bins=50, alpha=0.7, label='NoisyLR', color='#FF6B6B')
    if gt_stats:
        axes[0].hist(gt_stats['means'], bins=50, alpha=0.7, label='GT', color='#4ECDC4')
    axes[0].set_xlabel('Image Mean')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Mean Intensity Distribution')
    axes[0].legend()
    
    # Std distribution
    axes[1].hist(noisy_stats['stds'], bins=50, alpha=0.7, label='NoisyLR', color='#FF6B6B')
    if gt_stats:
        axes[1].hist(gt_stats['stds'], bins=50, alpha=0.7, label='GT', color='#4ECDC4')
    axes[1].set_xlabel('Image Std')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Std Deviation Distribution')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'dataset_distributions.png'), dpi=150)
    plt.close()
    print(f"\nSaved distribution plot to {save_dir}/dataset_distributions.png")


def plot_sample_pairs(noisy_dir: str, gt_dir: str, save_dir: str, n: int = 10):
    """Plot side-by-side comparisons of NoisyLR vs GT pairs."""
    os.makedirs(save_dir, exist_ok=True)
    
    noisy_files = sorted(Path(noisy_dir).glob("*.npy"))[:n]
    gt_files = sorted(Path(gt_dir).glob("*.npy"))[:n]
    
    fig, axes = plt.subplots(n, 2, figsize=(8, 4 * n))
    fig.suptitle('NoisyLR (128×128) vs GT (256×256)', fontsize=16, fontweight='bold')
    
    for i, (nf, gf) in enumerate(zip(noisy_files, gt_files)):
        noisy = np.load(str(nf))
        gt = np.load(str(gf))
        
        axes[i, 0].imshow(noisy, cmap='gray', vmin=0, vmax=1)
        axes[i, 0].set_title(f'NoisyLR: {nf.stem}', fontsize=10)
        axes[i, 0].axis('off')
        
        axes[i, 1].imshow(gt, cmap='gray', vmin=0, vmax=1)
        axes[i, 1].set_title(f'GT: {gf.stem}', fontsize=10)
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_pairs.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved sample pairs to {save_dir}/sample_pairs.png")


def main():
    parser = argparse.ArgumentParser(description='Analyze hackathon dataset')
    parser.add_argument('--data', type=str, default='Data-public/train/train',
                        help='Path to dataset directory')
    parser.add_argument('--output', type=str, default='results',
                        help='Output directory for plots')
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    print("=" * 60)
    print("  Dataset Analysis — SEMICON India Hackathon 2026")
    print("=" * 60)
    
    # Analyze NoisyLR
    noisy_dir = data_dir / 'NoisyLR'
    noisy_stats = {}
    if noisy_dir.exists():
        noisy_stats = analyze_directory(str(noisy_dir), 'NoisyLR')
    
    # Analyze GT
    gt_dir = data_dir / 'GT'
    gt_stats = {}
    if gt_dir.exists():
        gt_stats = analyze_directory(str(gt_dir), 'GT')
    
    # Scale factor
    if noisy_stats and gt_stats:
        scale = gt_stats['shape'][0] // noisy_stats['shape'][0]
        print(f"\n  Scale factor: {scale}x ({noisy_stats['shape']} -> {gt_stats['shape']})")
    
    # Plots
    if noisy_stats:
        plot_distributions(noisy_stats, gt_stats, args.output)
    
    if noisy_dir.exists() and gt_dir.exists():
        plot_sample_pairs(str(noisy_dir), str(gt_dir), args.output, n=8)
    
    print("\nAnalysis complete!")


if __name__ == "__main__":
    main()
