"""
WaveSemiNet — Evaluation Metrics

PSNR, SSIM, and LPIPS computation for semiconductor image restoration.
Also includes a custom Defect Preservation Score (DPS).
"""

import numpy as np
import torch
import torch.nn.functional as F
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_psnr(pred: np.ndarray, target: np.ndarray,
                 data_range: float = 1.0) -> float:
    """
    Compute Peak Signal-to-Noise Ratio (PSNR).
    
    Args:
        pred: Predicted image (H, W) or (H, W, C), float32 [0, 1]
        target: Ground truth image, same shape as pred
        data_range: Maximum pixel value (1.0 for normalized images)
    
    Returns:
        PSNR value in dB
    """
    return float(peak_signal_noise_ratio(target, pred, data_range=data_range))


def compute_ssim(pred: np.ndarray, target: np.ndarray,
                 data_range: float = 1.0) -> float:
    """
    Compute Structural Similarity Index (SSIM).
    
    Args:
        pred: Predicted image (H, W) or (H, W, C), float32 [0, 1]
        target: Ground truth image, same shape as pred
        data_range: Maximum pixel value
    
    Returns:
        SSIM value [0, 1]
    """
    return float(structural_similarity(target, pred, data_range=data_range,
                                        channel_axis=None if pred.ndim == 2 else -1))


def compute_lpips(pred: torch.Tensor, target: torch.Tensor,
                  lpips_fn=None) -> float:
    """
    Compute LPIPS (Learned Perceptual Image Patch Similarity).
    
    Requires the 'lpips' package. Creates a VGG-based LPIPS model
    on first call if lpips_fn is not provided.
    
    Args:
        pred: Predicted image tensor (B, C, H, W), values in [0, 1]
        target: Ground truth tensor, same shape
        lpips_fn: Pre-initialized LPIPS model (for efficiency)
    
    Returns:
        LPIPS distance (lower is better)
    """
    if lpips_fn is None:
        import lpips
        lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
        if pred.is_cuda:
            lpips_fn = lpips_fn.cuda()

    # LPIPS expects values in [-1, 1]
    pred_scaled = pred * 2.0 - 1.0
    target_scaled = target * 2.0 - 1.0

    # LPIPS expects 3-channel input; replicate grayscale
    if pred_scaled.shape[1] == 1:
        pred_scaled = pred_scaled.repeat(1, 3, 1, 1)
        target_scaled = target_scaled.repeat(1, 3, 1, 1)

    with torch.no_grad():
        distance = lpips_fn(pred_scaled, target_scaled)

    return float(distance.mean())


class MetricsCalculator:
    """
    Accumulates metrics over a dataset for final reporting.
    
    Usage:
        calc = MetricsCalculator()
        for pred, target in results:
            calc.update(pred, target)
        summary = calc.compute()
    """

    def __init__(self, use_lpips: bool = True):
        self.psnr_values = []
        self.ssim_values = []
        self.lpips_values = []
        self.use_lpips = use_lpips
        self.lpips_fn = None

        if use_lpips:
            try:
                import lpips
                self.lpips_fn = lpips.LPIPS(net='vgg', verbose=False)
            except ImportError:
                print("Warning: lpips package not installed. Skipping LPIPS.")
                self.use_lpips = False

    def update(self, pred: np.ndarray, target: np.ndarray):
        """
        Update metrics with a single prediction-target pair.
        
        Args:
            pred: Predicted image (H, W), float32 [0, 1]
            target: Ground truth image (H, W), float32 [0, 1]
        """
        # Clip to valid range
        pred = np.clip(pred, 0.0, 1.0)
        target = np.clip(target, 0.0, 1.0)

        self.psnr_values.append(compute_psnr(pred, target))
        self.ssim_values.append(compute_ssim(pred, target))

        if self.use_lpips and self.lpips_fn is not None:
            pred_t = torch.from_numpy(pred).unsqueeze(0).unsqueeze(0).float()
            target_t = torch.from_numpy(target).unsqueeze(0).unsqueeze(0).float()
            self.lpips_values.append(
                compute_lpips(pred_t, target_t, self.lpips_fn)
            )

    def compute(self) -> dict[str, float]:
        """
        Compute aggregate metrics.
        
        Returns:
            Dictionary with mean and std of all metrics
        """
        result = {
            'psnr_mean': float(np.mean(self.psnr_values)),
            'psnr_std': float(np.std(self.psnr_values)),
            'ssim_mean': float(np.mean(self.ssim_values)),
            'ssim_std': float(np.std(self.ssim_values)),
            'num_images': len(self.psnr_values),
        }

        if self.lpips_values:
            result['lpips_mean'] = float(np.mean(self.lpips_values))
            result['lpips_std'] = float(np.std(self.lpips_values))

        return result

    def print_summary(self):
        """Print a formatted metrics summary table."""
        metrics = self.compute()
        print("\n" + "=" * 50)
        print("  Evaluation Metrics Summary")
        print("=" * 50)
        print(f"  Images evaluated: {metrics['num_images']}")
        print(f"  PSNR:   {metrics['psnr_mean']:.4f} ± {metrics['psnr_std']:.4f} dB")
        print(f"  SSIM:   {metrics['ssim_mean']:.4f} ± {metrics['ssim_std']:.4f}")
        if 'lpips_mean' in metrics:
            print(f"  LPIPS:  {metrics['lpips_mean']:.4f} ± {metrics['lpips_std']:.4f}")
        print("=" * 50)
