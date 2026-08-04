"""
WaveSemiNet — Composite Loss Functions

Multi-objective loss for semiconductor image restoration:
- L1 pixel loss (baseline reconstruction)
- FFT frequency loss (preserves frequency structure)
- Sobel edge loss (preserves sharp edges for defect detection)
- MS-SSIM loss (multi-scale structural similarity)
- LPIPS perceptual loss (optional, for perceptual quality)

The composite loss is critical because standard pixel-only losses
(L1, L2) tend to produce over-smoothed results that destroy
sub-pixel defects. The frequency and edge losses counteract this.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFTFrequencyLoss(nn.Module):
    """
    Frequency-domain L1 loss computed in FFT space.
    
    Compares the magnitude spectra of predicted and target images.
    This encourages the model to preserve the frequency structure
    of semiconductor patterns — periodic SRAM cells, grid lines,
    and regular interconnect patterns all have distinctive spectral
    signatures that must be maintained.
    
    Args:
        loss_type: 'l1' or 'l2' distance metric
        reduction: 'mean' or 'sum'
    """

    def __init__(self, loss_type: str = 'l1', reduction: str = 'mean'):
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predicted image (B, C, H, W)
            target: Ground truth image (B, C, H, W)
        Returns:
            Scalar frequency loss
        """
        # Compute 2D FFT
        pred_fft = torch.fft.rfft2(pred, norm='ortho')
        target_fft = torch.fft.rfft2(target, norm='ortho')

        # Compare magnitude spectra
        pred_mag = torch.abs(pred_fft)
        target_mag = torch.abs(target_fft)

        if self.loss_type == 'l1':
            loss = F.l1_loss(pred_mag, target_mag, reduction=self.reduction)
        else:
            loss = F.mse_loss(pred_mag, target_mag, reduction=self.reduction)

        return loss


class SobelEdgeLoss(nn.Module):
    """
    Edge-preservation loss using Sobel gradient operators.
    
    Computes horizontal and vertical gradients of both predicted
    and target images, then measures the L1 distance between them.
    This directly penalizes edge distortion and is critical for
    semiconductor images where edge sharpness determines defect
    detection accuracy.
    
    Sub-pixel defects often manifest as tiny gradient changes that
    standard L1/L2 losses would ignore.
    """

    def __init__(self):
        super().__init__()
        # Sobel kernels
        sobel_x = torch.tensor([[-1, 0, 1],
                                [-2, 0, 2],
                                [-1, 0, 1]], dtype=torch.float32)
        sobel_y = torch.tensor([[-1, -2, -1],
                                [0, 0, 0],
                                [1, 2, 1]], dtype=torch.float32)

        # Shape: (1, 1, 3, 3) for single-channel convolution
        self.register_buffer('sobel_x', sobel_x.unsqueeze(0).unsqueeze(0))
        self.register_buffer('sobel_y', sobel_y.unsqueeze(0).unsqueeze(0))

    def _compute_gradients(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute Sobel gradients per channel."""
        B, C, H, W = x.shape
        # Process each channel separately
        grad_x = F.conv2d(x.reshape(B * C, 1, H, W), self.sobel_x, padding=1)
        grad_y = F.conv2d(x.reshape(B * C, 1, H, W), self.sobel_y, padding=1)
        grad_x = grad_x.reshape(B, C, H, W)
        grad_y = grad_y.reshape(B, C, H, W)
        return grad_x, grad_y

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        pred_gx, pred_gy = self._compute_gradients(pred)
        target_gx, target_gy = self._compute_gradients(target)

        loss_x = F.l1_loss(pred_gx, target_gx)
        loss_y = F.l1_loss(pred_gy, target_gy)

        return loss_x + loss_y


class SSIMLoss(nn.Module):
    """
    Structural Similarity (SSIM) loss.
    
    1 - SSIM, so that minimizing this loss maximizes SSIM.
    Uses a Gaussian window for local statistics computation.
    
    SSIM is one of the primary evaluation metrics for the hackathon,
    so directly optimizing it during training is critical.
    
    Args:
        window_size: Size of the Gaussian window
        sigma: Standard deviation of Gaussian window
    """

    def __init__(self, window_size: int = 11, sigma: float = 1.5):
        super().__init__()
        self.window_size = window_size
        self.sigma = sigma
        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

        # Create Gaussian window
        window = self._create_window(window_size, sigma)
        self.register_buffer('window', window)

    def _create_window(self, size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = g.unsqueeze(1) @ g.unsqueeze(0)
        return window.unsqueeze(0).unsqueeze(0)

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        B, C, H, W = pred.shape
        window = self.window.expand(C, -1, -1, -1)

        mu_pred = F.conv2d(pred, window, padding=self.window_size // 2,
                           groups=C)
        mu_target = F.conv2d(target, window, padding=self.window_size // 2,
                             groups=C)

        mu_pred_sq = mu_pred ** 2
        mu_target_sq = mu_target ** 2
        mu_cross = mu_pred * mu_target

        sigma_pred_sq = F.conv2d(pred ** 2, window,
                                  padding=self.window_size // 2,
                                  groups=C) - mu_pred_sq
        sigma_target_sq = F.conv2d(target ** 2, window,
                                    padding=self.window_size // 2,
                                    groups=C) - mu_target_sq
        sigma_cross = F.conv2d(pred * target, window,
                                padding=self.window_size // 2,
                                groups=C) - mu_cross

        ssim_map = ((2 * mu_cross + self.C1) * (2 * sigma_cross + self.C2)) / \
                   ((mu_pred_sq + mu_target_sq + self.C1) *
                    (sigma_pred_sq + sigma_target_sq + self.C2))

        return 1 - ssim_map.mean()


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 variant).
    
    L(x) = sqrt(x^2 + eps^2)
    
    More robust than L1 near zero (differentiable everywhere)
    while maintaining similar gradient magnitude for large errors.
    Often preferred over L1 for image restoration.
    
    Args:
        eps: Smoothing constant
    """

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps_sq = eps ** 2

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        diff = pred - target
        loss = torch.sqrt(diff ** 2 + self.eps_sq)
        return loss.mean()


class CompositeLoss(nn.Module):
    """
    Combined loss function for WaveSemiNet training.
    
    L_total = λ₁·L_char + λ₂·L_fft + λ₃·L_edge + λ₄·L_ssim
    
    Each component targets a different aspect:
    - Charbonnier: pixel-level reconstruction accuracy
    - FFT: frequency structure preservation
    - Sobel edge: edge/defect preservation
    - SSIM: structural similarity (directly optimizes evaluation metric)
    
    LPIPS is optionally added when available but requires VGG features
    which increase memory. We use it only for fine-tuning.
    
    Args:
        pixel_weight: Weight for Charbonnier pixel loss
        freq_weight: Weight for FFT frequency loss
        edge_weight: Weight for Sobel edge loss
        ssim_weight: Weight for SSIM loss
    """

    def __init__(self, pixel_weight: float = 1.0,
                 freq_weight: float = 0.1,
                 edge_weight: float = 0.05,
                 ssim_weight: float = 0.1):
        super().__init__()

        self.pixel_loss = CharbonnierLoss()
        self.freq_loss = FFTFrequencyLoss()
        self.edge_loss = SobelEdgeLoss()
        self.ssim_loss = SSIMLoss()

        self.weights = {
            'pixel': pixel_weight,
            'freq': freq_weight,
            'edge': edge_weight,
            'ssim': ssim_weight,
        }

    def forward(self, pred: torch.Tensor,
                target: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            pred: Predicted restored image (B, C, H, W)
            target: Ground truth clean image (B, C, H, W)
        
        Returns:
            Tuple of:
            - total_loss: Weighted sum of all losses
            - loss_dict: Individual loss values for logging
        """
        l_pixel = self.pixel_loss(pred, target)
        l_freq = self.freq_loss(pred, target)
        l_edge = self.edge_loss(pred, target)
        l_ssim = self.ssim_loss(pred, target)

        total = (self.weights['pixel'] * l_pixel +
                 self.weights['freq'] * l_freq +
                 self.weights['edge'] * l_edge +
                 self.weights['ssim'] * l_ssim)

        loss_dict = {
            'pixel': l_pixel.item(),
            'freq': l_freq.item(),
            'edge': l_edge.item(),
            'ssim': l_ssim.item(),
            'total': total.item(),
        }

        return total, loss_dict


if __name__ == "__main__":
    # Smoke test
    loss_fn = CompositeLoss()

    pred = torch.randn(2, 1, 128, 128)
    target = torch.randn(2, 1, 128, 128)

    total, loss_dict = loss_fn(pred, target)
    print("Loss components:")
    for k, v in loss_dict.items():
        print(f"  {k}: {v:.4f}")
