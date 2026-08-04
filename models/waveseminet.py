"""
WaveSemiNet — Main Model Architecture

Wavelet-Guided Semiconductor Image Restoration Network

Complete pipeline:
    Input → DWT → [LL→NAFNet, LH/HL/HH→EdgeTransformer] → Fusion → IWT → Output

For super-resolution, the input is first upsampled to target resolution
using pixel-shuffle or bicubic interpolation before entering the wavelet
pipeline. The model then performs joint denoising + detail recovery.

This unified architecture handles:
- Gaussian noise removal
- Speckle noise removal
- Super-resolution (2x, 4x)
- Combined degradations
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.wavelet import DWT2d, IWT2d
from models.nafnet_branch import NAFNetBranch
from models.edge_transformer import EdgeTransformerBranch
from models.fusion import FeatureFusionModule


class PixelShuffleUpsampler(nn.Module):
    """
    Learnable upsampler using sub-pixel convolution (pixel shuffle).
    More effective than bicubic interpolation for super-resolution
    because it learns the upsampling kernels from data.
    
    Args:
        in_channels: Number of input channels
        scale_factor: Upsampling factor (2 or 4)
    """

    def __init__(self, in_channels: int = 1, scale_factor: int = 2):
        super().__init__()
        self.scale = scale_factor
        
        if scale_factor == 4:
            # Two-stage pixel shuffle: 2x → 2x
            self.conv1 = nn.Conv2d(in_channels, in_channels * 4,
                                   kernel_size=3, padding=1)
            self.ps1 = nn.PixelShuffle(2)
            self.conv2 = nn.Conv2d(in_channels, in_channels * 4,
                                   kernel_size=3, padding=1)
            self.ps2 = nn.PixelShuffle(2)
        else:
            self.conv1 = nn.Conv2d(in_channels, in_channels * scale_factor ** 2,
                                   kernel_size=3, padding=1)
            self.ps1 = nn.PixelShuffle(scale_factor)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 4:
            x = self.ps1(self.conv1(x))
            x = self.ps2(self.conv2(x))
        else:
            x = self.ps1(self.conv1(x))
        return x


class WaveSemiNet(nn.Module):
    """
    WaveSemiNet: Wavelet-Guided Semiconductor Image Restoration Network
    
    Architecture overview:
    1. (Optional) Upsample input for super-resolution
    2. DWT decomposition into LL, LH, HL, HH subbands
    3. NAFNet branch processes LL (low-frequency structure)
    4. Edge Transformer branch processes LH, HL, HH (high-frequency details)
    5. Feature fusion with cross-attention + FFT mixing
    6. IWT reconstruction from processed subbands
    7. Global residual connection (learn the degradation residual)
    
    Args:
        in_channels: Input image channels (1 for grayscale)
        out_channels: Output image channels (1 for grayscale)
        base_channels: Base feature dimension
        nafnet_blocks: Block counts per NAFNet encoder/decoder level
        nafnet_middle: NAFNet bottleneck blocks
        transformer_layers: Number of Edge Transformer layers
        transformer_dim: Transformer embedding dimension
        transformer_heads: Transformer attention heads
        window_size: Transformer window size
        mlp_ratio: Transformer FFN expansion ratio
        scale_factor: Super-resolution factor (1=no SR, 2=2x, 4=4x)
        num_tasks: Number of task types for conditioning
        dropout: Dropout rate
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 48,
        nafnet_blocks: list[int] | None = None,
        nafnet_middle: int = 2,
        transformer_layers: int = 4,
        transformer_dim: int = 96,
        transformer_heads: int = 4,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        scale_factor: int = 1,
        num_tasks: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()

        if nafnet_blocks is None:
            nafnet_blocks = [2, 2, 4, 4]

        self.scale_factor = scale_factor
        self.in_channels = in_channels
        self.out_channels = out_channels

        # Super-resolution upsampler (before wavelet decomposition)
        if scale_factor > 1:
            self.upsampler = PixelShuffleUpsampler(in_channels, scale_factor)
        else:
            self.upsampler = None

        # Wavelet transform
        self.dwt = DWT2d()
        self.iwt = IWT2d()

        # Low-frequency branch (NAFNet)
        self.nafnet = NAFNetBranch(
            in_channels=in_channels,
            out_channels=base_channels,
            channels=base_channels,
            num_blocks=nafnet_blocks,
            middle_blocks=nafnet_middle,
            dropout=dropout,
        )

        # High-frequency branch (Edge-Aware Transformer)
        self.edge_transformer = EdgeTransformerBranch(
            in_channels=in_channels,
            out_channels=base_channels,
            embed_dim=transformer_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            dropout=dropout,
        )

        # Feature fusion module
        self.fusion = FeatureFusionModule(
            dim=base_channels,
            num_heads=transformer_heads,
            num_tasks=num_tasks,
        )

        # Final refinement conv (post-IWT)
        self.refine = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, task_id: int = 0) -> torch.Tensor:
        """
        Args:
            x: Degraded input image (B, C, H, W)
            task_id: Degradation type (0=gaussian, 1=speckle, 2=superres, 3=mixed)
        
        Returns:
            Restored image (B, C, H_out, W_out)
            H_out = H * scale_factor, W_out = W * scale_factor
        """
        # Optional super-resolution upsampling
        if self.upsampler is not None:
            x_up = self.upsampler(x)
        else:
            x_up = x

        # Store for global residual
        identity = x_up

        # Wavelet decomposition
        ll, lh, hl, hh = self.dwt(x_up)

        # Low-frequency branch (NAFNet)
        ll_feat = self.nafnet(ll)

        # High-frequency branch (Edge Transformer)
        lh_feat, hl_feat, hh_feat = self.edge_transformer(lh, hl, hh)

        # Feature fusion
        ll_res, lh_res, hl_res, hh_res = self.fusion(
            ll_feat, lh_feat, hl_feat, hh_feat, task_id=task_id
        )

        # Add residuals to original subbands
        ll_out = ll + ll_res
        lh_out = lh + lh_res
        hl_out = hl + hl_res
        hh_out = hh + hh_res

        # Inverse wavelet reconstruction
        restored = self.iwt(ll_out, lh_out, hl_out, hh_out)

        # Ensure matching size (handle rounding from odd dims)
        if restored.shape != identity.shape:
            restored = restored[:, :, :identity.shape[2], :identity.shape[3]]

        # Final refinement + global residual
        restored = self.refine(restored) + identity

        return restored

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def get_branch_params(self) -> dict[str, int]:
        """Returns parameter count per branch for analysis."""
        return {
            'upsampler': sum(p.numel() for p in self.upsampler.parameters()) if self.upsampler else 0,
            'nafnet': self.nafnet.count_parameters(),
            'edge_transformer': self.edge_transformer.count_parameters(),
            'fusion': self.fusion.count_parameters(),
            'refine': sum(p.numel() for p in self.refine.parameters()),
        }


def build_waveseminet(config: dict | None = None) -> WaveSemiNet:
    """
    Factory function to build WaveSemiNet from a config dictionary.
    
    Args:
        config: Model configuration dict (from YAML config file).
                If None, uses default configuration.
    
    Returns:
        Initialized WaveSemiNet model
    """
    if config is None:
        config = {}

    model_cfg = config.get('model', {})

    return WaveSemiNet(
        in_channels=model_cfg.get('in_channels', 1),
        out_channels=model_cfg.get('out_channels', 1),
        base_channels=model_cfg.get('base_channels', 48),
        nafnet_blocks=model_cfg.get('nafnet_branch', {}).get('num_blocks', None),
        nafnet_middle=model_cfg.get('nafnet_branch', {}).get('middle_blocks', 2),
        transformer_layers=model_cfg.get('edge_transformer', {}).get('num_layers', 4),
        transformer_dim=model_cfg.get('edge_transformer', {}).get('embed_dim', 96),
        transformer_heads=model_cfg.get('edge_transformer', {}).get('num_heads', 4),
        window_size=model_cfg.get('edge_transformer', {}).get('window_size', 8),
        mlp_ratio=model_cfg.get('edge_transformer', {}).get('mlp_ratio', 2.0),
        scale_factor=model_cfg.get('scale_factor', 1),
        num_tasks=4,
        dropout=model_cfg.get('dropout', 0.0),
    )


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("WaveSemiNet — Smoke Test")
    print("=" * 60)

    # Test 1: Denoising mode (scale=1)
    model = WaveSemiNet(
        in_channels=1, out_channels=1,
        base_channels=32,
        nafnet_blocks=[2, 2, 2, 2],
        transformer_layers=2,
        transformer_dim=64,
        transformer_heads=4,
        window_size=8,
        scale_factor=1,
    )
    
    x = torch.randn(1, 1, 128, 128)
    with torch.no_grad():
        t0 = time.time()
        y = model(x, task_id=0)
        t1 = time.time()
    
    print(f"\n[Denoising] Input: {x.shape} → Output: {y.shape}")
    print(f"Time: {(t1-t0)*1000:.1f}ms")
    print(f"Total params: {model.count_parameters():,}")
    
    branch_params = model.get_branch_params()
    for name, count in branch_params.items():
        print(f"  {name}: {count:,}")

    # Test 2: Super-resolution mode (scale=2)
    model_sr = WaveSemiNet(
        in_channels=1, out_channels=1,
        base_channels=32,
        nafnet_blocks=[2, 2, 2, 2],
        transformer_layers=2,
        transformer_dim=64,
        transformer_heads=4,
        window_size=8,
        scale_factor=2,
    )
    
    x_lr = torch.randn(1, 1, 64, 64)
    with torch.no_grad():
        y_sr = model_sr(x_lr, task_id=2)
    
    print(f"\n[Super-Res 2x] Input: {x_lr.shape} → Output: {y_sr.shape}")
    print(f"Total params: {model_sr.count_parameters():,}")
