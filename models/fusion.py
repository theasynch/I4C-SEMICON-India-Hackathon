"""
WaveSemiNet — Cross-Attention + FFT Feature Fusion Module

Fuses features from the NAFNet (low-frequency) branch and the
Edge-Aware Transformer (high-frequency) branch using:
1. Cross-attention between low-freq and high-freq features
2. FFT mixing layer for frequency-domain feature interaction
3. Task-conditioned modulation for unified denoising + super-resolution

The FFT mixing is a key novelty: semiconductor images have strong periodic
structures (repeated SRAM cells, grid-aligned interconnects) that are best
captured in the frequency domain.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention between low-frequency and high-frequency features.
    
    Low-freq features provide global structural context (queries),
    high-freq features provide detail information (keys, values).
    This allows the model to selectively enhance edges and defects
    guided by the structural context.
    
    Args:
        dim: Feature dimension
        num_heads: Number of attention heads
    """

    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)  # Queries from low-freq
        self.k_proj = nn.Linear(dim, dim)  # Keys from high-freq
        self.v_proj = nn.Linear(dim, dim)  # Values from high-freq
        self.out_proj = nn.Linear(dim, dim)

        self.norm_lf = nn.LayerNorm(dim)
        self.norm_hf = nn.LayerNorm(dim)

    def forward(self, lf_feat: torch.Tensor,
                hf_feat: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lf_feat: Low-frequency features (B, L, C)
            hf_feat: High-frequency features (B, L, C)
        
        Returns:
            Fused features (B, L, C)
        """
        lf = self.norm_lf(lf_feat)
        hf = self.norm_hf(hf_feat)

        B, L, C = lf.shape
        
        q = self.q_proj(lf).reshape(B, L, self.num_heads,
                                     self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(hf).reshape(B, L, self.num_heads,
                                     self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(hf).reshape(B, L, self.num_heads,
                                     self.head_dim).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, C)
        out = self.out_proj(out)

        return lf_feat + out


class FFTMixingLayer(nn.Module):
    """
    Frequency-domain feature mixing using FFT.
    
    Transforms features to the frequency domain, applies learnable
    spectral filters, and transforms back. This is particularly
    effective for semiconductor images because:
    - Periodic structures (SRAM cells) become concentrated peaks in FFT
    - Noise is spread across all frequencies
    - Defects create specific spectral signatures
    
    The learnable spectral filter can selectively amplify defect-related
    frequencies while suppressing noise frequencies.
    
    Args:
        dim: Feature dimension
        fft_norm: FFT normalization mode
    """

    def __init__(self, dim: int, fft_norm: str = "ortho"):
        super().__init__()
        self.fft_norm = fft_norm

        # Learnable complex spectral filter (real + imaginary parts)
        self.spectral_weight_real = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.spectral_weight_imag = nn.Parameter(torch.zeros(1, dim, 1, 1))
        
        # Post-FFT 1x1 conv for channel mixing
        self.channel_mix = nn.Sequential(
            nn.Conv2d(dim, dim * 2, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(dim * 2, dim, kernel_size=1),
        )

        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Feature tensor (B, C, H, W)
        Returns:
            FFT-mixed features (B, C, H, W)
        """
        shortcut = x
        x = self.norm(x)

        # Forward FFT
        x_fft = torch.fft.rfft2(x, norm=self.fft_norm)

        # Apply learnable spectral filter (complex multiplication)
        weight = torch.complex(
            self.spectral_weight_real.expand_as(x_fft.real),
            self.spectral_weight_imag.expand_as(x_fft.imag)
        )
        x_fft = x_fft * weight

        # Inverse FFT
        x = torch.fft.irfft2(x_fft, s=(shortcut.shape[2], shortcut.shape[3]),
                              norm=self.fft_norm)

        # Channel mixing
        x = self.channel_mix(x)

        return shortcut + x


class TaskConditionedModulation(nn.Module):
    """
    Task-specific modulation for unified model handling.
    
    Generates scale and shift parameters based on the degradation type
    (gaussian noise, speckle noise, super-resolution). This allows
    a single model to adapt its behavior for different restoration tasks.
    
    Uses FiLM (Feature-wise Linear Modulation) conditioning.
    
    Args:
        dim: Feature dimension
        num_tasks: Number of degradation types
    """

    def __init__(self, dim: int, num_tasks: int = 4):
        super().__init__()
        # Task embeddings: 0=gaussian, 1=speckle, 2=superres, 3=mixed
        self.task_embedding = nn.Embedding(num_tasks, dim * 2)
        nn.init.zeros_(self.task_embedding.weight)

    def forward(self, x: torch.Tensor,
                task_id: int = 0) -> torch.Tensor:
        """
        Args:
            x: Feature tensor (B, C, H, W)
            task_id: Degradation type index
        Returns:
            Modulated features (B, C, H, W)
        """
        cond = self.task_embedding(
            torch.tensor(task_id, device=x.device)
        )
        scale, shift = cond.chunk(2, dim=-1)
        scale = scale.view(1, -1, 1, 1)
        shift = shift.view(1, -1, 1, 1)
        return x * (1 + scale) + shift


class FeatureFusionModule(nn.Module):
    """
    Complete feature fusion module combining:
    1. Cross-attention between branches
    2. FFT mixing for frequency-domain interaction
    3. Task-conditioned modulation
    4. Final projection to wavelet subband space
    
    This is the core integration point of WaveSemiNet where
    low-freq structural understanding meets high-freq detail preservation.
    
    Args:
        dim: Feature dimension (must match both branch outputs)
        num_heads: Attention heads for cross-attention
        fft_norm: FFT normalization mode
        num_tasks: Number of degradation task types
    """

    def __init__(self, dim: int = 48, num_heads: int = 4,
                 fft_norm: str = "ortho", num_tasks: int = 4):
        super().__init__()

        # Cross-attention: low-freq queries, high-freq keys/values
        self.cross_attn = CrossAttentionFusion(dim, num_heads)

        # FFT mixing for frequency-domain feature interaction
        self.fft_mix = FFTMixingLayer(dim, fft_norm)

        # Task conditioning
        self.task_mod = TaskConditionedModulation(dim, num_tasks)

        # Projection heads for each output subband
        self.proj_ll = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
        )
        self.proj_lh = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
        )
        self.proj_hl = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
        )
        self.proj_hh = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, 1, kernel_size=3, padding=1),
        )

        # High-freq subband aggregation (combine LH + HL + HH features)
        self.hf_aggregate = nn.Conv2d(dim * 3, dim, kernel_size=1)

    def forward(self, ll_feat: torch.Tensor,
                lh_feat: torch.Tensor, hl_feat: torch.Tensor,
                hh_feat: torch.Tensor,
                task_id: int = 0) -> tuple[torch.Tensor, ...]:
        """
        Args:
            ll_feat: NAFNet branch output for LL (B, C, H, W)
            lh_feat, hl_feat, hh_feat: Transformer branch outputs (B, C, H, W)
            task_id: Degradation type for task conditioning
        
        Returns:
            Tuple of (ll_out, lh_out, hl_out, hh_out) residuals
            to add to original wavelet subbands
        """
        B, C, H, W = ll_feat.shape

        # Aggregate high-frequency features
        hf_combined = self.hf_aggregate(
            torch.cat([lh_feat, hl_feat, hh_feat], dim=1)
        )

        # Cross-attention: structural context (LF) guides detail (HF)
        lf_flat = rearrange(ll_feat, 'b c h w -> b (h w) c')
        hf_flat = rearrange(hf_combined, 'b c h w -> b (h w) c')
        fused = self.cross_attn(lf_flat, hf_flat)
        fused = rearrange(fused, 'b (h w) c -> b c h w', h=H, w=W)

        # FFT mixing for frequency-domain interaction
        fused = self.fft_mix(fused)

        # Task conditioning
        fused = self.task_mod(fused, task_id)

        # Project to subband residuals
        ll_out = self.proj_ll(fused + ll_feat)
        lh_out = self.proj_lh(fused + lh_feat)
        hl_out = self.proj_hl(fused + hl_feat)
        hh_out = self.proj_hh(fused + hh_feat)

        return ll_out, lh_out, hl_out, hh_out

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    fusion = FeatureFusionModule(dim=48, num_heads=4)

    ll_feat = torch.randn(1, 48, 128, 128)
    lh_feat = torch.randn(1, 48, 128, 128)
    hl_feat = torch.randn(1, 48, 128, 128)
    hh_feat = torch.randn(1, 48, 128, 128)

    ll_out, lh_out, hl_out, hh_out = fusion(ll_feat, lh_feat, hl_feat,
                                             hh_feat, task_id=0)
    print(f"LL: {ll_feat.shape} -> {ll_out.shape}")
    print(f"LH: {lh_feat.shape} -> {lh_out.shape}")
    print(f"Params: {fusion.count_parameters():,}")
