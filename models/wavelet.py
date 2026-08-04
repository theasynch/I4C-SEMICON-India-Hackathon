"""
WaveSemiNet — Wavelet-Guided Semiconductor Image Restoration Network

Differentiable 2D Discrete Wavelet Transform (DWT) and Inverse Wavelet Transform (IWT)
using Haar wavelets for GPU-accelerated wavelet decomposition.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DWT2d(nn.Module):
    """
    2D Discrete Wavelet Transform using Haar wavelets.
    
    Decomposes an input image into four subbands:
        LL (approximation), LH (horizontal detail),
        HL (vertical detail), HH (diagonal detail)
    
    This is implemented as fixed convolution filters (non-learnable)
    for efficiency and mathematical correctness.
    """

    def __init__(self):
        super().__init__()
        # Haar wavelet filters
        # Low-pass:  [1/2,  1/2]
        # High-pass: [1/2, -1/2]
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]], dtype=torch.float32)

        # Stack into [4, 1, 2, 2] filter bank
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            Tuple of (LL, LH, HL, HH), each of shape (B, C, H//2, W//2)
        """
        B, C, H, W = x.shape

        # Pad if dimensions are odd
        pad_h = H % 2
        pad_w = W % 2
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h), mode='reflect')

        # Apply DWT per channel using grouped convolution
        # Reshape to (B*C, 1, H, W) for grouped conv
        x_flat = x.reshape(B * C, 1, x.shape[2], x.shape[3])
        
        # Convolve with stride 2 for downsampling
        coeffs = F.conv2d(x_flat, self.filters, stride=2)
        
        # Reshape back to (B, C, 4, H//2, W//2) then split
        _, _, h_out, w_out = coeffs.shape
        coeffs = coeffs.reshape(B, C, 4, h_out, w_out)
        
        ll = coeffs[:, :, 0, :, :]
        lh = coeffs[:, :, 1, :, :]
        hl = coeffs[:, :, 2, :, :]
        hh = coeffs[:, :, 3, :, :]

        return ll, lh, hl, hh


class IWT2d(nn.Module):
    """
    2D Inverse Wavelet Transform using Haar wavelets.
    
    Reconstructs an image from its four subbands:
        LL (approximation), LH (horizontal detail),
        HL (vertical detail), HH (diagonal detail)
    
    Uses transposed convolution for upsampling reconstruction.
    """

    def __init__(self):
        super().__init__()
        # Inverse Haar filters (transpose of forward filters)
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5, 0.5],
                           [-0.5, 0.5]], dtype=torch.float32)
        hh = torch.tensor([[0.5, -0.5],
                           [-0.5, 0.5]], dtype=torch.float32)

        # For IWT, we use transposed convolution
        # conv_transpose2d weight shape: [in_channels, out_channels, kH, kW]
        # Input: (B*C, 4, h, w) -> Output: (B*C, 1, 2h, 2w)
        # So filters shape: [4, 1, 2, 2]
        filters = torch.stack([ll, lh, hl, hh], dim=0).unsqueeze(1)
        self.register_buffer('filters', filters)

    def forward(self, ll: torch.Tensor, lh: torch.Tensor,
                hl: torch.Tensor, hh: torch.Tensor) -> torch.Tensor:
        """
        Args:
            ll, lh, hl, hh: Subband tensors, each of shape (B, C, H//2, W//2)
        
        Returns:
            Reconstructed tensor of shape (B, C, H, W)
        """
        B, C, h, w = ll.shape

        # Stack subbands: (B, C, 4, h, w) -> (B*C, 4, h, w)
        coeffs = torch.stack([ll, lh, hl, hh], dim=2)
        coeffs = coeffs.reshape(B * C, 4, h, w)

        # Transposed convolution for upsampling
        # Input: (B*C, 4, h, w) -> Output: (B*C, 1, 2h, 2w)
        x = F.conv_transpose2d(coeffs, self.filters, stride=2)
        
        # Reshape back to (B, C, H, W) — the output channel dim (1) merges
        # with batch*channels during reshape
        x = x.reshape(B, C, x.shape[2], x.shape[3])

        return x


class MultiLevelDWT(nn.Module):
    """
    Multi-level DWT decomposition for hierarchical feature extraction.
    Useful for progressive restoration at multiple scales.
    """

    def __init__(self, levels: int = 2):
        super().__init__()
        self.levels = levels
        self.dwt = DWT2d()

    def forward(self, x: torch.Tensor) -> list[tuple[torch.Tensor, ...]]:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
        
        Returns:
            List of (LL, LH, HL, HH) tuples for each decomposition level.
            Level 0 is the finest (original scale), level N-1 is coarsest.
        """
        coefficients = []
        current = x
        
        for _ in range(self.levels):
            ll, lh, hl, hh = self.dwt(current)
            coefficients.append((ll, lh, hl, hh))
            current = ll  # Recurse on approximation subband

        return coefficients


if __name__ == "__main__":
    # Quick verification
    dwt = DWT2d()
    iwt = IWT2d()

    x = torch.randn(2, 1, 256, 256)  # Batch of 2 grayscale images
    ll, lh, hl, hh = dwt(x)
    print(f"Input shape:  {x.shape}")
    print(f"LL shape:     {ll.shape}")
    print(f"LH shape:     {lh.shape}")
    print(f"HL shape:     {hl.shape}")
    print(f"HH shape:     {hh.shape}")

    x_recon = iwt(ll, lh, hl, hh)
    print(f"Recon shape:  {x_recon.shape}")
    print(f"Recon error:  {(x - x_recon).abs().max().item():.6e}")

    # Multi-level
    mdwt = MultiLevelDWT(levels=3)
    coeffs = mdwt(x)
    for i, (ll_i, lh_i, hl_i, hh_i) in enumerate(coeffs):
        print(f"Level {i}: LL={ll_i.shape}, LH={lh_i.shape}")
