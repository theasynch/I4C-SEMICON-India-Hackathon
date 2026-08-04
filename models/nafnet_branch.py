"""
WaveSemiNet — NAFNet Low-Frequency Branch

Simplified NAFNet (Nonlinear Activation Free Network) encoder-decoder
for processing the LL (low-frequency approximation) subband.

Key design choices:
- SimpleGate replaces GELU/ReLU (no nonlinear activations)
- Simplified Channel Attention (SCA) for lightweight attention
- Layer normalization for stable training
- Skip connections throughout

Reference: Chen et al., "Simple Baselines for Image Restoration" (ECCV 2022)
Adapted for single-channel semiconductor images and wavelet-domain processing.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleGate(nn.Module):
    """
    Gating mechanism that splits channels in half and multiplies them.
    Replaces traditional nonlinear activations (GELU, ReLU) with a
    learnable gating operation. Key insight: the gate itself provides
    sufficient nonlinearity without explicit activation functions.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """
    Simplified Channel Attention (SCA).
    Uses global average pooling + 1x1 conv to generate channel-wise
    attention weights. Much lighter than SE-Net or CBAM.
    """

    def __init__(self, channels: int):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn = self.pool(x)
        attn = self.conv(attn)
        return x * attn


class NAFBlock(nn.Module):
    """
    Core NAFNet block.
    
    Structure:
        LayerNorm → Conv 1x1 (expand) → DWConv 3x3 → SimpleGate → 
        SCA → Conv 1x1 (project) → Residual
        +
        LayerNorm → Conv 1x1 (expand) → SimpleGate → Conv 1x1 (project) → Residual
    """

    def __init__(self, channels: int, expansion_ratio: int = 2,
                 dropout: float = 0.0):
        super().__init__()
        expanded = channels * expansion_ratio

        # Spatial mixing branch
        self.norm1 = nn.GroupNorm(1, channels)  # LayerNorm via GroupNorm(1, C)
        self.conv1 = nn.Conv2d(channels, expanded, kernel_size=1)
        self.dwconv = nn.Conv2d(expanded, expanded, kernel_size=3, padding=1,
                                groups=expanded)
        self.gate1 = SimpleGate()
        self.sca = SimplifiedChannelAttention(expanded // 2)
        self.proj1 = nn.Conv2d(expanded // 2, channels, kernel_size=1)
        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Channel mixing branch (FFN-like)
        self.norm2 = nn.GroupNorm(1, channels)
        self.conv2 = nn.Conv2d(channels, expanded, kernel_size=1)
        self.gate2 = SimpleGate()
        self.proj2 = nn.Conv2d(expanded // 2, channels, kernel_size=1)
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Learnable scaling factors (beta) for residual connections
        self.beta1 = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.beta2 = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Spatial mixing
        shortcut = x
        x_norm = self.norm1(x)
        x_sp = self.conv1(x_norm)
        x_sp = self.dwconv(x_sp)
        x_sp = self.gate1(x_sp)
        x_sp = self.sca(x_sp)
        x_sp = self.proj1(x_sp)
        x_sp = self.drop1(x_sp)
        x = shortcut + x_sp * self.beta1

        # Channel mixing
        shortcut = x
        x_ch = self.norm2(x)
        x_ch = self.conv2(x_ch)
        x_ch = self.gate2(x_ch)
        x_ch = self.proj2(x_ch)
        x_ch = self.drop2(x_ch)
        x = shortcut + x_ch * self.beta2

        return x


class NAFNetBranch(nn.Module):
    """
    NAFNet encoder-decoder branch for low-frequency (LL) subband processing.
    
    Architecture:
        Encoder: Conv → [NAFBlock × N] → Downsample → ...
        Middle:  [NAFBlock × M]
        Decoder: Upsample → [NAFBlock × N] → Conv → ...
    
    Uses U-Net style skip connections between encoder and decoder.
    
    Args:
        in_channels: Number of input channels (1 for grayscale LL subband)
        out_channels: Number of output channels
        channels: Base feature dimension
        num_blocks: List of block counts per encoder/decoder level
        middle_blocks: Number of blocks in the bottleneck
        dropout: Dropout rate
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1,
                 channels: int = 48, num_blocks: list[int] | None = None,
                 middle_blocks: int = 2, dropout: float = 0.0):
        super().__init__()

        if num_blocks is None:
            num_blocks = [2, 4, 4, 8]  # Blocks per level

        num_levels = len(num_blocks)

        # Input projection
        self.input_proj = nn.Conv2d(in_channels, channels, kernel_size=3,
                                    padding=1)

        # Encoder
        self.encoders = nn.ModuleList()
        self.downsamplers = nn.ModuleList()
        ch = channels
        for i, n_blocks in enumerate(num_blocks):
            self.encoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout)
                                for _ in range(n_blocks)])
            )
            if i < num_levels - 1:
                self.downsamplers.append(
                    nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2)
                )
                ch *= 2

        # Middle (bottleneck)
        self.middle = nn.Sequential(
            *[NAFBlock(ch, dropout=dropout) for _ in range(middle_blocks)]
        )

        # Decoder
        self.decoders = nn.ModuleList()
        self.upsamplers = nn.ModuleList()
        self.skip_projs = nn.ModuleList()
        for i, n_blocks in enumerate(reversed(num_blocks)):
            if i > 0:
                self.upsamplers.append(
                    nn.ConvTranspose2d(ch, ch // 2, kernel_size=2, stride=2)
                )
                ch //= 2
                # 1x1 conv to merge skip connection
                self.skip_projs.append(
                    nn.Conv2d(ch * 2, ch, kernel_size=1)
                )
            self.decoders.append(
                nn.Sequential(*[NAFBlock(ch, dropout=dropout)
                                for _ in range(n_blocks)])
            )

        # Output projection
        self.output_proj = nn.Conv2d(channels, out_channels, kernel_size=3,
                                     padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: LL subband tensor, shape (B, C_in, H, W)
        
        Returns:
            Processed LL subband, shape (B, C_out, H, W)
        """
        x = self.input_proj(x)

        # Encoder path
        skips = []
        for i, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if i < len(self.downsamplers):
                x = self.downsamplers[i](x)

        # Middle
        x = self.middle(x)

        # Decoder path
        # skip_idx starts at len(skips)-2 because skips[-1] is from the
        # deepest encoder level (same resolution as bottleneck), and the
        # first decoder (i=0) operates at that level without upsampling.
        # Subsequent decoders upsample and merge with progressively
        # shallower encoder skips.
        skip_idx = len(skips) - 2
        for i, decoder in enumerate(self.decoders):
            if i > 0:
                x = self.upsamplers[i - 1](x)
                # Handle size mismatch from odd dimensions
                skip = skips[skip_idx]
                if x.shape[2:] != skip.shape[2:]:
                    x = F.pad(x, (0, skip.shape[3] - x.shape[3],
                                  0, skip.shape[2] - x.shape[2]))
                x = self.skip_projs[i - 1](torch.cat([x, skip], dim=1))
                skip_idx -= 1
            x = decoder(x)

        x = self.output_proj(x)
        return x

    def count_parameters(self) -> int:
        """Returns total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Smoke test
    model = NAFNetBranch(
        in_channels=1,
        out_channels=48,  # Output features, not final image
        channels=48,
        num_blocks=[2, 2, 4, 4],
        middle_blocks=2
    )

    x = torch.randn(1, 1, 128, 128)  # LL subband (half resolution)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Params: {model.count_parameters():,}")
