"""Loss functions package for WaveSemiNet."""

from losses.composite_loss import (
    CompositeLoss,
    CharbonnierLoss,
    FFTFrequencyLoss,
    SobelEdgeLoss,
    SSIMLoss,
)

__all__ = [
    "CompositeLoss",
    "CharbonnierLoss",
    "FFTFrequencyLoss",
    "SobelEdgeLoss",
    "SSIMLoss",
]
