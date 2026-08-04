"""
WaveSemiNet — Semiconductor Dataset Loader

PyTorch Dataset for paired (degraded, clean) semiconductor inspection images.
Handles the hackathon dataset format:
- Images stored as .npy float32 arrays
- Grayscale, 128x128 (NoisyLR) / 256x256 or 512x512 (Clean HR)
- Paired by filename index

Dataset structure expected:
    train/
        NoisyLR/     # Low-resolution noisy images (128x128)
        CleanHR/     # High-resolution clean images (256x256 or 512x512)
    test/
        NoisyLR/     # Test degraded images
"""

import os
import glob
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SemiconductorDataset(Dataset):
    """
    Dataset for paired semiconductor image restoration.
    
    Loads NoisyLR and CleanHR image pairs, applies augmentations,
    and returns normalized tensors ready for training.
    
    Args:
        noisy_dir: Path to directory containing degraded .npy images
        clean_dir: Path to directory containing clean .npy images (None for test)
        patch_size: Random crop size for training (None = full image)
        augment: Whether to apply data augmentation
        normalize: Whether to normalize to [0, 1] range
    """

    def __init__(
        self,
        noisy_dir: str,
        clean_dir: str | None = None,
        patch_size: int | None = None,
        augment: bool = False,
        normalize: bool = True,
    ):
        self.noisy_dir = Path(noisy_dir)
        self.clean_dir = Path(clean_dir) if clean_dir else None
        self.patch_size = patch_size
        self.augment = augment
        self.normalize = normalize

        # Collect and sort file paths
        self.noisy_files = sorted(glob.glob(str(self.noisy_dir / "*.npy")))
        
        if self.clean_dir is not None:
            self.clean_files = sorted(glob.glob(str(self.clean_dir / "*.npy")))
            assert len(self.noisy_files) == len(self.clean_files), \
                f"Mismatch: {len(self.noisy_files)} noisy vs {len(self.clean_files)} clean images"
        else:
            self.clean_files = None

        if len(self.noisy_files) == 0:
            raise FileNotFoundError(f"No .npy files found in {noisy_dir}")

    def __len__(self) -> int:
        return len(self.noisy_files)

    def _load_image(self, path: str) -> np.ndarray:
        """Load a .npy image and ensure float32."""
        img = np.load(path).astype(np.float32)
        return img

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        """Clip to [0, 1] range (some noisy images exceed this)."""
        return np.clip(img, 0.0, 1.0)

    def _augment(self, noisy: np.ndarray,
                 clean: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Apply semiconductor-safe augmentations.
        Only 90° rotations and flips (preserves Manhattan geometry).
        """
        # Random 90° rotation (0, 90, 180, 270)
        k = random.randint(0, 3)
        noisy = np.rot90(noisy, k).copy()
        if clean is not None:
            clean = np.rot90(clean, k).copy()

        # Random horizontal flip
        if random.random() > 0.5:
            noisy = np.fliplr(noisy).copy()
            if clean is not None:
                clean = np.fliplr(clean).copy()

        # Random vertical flip
        if random.random() > 0.5:
            noisy = np.flipud(noisy).copy()
            if clean is not None:
                clean = np.flipud(clean).copy()

        return noisy, clean

    def _random_crop(self, noisy: np.ndarray,
                     clean: np.ndarray | None) -> tuple[np.ndarray, np.ndarray | None]:
        """
        Random crop from image. If clean is larger (HR), crop proportionally.
        """
        h, w = noisy.shape
        ps = self.patch_size

        if h <= ps or w <= ps:
            return noisy, clean

        top = random.randint(0, h - ps)
        left = random.randint(0, w - ps)
        noisy = noisy[top:top + ps, left:left + ps]

        if clean is not None:
            ch, cw = clean.shape
            scale = ch // h
            c_ps = ps * scale
            c_top = top * scale
            c_left = left * scale
            clean = clean[c_top:c_top + c_ps, c_left:c_left + c_ps]

        return noisy, clean

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        """
        Returns:
            Dictionary with:
            - 'noisy': Degraded image tensor (1, H, W)
            - 'clean': Clean image tensor (1, H_hr, W_hr) if available
            - 'filename': Original filename stem
            - 'scale': Super-resolution scale factor
        """
        noisy = self._load_image(self.noisy_files[idx])
        clean = self._load_image(self.clean_files[idx]) if self.clean_files else None

        # Compute scale factor
        if clean is not None:
            scale = clean.shape[0] // noisy.shape[0]
        else:
            scale = 1

        # Random crop (if training)
        if self.patch_size is not None and self.patch_size < noisy.shape[0]:
            noisy, clean = self._random_crop(noisy, clean)

        # Augmentation (if training)
        if self.augment:
            noisy, clean = self._augment(noisy, clean)

        # Normalize
        if self.normalize:
            noisy = self._normalize(noisy)
            if clean is not None:
                clean = self._normalize(clean)

        # Convert to tensors (add channel dim)
        noisy_tensor = torch.from_numpy(noisy).unsqueeze(0)  # (1, H, W)

        result = {
            'noisy': noisy_tensor,
            'filename': Path(self.noisy_files[idx]).stem,
            'scale': scale,
        }

        if clean is not None:
            clean_tensor = torch.from_numpy(clean).unsqueeze(0)
            result['clean'] = clean_tensor

        return result


class InferenceDataset(Dataset):
    """
    Lightweight dataset for inference on test images only.
    No pairing with clean images, no augmentation.
    """

    def __init__(self, image_dir: str, normalize: bool = True):
        self.image_dir = Path(image_dir)
        self.files = sorted(glob.glob(str(self.image_dir / "*.npy")))
        self.normalize = normalize

        if len(self.files) == 0:
            raise FileNotFoundError(f"No .npy files in {image_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | str]:
        img = np.load(self.files[idx]).astype(np.float32)
        if self.normalize:
            img = np.clip(img, 0.0, 1.0)
        
        return {
            'noisy': torch.from_numpy(img).unsqueeze(0),
            'filename': Path(self.files[idx]).stem,
        }


def create_dataloaders(
    train_noisy_dir: str,
    train_clean_dir: str,
    val_noisy_dir: str | None = None,
    val_clean_dir: str | None = None,
    patch_size: int = 128,
    batch_size: int = 8,
    num_workers: int = 4,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader | None]:
    """
    Factory function to create training and validation dataloaders.
    
    Returns:
        Tuple of (train_loader, val_loader)
    """
    train_dataset = SemiconductorDataset(
        noisy_dir=train_noisy_dir,
        clean_dir=train_clean_dir,
        patch_size=patch_size,
        augment=True,
        normalize=True,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = None
    if val_noisy_dir and val_clean_dir:
        val_dataset = SemiconductorDataset(
            noisy_dir=val_noisy_dir,
            clean_dir=val_clean_dir,
            patch_size=None,  # Full images for validation
            augment=False,
            normalize=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return train_loader, val_loader


if __name__ == "__main__":
    import sys

    # Test with actual data
    test_dir = "Data-public/test/NoisyLR"
    if os.path.exists(test_dir):
        ds = InferenceDataset(test_dir)
        print(f"Test dataset: {len(ds)} images")
        sample = ds[0]
        print(f"  Shape: {sample['noisy'].shape}")
        print(f"  Dtype: {sample['noisy'].dtype}")
        print(f"  Range: [{sample['noisy'].min():.4f}, {sample['noisy'].max():.4f}]")
        print(f"  Name:  {sample['filename']}")
    else:
        print(f"Test directory not found: {test_dir}")
