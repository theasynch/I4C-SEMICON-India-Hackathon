"""
WaveSemiNet — Single Image Inference

Restore a single degraded semiconductor image.

Usage:
    python inference.py --input path/to/degraded.npy --output path/to/restored.npy --weights weights/best.pth
"""

import argparse
import time
from pathlib import Path

import numpy as np
import torch

from models.waveseminet import build_waveseminet
from evaluate import load_model, restore_image


def main():
    parser = argparse.ArgumentParser(description='WaveSemiNet Single Image Inference')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to degraded image (.npy or image file)')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to save restored image')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--config', type=str, default=None,
                        help='Model config YAML')
    parser.add_argument('--tile_size', type=int, default=None,
                        help='Tile size for large images')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    args = parser.parse_args()

    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')

    # Load config
    config = None
    if args.config:
        import yaml
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)

    # Load model
    model = load_model(args.weights, device, config)
    print(f"Model loaded ({model.count_parameters():,} parameters)")

    # Load input image
    input_path = Path(args.input)
    if input_path.suffix == '.npy':
        img = np.load(str(input_path)).astype(np.float32)
    else:
        from PIL import Image
        img = np.array(Image.open(str(input_path)).convert('L')).astype(np.float32) / 255.0

    img = np.clip(img, 0.0, 1.0)
    print(f"Input: {img.shape}, range [{img.min():.4f}, {img.max():.4f}]")

    # Prepare tensor
    noisy = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Restore
    t0 = time.time()
    restored = restore_image(model, noisy, device, args.tile_size)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    restored_np = restored.squeeze().cpu().numpy()
    restored_np = np.clip(restored_np, 0.0, 1.0)

    print(f"Output: {restored_np.shape}, range [{restored_np.min():.4f}, {restored_np.max():.4f}]")
    print(f"Inference time: {elapsed*1000:.1f} ms")

    # Save output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == '.npy':
        np.save(str(output_path), restored_np)
    else:
        from PIL import Image
        img_out = (restored_np * 255).astype(np.uint8)
        Image.fromarray(img_out, mode='L').save(str(output_path))

    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
