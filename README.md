# WaveSemiNet — Wavelet-Guided Semiconductor Image Restoration

> **SEMICON India Hackathon 2026** | AI-Based Restoration of Degraded Images for Semiconductor Inspection

## Overview

WaveSemiNet is a wavelet-domain dual-branch restoration network specifically designed for semiconductor inspection images. Unlike generic image restoration models, it exploits the unique structural properties of semiconductor layouts — Manhattan geometry, periodic patterns, and sub-pixel defects — to deliver superior restoration while faithfully preserving defect information.

### Key Features
- **Wavelet-domain processing** — Naturally separates structural content from edge/defect details
- **Dual-branch architecture** — NAFNet for low-frequency + Edge-Aware Transformer for high-frequency
- **Unified model** — Single architecture handles Gaussian denoising, Speckle denoising, and Super-Resolution
- **Defect preservation** — Explicit high-frequency branch + edge-aware losses prevent defect smoothing
- **Industrial deployment ready** — ONNX export, efficient inference, tiled processing for large images

## Architecture

```
Input Image → DWT → [LL, LH, HL, HH]
                         │
              ┌──────────┴──────────┐
              │                     │
        NAFNet Branch         Edge-Aware
        (Low-Freq)           Transformer
              │              (High-Freq)
              └──────────┬──────────┘
                         │
               Feature Fusion (FFT Mixing)
                         │
                  IWT Reconstruction
                         │
              Skip ──────┘
                         │
                   Restored Image
```

## Quick Start

### Installation
```bash
git clone https://github.com/<your-username>/I4C-SEMICON-India-Hackathon.git
cd I4C-SEMICON-India-Hackathon
pip install -r requirements.txt
```

### Training
```bash
# Train unified model
python train.py --config configs/train_unified.yaml

# Train task-specific models
python train.py --config configs/train_denoise_gaussian.yaml
python train.py --config configs/train_denoise_speckle.yaml
python train.py --config configs/train_superres.yaml
```

### Evaluation
```bash
# Run full evaluation (this is what judges execute)
python evaluate.py --weights weights/best.pth --data data/test/ --output results/
```

### Single Image Inference
```bash
python inference.py --input path/to/degraded.png --output path/to/restored.png --weights weights/best.pth
```

## Project Structure
```
├── configs/              # Training configurations (YAML)
├── data/                 # Dataset loaders and degradation pipeline
├── models/               # WaveSemiNet architecture components
├── losses/               # Composite loss functions
├── utils/                # Metrics, visualization, checkpointing
├── scripts/              # Dataset analysis, ONNX export
├── train.py              # Training entry point
├── evaluate.py           # Evaluation script (judges run this)
├── inference.py          # Single-image inference
├── weights/              # Trained model weights
└── results/              # Sample restoration outputs
```

## Performance Metrics

| Metric | Description |
|--------|-------------|
| PSNR   | Peak Signal-to-Noise Ratio (dB) |
| SSIM   | Structural Similarity Index |
| LPIPS  | Learned Perceptual Image Patch Similarity |
| DPS    | Defect Preservation Score (custom) |

## Requirements
- Python 3.10+
- PyTorch 2.1+
- CUDA-capable GPU (8GB+ VRAM recommended)

## Team
SEMICON India Hackathon 2026

## License
MIT
