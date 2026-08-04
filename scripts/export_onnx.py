"""
WaveSemiNet — ONNX Export Script

Exports the trained WaveSemiNet model to ONNX format for
optimized inference and deployment.

Usage:
    python scripts/export_onnx.py --weights weights/best.pth --output weights/waveseminet.onnx
"""

import argparse
import time

import numpy as np
import torch

from models.waveseminet import build_waveseminet


class WaveSemiNetWrapper(torch.nn.Module):
    """Wrapper that removes task_id from forward for ONNX export."""
    
    def __init__(self, model):
        super().__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x, task_id=0)


def export_onnx(weights_path: str, output_path: str,
                input_size: tuple = (1, 1, 128, 128),
                config: dict | None = None):
    """Export model to ONNX format."""
    
    device = torch.device('cpu')
    
    # Build and load model
    model = build_waveseminet(config)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    model.eval()
    wrapper = WaveSemiNetWrapper(model)
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export
    print(f"Exporting to ONNX: {output_path}")
    torch.onnx.export(
        wrapper,
        dummy_input,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch', 2: 'height', 3: 'width'},
            'output': {0: 'batch', 2: 'height', 3: 'width'},
        },
        opset_version=17,
        do_constant_folding=True,
    )
    
    print(f"ONNX model saved: {output_path}")
    
    # Verify
    import onnx
    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")
    
    # Benchmark with ONNX Runtime
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        input_data = dummy_input.numpy()
        
        # Warmup
        for _ in range(3):
            session.run(None, {'input': input_data})
        
        # Benchmark
        times = []
        for _ in range(20):
            t0 = time.time()
            session.run(None, {'input': input_data})
            times.append(time.time() - t0)
        
        avg_time = np.mean(times) * 1000
        print(f"\nONNX Runtime inference: {avg_time:.1f} ms/image")
        print(f"Throughput: {1000/avg_time:.1f} images/sec")
        
    except ImportError:
        print("ONNX Runtime not installed, skipping benchmark")


def main():
    parser = argparse.ArgumentParser(description='Export WaveSemiNet to ONNX')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--output', type=str, default='weights/waveseminet.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--config', type=str, default=None,
                        help='Model config YAML')
    args = parser.parse_args()
    
    config = None
    if args.config:
        import yaml
        with open(args.config) as f:
            config = yaml.safe_load(f)
    
    export_onnx(args.weights, args.output, config=config)


if __name__ == "__main__":
    main()
