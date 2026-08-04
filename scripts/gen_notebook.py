"""Generate the Colab training notebook as .ipynb"""
import json

cells = []

def md(source):
    cells.append({
        'cell_type': 'markdown',
        'metadata': {},
        'source': source if isinstance(source, list) else [source]
    })

def code(source):
    cells.append({
        'cell_type': 'code',
        'metadata': {},
        'source': source if isinstance(source, list) else [source],
        'outputs': [],
        'execution_count': None
    })

# Title
md([
    '# 🔬 WaveSemiNet — Semiconductor Image Restoration\n',
    '**SEMICON India Hackathon 2026** | Wavelet-Guided Dual-Branch Restoration Network\n',
    '\n',
    '> **Setup:** Go to `Runtime → Change runtime type → T4 GPU` before running\n',
])

# Setup
md(['## 1. Setup & Installation'])

code([
    '# Verify GPU\n',
    'import torch\n',
    'print(f"PyTorch: {torch.__version__}")\n',
    'print(f"CUDA available: {torch.cuda.is_available()}")\n',
    'if torch.cuda.is_available():\n',
    '    print(f"GPU: {torch.cuda.get_device_name(0)}")\n',
    '    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")\n',
    'else:\n',
    '    raise RuntimeError("No GPU! Go to Runtime > Change runtime type > T4 GPU")\n',
])

code([
    '# Clone repo and install deps\n',
    '!git clone https://github.com/theasynch/I4C-SEMICON-India-Hackathon.git\n',
    '%cd I4C-SEMICON-India-Hackathon\n',
    '!pip install -q einops lpips PyWavelets albumentations timm torchmetrics scikit-image tensorboard onnx onnxruntime\n',
])

# Dataset
md([
    '## 2. Dataset Setup\n',
    'The data is in `.gitignore` so we need to upload it. Choose one option below.',
])

code([
    'import os\n',
    '\n',
    '# Check if data already exists\n',
    'train_ok = os.path.exists("Data-public/train/train/NoisyLR")\n',
    'test_ok = os.path.exists("Data-public/test/NoisyLR")\n',
    '\n',
    'if train_ok and test_ok:\n',
    '    print("Dataset already present!")\n',
    'else:\n',
    '    print("Dataset not found. Run one of the upload cells below.")\n',
])

code([
    '# OPTION A: Google Drive (recommended for ~900MB files)\n',
    '# 1. Upload train.zip and Test_NoisyLR.zip to your Google Drive root\n',
    '# 2. Uncomment and run:\n',
    '\n',
    '# from google.colab import drive\n',
    '# drive.mount("/content/drive")\n',
    '# !cp "/content/drive/MyDrive/train.zip" Data-public/\n',
    '# !cp "/content/drive/MyDrive/Test_NoisyLR.zip" Data-public/\n',
    '# !cd Data-public && unzip -q train.zip -d train/ && unzip -q Test_NoisyLR.zip -d test/\n',
    '# print("Dataset extracted!")\n',
])

code([
    '# OPTION B: Direct browser upload\n',
    '\n',
    '# from google.colab import files\n',
    '# import zipfile\n',
    '# print("Upload train.zip:")\n',
    '# uploaded = files.upload()\n',
    '# os.makedirs("Data-public/train", exist_ok=True)\n',
    '# with zipfile.ZipFile("train.zip", "r") as z:\n',
    '#     z.extractall("Data-public/train/")\n',
    '# print("Upload Test_NoisyLR.zip:")\n',
    '# uploaded = files.upload()\n',
    '# os.makedirs("Data-public/test", exist_ok=True)\n',
    '# with zipfile.ZipFile("Test_NoisyLR.zip", "r") as z:\n',
    '#     z.extractall("Data-public/test/")\n',
])

code([
    '# Verify dataset\n',
    'train_noisy = os.listdir("Data-public/train/train/NoisyLR")\n',
    'train_gt = os.listdir("Data-public/train/train/GT")\n',
    'test_noisy = os.listdir("Data-public/test/NoisyLR")\n',
    'print(f"Train NoisyLR: {len(train_noisy)} files")\n',
    'print(f"Train GT:      {len(train_gt)} files")\n',
    'print(f"Test NoisyLR:  {len(test_noisy)} files")\n',
    'assert len(train_noisy) == len(train_gt), "Train/GT count mismatch!"\n',
    'print("Dataset OK!")\n',
])

# Smoke test
md(['## 3. Quick Smoke Test'])

code([
    'from models.waveseminet import build_waveseminet\n',
    'import yaml\n',
    '\n',
    'with open("configs/train_unified.yaml", "r") as f:\n',
    '    config = yaml.safe_load(f)\n',
    '\n',
    'model = build_waveseminet(config).cuda()\n',
    'x = torch.randn(1, 1, 128, 128, device="cuda")\n',
    'with torch.no_grad():\n',
    '    y = model(x, task_id=0)\n',
    'print(f"Model: {model.count_parameters():,} params")\n',
    'print(f"Input: {x.shape} -> Output: {y.shape}")\n',
    'for name, count in model.get_branch_params().items():\n',
    '    print(f"  {name}: {count:,}")\n',
    'del model, x, y; torch.cuda.empty_cache()\n',
])

# Training
md(['## 4. Train! (~2-3 hours on T4)'])

code(['!python train.py --config configs/train_unified.yaml --gpu 0\n'])

# Evaluation
md(['## 5. Evaluate on Test Set'])

code([
    '!python evaluate.py \\\n',
    '    --weights weights/best.pth \\\n',
    '    --data Data-public/test/NoisyLR \\\n',
    '    --config configs/train_unified.yaml \\\n',
    '    --output results/ \\\n',
    '    --save_viz \\\n',
    '    --num_viz 20\n',
])

# Visualize
md(['## 6. Visualize Results'])

code([
    'import matplotlib.pyplot as plt\n',
    'import matplotlib.image as mpimg\n',
    'from pathlib import Path\n',
    '\n',
    'viz_dir = Path("results/visualizations")\n',
    'viz_files = sorted(viz_dir.glob("*.png"))[:6]\n',
    'if viz_files:\n',
    '    fig, axes = plt.subplots(2, 3, figsize=(18, 12))\n',
    '    for ax, vf in zip(axes.flat, viz_files):\n',
    '        ax.imshow(mpimg.imread(str(vf)))\n',
    '        ax.set_title(vf.stem.replace("_comparison", ""), fontsize=10)\n',
    '        ax.axis("off")\n',
    '    plt.suptitle("WaveSemiNet Restoration Results", fontsize=16, fontweight="bold")\n',
    '    plt.tight_layout()\n',
    '    plt.show()\n',
    'else:\n',
    '    print("No visualizations yet. Run evaluation first.")\n',
])

# ONNX
md(['## 7. Export ONNX (Optional)'])

code([
    '!python scripts/export_onnx.py \\\n',
    '    --weights weights/best.pth \\\n',
    '    --config configs/train_unified.yaml \\\n',
    '    --output weights/waveseminet.onnx\n',
])

# Download
md(['## 8. Download Trained Weights & Results'])

code([
    '!zip -r waveseminet_results.zip weights/ results/\n',
    'from google.colab import files\n',
    'files.download("waveseminet_results.zip")\n',
    'print("Download started! Contains trained weights + restoration results.")\n',
])

# Assemble notebook
nb = {
    'nbformat': 4,
    'nbformat_minor': 0,
    'metadata': {
        'colab': {'provenance': [], 'gpuType': 'T4'},
        'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
        'accelerator': 'GPU'
    },
    'cells': cells
}

with open('train_colab.ipynb', 'w') as f:
    json.dump(nb, f, indent=2)

print(f"Created train_colab.ipynb with {len(cells)} cells")
