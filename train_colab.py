# WaveSemiNet — Google Colab Training Notebook
# =============================================
# Upload this to Google Colab and run with GPU runtime.
# Runtime → Change runtime type → T4 GPU
#
# This notebook:
# 1. Clones your repo & installs dependencies
# 2. Uploads/extracts the dataset
# 3. Trains WaveSemiNet for 200 epochs (~2-3 hours on T4)
# 4. Evaluates on test set
# 5. Downloads trained weights

# %% [markdown]
# # 🔬 WaveSemiNet — Semiconductor Image Restoration
# **SEMICON India Hackathon 2026**
#
# > Wavelet-Guided Dual-Branch Restoration Network
#
# **Runtime Setup:** Go to `Runtime → Change runtime type → T4 GPU`

# %% [markdown]
# ## 1. Setup & Installation

# %%
# Verify GPU
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB")
else:
    raise RuntimeError("❌ No GPU detected! Go to Runtime → Change runtime type → T4 GPU")

# %%
# Clone the repository
!git clone https://github.com/theasynch/I4C-SEMICON-India-Hackathon.git
%cd I4C-SEMICON-India-Hackathon

# %%
# Install dependencies
!pip install -q einops lpips PyWavelets albumentations timm torchmetrics scikit-image tensorboard onnx onnxruntime

# %% [markdown]
# ## 2. Dataset Setup
#
# **Option A:** If your data is already in the repo (Data-public/ was committed)
#
# **Option B:** Upload the zip files manually

# %%
import os

# Check if data already exists
train_exists = os.path.exists("Data-public/train/train/NoisyLR")
test_exists = os.path.exists("Data-public/test/NoisyLR")

if train_exists and test_exists:
    print("✅ Dataset already present!")
    print(f"   Train NoisyLR: {len(os.listdir('Data-public/train/train/NoisyLR'))} files")
    print(f"   Train GT: {len(os.listdir('Data-public/train/train/GT'))} files")
    print(f"   Test NoisyLR: {len(os.listdir('Data-public/test/NoisyLR'))} files")
else:
    print("⚠️ Dataset not found in repo. Upload your zip files below.")

# %%
# === OPTION B: Upload zip files ===
# Uncomment and run this cell if data is NOT in the repo

# from google.colab import files
# print("Upload train.zip and Test_NoisyLR.zip:")
# uploaded = files.upload()
#
# # Extract
# import zipfile
# os.makedirs("Data-public/train", exist_ok=True)
# os.makedirs("Data-public/test", exist_ok=True)
#
# if "train.zip" in uploaded:
#     with zipfile.ZipFile("train.zip", 'r') as z:
#         z.extractall("Data-public/train/")
#     print("✅ Training data extracted")
#
# if "Test_NoisyLR.zip" in uploaded:
#     with zipfile.ZipFile("Test_NoisyLR.zip", 'r') as z:
#         z.extractall("Data-public/test/")
#     print("✅ Test data extracted")

# %%
# === OPTION C: Upload to Google Drive and mount ===
# Uncomment if you prefer Google Drive

# from google.colab import drive
# drive.mount('/content/drive')
# !cp /content/drive/MyDrive/SEMICON/train.zip Data-public/
# !cp /content/drive/MyDrive/SEMICON/Test_NoisyLR.zip Data-public/
# !cd Data-public && unzip -q train.zip -d train/ && unzip -q Test_NoisyLR.zip -d test/

# %% [markdown]
# ## 3. Verify Setup — Quick Smoke Test

# %%
# Quick model test (just model build + forward)
from models.waveseminet import build_waveseminet
import yaml

with open('configs/train_unified.yaml', 'r') as f:
    config = yaml.safe_load(f)

model = build_waveseminet(config)
device = torch.device('cuda')
model = model.to(device)

x = torch.randn(1, 1, 128, 128, device=device)
with torch.no_grad():
    y = model(x, task_id=0)

print(f"✅ Model built: {model.count_parameters():,} parameters")
print(f"   Input:  {x.shape}")
print(f"   Output: {y.shape}")

branch_params = model.get_branch_params()
for name, count in branch_params.items():
    print(f"   {name}: {count:,}")

del model, x, y
torch.cuda.empty_cache()

# %% [markdown]
# ## 4. Training
#
# Full training: 200 epochs, ~2-3 hours on T4 GPU

# %%
# Train with the unified config
!python train.py --config configs/train_unified.yaml --gpu 0

# %% [markdown]
# ## 5. Evaluation on Test Set

# %%
# Evaluate using the best checkpoint
!python evaluate.py \
    --weights weights/best.pth \
    --data Data-public/test/NoisyLR \
    --config configs/train_unified.yaml \
    --output results/ \
    --save_viz \
    --num_viz 20

# %% [markdown]
# ## 6. Visualize Results

# %%
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path

viz_dir = Path("results/visualizations")
viz_files = sorted(viz_dir.glob("*.png"))[:6]

if viz_files:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    for ax, vf in zip(axes.flat, viz_files):
        img = mpimg.imread(str(vf))
        ax.imshow(img)
        ax.set_title(vf.stem.replace("_comparison", ""), fontsize=10)
        ax.axis('off')
    plt.suptitle("WaveSemiNet Restoration Results", fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print("No visualizations found. Run evaluation first.")

# %%
# Show training curves from logs
import numpy as np

# Load training history if available
log_dir = Path("logs")
if log_dir.exists():
    print("TensorBoard logs available. Run: %tensorboard --logdir logs/")
else:
    print("No TensorBoard logs. Training progress was printed to console.")

# %% [markdown]
# ## 7. Export to ONNX (Optional)

# %%
# Export for deployment
!python scripts/export_onnx.py \
    --weights weights/best.pth \
    --config configs/train_unified.yaml \
    --output weights/waveseminet.onnx

# %% [markdown]
# ## 8. Download Results

# %%
# Pack up everything for download
!zip -r waveseminet_results.zip weights/ results/

from google.colab import files
files.download('waveseminet_results.zip')
print("✅ Download started! Contains trained weights + restoration results.")

# %% [markdown]
# ## 9. Quick Inference on Custom Image (Optional)

# %%
# Upload and restore a custom image
# from google.colab import files
# uploaded = files.upload()
#
# import numpy as np
# for filename in uploaded:
#     !python inference.py --input {filename} --output restored_{filename} \
#         --weights weights/best.pth --config configs/train_unified.yaml
#     print(f"Restored: restored_{filename}")
