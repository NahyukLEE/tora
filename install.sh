#!/bin/bash
# Run this script with the 'tora' conda environment already activated:
#   conda create -n tora python=3.10 -y && conda activate tora
set -e

pip install --upgrade pip

echo "Installing PyTorch (CUDA 12.4)..."
pip install \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    xformers==0.0.29 \
    --index-url https://download.pytorch.org/whl/cu124

echo "Installing PyG packages..."
pip install \
    torch-scatter \
    torch-sparse \
    torch-cluster \
    torch-spline-conv \
    --find-links https://data.pyg.org/whl/torch-2.5.1+cu124.html

echo "Installing PyTorch3D..."
pip install "https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/pytorch3d-0.7.8-cp310-cp310-linux_x86_64.whl"

echo "Installing Flash Attention..."
pip install "https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl"

echo "Installing remaining dependencies..."
pip install \
    diffusers==0.33.0 \
    ninja \
    lightning==2.5.2 \
    torchmetrics==1.6.3 \
    trimesh==4.6.4 \
    addict==2.4.0 \
    scipy==1.15.2 \
    h5py==3.13.0 \
    tqdm==4.67.1 \
    hydra-core==1.3.2 \
    wandb==0.20.1 \
    mitsuba==3.6.4 \
    matplotlib==3.10.3 \
    rich==14.0.0 \
    "huggingface-hub>=0.27.0" \
    "timm>=1.0.0" \
    "einops>=0.8.0" \
    "easydict>=1.13" \
    "spconv-cu124>=2.3" \
    "transformers>=4.45" \
    "opencv-python>=4.10" \
    "open-clip-torch>=2.26"

echo "Installing teacher extra dependencies..."
pip install deepspeed
pip install plyfile fvcore torch-geometric

# pointnet2_ops hardcodes sm_37 which is unsupported in CUDA 12.x — clone, patch, and install
echo "Installing pointnet2_ops..."
git clone https://github.com/erikwijmans/Pointnet2_PyTorch.git /tmp/Pointnet2_PyTorch
sed -i 's|os.environ\["TORCH_CUDA_ARCH_LIST"\] = ".*"|os.environ["TORCH_CUDA_ARCH_LIST"] = "5.0;6.0;6.1;6.2;7.0;7.5;8.0;8.6;8.9;9.0"|' /tmp/Pointnet2_PyTorch/pointnet2_ops_lib/setup.py
pip install --no-build-isolation /tmp/Pointnet2_PyTorch/pointnet2_ops_lib
rm -rf /tmp/Pointnet2_PyTorch

pip install --no-build-isolation "git+https://github.com/unlimblue/KNN_CUDA.git"

echo "Done!"
