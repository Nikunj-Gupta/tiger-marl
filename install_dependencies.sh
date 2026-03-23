#!/bin/bash
# Install dependencies for TIGER-MARL
#
# Prerequisites: conda environment with Python 3.10
#   conda create -n tiger python=3.10 -y
#   conda activate tiger

set -e

LOG=install_$(date +%Y%m%d_%H%M%S).log
exec > >(tee "$LOG") 2>&1
echo "Logging to $LOG"

# --- PyTorch ---
PLATFORM=$(uname -s)-$(uname -m)
echo "Detected platform: $PLATFORM"

if [[ "$PLATFORM" == "Linux-x86_64" ]]; then
    # Linux: install with CUDA 12.1
    conda install pytorch=2.5.1 torchvision torchaudio pytorch-cuda=12.1 \
        -c pytorch -c nvidia -y
    # Fix iJIT_NotifyEvent / MKL symbol conflict
    pip install mkl
else
    # macOS (Apple Silicon or Intel): CPU/MPS only
    conda install pytorch torchvision torchaudio -c pytorch -y
fi

# --- Core Python packages ---
pip install \
    protobuf==3.20.* \
    sacred \
    numpy \
    scipy \
    gymnasium \
    matplotlib \
    seaborn \
    pyyaml \
    pygame \
    pytest \
    probscale \
    imageio \
    snakeviz \
    tensorboard-logger \
    pymongo \
    setproctitle \
    torch-tb-profiler

# --- SMAC v2 ---
pip install git+https://github.com/oxwhirl/smacv2.git

# --- PyTorch Geometric ---
# Install prebuilt sparse/scatter wheels BEFORE torch_geometric_temporal,
# which would otherwise pull torch_sparse from source (requires torch at build time).
if [[ "$PLATFORM" == "Linux-x86_64" ]]; then
    pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
        -f https://data.pyg.org/whl/torch-2.5.1+cu121.html
else
    # pyg_lib has no macOS wheels — skip it
    pip install torch_scatter torch_sparse torch_cluster torch_spline_conv \
        --no-index -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
fi
pip install torch_geometric==2.4.0
pip install torch_geometric_temporal==0.54.0 --no-deps
pip install decorator==4.4.2 cython "pandas<=1.3.5"  # torch_geometric_temporal deps

# --- Additional environments ---
pip install pogema pogema-toolbox
pip install pettingzoo "vmas[gymnasium]"
