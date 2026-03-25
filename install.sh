#!/bin/bash

# Activate your virtual environment
# source gnn_env/bin/activate  # Uncomment if not already activated

echo "Installing PyTorch with CUDA 12.1..."
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121

echo "Installing torch-geometric and dependencies..."
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install torch-geometric==2.4.0

echo "Installing other packages..."
pip install numpy tqdm matplotlib

echo "Installation complete!"