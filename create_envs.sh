#!/bin/bash

set -e 

# conda create -n swift-tailor python=3.10 -y
# conda activate swift-tailor
conda install pytorch==1.13.1 torchvision torchaudio pytorch-cuda=11.7 -c pytorch -c nvidia -y

conda config --append channels conda-forge

conda install einops -y
conda install ffmpeg -y
conda install jupyterlab -y
conda install matplotlib -y
conda install munch -y
conda install networkx -y
conda install pandas pillow scikit-learn tqdm yaml -y
conda install -c iopath iopath -y

pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-1.13.1+cu117.html
pip install torch-geometric==2.4.0
pip install warp-lang

# Export CUDA_HOME or CUDA_PATH here if using multiple cuda version
# export CUDA_HOME=/user/local/cuda-xx
# export CUDA_PATH=/user/local/cuda-xx

pip install --extra-index-url=https://pypi.nvidia.com cudf-cu11
pip install cugraph-cu11 --extra-index-url=https://pypi.nvidia.com
pip install smplx aitviewer chumpy scikit-image scipy trimesh loguru


conda install conda-forge::timm -y
pip install --force-reinstall -v "numpy==1.25.2"

pip install svgwrite svgpathtools cairosvg nicegui libigl pyrender cgal wandb black isort flake8 imagesize pytest
pip install git+'https://github.com/otaheri/chamfer_distance'
pip install bitsandbytes
pip install accelerate --no-deps