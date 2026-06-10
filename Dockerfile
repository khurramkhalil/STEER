# Use NVIDIA PyTorch base image for A100 support (CUDA 12.4)
# This image already contains torch, torchvision, etc.
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

# Set working directory
WORKDIR /workspace/STEER

# Install system dependencies
# git: for installing packages from git
# ninja-build: for compiling CUDA extensions (adam-atan2, flash-attn etc)
RUN apt-get update && apt-get install -y \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

# 1. Install Python dependencies (Rarely changes)
COPY requirements.txt .
# We rely on the base image's torch. requirements.txt has torch commented out.
RUN pip install --no-cache-dir -r requirements.txt

# 2. Copy stable modules (Unlikely to change often)
COPY dataset/ dataset/
COPY utils/ utils/
COPY config/ config/
COPY evaluators/ evaluators/
COPY puzzle_dataset.py .

# 3. Copy Model definitions (Changes occasionally)
COPY models/ models/

# 4. Copy Research Code (Changes frequently)
# This layer will be rebuilt most often, but it's small/fast.
COPY steer/ steer/
COPY pretrain.py .

# Set environment variables for A100
ENV CUDA_HOME=/usr/local/cuda
ENV TORCH_CUDA_ARCH_LIST="7.0 7.5 8.0 8.6 9.0"

# Default command
CMD ["/bin/bash"]
