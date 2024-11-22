# 使用NVIDIA CUDA 12.4基础镜像
FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

# 设置环境变量
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
ENV CACHE_DIR=/app/.cache

# 安装基本依赖
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3.10-dev \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# 创建缓存目录
RUN mkdir -p ${CACHE_DIR}

# 升级pip并安装基本Python包
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# 安装PyTorch和CUDA相关包
RUN pip3 install --no-cache-dir \
    torch \
    torchvision \
    --index-url https://download.pytorch.org/whl/cu124

# 安装项目依赖
RUN pip3 install --no-cache-dir \
    numpy \
    accelerate \
    "lightning>=2.0" \
    "omegaconf>=2.1.1" \
    "transformers>=4.43.0" \
    "datasets>=2.17" \
    "diffusers>=0.29" \
    "bitsandbytes>=0.41" \
    wandb \
    tensorboard \
    ftfy \
    requests \
    timm \
    safetensors \
    h5py \
    "open-clip-torch==2.24.0" \
    einops \
    scipy \
    opencv-python \
    deepspeed

# 设置工作目录
WORKDIR /app

# 设置缓存卷
VOLUME ${CACHE_DIR}

# 设置容器启动命令
CMD ["bash"]