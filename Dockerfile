FROM nvidia/cuda:12.6.2-cudnn-devel-ubuntu22.04

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV TERM=xterm-256color

# Install basic system dependencies and Git
RUN apt update -y && apt install -y \
    software-properties-common \
    build-essential \
    libgl1 \
    libglib2.0-0 \
    zlib1g-dev \
    libncurses5-dev \
    libgdbm-dev \
    libnss3-dev \
    libssl-dev \
    libreadline-dev \
    libffi-dev \
    google-perftools \
    python3.10 \
    python3.10-venv \
    python3-pip \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt update -y \
    && apt install -y \
    git \
    git-lfs \
    sudo \
    nano \
    aria2 \
    curl \
    wget \
    unzip \
    unrar \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.10 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Verify Python version
RUN python3.10 --version && pip3.10 --version

# Install PyTorch and related packages
RUN pip3.10 install --no-cache-dir \
    torch==2.5.0+cu124 \
    torchvision==0.20.0+cu124 \
    torchaudio==2.5.0+cu124 \
    torchtext==0.18.0 \
    torchdata==0.8.0 \
    --extra-index-url https://download.pytorch.org/whl/cu124

# Install AI/ML specific packages
RUN pip3.10 install --no-cache-dir \
    xformers==0.0.28.post2 \
    https://github.com/camenduru/wheels/releases/download/torch-2.5.0-cu124/flash_attn-2.6.3-cp310-cp310-linux_x86_64.whl \
    opencv-python \
    imageio \
    imageio-ffmpeg \
    ffmpeg-python \
    av \
    runpod \
    torchsde \
    einops \
    diffusers \
    transformers \
    accelerate \
    git+https://github.com/timpietrusky/vercel_blob.git

# Clone ComfyUI and custom nodes
RUN git clone https://github.com/comfyanonymous/ComfyUI /ComfyUI && \
    git clone -b dev https://github.com/camenduru/ComfyUI-MochiWrapper /ComfyUI/custom_nodes/ComfyUI-MochiWrapper && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git clone https://github.com/kijai/ComfyUI-KJNodes /ComfyUI/custom_nodes/ComfyUI-KJNodes && \
    git clone https://github.com/ltdrdata/ComfyUI-Manager /ComfyUI/custom_nodes/ComfyUI-Manager && \
    git clone https://github.com/ciri/comfyui-model-downloader /ComfyUI/custom_nodes/comfyui-model-downloader

# Download model files
RUN mkdir -p /ComfyUI/models/diffusion_models/mochi && \
    mkdir -p /ComfyUI/models/vae/mochi && \
    mkdir -p /ComfyUI/models/clip && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/Kijai/Mochi_preview_comfy/resolve/main/mochi_preview_dit_bf16.safetensors \
    -d /ComfyUI/models/diffusion_models/mochi -o mochi_preview_dit_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/Kijai/Mochi_preview_comfy/resolve/main/mochi_preview_vae_decoder_bf16.safetensors \
    -d /ComfyUI/models/vae/mochi -o mochi_preview_vae_decoder_bf16.safetensors && \
    aria2c --console-log-level=error -c -x 16 -s 16 -k 1M \
    https://huggingface.co/mcmonkey/google_t5-v1_1-xxl_encoderonly/resolve/main/model.safetensors \
    -d /ComfyUI/models/clip -o google_t5-v1_1-xxl_encoderonly-fp16.safetensors

# Copy source code and start script
COPY ./src /ComfyUI/src
COPY ./src/start.sh /start.sh
RUN chmod +x /start.sh

ENTRYPOINT ["/start.sh"]