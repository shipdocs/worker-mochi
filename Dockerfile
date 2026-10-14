# CUDA 12.8 runtime: matches the cu128 PyTorch wheels (which bundle cuDNN etc.).
# Hosts need an NVIDIA driver that supports CUDA >= 12.8 (R570+).
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9

# Pinned upstream revisions. Bump deliberately; `src/selftest.py` runs at the end
# of the build and fails it if ComfyUI or the custom nodes no longer import.
ARG COMFYUI_REF=v0.3.68
ARG MOCHI_WRAPPER_REF=d84ce774e94d1963c5d236f4ffedd2a93c87390f
ARG VHS_REF=8923bd836bdab8b7bbdf4ed104b7d045e70c66e2

WORKDIR /

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV TERM=xterm-256color
ENV PIP_NO_CACHE_DIR=1
ENV COMFYUI_DIR=/ComfyUI

# Install basic system dependencies and Git
RUN apt update -y && apt install -y \
    software-properties-common \
    libgl1 \
    libglib2.0-0 \
    google-perftools \
    python3.10 \
    python3.10-venv \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt update -y \
    && apt install -y \
    git \
    aria2 \
    curl \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Isolated Python 3.10 environment, so pinned packages never clash with apt's.
RUN python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Python dependencies (PyTorch 2.8.0+cu128 and everything else, fully pinned).
# See requirements-worker.in for how to regenerate the lock file.
COPY requirements-worker.txt /tmp/requirements-worker.txt
RUN python3 -m pip install --upgrade pip==25.2 && \
    python3 -m pip install -r /tmp/requirements-worker.txt && \
    rm /tmp/requirements-worker.txt

# Clone ComfyUI and the custom nodes the handler uses, at pinned revisions
RUN git clone https://github.com/comfyanonymous/ComfyUI /ComfyUI && \
    git -C /ComfyUI checkout --detach ${COMFYUI_REF} && \
    git clone https://github.com/camenduru/ComfyUI-MochiWrapper /ComfyUI/custom_nodes/ComfyUI-MochiWrapper && \
    git -C /ComfyUI/custom_nodes/ComfyUI-MochiWrapper checkout --detach ${MOCHI_WRAPPER_REF} && \
    git clone https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite && \
    git -C /ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite checkout --detach ${VHS_REF}

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

# Smoke test (no GPU needed): ComfyUI imports the way the handler does them,
# custom nodes load, and torch has kernels for Ampere..Blackwell.
WORKDIR /ComfyUI
RUN python3 -m src.selftest

ENTRYPOINT ["/start.sh"]
