# Wan 2.2 A14B (text-to-video + image-to-video) with LightX2V 4-step LoRAs.
# Build context is the repository root:  docker build -f wan/Dockerfile .
# The model weights are NOT in the image; the worker downloads them on startup
# (see wan/README.md).

# CUDA 12.8 runtime: matches the cu128 PyTorch wheels (which bundle cuDNN etc.).
# Hosts need an NVIDIA driver that supports CUDA >= 12.8 (R570+).
FROM nvidia/cuda:12.8.1-runtime-ubuntu22.04@sha256:4a801ef9232d2b05e69df4eb8aa054dbbe2824e5499e1e6e857320bb01ac41a9

ARG COMFYUI_REF=v0.3.68

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=True
ENV PIP_NO_CACHE_DIR=1
ENV COMFYUI_DIR=/ComfyUI

RUN apt update -y && apt install -y \
    software-properties-common \
    libgl1 \
    libglib2.0-0 \
    google-perftools \
    python3.10 \
    python3.10-venv \
    && add-apt-repository -y ppa:git-core/ppa \
    && apt update -y \
    && apt install -y git curl ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Isolated Python 3.10 environment, so pinned packages never clash with apt's.
RUN python3.10 -m venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="/opt/venv/bin:${PATH}"

# Same locked dependencies as the Mochi worker (PyTorch 2.8.0+cu128, ComfyUI v0.3.68).
COPY requirements-worker.txt /tmp/requirements-worker.txt
RUN python3 -m pip install --upgrade pip==25.2 && \
    python3 -m pip install -r /tmp/requirements-worker.txt && \
    rm /tmp/requirements-worker.txt

RUN git clone https://github.com/comfyanonymous/ComfyUI /ComfyUI && \
    git -C /ComfyUI checkout --detach ${COMFYUI_REF}

COPY src /worker/src
COPY wan/__init__.py /worker/wan/__init__.py
COPY wan/src /worker/wan/src
COPY wan/start.sh /start.sh
RUN chmod +x /start.sh

# Smoke test (no GPU, no weights): ComfyUI starts and accepts both workflows,
# and torch has kernels for Ampere..Blackwell.
WORKDIR /worker
RUN python3 -m wan.src.selftest

ENTRYPOINT ["/start.sh"]
