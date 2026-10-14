"""Build-time smoke test for the worker image. Needs no GPU.

Runs the same ComfyUI import/node-loading path as the handler (in CPU mode),
and checks that PyTorch was built for the GPU architectures we deploy on.

Usage (inside the image): ``cd /ComfyUI && python -m src.selftest``
"""

import asyncio
import os

from .comfy_bootstrap import (
    COMFYUI_DIR,
    CUSTOM_NODES,
    REQUIRED_NODES,
    arch_supported,
    load_worker_nodes,
    prepare_comfyui_imports,
)

# Ampere, Ada, Hopper, Blackwell (B200 and RTX 50xx / RTX PRO 6000).
# Override with REQUIRED_CUDA_ARCHS="" (e.g. for a CPU-only dev environment).
REQUIRED_CUDA_ARCHS = os.environ.get(
    "REQUIRED_CUDA_ARCHS", "sm_80 sm_86 sm_89 sm_90 sm_100 sm_120"
).split()

# ComfyUI-MochiWrapper queries the GPU at import time, so it can only be loaded
# when one is present (not during `docker build`).
GPU_ONLY_NODES = {
    "ComfyUI-MochiWrapper": (
        "DownloadAndLoadMochiModel",
        "MochiTextEncode",
        "MochiSampler",
        "MochiDecode",
    ),
}


def check_torch() -> None:
    import torch

    # get_arch_list() returns [] without a GPU; the compiled flags are always there.
    flags = torch._C._cuda_getArchFlags() if torch.version.cuda else ""
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, archs: {flags}")
    missing = [
        arch
        for arch in REQUIRED_CUDA_ARCHS
        if not arch_supported((int(arch[3:-1]), int(arch[-1])), flags.split())
    ]
    if missing:
        msg = f"torch {torch.__version__} is not built for {missing}"
        raise SystemExit(msg)


def check_comfyui() -> None:
    prepare_comfyui_imports()

    from comfy.cli_args import args

    import torch

    custom_nodes, required_nodes = CUSTOM_NODES, REQUIRED_NODES
    if not torch.cuda.is_available():
        args.cpu = True  # no GPU during `docker build`
        skipped = [n for c in GPU_ONLY_NODES.values() for n in c]
        custom_nodes = tuple(n for n in custom_nodes if n not in GPU_ONLY_NODES)
        required_nodes = tuple(n for n in required_nodes if n not in skipped)
        print(f"No GPU: skipping {list(GPU_ONLY_NODES)}")

    import execution
    import server

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    execution.PromptQueue(server.PromptServer(loop))
    load_worker_nodes(loop, custom_nodes, required_nodes)
    print(f"ComfyUI imports and custom nodes OK ({COMFYUI_DIR})")


if __name__ == "__main__":
    check_torch()
    check_comfyui()
    print("selftest passed")
