"""Make ComfyUI importable from the worker and check the GPU is supported.

ComfyUI's ``nodes.py`` runs ``sys.path.insert(0, "<ComfyUI>/comfy")`` at import time.
From then on ``comfy/utils.py`` shadows ComfyUI's top-level ``utils/`` package, so a
later ``from utils.json_util import ...`` (``server`` -> ``app.custom_node_manager``)
fails with::

    ModuleNotFoundError: No module named 'utils.json_util'; 'utils' is not a package

ComfyUI's own ``main.py`` avoids this by importing ``utils`` before ``nodes``. The
worker imports ``server`` directly, so it has to do the same, which is what
:func:`prepare_comfyui_imports` does.
"""

import asyncio
import importlib
import inspect
import os
import sys
from pathlib import Path

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/ComfyUI"))

CUSTOM_NODES = ("ComfyUI-MochiWrapper", "ComfyUI-VideoHelperSuite")
REQUIRED_NODES = (
    "CLIPLoader",
    "DownloadAndLoadMochiModel",
    "MochiTextEncode",
    "MochiSampler",
    "MochiDecode",
    "VHS_VideoCombine",
)


def prepare_comfyui_imports(comfyui_dir: Path = COMFYUI_DIR) -> None:
    """Put ComfyUI first on ``sys.path`` and bind ``utils`` to ComfyUI's package.

    Must run before ``import server`` / ``import nodes``.
    """
    comfyui_dir = Path(comfyui_dir).resolve()
    if not (comfyui_dir / "utils" / "__init__.py").is_file():
        msg = f"ComfyUI 'utils' package not found under {comfyui_dir}"
        raise RuntimeError(msg)

    path = str(comfyui_dir)
    if path in sys.path:
        sys.path.remove(path)
    sys.path.insert(0, path)

    utils = sys.modules.get("utils")
    if utils is not None and not _is_comfyui_utils(utils, comfyui_dir):
        # Something else (e.g. comfy/utils.py) was already imported as ``utils``.
        del sys.modules["utils"]

    # Once cached in sys.modules, ``utils.<submodule>`` is looked up in the package's
    # own __path__, so the later sys.path change made by nodes.py no longer matters.
    utils = importlib.import_module("utils")
    if not _is_comfyui_utils(utils, comfyui_dir):
        msg = f"'utils' resolved to {getattr(utils, '__file__', utils)!r}, expected the package in {comfyui_dir}"
        raise RuntimeError(msg)


def load_worker_nodes(
    loop: asyncio.AbstractEventLoop,
    custom_nodes: tuple[str, ...] = CUSTOM_NODES,
    required_nodes: tuple[str, ...] = REQUIRED_NODES,
) -> dict:
    """Load the custom nodes the handler uses and return ``NODE_CLASS_MAPPINGS``."""
    from nodes import NODE_CLASS_MAPPINGS, load_custom_node

    for name in custom_nodes:
        result = load_custom_node(str(COMFYUI_DIR / "custom_nodes" / name))
        # load_custom_node became a coroutine in newer ComfyUI versions.
        if inspect.isawaitable(result):
            result = loop.run_until_complete(result)
        if not result:
            msg = f"Failed to load custom node {name}"
            raise RuntimeError(msg)

    missing = [n for n in required_nodes if n not in NODE_CLASS_MAPPINGS]
    if missing:
        msg = f"Missing ComfyUI nodes: {missing}"
        raise RuntimeError(msg)
    return NODE_CLASS_MAPPINGS


def _is_comfyui_utils(module, comfyui_dir: Path) -> bool:
    paths = getattr(module, "__path__", None)
    if paths is None:  # a plain module such as comfy/utils.py, not a package
        return False
    expected = comfyui_dir / "utils"
    return any(Path(p).resolve() == expected for p in paths)


def arch_supported(capability: tuple[int, int], arch_list: list[str]) -> bool:
    """Whether a torch build for ``arch_list`` can run on a GPU of ``capability``.

    Follows CUDA's compatibility rules: ``sm_XY`` binaries run on any ``X.Z`` GPU
    with ``Z >= Y`` (so sm_86 covers Ada sm_89), and ``compute_XY`` PTX can be
    JIT-compiled for any newer GPU. Blackwell (sm_120) needs sm_120 or PTX.
    """
    for entry in arch_list:
        kind, _, digits = entry.partition("_")
        digits = digits.rstrip("abcdefghijklmnopqrstuvwxyz")  # e.g. sm_90a
        if not digits.isdigit() or len(digits) < 2:
            continue
        built = (int(digits[:-1]), int(digits[-1]))
        if kind == "sm" and built[0] == capability[0] and built[1] <= capability[1]:
            return True
        if kind == "compute" and built <= capability:
            return True
    return False


def check_cuda_compatibility() -> None:
    """Fail fast if the installed PyTorch has no kernels for this GPU.

    Without this, an unsupported GPU (e.g. Blackwell sm_120 with a cu124 build)
    only logs a warning at startup and then every job fails with
    "no kernel image is available for execution on the device".
    """
    import torch

    if not torch.cuda.is_available():
        msg = "CUDA is not available; this worker needs an NVIDIA GPU"
        raise RuntimeError(msg)

    capability = torch.cuda.get_device_capability()
    arch = "sm_{}{}".format(*capability)
    supported = torch.cuda.get_arch_list()
    print(
        f"torch {torch.__version__} (CUDA {torch.version.cuda}), "
        f"GPU {torch.cuda.get_device_name()} ({arch}), built for {supported}"
    )
    if not arch_supported(capability, supported):
        msg = (
            f"GPU architecture {arch} is not supported by torch {torch.__version__} "
            f"(built for {supported}). Use a different GPU type for this endpoint "
            "or rebuild the image with a PyTorch build that includes this architecture."
        )
        raise RuntimeError(msg)
