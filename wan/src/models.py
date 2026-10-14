"""Wan 2.2 model files: which ones each mode needs, and downloading them.

The weights (~69 GB for both modes) are too big to bake into a Runpod image, so the
worker downloads them on startup. With a Runpod network volume attached they are
stored there once and reused by every worker; without one, each new worker
downloads them to its container disk.
"""

import os
from pathlib import Path

# Comfy-Org's repackaged Wan 2.2 files, pinned to a commit for reproducibility.
REPO_ID = "Comfy-Org/Wan_2.2_ComfyUI_Repackaged"
REVISION = "ee6f4a40737a995bf5818954cfce6d59443b0f04"

SHARED_FILES = (
    "split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors",
    "split_files/vae/wan_2.1_vae.safetensors",
)

MODE_FILES = {
    "t2v": (
        "split_files/diffusion_models/wan2.2_t2v_high_noise_14B_fp8_scaled.safetensors",
        "split_files/diffusion_models/wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_high_noise.safetensors",
        "split_files/loras/wan2.2_t2v_lightx2v_4steps_lora_v1.1_low_noise.safetensors",
    ),
    "i2v": (
        "split_files/diffusion_models/wan2.2_i2v_high_noise_14B_fp8_scaled.safetensors",
        "split_files/diffusion_models/wan2.2_i2v_low_noise_14B_fp8_scaled.safetensors",
        "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_high_noise.safetensors",
        "split_files/loras/wan2.2_i2v_lightx2v_4steps_lora_v1_low_noise.safetensors",
    ),
}

# ComfyUI model folders, relative to <models_dir>/split_files.
COMFY_FOLDERS = ("diffusion_models", "loras", "text_encoders", "vae")


def enabled_modes() -> tuple[str, ...]:
    """Modes to serve, from WAN_MODES (default: both)."""
    modes = tuple(
        m.strip() for m in os.environ.get("WAN_MODES", "t2v,i2v").split(",") if m.strip()
    )
    unknown = [m for m in modes if m not in MODE_FILES]
    if not modes or unknown:
        msg = f"WAN_MODES must be a comma-separated subset of {list(MODE_FILES)}, got {modes}"
        raise ValueError(msg)
    return modes


def models_dir() -> Path:
    """Network volume if attached (shared, persistent), else the container disk."""
    if "WAN_MODELS_DIR" in os.environ:
        return Path(os.environ["WAN_MODELS_DIR"])
    if Path("/runpod-volume").is_dir():
        return Path("/runpod-volume/wan-models")
    return Path("/ComfyUI/models/wan")


def files_for(modes: tuple[str, ...]) -> tuple[str, ...]:
    return SHARED_FILES + tuple(f for m in modes for f in MODE_FILES[m])


def ensure_models(modes: tuple[str, ...], target: Path) -> None:
    """Download any missing model files into ``target`` (resumable, lock-protected)."""
    from huggingface_hub import hf_hub_download

    for filename in files_for(modes):
        if (target / filename).is_file():
            continue
        print(f"Downloading {filename} ...", flush=True)
        hf_hub_download(REPO_ID, filename, revision=REVISION, local_dir=target)
    print(f"Models ready in {target}", flush=True)


def write_extra_model_paths(target: Path, config_path: Path) -> Path:
    """Point ComfyUI's model folders at the downloaded files."""
    lines = ["wan:", f"  base_path: {target / 'split_files'}"]
    lines += [f"  {folder}: {folder}" for folder in COMFY_FOLDERS]
    config_path.write_text("\n".join(lines) + "\n")
    return config_path
