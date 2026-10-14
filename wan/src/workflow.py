"""Job input validation and the ComfyUI (API format) workflows for Wan 2.2 A14B.

Both workflows follow ComfyUI's official "Wan 2.2 14B + LightX2V 4-step LoRA"
templates: the high-noise expert runs the first half of the steps and the
low-noise expert finishes, each with its distillation LoRA and cfg=1.
"""

import random
from dataclasses import dataclass
from pathlib import PurePosixPath

from .models import MODE_FILES, SHARED_FILES

FPS = 16  # Wan 2.2 A14B is trained at 16 fps
DEFAULT_NEGATIVE = (
    "blurry, low quality, worst quality, static, distorted, deformed, "
    "extra limbs, watermark, text, subtitles"
)
TARGET_AREA = 832 * 480  # default resolution (480p) when only an image is given


def _name(path: str) -> str:
    return PurePosixPath(path).name


def _files(mode: str) -> dict[str, str]:
    high, low, lora_high, lora_low = (_name(f) for f in MODE_FILES[mode])
    text_encoder, vae = (_name(f) for f in SHARED_FILES)
    return {
        "high": high,
        "low": low,
        "lora_high": lora_high,
        "lora_low": lora_low,
        "text_encoder": text_encoder,
        "vae": vae,
    }


@dataclass
class Job:
    prompt: str
    negative_prompt: str
    width: int
    height: int
    length: int
    steps: int
    cfg: float
    shift: float
    seed: int
    image: str | None  # base64 / data URL / http(s) URL; None means text-to-video

    @property
    def mode(self) -> str:
        return "i2v" if self.image else "t2v"


def _int(values: dict, key: str, default: int, lo: int, hi: int) -> int:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not lo <= value <= hi:
        msg = f"'{key}' must be an integer between {lo} and {hi}"
        raise ValueError(msg)
    return value


def _float(values: dict, key: str, default: float, lo: float, hi: float) -> float:
    value = values.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int | float) or not lo <= value <= hi:
        msg = f"'{key}' must be a number between {lo} and {hi}"
        raise ValueError(msg)
    return float(value)


def size_for_image(image_size: tuple[int, int]) -> tuple[int, int]:
    """Width/height with the image's aspect ratio and about TARGET_AREA pixels."""
    w, h = image_size
    scale = (TARGET_AREA / (w * h)) ** 0.5
    return max(256, round(w * scale / 16) * 16), max(256, round(h * scale / 16) * 16)


def parse_job(values: dict, image_size: tuple[int, int] | None = None) -> Job:
    """Validate the job input. ``image_size`` is the decoded input image's size."""
    prompt = values.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        msg = "'prompt' is required"
        raise ValueError(msg)

    default_w, default_h = size_for_image(image_size) if image_size else (832, 480)
    width = _int(values, "width", default_w, 256, 1280)
    height = _int(values, "height", default_h, 256, 1280)
    if width % 16 or height % 16:
        msg = "'width' and 'height' must be multiples of 16"
        raise ValueError(msg)

    length = _int(values, "length", 81, 5, 121)
    if (length - 1) % 4:
        msg = "'length' (frames) must be 4n+1, e.g. 33, 49, 81"
        raise ValueError(msg)

    seed = values.get("seed")
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    seed = _int({"seed": seed}, "seed", 0, 0, 2**63 - 1)

    return Job(
        prompt=prompt,
        negative_prompt=values.get("negative_prompt") or DEFAULT_NEGATIVE,
        width=width,
        height=height,
        length=length,
        steps=_int(values, "steps", 4, 2, 50),
        cfg=_float(values, "cfg", 1.0, 0.0, 20.0),
        shift=_float(values, "shift", 5.0, 0.0, 20.0),
        seed=seed,
        image=values.get("image"),
    )


def build_workflow(job: Job, image_name: str | None, filename_prefix: str) -> dict:
    """ComfyUI API-format workflow. ``image_name`` is a file in ComfyUI's input dir."""
    f = _files(job.mode)
    split = job.steps // 2

    wf: dict = {
        "1": {"class_type": "UNETLoader", "inputs": {"unet_name": f["high"], "weight_dtype": "default"}},
        "2": {"class_type": "UNETLoader", "inputs": {"unet_name": f["low"], "weight_dtype": "default"}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["1", 0], "lora_name": f["lora_high"], "strength_model": 1.0}},
        "4": {"class_type": "LoraLoaderModelOnly", "inputs": {"model": ["2", 0], "lora_name": f["lora_low"], "strength_model": 1.0}},
        "5": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["3", 0], "shift": job.shift}},
        "6": {"class_type": "ModelSamplingSD3", "inputs": {"model": ["4", 0], "shift": job.shift}},
        "7": {"class_type": "CLIPLoader", "inputs": {"clip_name": f["text_encoder"], "type": "wan", "device": "default"}},
        "8": {"class_type": "CLIPTextEncode", "inputs": {"text": job.prompt, "clip": ["7", 0]}},
        "9": {"class_type": "CLIPTextEncode", "inputs": {"text": job.negative_prompt, "clip": ["7", 0]}},
        "10": {"class_type": "VAELoader", "inputs": {"vae_name": f["vae"]}},
    }

    size = {"width": job.width, "height": job.height, "length": job.length, "batch_size": 1}
    if job.mode == "i2v":
        if not image_name:
            msg = "image-to-video needs an input image"
            raise ValueError(msg)
        wf["12"] = {"class_type": "LoadImage", "inputs": {"image": image_name}}
        wf["11"] = {
            "class_type": "WanImageToVideo",
            "inputs": {"positive": ["8", 0], "negative": ["9", 0], "vae": ["10", 0], "start_image": ["12", 0], **size},
        }
        positive, negative, latent = ["11", 0], ["11", 1], ["11", 2]
    else:
        wf["11"] = {"class_type": "EmptyHunyuanLatentVideo", "inputs": size}
        positive, negative, latent = ["8", 0], ["9", 0], ["11", 0]

    sampler = {
        "noise_seed": job.seed,
        "steps": job.steps,
        "cfg": job.cfg,
        "sampler_name": "euler",
        "scheduler": "simple",
        "positive": positive,
        "negative": negative,
    }
    wf["13"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            **sampler,
            "model": ["5", 0],
            "add_noise": "enable",
            "latent_image": latent,
            "start_at_step": 0,
            "end_at_step": split,
            "return_with_leftover_noise": "enable",
        },
    }
    wf["14"] = {
        "class_type": "KSamplerAdvanced",
        "inputs": {
            **sampler,
            "model": ["6", 0],
            "add_noise": "disable",
            "latent_image": ["13", 0],
            "start_at_step": split,
            "end_at_step": 10000,
            "return_with_leftover_noise": "disable",
        },
    }
    wf["15"] = {"class_type": "VAEDecode", "inputs": {"samples": ["14", 0], "vae": ["10", 0]}}
    wf["16"] = {"class_type": "CreateVideo", "inputs": {"images": ["15", 0], "fps": float(FPS)}}
    wf["17"] = {
        "class_type": "SaveVideo",
        "inputs": {"video": ["16", 0], "filename_prefix": filename_prefix, "format": "mp4", "codec": "h264"},
    }
    return wf
