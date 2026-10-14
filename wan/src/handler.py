"""Runpod serverless handler: Wan 2.2 A14B text-to-video and image-to-video."""

import os
import uuid

import runpod

from src.comfy_bootstrap import check_cuda_compatibility
from src.video_output import video_result

from . import comfy_client, models
from .inputs import load_image
from .workflow import build_workflow, parse_job

JOB_TIMEOUT = float(os.environ.get("WAN_JOB_TIMEOUT", "1800"))


def generate(job: dict) -> dict:
    values = job.get("input") or {}
    image_path = None
    try:
        image = load_image(values["image"]) if values.get("image") else None
        params = parse_job(values, image.size if image else None)
        if params.mode not in MODES:
            msg = f"This worker serves {list(MODES)} only (WAN_MODES); got {params.mode}"
            raise ValueError(msg)

        image_name = None
        if image is not None:
            image_name = f"wan-{uuid.uuid4().hex}.png"
            image_path = comfy_client.COMFYUI_DIR / "input" / image_name
            image.save(image_path)

        workflow = build_workflow(params, image_name, f"wan/{job.get('id', uuid.uuid4().hex)}")
        print(f"{params.mode}: {params.width}x{params.height}, {params.length} frames, "
              f"{params.steps} steps, seed {params.seed}", flush=True)
        entry = comfy_client.wait(comfy_client.queue(workflow), COMFY, JOB_TIMEOUT)

        videos = comfy_client.output_files(entry)
        if not videos:
            msg = "ComfyUI finished but saved no video"
            raise comfy_client.ComfyError(msg)
        result = video_result(videos[-1])
        result.update(mode=params.mode, seed=params.seed, width=params.width,
                      height=params.height, frames=params.length, fps=16)
        return result
    except Exception as e:
        print(f"Generation failed: {e}", flush=True)
        result = {"status": "ERROR", "error": str(e)}
        if COMFY.poll() is not None:
            result["refresh_worker"] = True  # ComfyUI died; let Runpod replace the worker
        return result
    finally:
        if image_path is not None:
            image_path.unlink(missing_ok=True)


check_cuda_compatibility()
MODES = models.enabled_modes()
MODELS_DIR = models.models_dir()
models.ensure_models(MODES, MODELS_DIR)
COMFY = comfy_client.start(
    models.write_extra_model_paths(MODELS_DIR, comfy_client.COMFYUI_DIR / "extra_model_paths.yaml")
)

if __name__ == "__main__":
    runpod.serverless.start({"handler": generate})
