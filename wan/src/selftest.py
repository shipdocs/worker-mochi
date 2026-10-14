"""Build-time smoke test for the Wan image. Needs no GPU and no model weights.

Starts ComfyUI (CPU mode) exactly like the handler does, with empty placeholder
files for every model, and submits both workflows. ComfyUI validates a workflow
before queueing it (node names, inputs, enum values and model file names), so
this catches any mismatch between our workflows and the pinned ComfyUI version.

Usage (inside the image): ``cd /worker && python -m wan.src.selftest``
"""

import tempfile
from pathlib import Path

import requests
from PIL import Image

from src.selftest import check_torch

from . import comfy_client, models
from .workflow import build_workflow, parse_job


def main() -> None:
    check_torch()

    with tempfile.TemporaryDirectory() as tmp:
        target = Path(tmp)
        for name in models.files_for(tuple(models.MODE_FILES)):
            (target / name).parent.mkdir(parents=True, exist_ok=True)
            (target / name).touch()
        config = models.write_extra_model_paths(target, target / "extra_model_paths.yaml")

        image_name = "wan-selftest.png"
        image_path = comfy_client.COMFYUI_DIR / "input" / image_name
        image_path.parent.mkdir(exist_ok=True)
        Image.new("RGB", (640, 480)).save(image_path)

        proc = comfy_client.start(config, ("--cpu",))
        try:
            for values, image in (
                ({"prompt": "a cat"}, None),
                ({"prompt": "a cat", "image": "x"}, image_name),
            ):
                job = parse_job(values, (640, 480) if image else None)
                prompt_id = comfy_client.queue(build_workflow(job, image, "selftest"))
                print(f"{job.mode} workflow accepted by ComfyUI ({prompt_id})")
            requests.post(f"{comfy_client.BASE_URL}/queue", json={"clear": True}, timeout=10)
            requests.post(f"{comfy_client.BASE_URL}/interrupt", timeout=10)
        finally:
            proc.terminate()
            proc.wait(timeout=60)
            image_path.unlink(missing_ok=True)
    print("selftest passed")


if __name__ == "__main__":
    main()
