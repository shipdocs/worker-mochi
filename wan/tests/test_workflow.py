"""Input validation and workflow shape for the Wan 2.2 worker (no ComfyUI needed)."""

import base64
import io

import pytest
from PIL import Image

from wan.src import models
from wan.src.inputs import load_image
from wan.src.workflow import build_workflow, parse_job, size_for_image


def test_defaults_text_to_video() -> None:
    job = parse_job({"prompt": "a cat", "seed": 7})
    assert (job.mode, job.width, job.height, job.length, job.steps, job.cfg) == ("t2v", 832, 480, 81, 4, 1.0)
    assert job.seed == 7


def test_random_seed_when_missing() -> None:
    assert 0 <= parse_job({"prompt": "a cat"}).seed < 2**32


def test_image_sets_mode_and_keeps_aspect_ratio() -> None:
    job = parse_job({"prompt": "a cat", "image": "x"}, image_size=(1080, 1920))
    assert job.mode == "i2v"
    assert (job.width, job.height) == size_for_image((1080, 1920))
    assert job.width < job.height
    assert job.width % 16 == 0 and job.height % 16 == 0


@pytest.mark.parametrize(
    "values",
    [
        {},
        {"prompt": "  "},
        {"prompt": "x", "length": 80},
        {"prompt": "x", "width": 840},
        {"prompt": "x", "steps": 1},
        {"prompt": "x", "cfg": "high"},
        {"prompt": "x", "seed": -1},
    ],
)
def test_invalid_input_is_rejected(values: dict) -> None:
    with pytest.raises(ValueError):
        parse_job(values)


def _class_types(wf: dict) -> set[str]:
    return {node["class_type"] for node in wf.values()}


def test_t2v_workflow_uses_t2v_models_and_splits_steps() -> None:
    wf = build_workflow(parse_job({"prompt": "a cat", "steps": 4, "seed": 1}), None, "out")
    assert "EmptyHunyuanLatentVideo" in _class_types(wf)
    assert "WanImageToVideo" not in _class_types(wf)
    assert wf["1"]["inputs"]["unet_name"].startswith("wan2.2_t2v_high_noise")
    assert wf["2"]["inputs"]["unet_name"].startswith("wan2.2_t2v_low_noise")
    assert (wf["13"]["inputs"]["end_at_step"], wf["14"]["inputs"]["start_at_step"]) == (2, 2)
    assert wf["13"]["inputs"]["noise_seed"] == wf["14"]["inputs"]["noise_seed"] == 1


def test_i2v_workflow_uses_input_image() -> None:
    job = parse_job({"prompt": "a cat", "image": "x"}, image_size=(640, 480))
    wf = build_workflow(job, "in.png", "out")
    assert wf["12"]["inputs"]["image"] == "in.png"
    assert wf["11"]["class_type"] == "WanImageToVideo"
    assert wf["13"]["inputs"]["positive"] == ["11", 0]
    assert wf["1"]["inputs"]["unet_name"].startswith("wan2.2_i2v_high_noise")
    with pytest.raises(ValueError):
        build_workflow(job, None, "out")


def test_every_referenced_model_file_is_downloaded() -> None:
    names = {f.rsplit("/", 1)[-1] for f in models.files_for(tuple(models.MODE_FILES))}
    for job in (parse_job({"prompt": "x"}), parse_job({"prompt": "x", "image": "y"}, (64, 64))):
        wf = build_workflow(job, "in.png" if job.image else None, "out")
        for node in wf.values():
            for key in ("unet_name", "lora_name", "clip_name", "vae_name"):
                if key in node["inputs"]:
                    assert node["inputs"][key] in names


def test_modes_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAN_MODES", "i2v")
    assert models.enabled_modes() == ("i2v",)
    monkeypatch.setenv("WAN_MODES", "t2v,video")
    with pytest.raises(ValueError):
        models.enabled_modes()


def test_extra_model_paths(tmp_path) -> None:
    text = models.write_extra_model_paths(tmp_path, tmp_path / "x.yaml").read_text()
    assert f"base_path: {tmp_path / 'split_files'}" in text
    assert "diffusion_models: diffusion_models" in text


def test_load_image_accepts_data_url() -> None:
    buf = io.BytesIO()
    Image.new("RGB", (32, 16)).save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode()
    assert load_image(f"data:image/png;base64,{encoded}").size == (32, 16)
    assert load_image(encoded).size == (32, 16)
    with pytest.raises(ValueError):
        load_image("not base64!!")
