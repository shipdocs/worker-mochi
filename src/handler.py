import os
import runpod
import torch
import asyncio

from .comfy_bootstrap import (
    check_cuda_compatibility,
    load_worker_nodes,
    prepare_comfyui_imports,
)
from .video_output import video_result

# Must run before importing any ComfyUI module (see comfy_bootstrap docstring).
prepare_comfyui_imports()
check_cuda_compatibility()

import execution  # noqa: E402
import server  # noqa: E402

# Setup ComfyUI server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server_instance)

# Load ComfyUI custom nodes
NODE_CLASS_MAPPINGS = load_worker_nodes(loop)

# Initialize ComfyUI nodes for the video generation pipeline
CLIPLoader = NODE_CLASS_MAPPINGS["CLIPLoader"]()
DownloadAndLoadMochiModel = NODE_CLASS_MAPPINGS["DownloadAndLoadMochiModel"]()
MochiTextEncode = NODE_CLASS_MAPPINGS["MochiTextEncode"]()
MochiSampler = NODE_CLASS_MAPPINGS["MochiSampler"]()
MochiDecode = NODE_CLASS_MAPPINGS["MochiDecode"]()
VHS_VideoCombine = NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()

# Load models at startup (only done once)
with torch.inference_mode():
    # Load T5 encoder for better prompt understanding
    clip = CLIPLoader.load_clip(
        "google_t5-v1_1-xxl_encoderonly-fp16.safetensors", type="sd3"
    )[0]
    # Load Mochi model and VAE decoder
    model, vae = DownloadAndLoadMochiModel.loadmodel(
        "mochi_preview_dit_bf16.safetensors",
        "mochi_preview_vae_decoder_bf16.safetensors",
        "bf16",
        # "sdpa" uses PyTorch's built-in attention and needs no extra wheel that has
        # to match the torch/CUDA version (flash_attn, sage_attn).
        os.environ.get("MOCHI_ATTENTION_MODE", "sdpa"),
    )


@torch.inference_mode()
def generate(input):
    values = input["input"]
    try:
        # Step 1: Extract parameters with defaults from test_input.json
        positive_prompt = values.get("positive_prompt", "")
        negative_prompt = values.get("negative_prompt", "")
        width = values.get("width", 848)
        height = values.get("height", 480)
        seed = values.get("seed", 1337)
        steps = values.get("steps", 40)
        cfg = values.get("cfg", 6)
        num_frames = values.get("num_frames", 31)

        # Step 2: Get VAE parameters from input with defaults
        vae_config = values.get("vae", {})
        enable_vae_tiling = vae_config.get("enable_vae_tiling", False)
        tile_sample_min_width = vae_config.get("tile_sample_min_width", 312)
        tile_sample_min_height = vae_config.get("tile_sample_min_height", 160)
        tile_overlap_factor_width = vae_config.get("tile_overlap_factor_width", 0.25)
        tile_overlap_factor_height = vae_config.get("tile_overlap_factor_height", 0.25)
        auto_tile_size = vae_config.get("auto_tile_size", False)
        frame_batch_size = vae_config.get("frame_batch_size", 8)

        # Step 3: Convert text prompts to embeddings
        positive = MochiTextEncode.process(
            clip, positive_prompt, strength=1.0, force_offload=True
        )[0]
        negative = MochiTextEncode.process(
            clip, negative_prompt, strength=1.0, force_offload=True
        )[0]

        # Step 4: Generate video frames in latent space using diffusion
        print(f"generating video.")
        samples = MochiSampler.process(
            model, positive, negative, steps, cfg, seed, height, width, num_frames
        )[0]

        # Step 5: Convert latent frames to images using VAE decoder
        print(f"decoding video.")
        frames = MochiDecode.decode(
            vae,
            samples,
            enable_vae_tiling,
            tile_sample_min_height,
            tile_sample_min_width,
            tile_overlap_factor_height,
            tile_overlap_factor_width,
            auto_tile_size,
            frame_batch_size,
        )[0]

        # Step 6: Combine frames into video
        print(f"combining frames into video.")
        out_video = VHS_VideoCombine.combine_video(
            images=frames,
            frame_rate=24,
            loop_count=0,
            filename_prefix="Mochi",
            format="video/h264-mp4",
            save_output=True,
            prompt=None,
            unique_id=None,
            pix_fmt="yuv420p",
            crf=17,
            save_metadata=True,
        )

        # Step 7: Return the video inline (base64), so it never leaves Runpod
        _, output_files = out_video["result"][0]
        return video_result(output_files[-1])

    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e),
            "result": None
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": generate})
