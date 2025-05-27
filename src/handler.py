import os
import runpod
import torch
import time
import asyncio
import server
from nodes import NODE_CLASS_MAPPINGS, load_custom_node
import random
import string
import hashlib
import mimetypes
import requests
from pathlib import Path

# Setup ComfyUI server
loop = asyncio.new_event_loop()
asyncio.set_event_loop(loop)
server_instance = server.PromptServer(loop)
execution.PromptQueue(server)

# Load ComfyUI custom nodes
load_custom_node("/ComfyUI/custom_nodes/ComfyUI-MochiWrapper")
load_custom_node("/ComfyUI/custom_nodes/ComfyUI-VideoHelperSuite")

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
        "flash_attn",
    )


def upload_file_to_uploadthing(
    file_path: str | Path,
    max_retries: int = 2,
    initial_delay: float = 5.0,
) -> tuple[requests.Response, requests.Response, str]:
    """
    Upload file to UploadThing with retry mechanism.

    Args:
        file_path: Path to the file to upload
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay between retries in seconds

    Returns:
        Tuple containing (presigned_response, upload_response, file_name)
    
    Raises:
        Exception: If upload fails after all retries
    """
    attempt = 0
    last_error = None
    file_path = Path(file_path)

    while attempt <= max_retries:
        try:
            if attempt > 0:
                delay = initial_delay * (2 ** (attempt - 1))
                print(f"Retry attempt {attempt}/{max_retries} after {delay:.1f}s delay...")
                time.sleep(delay)

            # Generate file info
            file_name = file_path.name
            file_extension = file_path.suffix
            random_string = ''.join(random.choice(string.ascii_letters + string.digits) for _ in range(8))
            md5_hash = hashlib.md5(random_string.encode()).hexdigest()
            new_file_name = f"{md5_hash}{file_extension}"
            file_size = file_path.stat().st_size
            file_type, _ = mimetypes.guess_type(str(file_path))

            with open(file_path, "rb") as file:
                file_content = file.read()

            file_info = {"name": new_file_name, "size": file_size, "type": file_type}
            uploadthing_api_key = os.getenv('UPLOADTHING_API_KEY')
            
            if not uploadthing_api_key:
                raise ValueError("UPLOADTHING_API_KEY environment variable not set")

            headers = {"x-uploadthing-api-key": uploadthing_api_key}
            data = {
                "contentDisposition": "inline",
                "acl": "public-read",
                "files": [file_info],
            }

            # Get presigned URL
            presigned_response = requests.post(
                "https://api.uploadthing.com/v6/uploadFiles",
                headers=headers,
                json=data,
            )
            presigned_response.raise_for_status()
            
            # Add response content logging
            print(f"Presigned response status: {presigned_response.status_code}")
            print(f"Presigned response content: {presigned_response.text}")
            
            presigned = presigned_response.json()["data"][0]
            upload_url = presigned["url"]
            fields = presigned["fields"]

            # Perform actual upload
            files = {"file": file_content}
            upload_response = requests.post(upload_url, data=fields, files=files)
            upload_response.raise_for_status()
            
            # Add upload response logging
            print(f"Upload response status: {upload_response.status_code}")
            print(f"Upload response content: {upload_response.text}")

            print(f"File uploaded successfully: {presigned['fileUrl']}")
            return presigned_response, upload_response, new_file_name

        except Exception as e:
            last_error = e
            print(f"Upload attempt {attempt + 1} failed: {str(e)}")
            # Add more detailed error information
            if isinstance(e, requests.exceptions.RequestException):
                print(f"Request error details: {e.response.text if e.response else 'No response'}")
            attempt += 1

    raise last_error


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

        # Step 7: Upload video with retry mechanism
        print(f"uploading video into storage.")
        try:
            # Get the path to the generated video file
            _, output_files = out_video["result"][0]
            video_path = output_files[-1]

            # Upload the file
            presigned_response, upload_response, file_name = upload_file_to_uploadthing(video_path)
            video_url = presigned_response.json()['data'][0]['fileUrl']

            # Clean up the temporary file
            if os.path.exists(video_path):
                os.remove(video_path)

            return {
                "result": video_url,
                "status": "SUCCESS"
            }
        except Exception as e:
            print(f"Upload failed: {str(e)}")
            # Clean up in case of error
            if "video_path" in locals() and os.path.exists(video_path):
                os.remove(video_path)
            raise  # Re-raise the exception to be caught by the outer try-except

    except Exception as e:
        print(f"Generation failed: {str(e)}")
        return {
            "status": "ERROR",
            "error": str(e),
            "result": None
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": generate})
