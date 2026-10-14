# Wan 2.2 worker (text-to-video and image-to-video)

A Runpod serverless worker for **Wan 2.2 A14B** with the **LightX2V 4-step LoRAs**:
4 sampling steps instead of ~40, at 480p–720p and 16 fps. It uses ComfyUI v0.3.68,
running as a child process, with ComfyUI's official Wan 2.2 workflows
(`wan/src/workflow.py`).

- Text-to-video: send a `prompt`.
- Image-to-video: send a `prompt` and an `image`. The video starts from that image.

## Model weights

The weights for both modes total about 69 GB, which is too large to bake into a
Runpod image (80 GB limit, 30-minute build step). The image contains no weights.
The worker downloads them from Hugging Face (`Comfy-Org/Wan_2.2_ComfyUI_Repackaged`,
pinned revision) when it starts:

| Where | When |
| --- | --- |
| `/runpod-volume/wan-models` | A network volume is attached (recommended): downloaded once, shared by all workers |
| `/ComfyUI/models/wan` | No network volume: every new worker downloads again (a few minutes) |

## Runpod setup

1. **Network volume** (recommended): Storage → New Network Volume, at least **100 GB**.
   Pick a datacenter that has the GPUs you want.
2. **Endpoint**: Serverless → New Endpoint → GitHub repo `shipdocs/worker-mochi`, branch `main`:
   - Dockerfile path: `wan/Dockerfile` (build context: repository root)
   - Network volume: the one from step 1 (this also pins the endpoint to that datacenter)
   - GPU: 48 GB or more (L40S, RTX 6000 Ada, RTX PRO 6000, A100, H100, B200)
   - Container disk: 20 GB with a network volume, **100 GB** without
   - Allowed CUDA versions: 12.8, 12.9, 13.0
   - Optional env: `WAN_MODES=t2v` or `i2v` to download and serve only one mode (half the weights)
3. Publish a GitHub release to trigger a build.

The first worker start downloads the weights. Watch the worker logs for `Models ready`.

## Input

```json
{
  "input": {
    "prompt": "a fluffy orange cat chasing a red ball of yarn across a wooden floor, soft morning light",
    "image": "<optional: https URL, data URL or base64 of a PNG/JPG>",
    "negative_prompt": "blurry, low quality",
    "width": 832,
    "height": 480,
    "length": 81,
    "steps": 4,
    "seed": 42
  }
}
```

| Field | Default | Notes |
| --- | --- | --- |
| `prompt` | required | Longer, descriptive English prompts work best |
| `image` | none | Turns on image-to-video |
| `width`, `height` | 832×480, or the image's aspect ratio at ~480p | Multiples of 16. 1280×720 for 720p (slower) |
| `length` | 81 | Frames, must be 4n+1. 81 frames = 5 s at 16 fps |
| `steps` | 4 | The LoRAs are tuned for 4 |
| `cfg` | 1.0 | Keep at 1 with the 4-step LoRAs (the negative prompt has no effect at cfg 1) |
| `shift` | 5.0 | Sampling shift |
| `seed` | random | Returned in the output, so a result can be reproduced |

## Output

```json
{
  "status": "SUCCESS",
  "video": "<base64 MP4>",
  "mime_type": "video/mp4",
  "mode": "t2v",
  "seed": 42,
  "width": 832, "height": 480, "frames": 81, "fps": 16
}
```

Errors come back as `{"status": "ERROR", "error": "..."}` while the Runpod job shows
`COMPLETED`, so always check `output.status`.

## Testing

- `pytest wan/tests`: input validation and workflow shape.
- `python -m wan.src.selftest` (runs as the last image build step and in CI): starts
  ComfyUI in CPU mode with empty placeholder model files and submits both workflows.
  ComfyUI validates node names, inputs and model file names before queueing, so a
  mismatch with the pinned ComfyUI version fails the build.
