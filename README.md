# worker-mochi

[![RunPod](https://api.runpod.io/badge/rachfop/worker-mochi)](https://www.runpod.io/console/hub/rachfop/worker-mochi)

> Generate videos with Mochi as an endpoint on RunPod

**New:** [`wan/`](wan/README.md) contains a faster, higher-quality worker for Wan 2.2 (text-to-video and image-to-video, 4-step LoRAs).

## Features

- Video generation using [Mochi 1](https://github.com/genmoai/mochi) by [Genmo](https://genmo.ai)
- Automatic model loading and initialization
- The generated video is returned directly in the job output (base64), so it is never uploaded to a third party

## API Reference

### Input Parameters

```json
{
  "input": {
    "positive_prompt": "a cat playing with yarn, cute, fluffy, detailed fur",
    "negative_prompt": "",
    "width": 848,
    "height": 480,
    "seed": 42,
    "steps": 40,
    "cfg": 6,
    "num_frames": 31,
    "vae": {
      "enable_vae_tiling": false,
      "tile_sample_min_width": 312,
      "tile_sample_min_height": 160,
      "tile_overlap_factor_width": 0.25,
      "tile_overlap_factor_height": 0.25,
      "auto_tile_size": false,
      "frame_batch_size": 8
    }
  }
}
```

### Output

```json
{
  "status": "SUCCESS",
  "video": "<base64-encoded MP4>",
  "filename": "Mochi_00001.mp4",
  "mime_type": "video/mp4",
  "size_bytes": 1234567
}
```

The video is deleted from the worker after it is encoded. Save it with, for example:

```bash
curl -s -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
  https://api.runpod.ai/v2/<endpoint-id>/runsync -d @test_input.json \
  | jq -r '.output.video' | base64 -d > video.mp4
```

Runpod limits the size of a job's output, so keep videos short (the defaults produce a few MB). On failure the output is `{"status": "ERROR", "error": "..."}` while the Runpod job itself still shows `COMPLETED`, so always check `output.status`.

#### Core Parameters

| Parameter         | Description                                                            | Default |
| ----------------- | ---------------------------------------------------------------------- | ------- |
| `positive_prompt` | Text description of what you want to generate                          | `""`    |
| `negative_prompt` | Text description of what you want to avoid in the generation           | `""`    |
| `width`           | Output video width in pixels                                           | `848`   |
| `height`          | Output video height in pixels                                          | `480`   |
| `seed`            | Random seed for reproducible results                                   | `1337`  |
| `steps`           | Number of denoising steps (higher = better quality, slower generation) | `40`    |
| `cfg`             | Classifier-free guidance scale (how closely to follow the prompt)      | `6`     |
| `num_frames`      | Number of frames to generate                                           | `31`    |

#### VAE Parameters

| Parameter                    | Description                                | Default |
| ---------------------------- | ------------------------------------------ | ------- |
| `enable_vae_tiling`          | Enable tiling for VAE decoding             | `false` |
| `tile_sample_min_width`      | Minimum tile width when tiling is enabled  | `312`   |
| `tile_sample_min_height`     | Minimum tile height when tiling is enabled | `160`   |
| `tile_overlap_factor_width`  | Overlap factor between tiles (width)       | `0.25`  |
| `tile_overlap_factor_height` | Overlap factor between tiles (height)      | `0.25`  |
| `auto_tile_size`             | Automatically determine tile size          | `false` |
| `frame_batch_size`           | Number of frames to process in parallel    | `8`     |

#### Environment variables

| Variable               | Description                                                                 | Default |
| ---------------------- | --------------------------------------------------------------------------- | ------- |
| `MOCHI_ATTENTION_MODE` | Mochi attention backend: `sdpa`, `flash_attn`, `sage_attn` or `comfy`. Only `sdpa` and `comfy` work without installing extra packages | `sdpa`  |

## Deployment

Deploy this worker on RunPod using the [GitHub Integration](https://docs.runpod.io/serverless/github-integration), or build the image yourself and push it to Docker Hub, as described below.

### GPU compatibility

The image uses PyTorch `2.8.0+cu128` on a CUDA 12.8 base image, so it runs on:

| Architecture | Examples                         | Compute capability |
| ------------ | -------------------------------- | ------------------ |
| Ampere       | A100, A40, RTX A6000, RTX 3090   | `sm_80`, `sm_86`   |
| Ada          | L4, L40S, RTX 6000 Ada, RTX 4090 | `sm_89`            |
| Hopper       | H100, H200                       | `sm_90`            |
| Blackwell    | B200, RTX 5090, RTX PRO 6000     | `sm_100`, `sm_120` |

CUDA 12.8 needs NVIDIA driver R570 or newer on the host. On Runpod, allow only CUDA 12.8 and newer on the endpoint (`.runpod/hub.json` does this for Hub deployments).

At startup the worker checks that PyTorch has kernels for the GPU it got and exits with a clear error if not, instead of failing every job with `no kernel image is available`. Earlier images (PyTorch `2.5.0+cu124`) support up to `sm_90` only; if you stay on one of them, limit the endpoint to Ampere, Ada and Hopper GPUs.

### Where to build and host the image

The image bakes in about 30 GB of model weights. The Mochi DiT file alone is a single 20 GB layer, which rules out GHCR: it limits each layer to 10 GB and each upload to 10 minutes. Use one of these instead:

- **Runpod GitHub integration (recommended).** Runpod builds the `Dockerfile` (images up to 80 GB) every time you publish a GitHub release, and you can roll back to any earlier build from the endpoint's **Builds** tab. No registry or credentials to manage. It can't be attached to an existing endpoint, so this gives you a new endpoint ID.
- **Docker Hub, pushed from your own machine** (at least 100 GB of free disk). This keeps an existing endpoint ID, because you point its template at the new tag:

  ```bash
  docker login                                  # Docker Hub access token
  docker build --platform linux/amd64 -t <dockerhub-user>/worker-mochi:v0.1.0 .
  docker push <dockerhub-user>/worker-mochi:v0.1.0
  ```

Either way, deploy an immutable tag (a release version or a commit SHA), never `latest`, so every worker runs the same build and rolling back means picking the previous tag.

The last build step runs `python -m src.selftest`, which imports ComfyUI the way the handler does, loads the custom nodes and checks the PyTorch CUDA architectures. A broken import fails the build instead of the Runpod worker. The `CI` workflow runs the startup regression tests on every pull request.

To run the full self-test, including ComfyUI-MochiWrapper (which needs a GPU), start a Runpod **GPU Pod** from the image with this container start command, then read the Pod logs:

```bash
bash -c "cd /ComfyUI && python -m src.selftest; sleep infinity"
```

On any machine with Docker and an NVIDIA GPU, run this instead:

```bash
docker run --rm --gpus all --entrypoint python3 -w /ComfyUI <image>:<tag> -m src.selftest
```

### Runpod endpoint settings

- Container disk: at least 50 GB.
- GPU types: any from the table above. The Hub config defaults to 80 GB H100-class GPUs.
- CUDA: allow only hosts with CUDA 12.8 or newer, using **one** of the endpoint's two CUDA settings (allowed versions 12.8, 12.9 and 13.0, or minimum version 12.8), not both.

### Updating ComfyUI or the custom nodes

ComfyUI and the custom nodes are pinned in the `Dockerfile` (`COMFYUI_REF`, `MOCHI_WRAPPER_REF`, `VHS_REF`) and the Python packages are locked in `requirements-worker.txt`. To update, change the refs, copy the new ComfyUI `requirements.txt` entries into `requirements-worker.in`, regenerate the lock file with the command at the top of that file, and rebuild. The self-test fails the build if something no longer imports.

### Troubleshooting

`ModuleNotFoundError: No module named 'utils.json_util'; 'utils' is not a package`: ComfyUI's `nodes.py` puts `ComfyUI/comfy/` first on `sys.path` when it's imported, so `comfy/utils.py` hides ComfyUI's `utils/` package. ComfyUI's own `main.py` imports `utils` before `nodes`; the worker does the same in `src/comfy_bootstrap.py`, which must run before any ComfyUI import. `tests/test_comfy_bootstrap.py` covers this.

## Development

For development and contribution guidelines, please see our [Contributing Guide](.github/CONTRIBUTING.md).

## License

[MIT License](LICENSE)
