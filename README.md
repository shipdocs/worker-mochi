# worker-mochi

[![RunPod](https://api.runpod.io/badge/rachfop/worker-mochi)](https://www.runpod.io/console/hub/rachfop/worker-mochi)

> Generate videos with Mochi as an endpoint on RunPod

## Features

- Video generation using [Mochi 1](https://github.com/genmoai/mochi) by [Genmo](https://genmo.ai)
- Automatic model loading and initialization
- [UploadThing](https://uploadthing.com/) integration for video upload

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
| `UPLOADTHING_API_KEY`  | UploadThing API key used to upload the generated video (required)           |         |
| `MOCHI_ATTENTION_MODE` | Mochi attention backend: `sdpa`, `flash_attn`, `sage_attn` or `comfy`. Only `sdpa` and `comfy` work without installing extra packages | `sdpa`  |

## Deployment

Deploy this worker on RunPod using the [GitHub Integration](https://docs.runpod.io/serverless/github-integration), or build the image yourself and push it to GHCR as described below.

### GPU compatibility

The image uses PyTorch `2.8.0+cu128` on a CUDA 12.8 base image, so it runs on:

| Architecture | Examples                         | Compute capability |
| ------------ | -------------------------------- | ------------------ |
| Ampere       | A100, A40, RTX A6000, RTX 3090   | `sm_80`, `sm_86`   |
| Ada          | L4, L40S, RTX 6000 Ada, RTX 4090 | `sm_89`            |
| Hopper       | H100, H200                       | `sm_90`            |
| Blackwell    | B200, RTX 5090, RTX PRO 6000     | `sm_100`, `sm_120` |

CUDA 12.8 needs NVIDIA driver R570 or newer on the host. On Runpod, set **Allowed CUDA Versions** on the endpoint to 12.8 and newer (`.runpod/hub.json` does this for Hub deployments).

At startup the worker checks that PyTorch has kernels for the GPU it got and exits with a clear error if not, instead of failing every job with `no kernel image is available`. Earlier images (PyTorch `2.5.0+cu124`) support up to `sm_90` only; if you stay on one of them, limit the endpoint to Ampere, Ada and Hopper GPUs.

### Build and push to GHCR

The image contains about 30 GB of model weights, so build it on a machine with at least 100 GB of free disk.

```bash
docker login ghcr.io -u <github-user>   # password: a PAT with the write:packages scope
docker build --platform linux/amd64 -t ghcr.io/shipdocs/worker-mochi:<tag> .
docker push ghcr.io/shipdocs/worker-mochi:<tag>
```

The `GHCR` workflow (`.github/workflows/ghcr.yml`) does the same on the self-hosted runner for every push to `main` and every `v*` tag, and tags the image `sha-<short-sha>`, `main`/`latest` and `<version>`. Pull requests only run the fast startup regression tests.

The last build step runs `python -m src.selftest`, which imports ComfyUI the way the handler does, loads the custom nodes and checks the PyTorch CUDA architectures. A broken import fails the build instead of the Runpod worker. To run the full test, including ComfyUI-MochiWrapper (which needs a GPU), on a GPU machine:

```bash
docker run --rm --gpus all --entrypoint python3 -w /ComfyUI ghcr.io/shipdocs/worker-mochi:<tag> -m src.selftest
```

### Use the image on Runpod Serverless

1. If the GHCR package is private, add a registry credential under **Settings → Container Registry Auth**: registry `ghcr.io`, your GitHub username, and a PAT with the `read:packages` scope. Alternatively, make the package public in its GitHub package settings.
2. Create or edit a **Serverless template**:
   - Container image: `ghcr.io/shipdocs/worker-mochi:sha-<short-sha>`. Use an immutable tag rather than `latest` so workers don't run mixed versions.
   - Container registry credentials: the credential from step 1 (private packages only).
   - Container disk: at least 50 GB.
   - Environment variables: `UPLOADTHING_API_KEY` (use a Runpod secret, e.g. `{{ RUNPOD_SECRET_uploadthing_api_key }}`).
3. On the endpoint, select the template, choose GPU types from the table above (the Hub config defaults to 80 GB H100-class GPUs), and set **Allowed CUDA Versions** to 12.8 and newer.
4. To roll out a new build later, edit the template's image tag. Runpod replaces the workers with the new image.

### Updating ComfyUI or the custom nodes

ComfyUI and the custom nodes are pinned in the `Dockerfile` (`COMFYUI_REF`, `MOCHI_WRAPPER_REF`, `VHS_REF`) and the Python packages are locked in `requirements-worker.txt`. To update, change the refs, copy the new ComfyUI `requirements.txt` entries into `requirements-worker.in`, regenerate the lock file with the command at the top of that file, and rebuild. The self-test fails the build if something no longer imports.

### Troubleshooting

`ModuleNotFoundError: No module named 'utils.json_util'; 'utils' is not a package`: ComfyUI's `nodes.py` puts `ComfyUI/comfy/` first on `sys.path` when it's imported, so `comfy/utils.py` hides ComfyUI's `utils/` package. ComfyUI's own `main.py` imports `utils` before `nodes`; the worker does the same in `src/comfy_bootstrap.py`, which must run before any ComfyUI import. `tests/test_comfy_bootstrap.py` covers this.

## Development

For development and contribution guidelines, please see our [Contributing Guide](.github/CONTRIBUTING.md).

## License

[MIT License](LICENSE)
