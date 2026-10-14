#!/usr/bin/env bash
set -e

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

# Run from the ComfyUI root, like ComfyUI's own main.py, so that `python -m`
# puts /ComfyUI first on sys.path and `src` is importable as a package.
cd "${COMFYUI_DIR:-/ComfyUI}"

# Serve the API and don't shutdown the container
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    exec python3 -u -m src.handler --rp_serve_api --rp_api_host=0.0.0.0
else
    exec python3 -u -m src.handler
fi
