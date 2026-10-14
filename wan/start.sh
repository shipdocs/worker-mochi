#!/usr/bin/env bash
set -e

# Use libtcmalloc for better memory management
TCMALLOC="$(ldconfig -p | grep -Po "libtcmalloc.so.\d" | head -n 1)"
export LD_PRELOAD="${TCMALLOC}"

cd /worker
if [ "$SERVE_API_LOCALLY" == "true" ]; then
    exec python3 -u -m wan.src.handler --rp_serve_api --rp_api_host=0.0.0.0
else
    exec python3 -u -m wan.src.handler
fi
