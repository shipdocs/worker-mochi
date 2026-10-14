"""Run ComfyUI as a child process and talk to it over its local HTTP API.

Running ComfyUI through its own ``main.py`` (instead of importing it into the
worker) keeps its import order, node loading and memory management exactly as
upstream intends; this is also how Runpod's official ComfyUI worker does it.
"""

import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import requests

COMFYUI_DIR = Path(os.environ.get("COMFYUI_DIR", "/ComfyUI"))
HOST = "127.0.0.1"
PORT = int(os.environ.get("COMFYUI_PORT", "8188"))
BASE_URL = f"http://{HOST}:{PORT}"


class ComfyError(RuntimeError):
    pass


def start(extra_model_paths: Path, extra_args: tuple[str, ...] = ()) -> subprocess.Popen:
    """Start ComfyUI and wait until its API answers. Its logs go to our stdout."""
    cmd = [
        sys.executable, "-u", "main.py",
        "--listen", HOST, "--port", str(PORT),
        "--disable-auto-launch", "--preview-method", "none",
        "--extra-model-paths-config", str(extra_model_paths),
        *extra_args,
    ]
    proc = subprocess.Popen(cmd, cwd=COMFYUI_DIR)
    deadline = time.monotonic() + float(os.environ.get("COMFYUI_START_TIMEOUT", "180"))
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            msg = f"ComfyUI exited during startup (code {proc.returncode})"
            raise ComfyError(msg)
        try:
            if requests.get(f"{BASE_URL}/system_stats", timeout=2).ok:
                return proc
        except requests.RequestException:
            pass
        time.sleep(1)
    proc.kill()
    msg = "ComfyUI did not start in time"
    raise ComfyError(msg)


def queue(workflow: dict) -> str:
    """Submit a workflow; ComfyUI validates it first and reports node errors."""
    resp = requests.post(
        f"{BASE_URL}/prompt",
        json={"prompt": workflow, "client_id": str(uuid.uuid4())},
        timeout=30,
    )
    body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}
    if not resp.ok or "prompt_id" not in body:
        msg = f"ComfyUI rejected the workflow: {body.get('error')} {body.get('node_errors')}"
        raise ComfyError(msg)
    return body["prompt_id"]


def wait(prompt_id: str, proc: subprocess.Popen, timeout: float) -> dict:
    """Wait for a queued prompt to finish and return its history entry."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            msg = f"ComfyUI stopped while running the job (code {proc.returncode})"
            raise ComfyError(msg)
        entry = requests.get(f"{BASE_URL}/history/{prompt_id}", timeout=10).json().get(prompt_id)
        if entry and entry.get("status", {}).get("completed") is not None:
            status = entry["status"]
            if status.get("status_str") == "error":
                errors = [m[1] for m in status.get("messages", []) if m[0] == "execution_error"]
                detail = errors[0].get("exception_message", "").strip() if errors else "unknown error"
                msg = f"ComfyUI execution failed: {detail}"
                raise ComfyError(msg)
            if status.get("completed"):
                return entry
        time.sleep(1)
    requests.post(f"{BASE_URL}/interrupt", timeout=10)
    msg = f"Job did not finish within {timeout:.0f}s"
    raise ComfyError(msg)


def output_files(entry: dict, suffix: str = ".mp4") -> list[Path]:
    """Files with ``suffix`` that the finished prompt saved to ComfyUI's output dir."""
    found = []
    for node_output in entry.get("outputs", {}).values():
        for items in node_output.values():
            if not isinstance(items, list):
                continue
            for item in items:
                if isinstance(item, dict) and str(item.get("filename", "")).endswith(suffix):
                    found.append(COMFYUI_DIR / "output" / item.get("subfolder", "") / item["filename"])
    return found
