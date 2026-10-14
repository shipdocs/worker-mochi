"""Return the generated video inline in the job output, instead of uploading it."""

import base64
import mimetypes
from pathlib import Path


def video_result(video_path: str | Path) -> dict:
    """Read the video, delete it from the worker and return it base64-encoded."""
    video_path = Path(video_path)
    try:
        data = video_path.read_bytes()
    finally:
        video_path.unlink(missing_ok=True)

    return {
        "status": "SUCCESS",
        "video": base64.b64encode(data).decode("ascii"),
        "filename": video_path.name,
        "mime_type": mimetypes.guess_type(video_path.name)[0] or "video/mp4",
        "size_bytes": len(data),
    }
