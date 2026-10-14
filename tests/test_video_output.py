"""The handler returns the video inline (base64) and removes it from the worker."""

import base64
from pathlib import Path

from src.video_output import video_result


def test_video_result_returns_base64_and_deletes_file(tmp_path: Path) -> None:
    video = tmp_path / "Mochi_00001.mp4"
    video.write_bytes(b"\x00\x00\x00\x18ftypmp42 fake video")

    result = video_result(video)

    assert result["status"] == "SUCCESS"
    assert base64.b64decode(result["video"]) == b"\x00\x00\x00\x18ftypmp42 fake video"
    assert result["filename"] == "Mochi_00001.mp4"
    assert result["mime_type"] == "video/mp4"
    assert result["size_bytes"] == len(b"\x00\x00\x00\x18ftypmp42 fake video")
    assert not video.exists()
