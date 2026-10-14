"""Decoding the job's input image."""

import base64
import binascii
import io

import requests
from PIL import Image

MAX_IMAGE_BYTES = 25 * 1024 * 1024


def load_image(value: str) -> Image.Image:
    """Decode an input image given as a http(s) URL, data URL or plain base64."""
    if value.startswith(("http://", "https://")):
        resp = requests.get(value, timeout=60)
        resp.raise_for_status()
        data = resp.content
    else:
        try:
            data = base64.b64decode(value.split(",", 1)[-1], validate=True)
        except (binascii.Error, ValueError) as e:
            msg = "'image' must be a http(s) URL or base64-encoded image"
            raise ValueError(msg) from e
    if len(data) > MAX_IMAGE_BYTES:
        msg = "'image' is larger than 25 MB"
        raise ValueError(msg)
    try:
        return Image.open(io.BytesIO(data)).convert("RGB")
    except Exception as e:
        msg = "'image' could not be read as an image"
        raise ValueError(msg) from e
