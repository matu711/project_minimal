"""Legacy RGB/BILINEAR semantics shared by both robot writers."""
from pathlib import Path
import io
import numpy as np
from PIL import Image

def load_rgb(path: Path, resize: tuple[int, int] | None) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"Missing image: {path}")

    with Image.open(path) as img:
        img = img.convert("RGB")
        if resize is not None:
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:  # Pillow < 9.1
                resample = Image.BILINEAR
            img = img.resize(resize, resample=resample)
        return np.asarray(img, dtype=np.uint8).copy()


def read_rgb(value, resize=None):
    if not isinstance(value, bytes):
        return load_rgb(Path(value), resize)
    with Image.open(io.BytesIO(value)) as image:
        image = image.convert('RGB')
        if resize is not None:
            try:
                resample = Image.Resampling.BILINEAR
            except AttributeError:
                resample = Image.BILINEAR
            image = image.resize(resize, resample=resample)
        return np.asarray(image, dtype=np.uint8).copy()


def resize_size(value):
    if value is None:
        return None
    if len(value) != 2 or any(int(v) != v or v <= 0 for v in value):
        raise ValueError('--resize WIDTH HEIGHT must both be positive integers')
    return tuple(map(int, value))
