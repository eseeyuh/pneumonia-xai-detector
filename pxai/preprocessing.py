"""Image loading and preprocessing shared by training, evaluation and the apps.

Keeping this in one place guarantees that the demo sees exactly the same
pixels the model was evaluated on.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
DICOM_SUFFIXES = {".dcm", ".dicom"}


def _to_uint8(arr: np.ndarray) -> np.ndarray:
    """Min-max scale an arbitrary-depth array to uint8 (handles 12/16-bit X-rays)."""
    arr = arr.astype(np.float32)
    lo, hi = np.percentile(arr, [0.5, 99.5])
    if hi <= lo:
        hi = lo + 1.0
    arr = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (arr * 255.0).round().astype(np.uint8)


def _load_dicom(source: str | Path | bytes) -> Image.Image:
    try:
        import pydicom
    except ImportError as e:  # pragma: no cover - optional dependency
        raise ImportError("Reading DICOM needs `pip install pydicom`.") from e
    ds = pydicom.dcmread(io.BytesIO(source) if isinstance(source, bytes) else source)
    arr = ds.pixel_array
    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr  # inverted grayscale
    return Image.fromarray(_to_uint8(arr), mode="L")


def load_image(source: str | Path | bytes, filename: str | None = None) -> Image.Image:
    """Load an X-ray from a path or raw bytes and return an 8-bit grayscale PIL image.

    Supports JPEG/PNG (8- and 16-bit) and DICOM.
    """
    name = filename or (str(source) if not isinstance(source, bytes) else "")
    if Path(name).suffix.lower() in DICOM_SUFFIXES:
        return _load_dicom(source)

    img = Image.open(io.BytesIO(source) if isinstance(source, bytes) else source)
    img = ImageOps.exif_transpose(img)
    if img.mode in ("I", "I;16", "I;16B", "I;16L", "F"):
        return Image.fromarray(_to_uint8(np.array(img)), mode="L")
    return img.convert("L")


def _resize(img: Image.Image, size: int, mode: str) -> Image.Image:
    if mode == "squash":
        return img.resize((size, size), Image.BILINEAR)
    if mode == "letterbox":
        return ImageOps.pad(img, (size, size), method=Image.BILINEAR, color=0)
    raise ValueError(f"Unknown resize mode {mode!r}; use 'squash' or 'letterbox'.")


class Resize:
    """Deterministic resize used everywhere (train + eval + app)."""

    def __init__(self, size: int, mode: str = "squash"):
        self.size, self.mode = size, mode

    def __call__(self, img: Image.Image) -> Image.Image:
        return _resize(img, self.size, self.mode)


def eval_transform(size: int = 224, resize_mode: str = "squash") -> transforms.Compose:
    return transforms.Compose(
        [
            Resize(size, resize_mode),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def train_transform(
    size: int = 224, resize_mode: str = "squash", hflip: bool = False
) -> transforms.Compose:
    """Mild, anatomy-preserving augmentation.

    Horizontal flips are off by default: they move the heart to the right side,
    which is not a plausible chest X-ray.
    """
    aug: list = [
        Resize(size, resize_mode),
        transforms.RandomAffine(degrees=7, translate=(0.04, 0.04), scale=(0.93, 1.07)),
        transforms.ColorJitter(brightness=0.15, contrast=0.15),
    ]
    if hflip:
        aug.append(transforms.RandomHorizontalFlip())
    aug += [
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    return transforms.Compose(aug)


def tta_transform(size: int = 224, resize_mode: str = "squash") -> transforms.Compose:
    """Random but label-preserving transform for test-time augmentation."""
    return train_transform(size, resize_mode, hflip=False)


def to_display_rgb(img: Image.Image, size: int = 224, resize_mode: str = "squash") -> np.ndarray:
    """Return the resized image as float RGB in [0, 1] for overlays."""
    small = _resize(img.convert("L"), size, resize_mode).convert("RGB")
    return np.asarray(small, dtype=np.float32) / 255.0


def looks_like_chest_xray(img: Image.Image) -> tuple[bool, list[str]]:
    """Cheap sanity checks that catch obviously wrong uploads.

    This is *not* an out-of-distribution detector; it only flags colour photos,
    extreme aspect ratios and near-blank images.
    """
    warnings: list[str] = []
    rgb = np.asarray(img.convert("RGB").resize((128, 128)), dtype=np.float32)
    chroma = np.abs(rgb - rgb.mean(axis=2, keepdims=True)).mean()
    if chroma > 8.0:
        warnings.append("The image is in colour; chest X-rays are grayscale.")
    w, h = img.size
    if max(w, h) / max(1, min(w, h)) > 1.8:
        warnings.append("Unusual aspect ratio for a chest X-ray.")
    if rgb.std() < 10.0:
        warnings.append("The image has almost no contrast.")
    return (not warnings), warnings


def batch_from_pil(img: Image.Image, size: int = 224, resize_mode: str = "squash") -> torch.Tensor:
    return eval_transform(size, resize_mode)(img).unsqueeze(0)
