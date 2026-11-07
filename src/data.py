"""Deterministic loading of 4-channel target-image quads.

The training manifests are JSON Lines files.  Each line has this shape::

    {
      "id": "train-000001",
      "items": [
        {"source": "synthetic", "seed": 731},
        {"source": "bsds", "path": "images/train/100075.jpg",
         "crop": {"x": 16, "y": 48}},
        ... two more items ...
      ]
    }

``bsds_root`` must point to ``BSDS500/data``: the directory that contains
``images/train``, ``images/val`` and ``images/test``.  No sample is downloaded
by this module; download logic deliberately lives in the smoke-test script.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset


IMAGE_SIZE = 128
_RESAMPLE = Image.Resampling.LANCZOS if hasattr(Image, "Resampling") else Image.LANCZOS


class ManifestError(ValueError):
    """Raised when a dataset manifest cannot describe one valid quad."""


def normalize_per_image(image: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Normalize a 2-D image or each channel of a `(C, H, W)` tensor to [0, 1]."""
    if image.ndim == 2:
        maximum = image.amax()
    elif image.ndim == 3:
        maximum = image.amax(dim=(-2, -1), keepdim=True)
    else:
        raise ValueError(f"Expected 2-D or 3-D tensor, got shape {tuple(image.shape)}")

    if torch.any(maximum <= eps):
        raise ValueError("Cannot normalize an all-black target image")
    return image / maximum


def _to_tensor(image: Image.Image) -> torch.Tensor:
    """Convert an 8-bit grayscale PIL image to a normalized float32 tensor."""
    if image.mode != "L":
        raise ValueError(f"Expected an 8-bit grayscale image (mode 'L'), got {image.mode!r}")
    values = np.asarray(image, dtype=np.float32) / 255.0
    return normalize_per_image(torch.from_numpy(values))


def _resize_short_side(image: Image.Image, size: int) -> Image.Image:
    width, height = image.size
    short_side = min(width, height)
    if short_side >= size:
        return image
    scale = size / short_side
    return image.resize((math.ceil(width * scale), math.ceil(height * scale)), _RESAMPLE)


def _crop_from_spec(image: Image.Image, crop: Mapping[str, Any] | Sequence[int] | None, size: int) -> Image.Image:
    image = _resize_short_side(image, size)
    width, height = image.size
    if crop is None:
        x = (width - size) // 2
        y = (height - size) // 2
    elif isinstance(crop, Mapping):
        x, y = int(crop["x"]), int(crop["y"])
    elif len(crop) == 2:
        x, y = int(crop[0]), int(crop[1])
    else:
        raise ManifestError("crop must be {'x': int, 'y': int} or [x, y]")

    if x < 0 or y < 0 or x + size > width or y + size > height:
        raise ManifestError(
            f"Crop {(x, y, size, size)} lies outside source image of size {(width, height)}"
        )
    return image.crop((x, y, x + size, y + size))


def _safe_image_path(bsds_root: Path, relative_path: str) -> Path:
    root = bsds_root.resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"BSDS path escapes dataset root: {relative_path!r}") from exc
    if not candidate.is_file():
        raise FileNotFoundError(f"BSDS image not found: {candidate}")
    return candidate


def load_bsds_item(spec: Mapping[str, Any], bsds_root: str | Path, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Load one BSDS item specified by a manifest record."""
    if "path" not in spec:
        raise ManifestError("A BSDS item requires a relative 'path'")
    path = _safe_image_path(Path(bsds_root), str(spec["path"]))
    with Image.open(path) as source:
        grayscale = source.convert("L")
        crop = _crop_from_spec(grayscale, spec.get("crop"), image_size)
    return _to_tensor(crop)


def _load_font(size: int) -> ImageFont.ImageFont:
    """Use DejaVu when present; Pillow's built-in font keeps the fallback portable."""
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def _draw_glyph(canvas: Image.Image, rng: np.random.Generator, intensity: int) -> None:
    char = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"[int(rng.integers(0, 34))]
    size = int(rng.integers(22, 66))
    glyph = Image.new("L", (size * 2, size * 2), color=0)
    draw = ImageDraw.Draw(glyph)
    font = _load_font(size)
    bbox = draw.textbbox((0, 0), char, font=font)
    draw.text((-bbox[0], -bbox[1]), char, fill=intensity, font=font)
    bbox = glyph.getbbox()
    if bbox is None:
        return
    glyph = glyph.crop(bbox)
    angle = float(rng.uniform(-25.0, 25.0))
    glyph = glyph.rotate(angle, resample=Image.Resampling.BILINEAR, expand=True)
    max_x = max(0, canvas.width - glyph.width)
    max_y = max(0, canvas.height - glyph.height)
    canvas.paste(glyph, (int(rng.integers(0, max_x + 1)), int(rng.integers(0, max_y + 1))), glyph)


def generate_synthetic_image(seed: int, image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Generate one nonempty deterministic grayscale target image from a seed."""
    rng = np.random.default_rng(int(seed))
    canvas = Image.new("L", (image_size, image_size), color=0)
    draw = ImageDraw.Draw(canvas)

    for _ in range(int(rng.integers(1, 5))):
        primitive = str(rng.choice(["ellipse", "rectangle", "line", "ring", "point", "glyph"]))
        intensity = int(rng.integers(80, 256))
        width = int(rng.integers(1, 7))
        x0, x1 = sorted(rng.integers(4, image_size - 4, size=2).tolist())
        y0, y1 = sorted(rng.integers(4, image_size - 4, size=2).tolist())
        if x1 - x0 < 8:
            x1 = min(image_size - 1, x0 + 8)
        if y1 - y0 < 8:
            y1 = min(image_size - 1, y0 + 8)

        if primitive == "ellipse":
            draw.ellipse((x0, y0, x1, y1), fill=intensity if rng.random() < 0.5 else None, outline=intensity, width=width)
        elif primitive == "rectangle":
            draw.rectangle((x0, y0, x1, y1), fill=intensity if rng.random() < 0.5 else None, outline=intensity, width=width)
        elif primitive == "line":
            draw.line((x0, y0, x1, y1), fill=intensity, width=width)
        elif primitive == "ring":
            draw.ellipse((x0, y0, x1, y1), outline=intensity, width=width)
        elif primitive == "point":
            radius = int(rng.integers(2, 10))
            center_x, center_y = int(rng.integers(radius, image_size - radius, endpoint=False)), int(rng.integers(radius, image_size - radius, endpoint=False))
            draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=intensity)
        else:
            _draw_glyph(canvas, rng, intensity)

    tensor = _to_tensor(canvas)
    if tensor.max() <= 0:  # Defensive guard against a font-rendering edge case.
        raise RuntimeError("Synthetic generator produced an all-black image")
    return tensor


def _read_manifest(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ManifestError(f"Invalid JSON in {path}:{line_number}") from exc
            items = record.get("items")
            if not isinstance(items, list) or len(items) != 4:
                raise ManifestError(f"{path}:{line_number} must contain exactly four 'items'")
            fingerprints = [json.dumps(item, sort_keys=True, ensure_ascii=False) for item in items]
            if len(set(fingerprints)) != 4:
                raise ManifestError(f"{path}:{line_number} contains duplicate target specifications")
            records.append(record)
    if not records:
        raise ManifestError(f"Manifest is empty: {path}")
    return records


class QuadDataset(Dataset[torch.Tensor]):
    """Dataset of deterministic `(4, N, N)` target-image quads."""

    def __init__(self, manifest_path: str | Path, bsds_root: str | Path, image_size: int = IMAGE_SIZE) -> None:
        self.records = _read_manifest(manifest_path)
        self.bsds_root = Path(bsds_root)
        self.image_size = int(image_size)
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")

    def __len__(self) -> int:
        return len(self.records)

    def _load_item(self, spec: Mapping[str, Any]) -> torch.Tensor:
        source = spec.get("source")
        if source == "synthetic":
            if "seed" not in spec:
                raise ManifestError("A synthetic item requires a 'seed'")
            return generate_synthetic_image(int(spec["seed"]), self.image_size)
        if source == "bsds":
            return load_bsds_item(spec, self.bsds_root, self.image_size)
        raise ManifestError(f"Unsupported source {source!r}; expected 'synthetic' or 'bsds'")

    def __getitem__(self, index: int) -> torch.Tensor:
        record = self.records[index]
        images = [self._load_item(spec) for spec in record["items"]]
        quad = torch.stack(images, dim=0).to(torch.float32)
        if quad.shape != (4, self.image_size, self.image_size):
            raise RuntimeError(f"Unexpected quad shape: {tuple(quad.shape)}")
        return quad


def load_quad_from_png(paths: Iterable[str | Path], image_size: int = IMAGE_SIZE) -> torch.Tensor:
    """Load exactly four 8-bit grayscale PNG files as `(1, 4, N, N)` float32."""
    paths = [Path(path) for path in paths]
    if len(paths) != 4:
        raise ValueError(f"Expected exactly 4 input PNG files, got {len(paths)}")
    images: list[torch.Tensor] = []
    for path in paths:
        with Image.open(path) as image:
            if image.format != "PNG":
                raise ValueError(f"Expected PNG input, got {image.format!r}: {path}")
            if image.mode != "L":
                raise ValueError(f"Expected one-channel PNG (mode 'L'), got {image.mode!r}: {path}")
            if image.size != (image_size, image_size):
                raise ValueError(f"Expected {image_size}x{image_size}, got {image.size}: {path}")
            images.append(_to_tensor(image.copy()))
    return torch.stack(images, dim=0).unsqueeze(0).to(torch.float32)
