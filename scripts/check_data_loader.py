"""Download only two BSDS500 JPEGs and smoke-test the manifest data loader.

Usage:
    python -m scripts.check_data_loader

The two images are fetched from a public BSDS500 mirror because the official
distribution is one 67 MB archive.  This command never requests that archive.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path
from urllib.request import Request, urlopen

import torch

from src.data import QuadDataset, load_bsds_item


ROOT = Path(__file__).resolve().parents[1]
SMOKE_ROOT = ROOT / "data" / "smoke_bsds" / "BSDS500" / "data"
MANIFEST = ROOT / "data" / "manifests" / "smoke.jsonl"
MIRROR = "https://raw.githubusercontent.com/BIDS/BSDS500/master/BSDS500/data"
SAMPLES = ("images/test/100007.jpg", "images/test/100039.jpg")


def download_one(relative_path: str) -> Path:
    destination = SMOKE_ROOT / relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination
    request = Request(f"{MIRROR}/{relative_path}", headers={"User-Agent": "ai-phonetics-smoke-test"})
    with urlopen(request, timeout=30) as response, destination.open("wb") as stream:
        shutil.copyfileobj(response, stream)
    return destination


def write_manifest() -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "id": "smoke-0000",
        "items": [
            {"source": "bsds", "path": SAMPLES[0], "crop": {"x": 24, "y": 32}},
            {"source": "bsds", "path": SAMPLES[1], "crop": {"x": 180, "y": 80}},
            {"source": "synthetic", "seed": 42},
            {"source": "synthetic", "seed": 31415},
        ],
    }
    MANIFEST.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    for relative_path in SAMPLES:
        path = download_one(relative_path)
        image = load_bsds_item({"path": relative_path, "crop": {"x": 0, "y": 0}}, SMOKE_ROOT)
        print(f"{path.relative_to(ROOT)}: shape={tuple(image.shape)}, min={image.min():.3f}, max={image.max():.3f}")

    write_manifest()
    quad = QuadDataset(MANIFEST, SMOKE_ROOT)[0]
    assert quad.shape == (4, 128, 128)
    assert quad.dtype == torch.float32
    assert torch.allclose(quad.amax(dim=(-2, -1)), torch.ones(4))
    assert torch.all(quad >= 0) and torch.all(quad <= 1)
    print(f"quad: shape={tuple(quad.shape)}, dtype={quad.dtype}, channel_max={quad.amax(dim=(-2, -1)).tolist()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
