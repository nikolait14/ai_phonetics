from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.data import QuadDataset, generate_synthetic_image, load_quad_from_png


class DataLoadingTest(unittest.TestCase):
    def test_synthetic_generation_is_deterministic_and_normalized(self) -> None:
        first = generate_synthetic_image(123)
        second = generate_synthetic_image(123)
        self.assertEqual(tuple(first.shape), (128, 128))
        self.assertTrue((first == second).all())
        self.assertEqual(float(first.max()), 1.0)
        self.assertGreater(float(first.sum()), 0.0)

    def test_quad_manifest_and_png_loader(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "quad.jsonl"
            manifest.write_text(
                json.dumps(
                    {
                        "items": [
                            {"source": "synthetic", "seed": 1},
                            {"source": "synthetic", "seed": 2},
                            {"source": "synthetic", "seed": 3},
                            {"source": "synthetic", "seed": 4},
                        ]
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            quad = QuadDataset(manifest, root)[0]
            self.assertEqual(tuple(quad.shape), (4, 128, 128))
            self.assertTrue((quad.amax(dim=(-2, -1)) == 1).all())

            paths = []
            for index in range(4):
                path = root / f"input_{index}.png"
                Image.fromarray((quad[index].numpy() * 255).round().astype("uint8")).save(path)
                paths.append(path)
            loaded = load_quad_from_png(paths)
            self.assertEqual(tuple(loaded.shape), (1, 4, 128, 128))
            self.assertTrue((loaded.amax(dim=(-2, -1)) == 1).all())


if __name__ == "__main__":
    unittest.main()
