import csv
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.build_readability_ladder import finalize


class ReadabilityLadderTests(unittest.TestCase):
    def test_gallery_requires_reconstruction_and_preserves_annotations(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            item = output / "items" / "000_training_sample"
            item.mkdir(parents=True)
            row = dict(ordinal=0, kind="training", id="sample", stratum="source",
                       subgroup="square", source_size=[16, 16], processed_size=[16, 16],
                       relative_dir="items/000_training_sample")
            (output / "manifest.jsonl").write_text(json.dumps(row) + "\n")
            pixels = Image.new("RGB", (16, 16), (12, 34, 56))
            for name in ("source", "processed"):
                pixels.save(item / f"{name}.png")
            args = SimpleNamespace(output_dir=directory)
            with self.assertRaisesRegex(RuntimeError, "Missing 1 reconstructions"):
                finalize(args)
            self.assertFalse((output / "review.csv").exists())

            pixels.save(item / "reconstructed.png")
            finalize(args)
            self.assertTrue((output / "index.html").is_file())
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["total"], 1)
            review = output / "review.csv"
            with review.open(newline="") as handle:
                reader = csv.DictReader(handle)
                fields = reader.fieldnames
                rows = list(reader)
            rows[0]["source_readability"] = "clean"
            rows[0]["notes"] = "Manual review must survive a repeated finalize."
            with review.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            annotated = review.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "manual annotations"):
                finalize(args)
            self.assertEqual(review.read_bytes(), annotated)


if __name__ == "__main__":
    unittest.main()
