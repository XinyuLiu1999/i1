"""Regression checks for corrected Parquet references and pixel-budgeted SFT."""

from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configs.sft_512 import get_config as config_512
from configs.sft_1024 import get_config as config_1024
from datasets.bucketed import BucketedImages
from datasets.build_image_index import build_index
from datasets.build_inspection_gallery import _allocate, main as build_gallery
from datasets.data_sources import iter_image_records, open_record_image
from datasets.image_geometry import generate_buckets


class ImageGeometryTests(unittest.TestCase):
    def test_gallery_marks_ineligible_1024_sources_and_matches_training(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (600, 600), (12, 34, 56)).save(root / 'small.png')
            Image.new('RGB', (1536, 1024), (78, 90, 12)).save(root / 'large.png')
            rows = [dict(id=str(i), image_path='small.png', caption='small', width=600, height=600)
                    for i in range(99)]
            rows.append(dict(id='large', image_path='large.png', caption='large', width=1536, height=1024))
            manifest = root / 'source.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            output = root / 'gallery'
            with patch.object(sys, 'argv', ['gallery', '--manifest', str(manifest),
                                           '--output_dir', str(output), '--count', '100', '--skip_tokenizer']):
                build_gallery()
            entries = [json.loads(line) for line in (output / 'samples.jsonl').read_text().splitlines()]
            self.assertEqual(len(entries), 100)
            for entry in entries:
                if entry['id'] != 'large':
                    self.assertEqual(entry['exclusion_1024'], 'source_too_small')
                    self.assertIsNone(entry['bucket_1024'])
                    self.assertIsNone(entry['processed_1024'])
            config = config_1024().input
            config.manifest = str(manifest)
            training = BucketedImages(config)
            self.assertEqual(len(training), 1)
            large = next(entry for entry in entries if entry['id'] == 'large')
            self.assertIsNone(large['exclusion_1024'])
            with Image.open(output / large['processed_1024']) as preview:
                pixels = np.asarray(preview, dtype=np.float32) / 127.5 - 1
            np.testing.assert_array_equal(pixels, training[0][0].numpy())
            self.assertEqual(len(list((output / 'images').glob('*_1024.png'))), 1)
            self.assertIn('Excluded from 1024 training', (output / 'index.html').read_text())

    def test_shapes_fit_model_and_preserve_dominant_aspects(self):
        small, large = config_512().input.buckets, config_1024().input.buckets
        self.assertEqual(len(small), len(set(small)))
        self.assertEqual(large, [(h * 2, w * 2) for h, w in small])
        self.assertIn((416, 624), small)
        self.assertIn((624, 416), small)
        self.assertIn((512, 512), small)
        for h, w in small:
            self.assertEqual(h % 16, 0)
            self.assertEqual(w % 16, 0)
            self.assertLessEqual(h * w, 512 ** 2)
            self.assertLessEqual(max(h, w) / min(h, w), 3)
            self.assertIn((w, h), small)
        # Cover the entire observed aspect range without losing >8% to padding.
        for ratio in [2/3, .83, .95, 1, 1.2, 1.5, 16/9, 2, 2.4, 3]:
            retained = max(min((w/h)/ratio, ratio/(w/h)) for h, w in small)
            self.assertGreater(retained, .92)

    def test_invalid_budgets_and_sparse_gallery(self):
        for kwargs in [dict(step=15), dict(step=0), dict(max_ratio=.5),
                       dict(extra_shapes=[(1024, 1024)])]:
            with self.assertRaises(ValueError):
                generate_buckets(512, **kwargs)
        self.assertEqual(_allocate(10, [1, 100, 100]), [1, 5, 4])
        with self.assertRaises(ValueError):
            _allocate(10, [1, 2])


class ImageIndexTests(unittest.TestCase):
    def setUp(self):
        try:
            import pyarrow as pa
            import pyarrow.parquet as pq
        except ImportError:
            self.skipTest("pyarrow is not installed")
        self.pa, self.pq = pa, pq

    @staticmethod
    def pixels(size, oriented=False):
        image = Image.new("RGB", size, (200, 20, 10))
        buffer = BytesIO()
        if oriented:
            exif = Image.Exif()
            exif[274] = 6
            image.save(buffer, format="PNG", exif=exif)
        else:
            image.save(buffer, format="PNG")
        return buffer.getvalue()

    def write_shard(self, root, name, rows):
        path = root / name
        self.pq.write_table(self.pa.Table.from_pylist(rows), path, row_group_size=2)
        return path

    def test_corrected_index_keeps_other_aspects_exif_and_excludes_bad_records(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [
                dict(id="reversed", prompt="edge text", size="64x96", image_bytes=self.pixels((96, 64))),
                dict(id="wide", prompt="wide text", size="96x64", image_bytes=self.pixels((128, 64))),
                dict(id="exif", prompt="rotated text", size="96x64", image_bytes=self.pixels((96, 64), True)),
                dict(id="broken", prompt="bad", size="96x64", image_bytes=b"broken PNG"),
                dict(id="empty", prompt=" ", size="96x64", image_bytes=self.pixels((96, 64))),
                dict(id="bad_size", prompt="recoverable", size="unknown", image_bytes=self.pixels((96, 64))),
            ]
            shard = self.write_shard(root, "a.parquet", rows)
            self.write_shard(root, "b.parquet", [dict(rows[0], id="second_shard")])
            output = root / "index.jsonl"
            report = build_index(root, output, workers=2)
            self.assertEqual(report["indexed"], 5)
            self.assertEqual(report["decode_error_count"], 1)
            self.assertEqual(report["invalid_caption_count"], 1)
            self.assertEqual(report["dimension_mismatch_count"], 4)
            records = list(iter_image_records(output))
            self.assertEqual([r.identifier for r in records],
                             ["reversed", "wide", "exif", "bad_size", "second_shard"])
            self.assertEqual([(r.width, r.height) for r in records[:3]], [(96, 64), (128, 64), (64, 96)])
            dataset = BucketedImages(dict(manifest=str(output), buckets=[(32, 48), (32, 64), (48, 32)]))
            self.assertEqual([len(g) for g in dataset.groups], [3, 1, 1])
            for index in range(len(dataset)):
                pixels, caption = dataset[index]
                self.assertTrue(caption.strip())
                self.assertEqual(tuple(pixels.shape[:2]), dataset.buckets[dataset.records[index][3]])
            original = output.read_bytes()
            build_index(root, output, workers=1)
            self.assertEqual(original, output.read_bytes())
            # Relative Parquet locations work independently of image_root.
            line = json.loads(output.read_text().splitlines()[0])
            line["parquet_path"] = shard.name
            output.write_text(json.dumps(line) + "\n")
            self.assertEqual(open_record_image(next(iter_image_records(output, image_root="/irrelevant"))).size, (96, 64))
            line["caption"] = "wrong caption"
            output.write_text(json.dumps(line) + "\n")
            with self.assertRaisesRegex(ValueError, "Stale Parquet"):
                open_record_image(next(iter_image_records(output)))
            line["caption"] = "edge text"
            line["width"] = 100
            output.write_text(json.dumps(line) + "\n")
            with self.assertRaisesRegex(ValueError, "Stale dimensions"):
                BucketedImages(dict(manifest=str(output), buckets=[(32, 48)]))[0]

    def test_incomplete_source_does_not_replace_existing_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.pq.write_table(self.pa.table(dict(id=["missing_columns"])), root / "bad.parquet")
            output = root / "index.jsonl"
            output.write_text("previous index\n")
            with self.assertRaisesRegex(RuntimeError, "incomplete index"):
                build_index(root, output, workers=1)
            self.assertEqual(output.read_text(), "previous index\n")


if __name__ == "__main__":
    unittest.main()
