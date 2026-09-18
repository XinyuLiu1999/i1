from io import BytesIO
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.bucketed import BucketedImages
from datasets.check_precompute import check_cache
from datasets.precompute_captioned import (
    DEFAULT_MAX_SOURCE_PIXELS,
    DEFAULT_MAX_SOURCE_SIDE,
    precompute_captioned,
    source_size_exclusion,
)
from configs.sft_1024_captioned import get_config as captioned_config_1024


class FakeTokenizer:
    def __call__(self, captions, **_):
        return {"input_ids": [[0] + caption.split() for caption in captions]}


def png(size=(80, 64)):
    buffer = BytesIO()
    Image.new("RGB", size, (12, 34, 56)).save(buffer, format="PNG")
    return buffer.getvalue()


class CaptionedPrecomputeTests(unittest.TestCase):
    def test_source_size_limits_include_the_4096_boundary(self):
        self.assertEqual(DEFAULT_MAX_SOURCE_PIXELS, 4096 * 4096)
        self.assertEqual(DEFAULT_MAX_SOURCE_SIDE, 4096)
        self.assertEqual(
            source_size_exclusion(4096, 4096, 4096 * 4096, 4096),
            (None, None),
        )
        self.assertEqual(
            source_size_exclusion(4097, 4096, 4096 * 4096, 4096)[0],
            "source_too_many_pixels",
        )
        self.assertEqual(
            source_size_exclusion(4097, 1, 4096 * 4096, 4096)[0],
            "source_side_too_large",
        )

    def test_captioned_config_uses_finer_production_frontier(self):
        buckets = captioned_config_1024().input.buckets
        self.assertEqual(len(buckets), 45)
        self.assertIn((1024, 1024), buckets)
        self.assertIn((1728, 576), buckets)
        self.assertEqual(buckets, sorted(set(buckets), key=lambda shape: (shape[1] / shape[0], shape)))

    def test_only_valid_caption_and_image_enter_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "captioned" / "data" / "danqing"
            source.mkdir(parents=True)
            rows = [
                dict(id="valid", caption="short caption", caption_status="ok",
                     image_bytes=png(), declared_width=80, declared_height=64),
                dict(id="long", caption="word " * 20, caption_status="ok",
                     image_bytes=png(), declared_width=80, declared_height=64),
                dict(id="failed", caption="", caption_status="error",
                     image_bytes=png(), declared_width=80, declared_height=64),
                dict(id="broken", caption="valid words", caption_status="ok",
                     image_bytes=b"not an image", declared_width=None, declared_height=None),
                dict(id="dimensions", caption="valid words", caption_status="ok",
                     image_bytes=png(), declared_width=81, declared_height=64),
                # Invalid bytes prove the declared-size rejection happens before decode.
                dict(id="too_many_pixels", caption="valid words", caption_status="ok",
                     image_bytes=b"not an image", declared_width=4097, declared_height=4096),
                dict(id="side_too_large", caption="valid words", caption_status="ok",
                     image_bytes=b"not an image", declared_width=4097, declared_height=1),
            ]
            for row in rows:
                row.update(source_dataset="danqing", image_decode_error=None)
            schema = pa.schema([
                pa.field("id", pa.string()), pa.field("source_dataset", pa.string()),
                pa.field("image_bytes", pa.binary()), pa.field("declared_width", pa.int32()),
                pa.field("declared_height", pa.int32()), pa.field("caption", pa.string()),
                pa.field("caption_status", pa.string()), pa.field("image_decode_error", pa.string()),
            ])
            pq.write_table(pa.Table.from_pylist(rows, schema=schema),
                           source / "part-00000.parquet", row_group_size=2)
            output = root / "cache"
            config = dict(buckets=[(64, 80)], min_image_area=0, min_image_side=0,
                          allow_upscale=False, resize_mode="pad", image_root="")
            with patch("transformers.AutoTokenizer.from_pretrained",
                       return_value=FakeTokenizer()):
                summary = precompute_captioned(
                    root / "captioned", output, config, workers=1, token_limit=8,
                    tokenizer_name="fake", records_per_part=10, max_shard_bytes=1024**2)
            self.assertEqual(summary["count"], 1)
            self.assertEqual(summary["rejected"], 6)
            self.assertEqual(summary["rejected_counts"], {
                "caption_too_long": 1, "caption_status": 1,
                "broken_image": 1, "declared_dimension_mismatch": 1,
                "source_too_many_pixels": 1, "source_side_too_large": 1,
            })
            self.assertEqual(summary["bucket_plan"]["exclusions"], {
                "source_too_many_pixels": 1, "source_side_too_large": 1,
            })
            cache_rows = [json.loads(line) for line in (output / "cache.jsonl").read_text().splitlines()]
            self.assertEqual([row["id"] for row in cache_rows], ["valid"])
            dataset = BucketedImages(dict(config, manifest=str(output / "cache.jsonl")))
            self.assertEqual(len(dataset), 1)
            self.assertEqual(dataset[0][1], "short caption")
            audit = check_cache(output / "cache.jsonl", config, root / "audit.json", parity_samples=1)
            self.assertTrue(audit["pixel_parity_passed"])


if __name__ == "__main__":
    unittest.main()
