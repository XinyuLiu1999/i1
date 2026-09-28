import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "torch_train"))

import numpy as np
from PIL import Image

from configs.sft_1024_captioned import get_config as base_config
from configs.sft_multires_captioned import get_config
from datasets.bucketed import BucketedImages
from datasets.image_geometry import select_bucket
from datasets.pixel_cache import fingerprint, transform_spec
from datasets.precompute_images import precompute


class HighResolutionTests(unittest.TestCase):
    def test_budgets_and_source_boundaries(self):
        original = base_config().input.buckets
        for cap in (1536, 2048):
            config = get_config(cap).input
            self.assertEqual(config.buckets[:len(original)], original)
            self.assertEqual(len(set(config.buckets)), len(config.buckets))
            for (h, w), resolution in zip(config.buckets, config.bucket_resolutions):
                self.assertEqual(h % 16, 0)
                self.assertEqual(w % 16, 0)
                self.assertLessEqual(h * w, resolution ** 2)
                self.assertLessEqual(resolution, cap)
                self.assertLessEqual(max(h, w) / min(h, w), 3)
                self.assertIn((w, h), config.buckets)
            for source, tier in [(1024, 1024), (1535, 1024), (1536, 1536),
                                 (2047, 1536), (2048, min(cap, 2048)), (4096, cap)]:
                index, reason = select_bucket(source, source, config)
                self.assertIsNone(reason)
                self.assertEqual(config.buckets[index], (tier, tier))
            self.assertEqual(select_bucket(1023, 1023, config), (None, "source_too_small"))

    def test_high_tier_wins_over_exact_lower_aspect(self):
        config = get_config().input
        # Scale a non-anchor 1024 frontier shape: its exact aspect exists in the
        # low tier, but must not pull this 2048+ source back down to 1024.
        h, w = next((h, w) for h, w in base_config().input.buckets
                    if h != w and (h * 2, w * 2) not in config.buckets)
        for height, width in [(h * 3, w * 3), (w * 3, h * 3)]:
            index, reason = select_bucket(height, width, config)
            self.assertIsNone(reason)
            self.assertEqual(config.bucket_resolutions[index], 2048)
            bh, bw = config.buckets[index]
            self.assertLessEqual(min(bh / height, bw / width), 1)
            self.assertEqual(bh > bw, height > width)

    def test_no_upscale_and_legacy_selection(self):
        config = dict(buckets=[(32, 32), (48, 48), (64, 64)],
                      bucket_resolutions=[32, 48, 64], allow_upscale=False)
        for h, w in [(32, 32), (40, 60), (64, 96), (16, 512)]:
            index, reason = select_bucket(h, w, config)
            self.assertIsNone(reason)
            bh, bw = config['buckets'][index]
            self.assertLessEqual(min(bh / h, bw / w), 1)
            self.assertLessEqual(config['bucket_resolutions'][index] ** 2, h * w)
        legacy = dict(buckets=[(32, 48), (64, 80)], allow_upscale=False)
        self.assertEqual(select_bucket(128, 192, legacy), (0, None))
        with self.assertRaisesRegex(ValueError, 'bucket_resolutions'):
            select_bucket(128, 128, dict(config, bucket_resolutions=[32]))

    def test_fingerprint_preserves_legacy_and_distinguishes_tiers(self):
        config = base_config().input
        legacy_spec = transform_spec(config)
        self.assertNotIn('bucket_resolutions', legacy_spec)
        self.assertEqual(fingerprint(config), hashlib.sha256(
            json.dumps(legacy_spec, sort_keys=True).encode()).hexdigest())
        tiers = dict(config, bucket_resolutions=[1024] * len(config.buckets))
        self.assertNotEqual(fingerprint(config), fingerprint(tiers))
        self.assertNotEqual(fingerprint(get_config(1536).input), fingerprint(get_config(2048).input))

    def test_cache_matches_online_across_tiers_and_rejects_policy_change(self):
        # Scaled-down tiers exercise the same real exporter and loader without
        # creating a production-sized cache or downloading frozen encoders.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for side in (32, 48, 64):
                pixels = np.random.default_rng(side).integers(0, 256, (side, side, 3), dtype=np.uint8)
                Image.fromarray(pixels).save(root / f'{side}.png')
                rows.append(dict(image_path=str(root / f'{side}.png'), caption='text',
                                 width=side, height=side))
            manifest = root / 'source.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            config = dict(manifest=str(manifest), buckets=[(32, 32), (48, 48), (64, 64)],
                          bucket_resolutions=[32, 48, 64], min_image_area=32**2,
                          allow_upscale=False, resize_mode='pad')
            online = BucketedImages(config)
            output = root / 'cache'
            report = precompute(manifest, output, config, workers=1)
            self.assertEqual(report['count'], 3)
            cached_config = dict(config, manifest=str(output / 'cache.jsonl'))
            cached = BucketedImages(cached_config)
            self.assertEqual(online.groups, [[0], [1], [2]])
            self.assertEqual(online.groups, cached.groups)
            for index in range(3):
                np.testing.assert_array_equal(online[index][0].numpy(), cached[index][0].numpy())
            del cached_config['bucket_resolutions']
            with self.assertRaisesRegex(ValueError, 'transform mismatch'):
                BucketedImages(cached_config)


if __name__ == '__main__':
    unittest.main()
