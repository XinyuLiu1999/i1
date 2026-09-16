from dataclasses import replace
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets.bucketed import BucketedImages, BucketBatchSampler
from datasets.pixel_cache import cached_pixels
from datasets.precompute_images import precompute
from datasets.check_precompute import check_cache
from datasets.benchmark_images import benchmark


class PixelCacheTests(unittest.TestCase):
    def test_filtered_parts_and_image_root_match_online_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / 'images'
            images.mkdir()
            rows = []
            for name, size in [('small', (16, 16)), ('eligible', (64, 64))]:
                Image.new('RGB', size, (12, 34, 56)).save(images / f'{name}.png')
                # A same-named file beside the manifest must not override image_root.
                Image.new('RGB', size, (250, 0, 0)).save(root / f'{name}.png')
                rows.append(dict(id=name, image_path=f'{name}.png', caption=name,
                                 width=size[0], height=size[1]))
            manifest = root / 'source.jsonl'
            manifest.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            config = dict(manifest=str(manifest), image_root=str(images),
                          buckets=[(32, 32)], min_image_area=1024)
            online = BucketedImages(config)
            output = root / 'cache'
            report = precompute(manifest, output, config, workers=1, records_per_part=1)
            self.assertEqual(report['count'], 1)
            self.assertEqual(report['filtered'], {'source_too_small': 1})
            cached = BucketedImages(dict(config, manifest=str(output / 'cache.jsonl')))
            self.assertTrue(torch.equal(online[0][0], cached[0][0]))
            self.assertEqual(online[0][1], cached[0][1])
            empty = output / 'parts' / '00000'
            self.assertEqual((empty / 'records.jsonl').read_text(), '')
            self.assertEqual(json.loads((empty / 'complete.json').read_text())['files'], [])
            before = (output / 'cache.jsonl').read_bytes()
            precompute(manifest, output, config, workers=1, records_per_part=1)
            self.assertEqual(before, (output / 'cache.jsonl').read_bytes())

    def test_all_filtered_input_does_not_publish_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.new('RGB', (16, 16)).save(root / 'small.png')
            manifest = root / 'source.jsonl'
            manifest.write_text(json.dumps(dict(image_path='small.png', caption='small',
                                               width=16, height=16)) + '\n')
            config = dict(manifest=str(manifest), buckets=[(32, 32)])
            with self.assertRaisesRegex(ValueError, 'No eligible images'):
                BucketedImages(config)
            output = root / 'cache'
            output.mkdir()
            published = output / 'cache.jsonl'
            published.write_text('previous manifest\n')
            with self.assertRaisesRegex(ValueError, 'No eligible images'):
                precompute(manifest, output, config, workers=1)
            self.assertEqual(published.read_text(), 'previous manifest\n')
            self.assertFalse((output / 'summary.json').exists())

    def test_pixels_sampling_resume_and_stale_config(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for i, (width, height) in enumerate([(96, 64), (129, 64), (64, 96), (64, 96)]):
                rng = np.random.default_rng(min(i, 2))
                Image.fromarray(rng.integers(0, 256, (height, width, 3), dtype=np.uint8)).save(root / f'{i}.png')
                rows.append(dict(id=str(i), image_path=str(root / f'{i}.png'), caption=f'caption {min(i, 2)}',
                                 width=width, height=height))
            source = root / 'source.jsonl'
            source.write_text(''.join(json.dumps(row) + '\n' for row in rows))
            config = dict(manifest=str(source), buckets=[(32, 48), (32, 64), (48, 32)],
                          min_image_area=4096, resize_mode='pad', allow_upscale=False)
            output = root / 'cache'
            report = precompute(source, output, config, workers=2, records_per_part=2,
                                max_shard_bytes=7000)
            self.assertEqual(report['count'], 4)
            self.assertTrue(report['verified_all_written_bytes'])
            raw = BucketedImages(config)
            cached_config = dict(config, manifest=str(output / 'cache.jsonl'))
            cached = BucketedImages(cached_config)
            self.assertEqual(raw.groups, cached.groups)
            for i in range(len(raw)):
                self.assertTrue(torch.equal(raw[i][0], cached[i][0]))
                self.assertEqual(raw[i][1], cached[i][1])
                self.assertEqual(raw.records[i][1:], cached.records[i][1:])
            for rank in range(2):
                self.assertEqual(list(BucketBatchSampler(raw.groups, 2, rank, 2, 8, seed=5)),
                                 list(BucketBatchSampler(cached.groups, 2, rank, 2, 8, seed=5)))
            metadata = (output / 'cache.jsonl').read_bytes()
            audit = check_cache(output / 'cache.jsonl', config, root / 'audit.json', parity_samples=4)
            self.assertEqual(audit['pixel_parity_checked'], 4)
            self.assertTrue(audit['pixel_parity_passed'])
            details = json.loads((root / 'audit.json').read_text())
            self.assertEqual(details['exact_image_duplicate_groups'], [['2', '3']])
            self.assertEqual(details['exact_caption_duplicate_groups'], [['2', '3']])
            measurements = benchmark(source, output / 'cache.jsonl', config, root / 'benchmark.json',
                                     readers=(1, 2), steps=1, warmup=0)
            self.assertEqual(len(measurements['results']), 4)
            self.assertTrue(all(row['measured_images'] == 32 for row in measurements['results']))
            precompute(source, output, config, workers=1, records_per_part=2, max_shard_bytes=7000)
            self.assertEqual(metadata, (output / 'cache.jsonl').read_bytes())
            with self.assertRaisesRegex(ValueError, 'transform mismatch'):
                BucketedImages(dict(cached_config, resize_mode='crop'))
            with self.assertRaisesRegex(ValueError, 'Truncated pixel cache'):
                cached_pixels(replace(cached.records[0][0], cache_offset=10**12))
            # A corrupt completed part cannot be silently reused.
            payload = next(output.glob('parts/*/*.bin'))
            with payload.open('r+b') as handle:
                handle.write(b'bad bytes')
            with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
                precompute(source, output, config, workers=1, records_per_part=2, max_shard_bytes=7000)


if __name__ == '__main__':
    unittest.main()
