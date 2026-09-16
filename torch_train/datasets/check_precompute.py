"""CPU audits for a completed pixel cache: duplicates, geometry, and pixel parity."""

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import numpy as np

from .bucketed import BucketedImages
from .data_sources import ImageRecord, open_record_image
from .pixel_cache import cached_pixels


def _rows(path):
    with path.open() as handle:
        for line in handle:
            yield json.loads(line)


def check_cache(manifest, config, report_path, parity_samples=128):
    manifest = Path(manifest).resolve()
    ids, pixels, captions = defaultdict(list), defaultdict(list), defaultdict(list)
    bands = defaultdict(list)
    compact, selected = [], {}
    by_bucket = defaultdict(list)
    candidates = []
    skipped_dense = 0
    # Keep hashes and dimensions for the near-duplicate screen, not decoded images.
    for index, row in enumerate(_rows(manifest)):
        identifier = row['id']
        ids[identifier].append(index)
        pixels[row['source_pixel_sha256']].append(identifier)
        captions[hashlib.sha256(row['caption'].encode()).hexdigest()].append(identifier)
        key = (row['cache_height'], row['cache_width'])
        by_bucket[key].append(index)
        value = int(row['source_dhash'], 16)
        possible = set()
        for band in range(4):
            matches = bands[(band, (value >> (16 * band)) & 65535)]
            if len(matches) <= 256:
                possible.update(matches)
            else:
                skipped_dense += 1
        if len(candidates) < 10000:
            ratio = row['width'] / row['height']
            for other in sorted(possible):
                oid, dhash, aspect, pixel_hash = compact[other]
                if pixel_hash == row['source_pixel_sha256'] or abs(math.log(ratio / aspect)) > .05:
                    continue
                distance = (value ^ dhash).bit_count()
                if distance <= 3:
                    candidates.append(dict(first=oid, second=identifier, hamming_distance=distance))
                    if len(candidates) == 10000:
                        break
        compact.append((identifier, value, row['width'] / row['height'], row['source_pixel_sha256']))
        for band in range(4):
            matches = bands[(band, (value >> (16 * band)) & 65535)]
            # A sentinel-length list identifies dense bins without unbounded storage.
            if len(matches) <= 256:
                matches.append(index)
    dataset = BucketedImages(dict(config, manifest=str(manifest)))
    if len(dataset) != len(compact):
        raise ValueError('The cache unexpectedly filters records under its training config.')
    rng = random.Random(20260916)
    sample_indices = {rng.choice(group) for group in by_bucket.values()}
    sample_indices.update(rng.sample(range(len(dataset)), min(parity_samples, len(dataset))))
    for index, row in enumerate(_rows(manifest)):
        if index in sample_indices:
            selected[index] = row
    for index in sorted(selected):
        entry = selected[index]
        record, h, w, bucket = dataset.records[index]
        source_record = ImageRecord(**entry['source'])
        with open_record_image(source_record) as source:
            if source.size != (w, h):
                raise ValueError(f'Source dimensions changed for {entry["id"]}')
            transformed = dataset.process_image(source, dataset.buckets[bucket])
        cached = cached_pixels(record)
        if not np.array_equal(np.asarray(transformed), cached):
            raise ValueError(f'Cached pixels differ from online transform: {entry["id"]}')
        transformed.close()
        if hashlib.sha256(cached.tobytes()).hexdigest() != entry['pixel_sha256']:
            raise ValueError(f'Cached pixel hash mismatch: {entry["id"]}')
        normalized, caption = dataset[index]
        if (not np.isfinite(normalized.numpy()).all() or normalized.min() < -1 or normalized.max() > 1
                or caption != entry['caption']):
            raise ValueError(f'Invalid training sample: {entry["id"]}')
    duplicate_ids = {key: value for key, value in ids.items() if len(value) > 1}
    duplicate_pixels = [value for value in pixels.values() if len(value) > 1]
    duplicate_captions = [value for value in captions.values() if len(value) > 1]
    report = dict(count=len(dataset), pixel_parity_checked=len(selected), pixel_parity_passed=True,
                  populated_buckets=len(by_bucket),
                  bucket_counts={f'{h}x{w}': len(group) for (h, w), group in by_bucket.items()},
                  duplicate_ids=duplicate_ids, exact_image_duplicate_groups=duplicate_pixels,
                  exact_caption_duplicate_groups=duplicate_captions,
                  near_duplicate_screen=dict(method='64-bit dHash, distance <=3, aspect within 5%; candidates require review',
                                             candidates=candidates, candidate_limit=10000,
                                             candidate_limit_reached=len(candidates) >= 10000,
                                             dense_band_lookups_skipped=skipped_dense,
                                             exhaustive=False),
                  shm_bytes=shutil.disk_usage('/dev/shm').total)
    Path(report_path).write_text(json.dumps(report, indent=2) + '\n')
    return {key: value for key, value in report.items() if key not in
            ('duplicate_ids', 'exact_image_duplicate_groups', 'exact_caption_duplicate_groups', 'near_duplicate_screen')}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--report', required=True)
    parser.add_argument('--resolution', type=int, choices=[512, 1024], default=1024)
    parser.add_argument('--parity_samples', type=int, default=128)
    args = parser.parse_args()
    from configs.sft_512 import get_config as config512
    from configs.sft_1024 import get_config as config1024
    config = (config512 if args.resolution == 512 else config1024)().input
    print(json.dumps(check_cache(args.manifest, config, args.report, args.parity_samples), indent=2))


if __name__ == '__main__':
    main()
