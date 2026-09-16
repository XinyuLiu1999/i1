"""Precompute exact SFT transforms into resumable, bounded uint8 RGB shards.

Reads a corrected JSONL index in source order. Every completed shard is read back
and SHA256-checked before publication. Training uses the resulting cache.jsonl.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import tempfile
import time

import numpy as np
from PIL import Image

from .bucketed import BucketedImages
from .data_sources import iter_image_records, open_record_image
from .pixel_cache import fingerprint, transform_spec


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def difference_hash(image):
    thumbnail = np.asarray(image.convert('L').resize((9, 8), Image.Resampling.LANCZOS))
    return np.packbits(thumbnail[:, 1:] > thumbnail[:, :-1]).tobytes().hex()


def _export_part(input_path, output, ordinal, config, max_shard_bytes):
    started = time.monotonic()
    output = Path(output)
    part = output / 'parts' / f'{ordinal:05d}'
    source_stats = {}
    for record in iter_image_records(input_path):
        source = record.parquet_path or record.image_path
        stat = Path(source).stat()
        source_stats[source] = [stat.st_size, stat.st_mtime_ns]
    signature = dict(input_sha256=file_hash(input_path), transform=fingerprint(config),
                     source_stats=source_stats, max_shard_bytes=max_shard_bytes)
    if (part / 'complete.json').exists():
        report = json.loads((part / 'complete.json').read_text())
        if report['signature'] != signature:
            raise ValueError(f'Existing cache part has different inputs/settings: {part}; use a new output directory.')
        if file_hash(part / 'records.jsonl') != report['records_sha256']:
            raise ValueError(f'Cache metadata checksum mismatch: {part}')
        for item in report['files']:
            if file_hash(part / item['name']) != item['sha256']:
                raise ValueError(f'Cache checksum mismatch: {part / item["name"]}')
        return dict(report, resumed=True)
    config = dict(config, manifest=str(input_path))
    # A source part may be entirely filtered even when the full dataset is valid.
    dataset = BucketedImages(config, allow_empty=True)
    handles, digests, byte_counts, file_names = {}, {}, {}, {}
    bucket_counts = Counter()
    pad_sum = 0.0
    pad_max = 0.0
    part.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f'.part-{ordinal:05d}-', dir=output) as temporary:
        stage = Path(temporary) / 'data'
        stage.mkdir()
        try:
            with (stage / 'records.jsonl').open('w') as manifest:
                for completed, (record, height, width, bucket) in enumerate(dataset.records, 1):
                    with open_record_image(record) as image:
                        if image.size != (width, height):
                            raise ValueError(f'Stale source dimensions: {record.identifier}')
                        raw_digest = hashlib.sha256(
                            f'{width}x{height}:RGB:'.encode() + image.tobytes()).hexdigest()
                        dhash = difference_hash(image)
                        transformed = dataset.process_image(image, dataset.buckets[bucket])
                    bh, bw = dataset.buckets[bucket]
                    pixels = np.asarray(transformed, dtype=np.uint8)
                    if pixels.shape != (bh, bw, 3):
                        raise ValueError(f'Unexpected transformed shape: {record.identifier}')
                    payload = pixels.tobytes()
                    pixel_digest = hashlib.sha256(payload).hexdigest()
                    transformed.close()
                    if bucket not in handles or byte_counts[file_names[bucket]] + len(payload) > max_shard_bytes:
                        if bucket in handles:
                            handles[bucket].close()
                        name = f'b{bucket:02d}_{len(digests):04d}.bin'
                        handles[bucket] = (stage / name).open('wb')
                        digests[name] = hashlib.sha256()
                        byte_counts[name] = 0
                        file_names[bucket] = name
                    name = file_names[bucket]
                    offset = byte_counts[name]
                    handles[bucket].write(payload)
                    digests[name].update(payload)
                    byte_counts[name] += len(payload)
                    bucket_counts[f'{bh}x{bw}'] += 1
                    scale = dataset.resize_scale(height, width, bh, bw)
                    padding = (1 - round(width * scale) * round(height * scale) / (bh * bw)
                               if dataset.resize_mode == 'pad' else 0.0)
                    pad_sum += padding
                    pad_max = max(pad_max, padding)
                    source_ref = {key: value for key, value in asdict(record).items() if value is not None}
                    entry = dict(id=record.identifier, caption=record.caption, width=width, height=height,
                                 cache_path=str(Path('parts') / part.name / name), cache_offset=offset,
                                 cache_height=bh, cache_width=bw, transform_fingerprint=signature['transform'],
                                 pixel_sha256=pixel_digest, source_pixel_sha256=raw_digest,
                                 source_dhash=dhash, source=source_ref)
                    manifest.write(json.dumps(entry, ensure_ascii=False) + '\n')
                    if completed % 500 == 0:
                        print(f'part {ordinal:05d}: transformed {completed}/{len(dataset)} images', flush=True)
        finally:
            for handle in handles.values():
                handle.close()
        files = []
        for name, digest in digests.items():
            expected = digest.hexdigest()
            if (stage / name).stat().st_size != byte_counts[name] or file_hash(stage / name) != expected:
                raise ValueError(f'Pixel readback differs from online transforms: {name}')
            files.append(dict(name=name, bytes=byte_counts[name], sha256=expected))
        # Detect source mutation during the export, not just during resume.
        for path, expected in source_stats.items():
            stat = Path(path).stat()
            if [stat.st_size, stat.st_mtime_ns] != expected:
                raise ValueError(f'Source changed while precomputing: {path}')
        report = dict(signature=signature, count=len(dataset), filtered=dict(dataset.filtered),
                      bucket_counts=dict(bucket_counts), files=files, bytes=sum(byte_counts.values()),
                      records_sha256=file_hash(stage / 'records.jsonl'),
                      padding_sum=pad_sum, max_padding_fraction=pad_max,
                      verified_all_written_bytes=True, seconds=time.monotonic() - started)
        (stage / 'complete.json').write_text(json.dumps(report, indent=2) + '\n')
        os.rename(stage, part)
    return report


def precompute(manifest, output, config, workers=8, records_per_part=2000, max_shard_bytes=256 * 1024**2):
    if workers <= 0 or records_per_part <= 0 or max_shard_bytes <= 0:
        raise ValueError('workers, records_per_part and max_shard_bytes must be positive.')
    manifest = Path(manifest).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    source_hash = file_hash(manifest)
    specification = dict(source_manifest=str(Path(manifest).resolve()), source_sha256=source_hash,
                         transform=transform_spec(config), records_per_part=records_per_part,
                         max_shard_bytes=max_shard_bytes)
    settings = output / 'settings.json'
    if settings.exists() and json.loads(settings.read_text()) != specification:
        raise ValueError('Cache output has different settings; use a new directory.')
    settings.write_text(json.dumps(specification, indent=2) + '\n')
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix='.inputs-', dir=output) as temporary:
        parts, handle = [], None
        try:
            last_source = None
            in_part = 0
            for record in iter_image_records(manifest, image_root=config.get('image_root')):
                if record.cache_path is not None:
                    raise ValueError('Precompute requires original sources, not another pixel cache.')
                source = record.parquet_path or '__image_files__'
                if handle is None or source != last_source or in_part >= records_per_part:
                    if handle is not None:
                        handle.close()
                    parts.append(Path(temporary) / f'{len(parts):05d}.jsonl')
                    handle = parts[-1].open('w')
                    in_part = 0
                    last_source = source
                entry = {key: value for key, value in asdict(record).items() if value is not None}
                entry['id'] = entry.pop('identifier')
                handle.write(json.dumps(entry, ensure_ascii=False) + '\n')
                in_part += 1
        finally:
            if handle is not None:
                handle.close()
        if not parts:
            raise ValueError('No records to precompute.')
        reports = {}
        context = multiprocessing.get_context('spawn')
        with ProcessPoolExecutor(max_workers=min(workers, len(parts)), mp_context=context) as executor:
            futures = {executor.submit(_export_part, path, output, i, dict(config), max_shard_bytes): i
                       for i, path in enumerate(parts)}
            for future in as_completed(futures):
                i = futures[future]
                reports[i] = future.result()
                done = sum(item['count'] for item in reports.values())
                print(f'cached {len(reports)}/{len(parts)} parts; {done} images verified; '
                      f'{time.monotonic() - started:.1f}s', flush=True)
        count = sum(item['count'] for item in reports.values())
        if not count:
            raise ValueError('No eligible images to precompute; cache manifest was not replaced.')
        counts, filtered = Counter(), Counter()
        with (output / 'cache.jsonl.partial').open('wb') as destination:
            for i in range(len(parts)):
                report = reports[i]
                counts.update(report['bucket_counts'])
                filtered.update(report['filtered'])
                with (output / 'parts' / f'{i:05d}' / 'records.jsonl').open('rb') as source:
                    shutil.copyfileobj(source, destination)
        if file_hash(manifest) != source_hash:
            raise ValueError('Source manifest changed during export.')
        os.replace(output / 'cache.jsonl.partial', output / 'cache.jsonl')
    summary = dict(count=count, parts=len(parts), bytes=sum(item['bytes'] for item in reports.values()),
                   bucket_counts=dict(counts), filtered=dict(filtered),
                   mean_padding_fraction=sum(item['padding_sum'] for item in reports.values()) / count,
                   max_padding_fraction=max(item['max_padding_fraction'] for item in reports.values()),
                   verified_all_written_bytes=True, seconds=time.monotonic() - started,
                   settings=specification, cache_manifest_sha256=file_hash(output / 'cache.jsonl'))
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--resolution', type=int, choices=[512, 1024], default=1024)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    from configs.sft_512 import get_config as config512
    from configs.sft_1024 import get_config as config1024
    config = (config512 if args.resolution == 512 else config1024)().input
    print(json.dumps(precompute(args.manifest, args.output_dir, config, args.workers), indent=2))


if __name__ == '__main__':
    main()
