"""Audit captioned production Parquet and build an accepted-only pixel cache.

Caption validation, image decoding, dimension verification, bucket assignment and
pixel transformation happen in the same worker pass. Rejected records are written
to ``rejected.jsonl`` and never appear in the training ``cache.jsonl``.
"""

from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
import math
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import tempfile
import time

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageOps

from .image_geometry import select_bucket
from .pixel_cache import fingerprint, transform_spec


DEFAULT_SOURCE = Path("/nfs_yaoyuan/liuxinyu/textdense_primary_english_captioned_v4")
DEFAULT_OUTPUT = Path(
    "/cephfs/liuxinyu/DenseText-Project/artifacts/"
    "textdense_primary_english_captioned_v4_precompute/cache_1024"
)
DEFAULT_MAX_SOURCE_SIDE = 4096
DEFAULT_MAX_SOURCE_PIXELS = 4096 * 4096
REQUIRED_COLUMNS = {
    "id", "source_dataset", "image_bytes", "declared_width", "declared_height",
    "caption", "caption_status", "image_decode_error",
}


def file_hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def captioned_files(source):
    source = Path(source).expanduser().resolve()
    data = source / "data" if (source / "data").is_dir() else source
    files = sorted(path for path in data.rglob("*.parquet") if not path.name.startswith("."))
    if not files:
        raise FileNotFoundError(f"no captioned Parquet files found under {source}")
    for path in files:
        missing = REQUIRED_COLUMNS.difference(pq.ParquetFile(path).schema_arrow.names)
        if missing:
            raise ValueError(f"{path} is missing captioned columns {sorted(missing)}")
    return files


def difference_hash(image):
    thumbnail = np.asarray(image.convert("L").resize((9, 8), Image.Resampling.LANCZOS))
    return np.packbits(thumbnail[:, 1:] > thumbnail[:, :-1]).tobytes().hex()


def process_image(image, bucket, resize_mode="pad"):
    bh, bw = bucket
    width, height = image.size
    scale = (min if resize_mode == "pad" else max)(bh / height, bw / width)
    if resize_mode == "pad":
        new_w = max(1, min(bw, round(width * scale)))
        new_h = max(1, min(bh, round(height * scale)))
    else:
        new_w = max(bw, math.ceil(width * scale))
        new_h = max(bh, math.ceil(height * scale))
    resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
    if resize_mode == "pad":
        transformed = Image.new("RGB", (bw, bh), (255, 255, 255))
        transformed.paste(resized, ((bw - new_w) // 2, (bh - new_h) // 2))
        resized.close()
        return transformed
    left, top = (new_w - bw) // 2, (new_h - bh) // 2
    transformed = resized.crop((left, top, left + bw, top + bh))
    resized.close()
    return transformed


def reject(handle, counts, uid, source, reason, **details):
    counts[reason] += 1
    value = dict(id=str(uid), source_dataset=source, reason=reason, **details)
    handle.write(json.dumps(value, ensure_ascii=False) + "\n")


def source_size_exclusion(width, height, max_source_pixels, max_source_side):
    """Return a deterministic oversized-source rejection and audit details."""
    pixels = width * height
    if pixels > max_source_pixels:
        return "source_too_many_pixels", dict(
            width=width, height=height, pixels=pixels,
            max_source_pixels=max_source_pixels,
        )
    if width > max_source_side or height > max_source_side:
        return "source_side_too_large", dict(
            width=width, height=height, max_source_side=max_source_side,
        )
    return None, None


def tokenize_lengths(tokenizer, captions, batch_size=128):
    lengths = {}
    for start in range(0, len(captions), batch_size):
        batch = captions[start:start + batch_size]
        encoded = tokenizer([caption for _, caption in batch], truncation=False,
                            padding=False, add_special_tokens=True)["input_ids"]
        lengths.update((uid, len(ids)) for (uid, _), ids in zip(batch, encoded))
    return lengths


def caption_prefix_matches(caption, word, word_count=10):
    """Case-insensitive whole-word match in the first N regex word tokens."""
    return word.casefold() in re.findall(r"\b\w+\b", caption.casefold())[:word_count]


def _part_signature(files, config, token_limit, tokenizer_name, max_shard_bytes,
                    max_source_pixels, max_source_side, reject_prefix_word=None,
                    prefix_word_count=10):
    result = dict(
        format="densetext-accepted-cache-v1",
        inputs=[dict(path=str(path), bytes=path.stat().st_size,
                     mtime_ns=path.stat().st_mtime_ns) for path in files],
        transform=fingerprint(config), token_limit=token_limit,
        tokenizer=tokenizer_name, max_shard_bytes=max_shard_bytes,
        max_source_pixels=max_source_pixels, max_source_side=max_source_side,
    )
    if reject_prefix_word is not None:
        result["caption_prefix_filter"] = dict(word=reject_prefix_word, count=prefix_word_count,
                                               tokenizer="unicode-regex-word-v1")
    return result


def _audit_export_part(files, output, ordinal, config, token_limit, tokenizer_name,
                       max_shard_bytes, max_source_pixels, max_source_side,
                       reject_prefix_word=None, prefix_word_count=10):
    from transformers import AutoTokenizer

    started = time.monotonic()
    files = [Path(path) for path in files]
    output = Path(output)
    part = output / "parts" / f"{ordinal:05d}"
    signature = _part_signature(
        files, config, token_limit, tokenizer_name, max_shard_bytes,
        max_source_pixels, max_source_side, reject_prefix_word, prefix_word_count)
    if (part / "complete.json").exists():
        report = json.loads((part / "complete.json").read_text())
        if report["signature"] != signature:
            raise ValueError(f"existing cache part has different inputs/settings: {part}")
        if file_hash(part / "records.jsonl") != report["records_sha256"]:
            raise ValueError(f"cache records checksum mismatch: {part}")
        if file_hash(part / "rejected.jsonl") != report["rejected_sha256"]:
            raise ValueError(f"rejection records checksum mismatch: {part}")
        for item in report["files"]:
            if file_hash(part / item["name"]) != item["sha256"]:
                raise ValueError(f"cache payload checksum mismatch: {part / item['name']}")
        return dict(report, resumed=True)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    handles = {}
    digests = {}
    byte_counts = {}
    file_names = {}
    bucket_counts = Counter()
    rejected_counts = Counter()
    source_counts = Counter()
    accepted_sources = Counter()
    token_histogram = Counter()
    pad_sum = 0.0
    pad_max = 0.0
    accepted = 0
    inspected = 0
    part.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix=f".part-{ordinal:05d}-", dir=output) as temporary:
        stage = Path(temporary) / "data"
        stage.mkdir()
        try:
            with (stage / "records.jsonl").open("w", encoding="utf-8") as manifest, \
                    (stage / "rejected.jsonl").open("w", encoding="utf-8") as rejected:
                for path in files:
                    path_text = str(path.resolve())
                    parquet = pq.ParquetFile(path)
                    metadata = parquet.read(columns=[
                        "id", "source_dataset", "caption", "caption_status",
                        "image_decode_error",
                    ]).to_pylist()
                    candidates = []
                    preliminary_reasons = {}
                    for row in metadata:
                        uid = str(row["id"])
                        source = str(row["source_dataset"])
                        inspected += 1
                        source_counts[source] += 1
                        caption = row.get("caption")
                        if row.get("caption_status") != "ok":
                            preliminary_reasons[uid] = ("caption_status", dict(
                                caption_status=row.get("caption_status")))
                        elif not isinstance(caption, str) or not caption.strip():
                            preliminary_reasons[uid] = ("empty_caption", {})
                        elif (reject_prefix_word is not None and caption_prefix_matches(
                                caption, reject_prefix_word, prefix_word_count)):
                            preliminary_reasons[uid] = ("caption_prefix_word", dict(
                                word=reject_prefix_word, first_words=prefix_word_count))
                        elif row.get("image_decode_error"):
                            preliminary_reasons[uid] = ("gpu_image_decode_error", dict(
                                error=row.get("image_decode_error")))
                        else:
                            candidates.append((uid, caption))
                    lengths = tokenize_lengths(tokenizer, candidates)
                    for uid, length in lengths.items():
                        token_histogram[str(length)] += 1
                        if length > token_limit:
                            preliminary_reasons[uid] = (
                                "caption_too_long", dict(tokens=length, token_limit=token_limit))

                    for group in range(parquet.num_row_groups):
                        table = parquet.read_row_group(group, columns=[
                            "id", "source_dataset", "image_bytes", "declared_width",
                            "declared_height", "caption",
                        ])
                        for row_in_group, row in enumerate(table.to_pylist()):
                            uid = str(row["id"])
                            source = str(row["source_dataset"])
                            if uid in preliminary_reasons:
                                reason, details = preliminary_reasons[uid]
                                reject(rejected, rejected_counts, uid, source, reason, **details)
                                continue
                            token_length = lengths[uid]
                            declared_width = row.get("declared_width")
                            declared_height = row.get("declared_height")
                            if (isinstance(declared_width, int) and declared_width > 0
                                    and isinstance(declared_height, int) and declared_height > 0):
                                reason, details = source_size_exclusion(
                                    declared_width, declared_height,
                                    max_source_pixels, max_source_side)
                                if reason is not None:
                                    reject(rejected, rejected_counts, uid, source, reason,
                                           dimension_source="declared", **details)
                                    continue
                            try:
                                with Image.open(BytesIO(row["image_bytes"])) as raw:
                                    reason, details = source_size_exclusion(
                                        raw.width, raw.height,
                                        max_source_pixels, max_source_side)
                                    if reason is not None:
                                        reject(rejected, rejected_counts, uid, source, reason,
                                               dimension_source="image_header", **details)
                                        continue
                                    image = ImageOps.exif_transpose(raw).convert("RGB")
                                    image.load()
                            except Exception as error:
                                reject(rejected, rejected_counts, uid, source, "broken_image",
                                       error=f"{type(error).__name__}: {error}")
                                continue
                            width, height = image.size
                            reason, details = source_size_exclusion(
                                width, height, max_source_pixels, max_source_side)
                            if reason is not None:
                                image.close()
                                reject(rejected, rejected_counts, uid, source, reason,
                                       dimension_source="decoded", **details)
                                continue
                            if (row.get("declared_width"), row.get("declared_height")) != (width, height):
                                image.close()
                                reject(rejected, rejected_counts, uid, source,
                                       "declared_dimension_mismatch",
                                       declared=[row.get("declared_width"), row.get("declared_height")],
                                       actual=[width, height])
                                continue
                            bucket, exclusion = select_bucket(height, width, config)
                            if exclusion is not None:
                                image.close()
                                reject(rejected, rejected_counts, uid, source, exclusion,
                                       width=width, height=height)
                                continue
                            bh, bw = config["buckets"][bucket]
                            raw_digest = hashlib.sha256(
                                f"{width}x{height}:RGB:".encode() + image.tobytes()).hexdigest()
                            dhash = difference_hash(image)
                            transformed = process_image(image, (bh, bw), config.get("resize_mode", "pad"))
                            image.close()
                            pixels = np.asarray(transformed, dtype=np.uint8)
                            if pixels.shape != (bh, bw, 3):
                                transformed.close()
                                reject(rejected, rejected_counts, uid, source,
                                       "unexpected_transform_shape", shape=list(pixels.shape))
                                continue
                            payload = pixels.tobytes()
                            pixel_digest = hashlib.sha256(payload).hexdigest()
                            transformed.close()
                            if (bucket not in handles
                                    or byte_counts[file_names[bucket]] + len(payload) > max_shard_bytes):
                                if bucket in handles:
                                    handles[bucket].close()
                                name = f"b{bucket:02d}_{len(digests):04d}.bin"
                                handles[bucket] = (stage / name).open("wb")
                                digests[name] = hashlib.sha256()
                                byte_counts[name] = 0
                                file_names[bucket] = name
                            name = file_names[bucket]
                            offset = byte_counts[name]
                            handles[bucket].write(payload)
                            digests[name].update(payload)
                            byte_counts[name] += len(payload)
                            scale = min(bh / height, bw / width)
                            padding = (1 - round(width * scale) * round(height * scale) / (bh * bw)
                                       if config.get("resize_mode", "pad") == "pad" else 0.0)
                            pad_sum += padding
                            pad_max = max(pad_max, padding)
                            bucket_counts[f"{bh}x{bw}"] += 1
                            accepted_sources[source] += 1
                            accepted += 1
                            source_record = dict(
                                identifier=uid, caption=row["caption"], width=width, height=height,
                                parquet_path=path_text, row_group=group,
                                row_in_group=row_in_group,
                            )
                            entry = dict(
                                id=uid, source_dataset=source, caption=row["caption"],
                                caption_tokens=token_length,
                                width=width, height=height,
                                cache_path=str(Path("parts") / part.name / name),
                                cache_offset=offset, cache_height=bh, cache_width=bw,
                                transform_fingerprint=signature["transform"],
                                pixel_sha256=pixel_digest, source_pixel_sha256=raw_digest,
                                source_dhash=dhash, source=source_record,
                            )
                            manifest.write(json.dumps(entry, ensure_ascii=False) + "\n")
        finally:
            for handle in handles.values():
                handle.close()

        payload_files = []
        for name, digest in digests.items():
            expected = digest.hexdigest()
            if (stage / name).stat().st_size != byte_counts[name] or file_hash(stage / name) != expected:
                raise ValueError(f"pixel payload verification failed: {stage / name}")
            payload_files.append(dict(name=name, bytes=byte_counts[name], sha256=expected))
        report = dict(
            signature=signature, inspected=inspected, count=accepted,
            source_counts=dict(source_counts), accepted_source_counts=dict(accepted_sources),
            rejected_counts=dict(rejected_counts), token_histogram=dict(token_histogram),
            bucket_counts=dict(bucket_counts), files=payload_files,
            bytes=sum(byte_counts.values()), records_sha256=file_hash(stage / "records.jsonl"),
            rejected_sha256=file_hash(stage / "rejected.jsonl"), padding_sum=pad_sum,
            max_padding_fraction=pad_max, verified_all_written_bytes=True,
            seconds=time.monotonic() - started,
        )
        (stage / "complete.json").write_text(json.dumps(report, indent=2) + "\n")
        os.rename(stage, part)
    return report


def group_files(files, records_per_part):
    groups = []
    current = []
    rows = 0
    for path in files:
        count = pq.ParquetFile(path).metadata.num_rows
        if current and rows + count > records_per_part:
            groups.append(current)
            current, rows = [], 0
        current.append(path)
        rows += count
    if current:
        groups.append(current)
    return groups


def plan_dimensions(files, config, max_source_pixels=DEFAULT_MAX_SOURCE_PIXELS,
                    max_source_side=DEFAULT_MAX_SOURCE_SIDE):
    """Cheap metadata-only coverage report that freezes buckets before decoding."""
    buckets = Counter()
    exclusions = Counter()
    sources = Counter()
    missing = 0
    padding_sum = 0.0
    padding_max = 0.0
    assigned = 0
    cache_bytes = 0
    min_width = min_height = None
    max_width = max_height = 0
    for path in files:
        for row in pq.ParquetFile(path).read(columns=[
                "source_dataset", "declared_width", "declared_height"]).to_pylist():
            source = str(row["source_dataset"])
            sources[source] += 1
            width, height = row.get("declared_width"), row.get("declared_height")
            if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
                missing += 1
                continue
            min_width = width if min_width is None else min(min_width, width)
            min_height = height if min_height is None else min(min_height, height)
            max_width, max_height = max(max_width, width), max(max_height, height)
            reason, _ = source_size_exclusion(
                width, height, max_source_pixels, max_source_side)
            if reason is not None:
                exclusions[reason] += 1
                continue
            bucket, reason = select_bucket(height, width, config)
            if reason is not None:
                exclusions[reason] += 1
                continue
            bh, bw = config["buckets"][bucket]
            buckets[f"{bh}x{bw}"] += 1
            cache_bytes += bh * bw * 3
            scale = min(bh / height, bw / width)
            padding = (1 - round(width * scale) * round(height * scale) / (bh * bw)
                       if config.get("resize_mode", "pad") == "pad" else 0.0)
            padding_sum += padding
            padding_max = max(padding_max, padding)
            assigned += 1
    return dict(
        count=sum(sources.values()), source_counts=dict(sources), missing_dimensions=missing,
        assigned=assigned, exclusions=dict(exclusions), configured_buckets=len(config["buckets"]),
        populated_buckets=len(buckets), bucket_counts=dict(buckets), estimated_cache_bytes=cache_bytes,
        mean_padding_fraction=padding_sum / assigned if assigned else None,
        max_padding_fraction=padding_max if assigned else None,
        max_source_pixels=max_source_pixels, max_source_side=max_source_side,
        dimensions=dict(min_width=min_width, max_width=max_width,
                        min_height=min_height, max_height=max_height),
    )


def percentile_from_histogram(histogram, percentile):
    total = sum(histogram.values())
    if not total:
        return None
    target = percentile / 100 * (total - 1)
    cumulative = 0
    for length in sorted(histogram):
        cumulative += histogram[length]
        if cumulative > target:
            return length
    return max(histogram)


def precompute_captioned(source, output, config, workers=8, token_limit=1024,
                         tokenizer_name="google/t5gemma-2b-2b-ul2-it",
                         records_per_part=2000, max_shard_bytes=256 * 1024**2,
                         max_source_pixels=DEFAULT_MAX_SOURCE_PIXELS,
                         max_source_side=DEFAULT_MAX_SOURCE_SIDE, *,
                         reject_prefix_word=None, prefix_word_count=10):
    if min(workers, token_limit, records_per_part, max_shard_bytes,
           max_source_pixels, max_source_side) <= 0:
        raise ValueError("worker, token, shard, and source-size limits must be positive")
    if reject_prefix_word is not None:
        if not re.fullmatch(r"\w+", reject_prefix_word) or prefix_word_count <= 0:
            raise ValueError("caption filter requires one word and a positive prefix length")
        reject_prefix_word = reject_prefix_word.casefold()
    source = Path(source).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    files = captioned_files(source)
    snapshot = [dict(path=str(path), bytes=path.stat().st_size,
                     mtime_ns=path.stat().st_mtime_ns,
                     rows=pq.ParquetFile(path).metadata.num_rows) for path in files]
    specification = dict(
        format="densetext-accepted-cache-v1", source=str(source), inputs=snapshot,
        transform=transform_spec(config), token_limit=token_limit, tokenizer=tokenizer_name,
        records_per_part=records_per_part, max_shard_bytes=max_shard_bytes,
        max_source_pixels=max_source_pixels, max_source_side=max_source_side,
    )
    if reject_prefix_word is not None:
        specification["caption_prefix_filter"] = dict(
            word=reject_prefix_word, count=prefix_word_count, tokenizer="unicode-regex-word-v1")
    settings = output / "settings.json"
    if settings.exists() and json.loads(settings.read_text()) != specification:
        raise ValueError("cache output has different settings; use a new output directory")
    settings.write_text(json.dumps(specification, indent=2) + "\n")
    bucket_plan = plan_dimensions(
        files, config, max_source_pixels, max_source_side)
    (output / "bucket_plan.json").write_text(json.dumps(bucket_plan, indent=2) + "\n")
    groups = group_files(files, records_per_part)
    started = time.monotonic()
    reports = {}
    if workers == 1:
        for ordinal, group in enumerate(groups):
            reports[ordinal] = _audit_export_part(
                [str(path) for path in group], str(output), ordinal, dict(config),
                token_limit, tokenizer_name, max_shard_bytes,
                max_source_pixels, max_source_side, reject_prefix_word, prefix_word_count)
            print(f"completed {len(reports)}/{len(groups)} parts; inspected "
                  f"{sum(report['inspected'] for report in reports.values())}; accepted "
                  f"{sum(report['count'] for report in reports.values())}", flush=True)
    else:
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=min(workers, len(groups)), mp_context=context) as executor:
            futures = {
                executor.submit(_audit_export_part, [str(path) for path in group], str(output), ordinal,
                                dict(config), token_limit, tokenizer_name, max_shard_bytes,
                                max_source_pixels, max_source_side,
                                reject_prefix_word, prefix_word_count): ordinal
                for ordinal, group in enumerate(groups)
            }
            for future in as_completed(futures):
                ordinal = futures[future]
                reports[ordinal] = future.result()
                inspected = sum(report["inspected"] for report in reports.values())
                accepted = sum(report["count"] for report in reports.values())
                print(f"completed {len(reports)}/{len(groups)} parts; inspected {inspected}; "
                      f"accepted {accepted}", flush=True)

    count = sum(report["count"] for report in reports.values())
    if not count:
        raise ValueError("no valid captioned images; cache manifest was not published")
    inspected = sum(report["inspected"] for report in reports.values())
    buckets, rejected, sources, accepted_sources, token_histogram = (Counter() for _ in range(5))
    with (output / "cache.jsonl.partial").open("wb") as accepted_handle, \
            (output / "rejected.jsonl.partial").open("wb") as rejected_handle:
        for ordinal in range(len(groups)):
            report = reports[ordinal]
            buckets.update(report["bucket_counts"])
            rejected.update(report["rejected_counts"])
            sources.update(report["source_counts"])
            accepted_sources.update(report["accepted_source_counts"])
            token_histogram.update({int(key): value for key, value in report["token_histogram"].items()})
            part = output / "parts" / f"{ordinal:05d}"
            with (part / "records.jsonl").open("rb") as handle:
                shutil.copyfileobj(handle, accepted_handle)
            with (part / "rejected.jsonl").open("rb") as handle:
                shutil.copyfileobj(handle, rejected_handle)
    os.replace(output / "cache.jsonl.partial", output / "cache.jsonl")
    os.replace(output / "rejected.jsonl.partial", output / "rejected.jsonl")
    summary = dict(
        format="densetext-accepted-cache-v1", status="complete", inspected=inspected,
        count=count, rejected=inspected - count, rejected_counts=dict(rejected),
        source_counts=dict(sources), accepted_source_counts=dict(accepted_sources),
        parts=len(groups), bytes=sum(report["bytes"] for report in reports.values()),
        bucket_counts=dict(buckets), populated_buckets=len(buckets),
        caption_tokens=dict(count=sum(token_histogram.values()),
                            p50=percentile_from_histogram(token_histogram, 50),
                            p95=percentile_from_histogram(token_histogram, 95),
                            p99=percentile_from_histogram(token_histogram, 99),
                            maximum=max(token_histogram, default=None), limit=token_limit),
        mean_padding_fraction=sum(report["padding_sum"] for report in reports.values()) / count,
        max_padding_fraction=max(report["max_padding_fraction"] for report in reports.values()),
        verified_all_written_bytes=True, seconds=time.monotonic() - started,
        cache_manifest_sha256=file_hash(output / "cache.jsonl"),
        rejected_manifest_sha256=file_hash(output / "rejected.jsonl"),
        completed_utc=datetime.now(timezone.utc).isoformat(), settings=specification,
        bucket_plan=bucket_plan,
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resolution", type=int, choices=(512, 1024), default=1024)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--token-limit", type=int, default=1024)
    parser.add_argument("--tokenizer", default="google/t5gemma-2b-2b-ul2-it")
    parser.add_argument("--records-per-part", type=int, default=2000)
    parser.add_argument("--max-shard-mib", type=int, default=256)
    parser.add_argument(
        "--max-source-pixels", type=int, default=DEFAULT_MAX_SOURCE_PIXELS,
        help=f"reject sources above this pixel area (default: {DEFAULT_MAX_SOURCE_PIXELS})")
    parser.add_argument(
        "--max-source-side", type=int, default=DEFAULT_MAX_SOURCE_SIDE,
        help=f"reject sources with either side above this value (default: {DEFAULT_MAX_SOURCE_SIDE})")
    args = parser.parse_args()
    from configs.sft_512 import get_config as config512
    from configs.sft_1024_captioned import get_config as config1024
    config = (config512 if args.resolution == 512 else config1024)().input
    summary = precompute_captioned(
        args.source, args.output_dir, config, args.workers, args.token_limit,
        args.tokenizer, args.records_per_part, args.max_shard_mib * 1024**2,
        args.max_source_pixels, args.max_source_side,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
