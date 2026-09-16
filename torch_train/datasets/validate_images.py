"""Fully decode GPT-Image Parquet images and verify their declared dimensions."""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from io import BytesIO
import json
import os
from pathlib import Path
import time

from PIL import Image, ImageOps

from .data_sources import _parse_size, parquet_files


def _validate_shard(path, max_reported_errors, index_path=None):
    import pyarrow.parquet as pq

    path = str(path)
    parquet = pq.ParquetFile(path)
    required = {"id", "size", "image_bytes"}
    if index_path is not None:
        required.add("prompt")
    missing = required.difference(parquet.schema.names)
    if missing:
        return dict(path=path, checked=0, decoded=0, decode_error_count=0,
                    dimension_mismatch_count=0, metadata_error_count=1,
                    row_group_error_count=0, actual_sizes={}, mismatches={}, decode_errors=[], errors=[
            dict(path=path, kind="metadata", error=f"missing required columns: {sorted(missing)}")
        ])
    checked = 0
    decoded = 0
    decode_error_count = 0
    dimension_mismatch_count = 0
    metadata_error_count = 0
    row_group_error_count = 0
    actual_sizes = Counter()
    mismatches = Counter()
    errors = []
    decode_errors = []
    caption_errors = []
    index_records = []
    invalid_caption_count = 0
    for row_group in range(parquet.num_row_groups):
        try:
            batch = parquet.read_row_group(row_group, columns=sorted(required)).to_pydict()
        except Exception as error:
            rows = parquet.metadata.row_group(row_group).num_rows
            checked += rows
            row_group_error_count += rows
            if len(errors) < max_reported_errors:
                errors.append(dict(path=path, kind="row_group_read", row_group=row_group,
                                   error=f"row-group read failed: {type(error).__name__}: {error}"))
            continue
        for row_in_group, (identifier, declared, image_bytes) in enumerate(
                zip(batch["id"], batch["size"], batch["image_bytes"])):
            checked += 1
            try:
                expected = _parse_size(declared, f"{path}:row_group {row_group}:{row_in_group}")
            except Exception as error:
                expected = None
                metadata_error_count += 1
                if len(errors) < max_reported_errors:
                    errors.append(dict(path=path, kind="metadata", row_group=row_group,
                                       row_in_group=row_in_group, id=str(identifier),
                                       declared_size=declared,
                                       error=f"{type(error).__name__}: {error}"))
            try:
                with Image.open(BytesIO(image_bytes)) as source:
                    image = ImageOps.exif_transpose(source)
                    image.load()  # Force complete decompression, including trailing chunks/checksums.
                    actual = image.size
                decoded += 1
                actual_label = f"{actual[0]}x{actual[1]}"
                actual_sizes[actual_label] += 1
            except Exception as error:
                decode_error_count += 1
                decode_errors.append(dict(path=path, kind="decode", row_group=row_group,
                                          row_in_group=row_in_group,
                                          id=str(identifier), declared_size=declared,
                                          error=f"{type(error).__name__}: {error}"))
                continue
            if index_path is not None:
                caption = batch["prompt"][row_in_group]
                if not isinstance(caption, str) or not caption.strip():
                    invalid_caption_count += 1
                    caption_error = dict(path=path, kind="caption", row_group=row_group,
                                         row_in_group=row_in_group, id=str(identifier))
                    caption_errors.append(caption_error)
                    if len(errors) < max_reported_errors:
                        errors.append(caption_error)
                else:
                    index_records.append(dict(
                        id=str(identifier), caption=caption, width=actual[0], height=actual[1],
                        parquet_path=path, row_group=row_group, row_in_group=row_in_group,
                    ))
            if expected is not None and actual != expected:
                dimension_mismatch_count += 1
                mismatch = f"{expected[0]}x{expected[1]} -> {actual_label}"
                mismatches[mismatch] += 1
                if len(errors) < max_reported_errors:
                    errors.append(dict(path=path, kind="dimension_mismatch", row_group=row_group,
                                       row_in_group=row_in_group, id=str(identifier),
                                       declared_size=declared, decoded_size=actual_label))
    if index_path is not None:
        with Path(index_path).open("w") as handle:
            for record in index_records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return dict(path=path, checked=checked, decoded=decoded,
                indexed=len(index_records), invalid_caption_count=invalid_caption_count,
                caption_errors=caption_errors,
                decode_error_count=decode_error_count,
                dimension_mismatch_count=dimension_mismatch_count,
                metadata_error_count=metadata_error_count,
                row_group_error_count=row_group_error_count,
                actual_sizes=dict(actual_sizes), mismatches=dict(mismatches),
                decode_errors=decode_errors, errors=errors)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="GPT-Image output directory or one Parquet shard.")
    parser.add_argument("--report", required=True)
    parser.add_argument("--workers", type=int, default=min(16, os.cpu_count() or 1))
    parser.add_argument("--max_reported_errors", type=int, default=1000)
    args = parser.parse_args()
    files = parquet_files(args.manifest)
    if not files:
        raise FileNotFoundError(f"No GPT-Image Parquet shards found at {args.manifest}")
    if args.workers <= 0 or args.max_reported_errors < 0:
        raise ValueError("--workers must be positive and --max_reported_errors nonnegative.")

    started = time.time()
    checked = 0
    decoded = 0
    decode_error_count = 0
    dimension_mismatch_count = 0
    metadata_error_count = 0
    row_group_error_count = 0
    worker_error_count = 0
    actual_sizes = Counter()
    mismatches = Counter()
    errors = []
    decode_errors = []
    completed = 0
    per_shard_limit = args.max_reported_errors
    with ProcessPoolExecutor(max_workers=min(args.workers, len(files))) as executor:
        futures = {executor.submit(_validate_shard, path, per_shard_limit): path for path in files}
        for future in as_completed(futures):
            path = futures[future]
            try:
                result = future.result()
            except Exception as error:
                worker_error_count += 1
                result = dict(path=str(path), checked=0, decoded=0, decode_error_count=0,
                              dimension_mismatch_count=0, metadata_error_count=0,
                              row_group_error_count=0, actual_sizes={}, mismatches={}, decode_errors=[], errors=[
                    dict(path=str(path), kind="worker", error=f"worker failed: {type(error).__name__}: {error}")
                ])
            checked += result["checked"]
            decoded += result["decoded"]
            decode_error_count += result["decode_error_count"]
            dimension_mismatch_count += result["dimension_mismatch_count"]
            metadata_error_count += result["metadata_error_count"]
            row_group_error_count += result["row_group_error_count"]
            actual_sizes.update(result["actual_sizes"])
            mismatches.update(result["mismatches"])
            decode_errors.extend(result["decode_errors"])
            room = max(0, args.max_reported_errors - len(errors))
            errors.extend(result["errors"][:room])
            completed += 1
            print(f"validated {completed}/{len(files)} shards; {checked} checked; {decoded} decoded; "
                  f"{decode_error_count} decode errors; {dimension_mismatch_count} size mismatches",
                  flush=True)

    issue_count = (decode_error_count + dimension_mismatch_count + metadata_error_count
                   + row_group_error_count + worker_error_count)
    report = dict(
        source=str(Path(args.manifest).expanduser().resolve()), shard_count=len(files),
        checked=checked, decoded=decoded, issue_count=issue_count,
        decode_error_count=decode_error_count,
        dimension_mismatch_count=dimension_mismatch_count,
        metadata_error_count=metadata_error_count,
        row_group_error_count=row_group_error_count,
        worker_error_count=worker_error_count,
        actual_size_counts=dict(sorted(actual_sizes.items())),
        declared_to_actual_mismatches=dict(sorted(mismatches.items())),
        workers=args.workers, elapsed_seconds=round(time.time() - started, 3),
        decode_errors=decode_errors,
        other_errors_truncated=(issue_count - decode_error_count) > len(errors),
        other_errors=errors,
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False)
    destination = Path(args.report).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(rendered + "\n")
    print(rendered)
    raise SystemExit(int(issue_count != 0))


if __name__ == "__main__":
    main()
