#!/usr/bin/env python3
"""CPU caption/image audit, pixel cache and OCR mask cache."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import argparse
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import tempfile

HERE = Path(__file__).resolve().parent
TRAIN = HERE.parents[1] / "torch_train"
sys.path.insert(0, str(TRAIN))

import pyarrow.parquet as pq
from configs.sft_1024_captioned import get_config
from datasets.precompute_captioned import precompute_captioned, file_hash
from datasets.pixel_cache import fingerprint
from region_masks import MASK_VERSION, LATENT_FACTOR, make_mask

CACHE_FORMAT = "region-weighted-flow-cache-v2"


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def mask_part(pixel_part, output, settings):
    pixel_part, output = Path(pixel_part), Path(output)
    rows_path = pixel_part / "records.jsonl"
    rows = [json.loads(line) for line in rows_path.read_text().splitlines()]
    source_paths = sorted({row["source"]["parquet_path"] for row in rows})
    signature = dict(settings=settings, records_sha256=file_hash(rows_path), sources=[
        dict(path=p, size=Path(p).stat().st_size, mtime_ns=Path(p).stat().st_mtime_ns)
        for p in source_paths])
    destination = output / "regions" / pixel_part.name
    complete = destination / "complete.json"
    if complete.exists():
        report = json.loads(complete.read_text())
        if report["signature"] != signature:
            raise ValueError(f"OCR inputs/settings changed: {destination}; use a new output")
        for name, digest in report["sha256"].items():
            if file_hash(destination / name) != digest:
                raise ValueError(f"OCR cache checksum mismatch: {destination / name}")
        return report
    if destination.exists():
        raise ValueError(f"Incomplete committed OCR part: {destination}")
    destination.parent.mkdir(exist_ok=True)
    counters = Counter()
    sources = Counter()
    buckets = Counter()
    masked_images = 0
    payload_digest = hashlib.sha256()
    with tempfile.TemporaryDirectory(prefix=".regions-", dir=output) as temporary:
        stage = Path(temporary) / "part"
        stage.mkdir()
        last_group = None
        group_rows = None
        with (stage / "masks.bin").open("wb") as masks, (stage / "records.jsonl").open("w") as records:
            for row in rows:
                ref = row["source"]
                group_key = (ref["parquet_path"], ref["row_group"])
                if group_key != last_group:
                    parquet = pq.ParquetFile(group_key[0])
                    available = set(parquet.schema_arrow.names)
                    if "ocr_raw_output" not in available:
                        raise ValueError(f"Missing ocr_raw_output column: {group_key[0]}")
                    columns = [c for c in ["id", "ocr_raw_output", "source_width", "source_height"]
                               if c in available]
                    group_rows = parquet.read_row_group(group_key[1], columns=columns).to_pylist()
                    last_group = group_key
                raw = group_rows[ref["row_in_group"]]
                if str(raw["id"]) != row["id"]:
                    raise ValueError(f"Stale OCR source pointer: {row['id']}")
                raw_ocr = raw["ocr_raw_output"]
                dimension_mismatch = any(raw.get(key) is not None and raw[key] != row[axis]
                                         for key, axis in [("source_width", "width"), ("source_height", "height")])
                if dimension_mismatch:
                    raw_ocr = None
                    counters["ocr_dimension_mismatch_images"] += 1
                weights, stats = make_mask(raw_ocr, row["width"], row["height"],
                                           (row["cache_height"], row["cache_width"]),
                                           settings["min_confidence"], settings["min_height"],
                                           settings["reference_height"])
                counters.update(stats)
                nonempty = bool(weights.any())
                masked_images += nonempty
                sources[row["source_dataset"]] += 1
                buckets[f"{row['cache_height']}x{row['cache_width']}"] += 1
                offset = masks.tell()
                payload = weights.tobytes()
                masks.write(payload)
                payload_digest.update(payload)
                row["cache_path"] = str(Path("pixels") / row["cache_path"])
                row["region_mask"] = dict(path=str(Path("regions") / pixel_part.name / "masks.bin"),
                                           offset=offset, height=weights.shape[0], width=weights.shape[1],
                                           dtype="<f2", version=MASK_VERSION)
                row["region_stats"] = dict(stats, has_mask=nonempty,
                                            dimension_mismatch=dimension_mismatch)
                records.write(json.dumps(row, ensure_ascii=False) + "\n")
        if file_hash(stage / "masks.bin") != payload_digest.hexdigest():
            raise ValueError("Written OCR mask bytes did not match the computed weights")
        report = dict(signature=signature, count=len(rows), masked_images=masked_images,
                      counters=dict(counters), source_counts=dict(sources), bucket_counts=dict(buckets),
                      mask_bytes=(stage / "masks.bin").stat().st_size,
                      sha256={name: file_hash(stage / name) for name in ["masks.bin", "records.jsonl"]})
        atomic_json(stage / "complete.json", report)
        os.rename(stage, destination)
    return report


def build_regions(output, config, *, workers=8, min_confidence=0.8, min_height=4.0,
                  reference_height=4.0):
    output = Path(output).resolve()
    base = json.loads((output / "pixels/summary.json").read_text())
    if base["status"] != "complete" or not base["verified_all_written_bytes"]:
        raise ValueError("Pixel stage must complete first")
    settings = dict(format=CACHE_FORMAT, mask_version=MASK_VERSION, latent_factor=LATENT_FACTOR,
                    transform=fingerprint(config), min_confidence=min_confidence,
                    min_height=min_height, reference_height=reference_height,
                    region_weight="min(1, reference_height / transformed_region_height)",
                    overlap="maximum", stored_weight="unsquared; squared by objective",
                    caption_filter=base["settings"].get("caption_prefix_filter"),
                    token_limit=base["settings"]["token_limit"],
                    pixel_manifest_sha256=file_hash(output / "pixels/cache.jsonl"))
    saved_settings = output / "region_settings.json"
    if saved_settings.exists() and json.loads(saved_settings.read_text()) != settings:
        raise ValueError("Region settings changed; use a new output directory")
    atomic_json(saved_settings, settings)
    # A failed rebuild must not leave a stale final completion marker behind.
    (output / "summary.json").unlink(missing_ok=True)
    parts = sorted((output / "pixels/parts").glob("*/complete.json"))
    if len(parts) != base["parts"]:
        raise ValueError("Pixel part count mismatch")
    reports = {}
    if workers == 1:
        for part in parts:
            reports[part.parent.name] = mask_part(part.parent, output, settings)
    else:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            jobs = {pool.submit(mask_part, part.parent, output, settings): part.parent.name for part in parts}
            for future in as_completed(jobs):
                reports[jobs[future]] = future.result()
                print(f"OCR masks: {len(reports)}/{len(parts)} parts", flush=True)
    count = sum(r["count"] for r in reports.values())
    if count != base["count"]:
        raise ValueError("Pixel and mask sample counts differ")
    with (output / "cache.jsonl.partial").open("wb") as handle:
        seen = set()
        for name in sorted(reports):
            with (output / "regions" / name / "records.jsonl").open("rb") as source:
                for line in source:
                    uid = json.loads(line)["id"]
                    if uid in seen:
                        raise ValueError(f"Duplicate training ID: {uid}")
                    seen.add(uid)
                    handle.write(line)
    os.replace(output / "cache.jsonl.partial", output / "cache.jsonl")
    shutil.copyfile(output / "pixels/rejected.jsonl", output / "rejected.jsonl.partial")
    os.replace(output / "rejected.jsonl.partial", output / "rejected.jsonl")
    masked_images = sum(r["masked_images"] for r in reports.values())
    if not masked_images:
        raise ValueError("No usable text regions; inspect OCR fields/coordinates before training")
    counters = Counter()
    for r in reports.values():
        counters.update(r["counters"])
    summary = dict(format=CACHE_FORMAT, status="complete", count=count, masked_images=masked_images,
                   unmasked_images=count-masked_images, inspected=base["inspected"],
                   rejected=base["rejected"], rejected_counts=base["rejected_counts"],
                   accepted_source_counts=base["accepted_source_counts"], bucket_counts=base["bucket_counts"],
                   mask_stats=dict(counters), settings=settings,
                   mask_bytes=sum(r["mask_bytes"] for r in reports.values()),
                   verified_all_written_bytes=True, cache_manifest_sha256=file_hash(output / "cache.jsonl"),
                   mask_files=[dict(path=str(Path("regions") / name / "masks.bin"),
                                    bytes=r["mask_bytes"], sha256=r["sha256"]["masks.bin"])
                               for name, r in sorted(reports.items())])
    atomic_json(output / "summary.json", summary)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Completed captioned Parquet export")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--tokenizer", default="google/t5gemma-2b-2b-ul2-it")
    parser.add_argument("--token-limit", type=int, default=1024)
    parser.add_argument("--records-per-part", type=int, default=2000)
    parser.add_argument("--max-shard-mib", type=int, default=256)
    parser.add_argument("--min-ocr-confidence", type=float, default=0.8)
    parser.add_argument("--min-region-height", type=float, default=4.0,
                        help="Minimum box height in training pixels (not measured glyph height)")
    parser.add_argument("--reference-region-height", type=float, default=4.0,
                        help="h_ref in training pixels for min(1, h_ref / h_i)")
    args = parser.parse_args(argv)
    if (args.workers <= 0 or not 0 <= args.min_ocr_confidence <= 1
            or not 0 <= args.min_region_height < float("inf")
            or not 0 < args.reference_region_height < float("inf")):
        parser.error("Invalid workers or OCR thresholds")
    config = get_config().input
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    # Prevent two launchers from publishing interleaved completion artifacts.
    import fcntl
    with (output / ".precompute.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (output / "summary.json").unlink(missing_ok=True)
        precompute_captioned(args.source, output / "pixels", config, workers=args.workers,
                             token_limit=args.token_limit, tokenizer_name=args.tokenizer,
                             records_per_part=args.records_per_part,
                             max_shard_bytes=args.max_shard_mib * 1024**2)
        summary = build_regions(output, config, workers=args.workers,
                                min_confidence=args.min_ocr_confidence,
                                min_height=args.min_region_height,
                                reference_height=args.reference_region_height)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
