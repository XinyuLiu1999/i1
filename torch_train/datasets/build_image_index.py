"""Fully decode Parquet images and write a corrected, deterministic JSONL index.

Only metadata is written. Image bytes remain in the original Parquet row groups.
Unreadable images and empty captions are excluded; incorrect sizes are corrected.
Shard/read failures abort publication, so an incomplete index cannot replace one.
"""

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
from pathlib import Path
import shutil
import tempfile

from .data_sources import parquet_files
from .validate_images import _validate_shard


def build_index(source, output, workers=16):
    files = parquet_files(source)
    if not files:
        raise FileNotFoundError(f"No Parquet shards at {source}")
    if workers <= 0:
        raise ValueError("workers must be positive.")
    output = Path(output).expanduser().resolve()
    if output.suffix != ".jsonl":
        raise ValueError("The corrected index must have a .jsonl suffix.")
    output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    sizes = Counter()
    issues = []
    with tempfile.TemporaryDirectory(prefix=".sft-index-", dir=output.parent) as temporary:
        parts = [Path(temporary) / f"{i:06d}.jsonl" for i in range(len(files))]
        results = {}
        with ProcessPoolExecutor(max_workers=min(workers, len(files))) as executor:
            futures = {executor.submit(_validate_shard, path, 20, part): i
                       for i, (path, part) in enumerate(zip(files, parts))}
            for future in as_completed(futures):
                i = futures[future]
                result = future.result()
                if result["row_group_error_count"] or not parts[i].exists():
                    raise RuntimeError(f"Cannot publish an incomplete index: {files[i]}: {result['errors'][:3]}")
                results[i] = result
                print(f"indexed {len(results)}/{len(files)} shards; "
                      f"{result['indexed']} retained from {files[i].name}", flush=True)
        # Sorted shard order and original row order are independent of worker completion.
        for i in range(len(files)):
            result = results[i]
            for key in ("checked", "decoded", "indexed", "decode_error_count",
                        "dimension_mismatch_count", "metadata_error_count", "invalid_caption_count"):
                counts[key] += result[key]
            sizes.update(result["actual_sizes"])
            issues.extend(result["decode_errors"])
            issues.extend(result["caption_errors"])
        if not counts["indexed"]:
            raise ValueError("No usable records; existing index was not replaced.")
        merged = Path(temporary) / "complete.jsonl"
        with merged.open("wb") as destination:
            for part in parts:
                with part.open("rb") as handle:
                    shutil.copyfileobj(handle, destination)
        os.replace(merged, output)
    report = dict(source=str(Path(source).expanduser().resolve()), index=str(output),
                  shard_count=len(files), **counts,
                  actual_size_counts=dict(sorted(sizes.items())),
                  excluded_examples=issues)
    output.with_suffix(".report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Parquet shard or shard directory.")
    parser.add_argument("--output", required=True, help="Corrected .jsonl index destination.")
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    report = build_index(args.manifest, args.output, args.workers)
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("actual_size_counts", "excluded_examples")}, indent=2))


if __name__ == "__main__":
    main()
