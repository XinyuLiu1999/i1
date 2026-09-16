"""Build a CPU-only gallery using the exact bucketed SFT image transform."""

import argparse
import html
import json
from pathlib import Path
import re

import numpy as np

from configs.sft_512 import get_config
from configs.sft_1024 import get_config as get_config_1024
from datasets.bucketed import BucketedImages
from datasets.data_sources import open_record_image
from datasets.image_geometry import select_bucket


def _allocate(total, capacities):
    """Spread samples evenly, redistributing quotas from sparse buckets."""
    if total > sum(capacities):
        raise ValueError(f"Requested {total} samples but only {sum(capacities)} are eligible.")
    counts = [0] * len(capacities)
    while sum(counts) < total:
        for index, capacity in enumerate(capacities):
            if counts[index] < capacity and sum(counts) < total:
                counts[index] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True,
                        help="JSONL manifest, Parquet shard, or GPT-Image output directory.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--count", type=int, default=150)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--tokenizer", default="google/t5gemma-2b-2b-ul2-it")
    parser.add_argument("--skip_tokenizer", action="store_true")
    parser.add_argument("--require_exact_source_aspect", action="store_true",
                        help="Select only sources whose decoded pixels exactly match a configured bucket ratio.")
    args = parser.parse_args()
    if not 100 <= args.count <= 200:
        raise ValueError("--count must be between 100 and 200.")

    config = get_config().input
    config_1024 = get_config_1024().input
    config.manifest = args.manifest
    dataset = BucketedImages(config)
    groups = dataset.groups
    if args.require_exact_source_aspect:
        groups = [[index for index in group
                   if dataset.records[index][2] * dataset.buckets[bucket_index][0]
                   == dataset.records[index][1] * dataset.buckets[bucket_index][1]]
                  for bucket_index, group in enumerate(groups)]
    populated = [(bucket_index, group) for bucket_index, group in enumerate(groups) if group]
    allocations = _allocate(args.count, [len(group) for _, group in populated])
    rng = np.random.default_rng(args.seed)
    selected = []
    skipped_nonexact = 0
    skipped_decode = 0
    if args.require_exact_source_aspect:
        skipped_nonexact = len(dataset) - sum(len(group) for group in groups)
    for (bucket_index, group), count in zip(populated, allocations):
        selected.extend((int(index), bucket_index) for index in rng.choice(group, count, replace=False))
    selected.sort(key=lambda item: (item[1], item[0]))

    tokenizer = None
    if not args.skip_tokenizer:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    output = Path(args.output_dir).expanduser().resolve()
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    for ordinal, (index, bucket_index) in enumerate(selected, 1):
        record, declared_height, declared_width, _ = dataset.records[index]
        source = open_record_image(record)
        width, height = source.size
        if (width, height) != (declared_width, declared_height):
            raise ValueError(f"Stale dimensions for {record.identifier}; build a corrected index first.")
        bucket_512 = dataset.buckets[bucket_index]
        index_1024, exclusion_1024 = select_bucket(height, width, config_1024)
        bucket_1024 = config_1024.buckets[index_1024] if index_1024 is not None else None
        processed_512 = dataset.process_image(source, bucket_512)
        # Both supplied configs use the same resize/pad transform; eligibility
        # and bucket selection must still be evaluated at each resolution.
        processed_1024 = dataset.process_image(source, bucket_1024) if bucket_1024 else None
        safe_identifier = re.sub(r"[^A-Za-z0-9_.-]+", "_", record.identifier).strip("._") or "sample"
        stem = f"{ordinal:03d}_{safe_identifier}"
        paths = {
            "original": images_dir / f"{stem}_original.png",
            "processed_512": images_dir / f"{stem}_512.png",
            "processed_1024": images_dir / f"{stem}_1024.png",
        }
        # Low PNG compression keeps the diagnostic lossless without making this
        # CPU-only export spend most of its time searching for smaller deflate output.
        source.save(paths["original"], compress_level=1)
        processed_512.save(paths["processed_512"], compress_level=1)
        if processed_1024 is not None:
            processed_1024.save(paths["processed_1024"], compress_level=1)
        token_count = None
        decoded = None
        if tokenizer is not None:
            ids = tokenizer(record.caption, padding=False, truncation=False,
                            add_special_tokens=True)["input_ids"]
            token_count = len(ids)
            decoded = tokenizer.decode(ids, skip_special_tokens=True)
        entries.append(dict(
            ordinal=ordinal, id=record.identifier, caption=record.caption,
            decoded=decoded, token_count=token_count, source_size=[width, height],
            declared_source_size=[declared_width, declared_height],
            dimension_mismatch=[width, height] != [declared_width, declared_height],
            bucket_512=list(bucket_512), bucket_1024=list(bucket_1024) if bucket_1024 else None,
            exclusion_1024=exclusion_1024,
            original=str(paths["original"].relative_to(output)),
            processed_512=str(paths["processed_512"].relative_to(output)),
            processed_1024=(str(paths["processed_1024"].relative_to(output))
                            if processed_1024 is not None else None),
        ))
        source.close()
        processed_512.close()
        if processed_1024 is not None:
            processed_1024.close()
        if ordinal % 10 == 0 or ordinal == len(selected):
            print(f"wrote {ordinal}/{len(selected)} examples", flush=True)

    metadata = output / "samples.jsonl"
    metadata.write_text("".join(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries))
    metadata_bucket_counts = {
        f"{dataset.buckets[index][0]}x{dataset.buckets[index][1]}": len(group)
        for index, group in populated
    }
    sampled_bucket_counts = {}
    for entry in entries:
        key = f'{entry["bucket_512"][0]}x{entry["bucket_512"][1]}'
        sampled_bucket_counts[key] = sampled_bucket_counts.get(key, 0) + 1
    cards = []
    for entry in entries:
        caption = html.escape(entry["caption"])
        token_text = "not computed" if entry["token_count"] is None else str(entry["token_count"])
        declared = entry["declared_source_size"]
        mismatch = (f' <strong class="warning">Declared {declared[0]}×{declared[1]} does not match pixels.</strong>'
                    if entry["dimension_mismatch"] else "")
        previews = [("original", f'original {entry["source_size"][0]}×{entry["source_size"][1]}'),
                    ("processed_512", f'512 config {entry["bucket_512"][0]}×{entry["bucket_512"][1]}')]
        if entry["bucket_1024"] is not None:
            previews.append(("processed_1024", f'1024 config {entry["bucket_1024"][0]}×{entry["bucket_1024"][1]}'))
        pictures = "".join(
            f'<figure><a href="{html.escape(entry[key])}"><img loading="lazy" src="{html.escape(entry[key])}"></a>'
            f'<figcaption>{label}</figcaption></figure>'
            for key, label in previews
        )
        if entry["exclusion_1024"] is not None:
            pictures += ('<figure><p>Excluded from 1024 training: '
                         f'{html.escape(entry["exclusion_1024"])}</p></figure>')
        cards.append(
            f'<article><h2>{entry["ordinal"]:03d} · {html.escape(entry["id"])}</h2>'
            f'<p><b>Actual T5Gemma tokens:</b> {token_text}.{mismatch}</p><div class="images">{pictures}</div>'
            f'<details><summary>Full caption</summary><pre>{caption}</pre></details></article>'
        )
    page = f"""<!doctype html><meta charset="utf-8"><title>SFT preprocessing inspection</title>
<style>
body {{ font: 15px/1.45 system-ui, sans-serif; margin: 2rem; background: #eee; color: #111 }}
header, article {{ max-width: 1500px; margin: 0 auto 2rem; background: white; padding: 1.25rem }}
.images {{ display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 1rem }}
figure {{ margin: 0 }} img {{ width: 100%; height: auto; border: 1px solid #bbb }}
figcaption {{ font-weight: 600 }} pre {{ white-space: pre-wrap; overflow-wrap: anywhere }}
.warning {{ color: #a00 }}
@media (max-width: 900px) {{ .images {{ grid-template-columns: 1fr }} }}
</style>
<header><h1>SFT preprocessing inspection</h1>
<p>{len(entries)} examples sampled evenly across every populated 512 bucket. Ineligible 1024 examples are marked as excluded. Click an image for native-pixel inspection.</p>
<p><b>Metadata-derived dataset bucket counts:</b> {html.escape(json.dumps(metadata_bucket_counts, sort_keys=True))}</p>
<p><b>Actual-dimension bucket counts in this sample:</b> {html.escape(json.dumps(sampled_bucket_counts, sort_keys=True))}</p></header>
{''.join(cards)}
"""
    (output / "index.html").write_text(page)
    summary = dict(count=len(entries), source=str(Path(args.manifest).expanduser().resolve()),
                   output=str(output), metadata_bucket_counts=metadata_bucket_counts,
                   sampled_actual_bucket_counts=sampled_bucket_counts,
                   sampled_dimension_mismatches=sum(entry["dimension_mismatch"] for entry in entries),
                   sampled_excluded_1024=sum(entry["exclusion_1024"] is not None for entry in entries),
                   require_exact_source_aspect=args.require_exact_source_aspect,
                   selection_skipped_nonexact=skipped_nonexact,
                   selection_skipped_decode=skipped_decode,
                   tokenizer=None if tokenizer is None else args.tokenizer)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
