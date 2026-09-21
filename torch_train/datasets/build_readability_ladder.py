"""Build a deterministic manual-review readability ladder.

The ladder compares native source pixels, the exact accepted 1024-bucket cache
used for SFT, and a posterior-mode FLUX.2 VAE reconstruction.  It deliberately
does not run OCR: the output is an HTML gallery plus a blank annotation CSV.

The commands are split so CPU-only reconstruction can be sharded safely::

    python -m datasets.build_readability_ladder prepare ...
    python -m datasets.build_readability_ladder reconstruct ... --shard-index 0 --shard-count 16
    python -m datasets.build_readability_ladder finalize ...
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from functools import lru_cache
import hashlib
import html
from io import BytesIO
import json
import math
from pathlib import Path
import random
import re
import time

import numpy as np
from PIL import Image, ImageOps


DOMAINS = ("webpage", "slides", "poster", "chart", "scientific_figure")
DIMENSIONS = ("layout", "attribute", "knowledge", "text")
OUTPUT_PATTERN = re.compile(
    r"^(webpage|slides|poster|chart|scientific_figure)_(layout|attribute|knowledge|text)_(\d+)\.png$"
)


def read_jsonl(path):
    with Path(path).open() as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def aspect_group(height, width):
    ratio = width / height
    if ratio < 0.85:
        return "portrait"
    if ratio > 1.18:
        return "landscape"
    return "square"


def aspect_targets(count):
    base, remainder = divmod(count, 3)
    groups = ("portrait", "square", "landscape")
    return {group: base + (index < remainder) for index, group in enumerate(groups)}


def stable_seed(seed, *parts):
    payload = "|".join([str(seed), *map(str, parts)]).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def systematic_bucket_sample(records, count, seed, label):
    """Cover the available bucket shapes, then fill deterministically."""
    groups = defaultdict(list)
    for row in records:
        groups[(row["cache_height"], row["cache_width"])].append(row)
    rng = random.Random(stable_seed(seed, label))
    for rows in groups.values():
        rng.shuffle(rows)
    buckets = sorted(groups, key=lambda shape: (shape[1] / shape[0], shape))
    chosen = []
    cursor = 0
    while len(chosen) < count and buckets:
        bucket = buckets[cursor % len(buckets)]
        if groups[bucket]:
            chosen.append(groups[bucket].pop())
        buckets = [shape for shape in buckets if groups[shape]]
        cursor += 1
    return chosen


def select_training(cache_manifest, count_per_source, seed):
    by_source_aspect = defaultdict(list)
    sources = set()
    for row in read_jsonl(cache_manifest):
        source = row["source_dataset"]
        sources.add(source)
        group = aspect_group(row["cache_height"], row["cache_width"])
        by_source_aspect[(source, group)].append(row)
    selected = []
    targets = aspect_targets(count_per_source)
    for source in sorted(sources):
        remaining = count_per_source
        for group in ("portrait", "square", "landscape"):
            target = min(targets[group], remaining)
            candidates = by_source_aspect[(source, group)]
            take = min(target, len(candidates))
            selected.extend(systematic_bucket_sample(candidates, take, seed, f"{source}:{group}"))
            remaining -= take
        if remaining:
            already = {row["id"] for row in selected if row["source_dataset"] == source}
            pool = [row for group in ("portrait", "square", "landscape")
                    for row in by_source_aspect[(source, group)] if row["id"] not in already]
            selected.extend(systematic_bucket_sample(pool, remaining, seed, f"{source}:remainder"))
        actual = sum(row["source_dataset"] == source for row in selected)
        if actual != count_per_source:
            raise ValueError(f"Could only select {actual}/{count_per_source} rows for {source}")
    return selected


def select_bizgeneval(image_dir, prompt_manifest, per_domain, seed):
    prompts = {int(row["id"]): row for row in read_jsonl(prompt_manifest)}
    by_cell = defaultdict(list)
    for path in Path(image_dir).glob("*.png"):
        match = OUTPUT_PATTERN.match(path.name)
        if match:
            domain, dimension, identifier = match.groups()
            by_cell[(domain, dimension)].append((path, int(identifier)))
    if per_domain % len(DIMENSIONS):
        raise ValueError("BizGenEval examples per domain must be divisible by four dimensions.")
    per_cell = per_domain // len(DIMENSIONS)
    selected = []
    for domain in DOMAINS:
        for dimension in DIMENSIONS:
            candidates = sorted(by_cell[(domain, dimension)], key=lambda pair: pair[1])
            if len(candidates) < per_cell:
                raise ValueError(f"Not enough outputs for {domain}/{dimension}")
            rng = random.Random(stable_seed(seed, domain, dimension))
            for path, identifier in rng.sample(candidates, per_cell):
                selected.append((path, prompts[identifier]))
    return selected


@lru_cache(maxsize=16)
def parquet_file(path):
    import pyarrow.parquet as pq
    return pq.ParquetFile(path)


def source_image(row):
    source = row["source"]
    parquet = parquet_file(source["parquet_path"])
    table = parquet.read_row_group(source["row_group"], columns=["id", "image_bytes"])
    index = source["row_in_group"]
    actual_id = str(table["id"][index].as_py())
    if actual_id != source["identifier"]:
        raise ValueError(f"Stale source pointer for {row['id']}: got {actual_id}")
    image = Image.open(BytesIO(table["image_bytes"][index].as_py()))
    try:
        return ImageOps.exif_transpose(image).convert("RGB")
    finally:
        image.close()


def cached_image(row, cache_root):
    path = Path(row["cache_path"])
    if not path.is_absolute():
        path = cache_root / path
    pixels = np.memmap(path, mode="r", dtype=np.uint8)
    size = row["cache_height"] * row["cache_width"] * 3
    start = row["cache_offset"]
    array = np.asarray(pixels[start:start + size]).reshape(row["cache_height"], row["cache_width"], 3)
    return Image.fromarray(array).copy()


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False)


def prepare(args):
    output = Path(args.output_dir).resolve()
    items_dir = output / "items"
    output.mkdir(parents=True, exist_ok=True)
    cache_manifest = Path(args.cache_manifest).resolve()
    cache_root = cache_manifest.parent
    training = select_training(cache_manifest, args.training_per_source, args.seed)
    generated = select_bizgeneval(args.bizgeneval_images, args.bizgeneval_prompts,
                                  args.bizgeneval_per_domain, args.seed)
    manifest = []
    ordinal = 0
    for row in training:
        slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", row["id"])[-100:]
        relative_dir = Path("items") / f"{ordinal:03d}_training_{slug}"
        item_dir = output / relative_dir
        native = source_image(row)
        processed = cached_image(row, cache_root)
        save_png(native, item_dir / "source.png")
        save_png(processed, item_dir / "processed.png")
        manifest.append({
            "ordinal": ordinal,
            "kind": "training",
            "id": row["id"],
            "stratum": row["source_dataset"],
            "subgroup": aspect_group(row["cache_height"], row["cache_width"]),
            "source_dataset": row["source_dataset"],
            "source_size": [native.height, native.width],
            "processed_size": [processed.height, processed.width],
            "caption_tokens": row.get("caption_tokens"),
            "caption": row["caption"],
            "source_reference": row["source"],
            "relative_dir": str(relative_dir),
        })
        ordinal += 1
    for path, prompt in generated:
        relative_dir = Path("items") / f"{ordinal:03d}_bizgeneval_{path.stem}"
        item_dir = output / relative_dir
        with Image.open(path) as image:
            native = image.convert("RGB")
        expected = (prompt["_i1_width"], prompt["_i1_height"])
        if native.size != expected:
            raise ValueError(f"Unexpected output geometry for {path}: {native.size} != {expected}")
        save_png(native, item_dir / "source.png")
        save_png(native, item_dir / "processed.png")
        manifest.append({
            "ordinal": ordinal,
            "kind": "bizgeneval",
            "id": path.stem,
            "stratum": prompt["domain"],
            "subgroup": prompt["dimension"],
            "domain": prompt["domain"],
            "dimension": prompt["dimension"],
            "source_size": [native.height, native.width],
            "processed_size": [native.height, native.width],
            "caption_tokens": prompt.get("_i1_text_tokens"),
            "caption": prompt["prompt"],
            "source_reference": str(path.resolve()),
            "relative_dir": str(relative_dir),
        })
        ordinal += 1
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w") as handle:
        for row in manifest:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    settings = {
        "format": "manual-readability-ladder-v1",
        "seed": args.seed,
        "posterior": "mode",
        "ocr": False,
        "training_cache_manifest": str(cache_manifest),
        "training_per_source": args.training_per_source,
        "training_count": len(training),
        "bizgeneval_setting": "04_captioned_sft_6245_truncate_1024",
        "bizgeneval_per_domain": args.bizgeneval_per_domain,
        "bizgeneval_count": len(generated),
        "total": len(manifest),
        "training_aspect_targets_per_source": aspect_targets(args.training_per_source),
        "note": "Generated outputs already have exact supported bucket geometry, so source and processed pixels are identical.",
    }
    (output / "settings.json").write_text(json.dumps(settings, indent=2) + "\n")
    print(json.dumps(settings, indent=2))


def reconstruct(args):
    import torch
    from configs.sft_1024_captioned import get_config
    from vae.vae import (encode_images_to_latents, load_vae, reverse_scale_latents,
                         scale_latents)

    output = Path(args.output_dir).resolve()
    rows = list(read_jsonl(output / "manifest.jsonl"))
    assigned = [row for row in rows if row["ordinal"] % args.shard_count == args.shard_index]
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    config = get_config()
    # oneDNN's channels-last convolution path is materially faster for this VAE
    # on the CPU preparation host and is numerically the same float32 model.
    vae = load_vae(config, torch.device("cpu"), dtype=torch.float32).to(
        memory_format=torch.channels_last)
    report = []
    with torch.inference_mode():
        for row in assigned:
            item_dir = output / row["relative_dir"]
            target = item_dir / "reconstructed.png"
            if target.exists() and not args.overwrite:
                report.append({"ordinal": row["ordinal"], "status": "existing"})
                continue
            with Image.open(item_dir / "processed.png") as image:
                array = np.asarray(image.convert("RGB"), dtype=np.float32) / 127.5 - 1.0
            pixels = torch.from_numpy(array).unsqueeze(0)
            started = time.monotonic()
            latent = encode_images_to_latents(vae, pixels, sample=False)
            packed = scale_latents(latent, config)
            restored = reverse_scale_latents(packed, config.vae_type)
            decoded = vae.decode(restored).sample[0].permute(1, 2, 0)
            reconstructed = ((decoded + 1) / 2).clamp(0, 1).cpu().numpy()
            save_png(Image.fromarray((reconstructed * 255).round().astype(np.uint8)), target)
            report.append({"ordinal": row["ordinal"], "status": "written",
                           "seconds": time.monotonic() - started})
            print(json.dumps(report[-1]), flush=True)
    path = output / f"reconstruct_shard_{args.shard_index:02d}_of_{args.shard_count:02d}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")


def thumbnail(source, target, size=(480, 480)):
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail(size, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", size, "white")
        canvas.paste(image, ((size[0] - image.width) // 2, (size[1] - image.height) // 2))
        canvas.save(target, "JPEG", quality=88, subsampling=0)


def finalize(args):
    output = Path(args.output_dir).resolve()
    review_path = output / "review.csv"
    if review_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite manual annotations: {review_path}. "
            "Move the existing CSV aside before finalizing again."
        )
    rows = list(read_jsonl(output / "manifest.jsonl"))
    missing = []
    enriched = []
    for row in rows:
        item_dir = output / row["relative_dir"]
        reconstruction = item_dir / "reconstructed.png"
        if not reconstruction.exists():
            missing.append(row["ordinal"])
            continue
        thumbnail(item_dir / "source.png", item_dir / "source_thumb.jpg")
        thumbnail(item_dir / "processed.png", item_dir / "processed_thumb.jpg")
        thumbnail(reconstruction, item_dir / "reconstructed_thumb.jpg")
        with Image.open(item_dir / "processed.png") as left, Image.open(reconstruction) as right:
            a = np.asarray(left.convert("RGB"), dtype=np.float32) / 255
            b = np.asarray(right.convert("RGB"), dtype=np.float32) / 255
        mse = float(np.square(a - b).mean())
        row = dict(row, reconstruction_mse=mse,
                   reconstruction_psnr_db=float(-10 * math.log10(max(mse, 1e-12))))
        enriched.append(row)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} reconstructions: {missing}")
    with review_path.open("x", newline="") as handle:
        fields = ["ordinal", "kind", "id", "stratum", "subgroup", "source_size",
                  "processed_size", "caption_tokens", "source_readability", "processed_readability",
                  "reconstruction_readability", "first_failure_stage", "notes"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in enriched:
            writer.writerow({key: row.get(key, "") for key in fields})
    cards = []
    for row in enriched:
        rel = row["relative_dir"]
        title = f"{row['ordinal']:03d} · {row['kind']} · {row['id']}"
        meta = (f"{row['stratum']} / {row['subgroup']} · source {row['source_size'][1]}×{row['source_size'][0]} "
                f"→ bucket {row['processed_size'][1]}×{row['processed_size'][0]} · "
                f"VAE PSNR {row['reconstruction_psnr_db']:.2f} dB")
        images = []
        for name, label in (("source", "1. Source/generated pixels"),
                            ("processed", "2. Bucketed training pixels"),
                            ("reconstructed", "3. VAE reconstruction")):
            images.append(
                f'<figure><a href="{rel}/{name}.png"><img loading="lazy" src="{rel}/{name}_thumb.jpg"></a>'
                f'<figcaption>{label}</figcaption></figure>')
        cards.append(f'<article data-kind="{row["kind"]}" data-stratum="{html.escape(row["stratum"])}">'
                     f'<h2>{html.escape(title)}</h2><p>{html.escape(meta)}</p>' + "".join(images) + "</article>")
    css = """
body{font:15px system-ui,sans-serif;margin:20px;background:#f4f5f7;color:#18202a}header{position:sticky;top:0;background:#fff;padding:12px;z-index:2;border:1px solid #ddd}article{background:#fff;margin:18px 0;padding:14px;border:1px solid #ccd3da;border-radius:8px}h2{font-size:17px;margin:0 0 4px}p{margin:4px 0 12px;color:#52606d}figure{display:inline-block;width:32.7%;margin:0;text-align:center;vertical-align:top}img{width:96%;height:auto;border:1px solid #aaa}figcaption{font-weight:600;margin-top:5px}@media(max-width:900px){figure{width:100%;margin-bottom:12px}}
"""
    script = """
function applyFilter(){const k=document.getElementById('kind').value,s=document.getElementById('stratum').value;document.querySelectorAll('article').forEach(x=>x.hidden=(k&&x.dataset.kind!==k)||(s&&!x.dataset.stratum.includes(s)));}
"""
    options = "".join(f'<option value="{x}">{x}</option>' for x in
                      sorted({row["stratum"] for row in enriched}))
    page = (f'<!doctype html><meta charset="utf-8"><title>Readability ladder manual review</title>'
            f'<style>{css}</style><header><b>Readability ladder: {len(enriched)} examples</b> · '
            f'<a href="review.csv">annotation CSV</a> · click any image for native pixels<br>'
            f'<label>Kind <select id="kind" onchange="applyFilter()"><option value="">all</option><option>training</option><option>bizgeneval</option></select></label> '
            f'<label>Stratum <select id="stratum" onchange="applyFilter()"><option value="">all</option>{options}</select></label></header>'
            + "".join(cards) + f'<script>{script}</script>')
    (output / "index.html").write_text(page)
    summary = {
        "status": "complete",
        "total": len(enriched),
        "training": sum(row["kind"] == "training" for row in enriched),
        "bizgeneval": sum(row["kind"] == "bizgeneval" for row in enriched),
        "mean_vae_psnr_db": float(np.mean([row["reconstruction_psnr_db"] for row in enriched])),
        "note": "PSNR is descriptive only and is not a readability metric; use review.csv for manual annotation.",
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    readme = f"""# Manual readability ladder

This package contains **{len(enriched)} deterministic examples**: {summary['training']} accepted examples
from the recaptioned 200K SFT dataset and {summary['bizgeneval']} outputs from setting 04.

Open `index.html`, zoom by clicking any thumbnail, and record judgments in `review.csv`.
The three columns are native source/generated pixels, the exact production 1024-bucket
training pixels, and the deterministic posterior-mode FLUX.2 VAE reconstruction.
No OCR was run. Generated outputs already use exact supported bucket sizes, so their
first two columns are identical by design. PSNR is included only as a file-integrity
and gross-distortion aid; it is not a substitute for judging text readability.

Suggested values for each readability column: `clean`, `minor_damage`, `unreadable`.
Suggested `first_failure_stage`: `source`, `resize`, `vae`, or `none`.
"""
    (output / "README.md").write_text(readme)
    print(json.dumps(summary, indent=2))


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    sub = root.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--cache-manifest", required=True)
    prep.add_argument("--bizgeneval-images", required=True)
    prep.add_argument("--bizgeneval-prompts", required=True)
    prep.add_argument("--output-dir", required=True)
    prep.add_argument("--training-per-source", type=int, default=30)
    prep.add_argument("--bizgeneval-per-domain", type=int, default=8)
    prep.add_argument("--seed", type=int, default=20260921)
    prep.set_defaults(func=prepare)
    recon = sub.add_parser("reconstruct")
    recon.add_argument("--output-dir", required=True)
    recon.add_argument("--shard-index", type=int, required=True)
    recon.add_argument("--shard-count", type=int, required=True)
    recon.add_argument("--threads", type=int, default=8)
    recon.add_argument("--overwrite", action="store_true")
    recon.set_defaults(func=reconstruct)
    final = sub.add_parser("finalize")
    final.add_argument("--output-dir", required=True)
    final.set_defaults(func=finalize)
    return root


def main():
    args = parser().parse_args()
    if args.command == "reconstruct" and not 0 <= args.shard_index < args.shard_count:
        raise ValueError("shard-index must be in [0, shard-count).")
    args.func(args)


if __name__ == "__main__":
    main()
