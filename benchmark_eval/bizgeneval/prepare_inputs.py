"""Prepare BizGenEval prompts and native i1 generation sizes.

The output JSONL remains a valid BizGenEval dataset. Two private fields are
added for torch_inference/generate.py, and output_names.txt follows the exact
filename convention used by BizGenEval's evaluator.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path


HEIGHT_KEY = "_i1_height"
WIDTH_KEY = "_i1_width"


def generate_buckets(
    resolution: int = 1024,
    step: int = 64,
    max_ratio: float = 3.0,
) -> list[tuple[int, int]]:
    """Reproduce the 1024 SFT configuration's (height, width) buckets."""
    budget = (resolution // step) ** 2
    shapes: set[tuple[int, int]] = set()
    for height_units in range(1, budget + 1):
        width_units = budget // height_units
        if max(width_units, height_units) / min(width_units, height_units) <= max_ratio:
            shapes.add((height_units * step, width_units * step))
            shapes.add((width_units * step, height_units * step))
    # The SFT configuration includes exact 3:2 and 2:3 anchors.
    shapes.update(((832, 1248), (1248, 832)))
    return sorted(shapes, key=lambda shape: (shape[1] / shape[0], shape))


def target_aspect_ratio(item: dict) -> float:
    declared = str(item.get("aspect_ratio") or "").strip()
    match = re.fullmatch(r"(\d+(?:\.\d+)?):(\d+(?:\.\d+)?)", declared)
    if match:
        width, height = map(float, match.groups())
        if width > 0 and height > 0:
            return width / height

    reference = str(item.get("reference_image_wh") or "").strip()
    match = re.fullmatch(r"(\d+)[xX*×](\d+)", reference)
    if match:
        width, height = map(int, match.groups())
        if width > 0 and height > 0:
            return width / height
    return 1.0


def select_bucket(item: dict, buckets: list[tuple[int, int]]) -> tuple[int, int]:
    target = target_aspect_ratio(item)
    return min(
        buckets,
        key=lambda shape: (
            abs(math.log((shape[1] / shape[0]) / target)),
            -(shape[0] * shape[1]),
        ),
    )


def image_name(item: dict) -> str:
    for key in ("reference_image", "reference image", "image_path", "image", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).name
    return f"{item.get('domain', '')}_{item.get('dimension', '')}_{item.get('id')}.png"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Use only the first N rows; 0 (the default) uses the full benchmark.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.limit < 0:
        raise ValueError("--limit must be non-negative")

    with args.input.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if args.limit:
        rows = rows[: args.limit]
    if not rows:
        raise ValueError(f"No benchmark rows found in {args.input}")

    buckets = generate_buckets()
    names: list[str] = []
    prepared: list[dict] = []
    for line_number, item in enumerate(rows, 1):
        if not isinstance(item.get("prompt"), str) or not item["prompt"].strip():
            raise ValueError(f"Input row {line_number} has no non-empty prompt")
        name = image_name(item)
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError(f"Unsafe or non-PNG output name on row {line_number}: {name!r}")
        height, width = select_bucket(item, buckets)
        output_item = dict(item)
        output_item[HEIGHT_KEY] = height
        output_item[WIDTH_KEY] = width
        prepared.append(output_item)
        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("BizGenEval output filenames are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "bizgeneval_i1.jsonl"
    names_path = args.output_dir / "output_names.txt"
    with dataset_path.open("w", encoding="utf-8") as handle:
        for item in prepared:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")

    shape_counts: dict[tuple[int, int], int] = {}
    for item in prepared:
        shape = (item[HEIGHT_KEY], item[WIDTH_KEY])
        shape_counts[shape] = shape_counts.get(shape, 0) + 1
    print(f"Prepared {len(prepared)} prompts in {dataset_path}")
    print(f"Output names: {names_path}")
    print("Shapes: " + ", ".join(f"{h}x{w}={count}" for (h, w), count in sorted(shape_counts.items())))


if __name__ == "__main__":
    main()
