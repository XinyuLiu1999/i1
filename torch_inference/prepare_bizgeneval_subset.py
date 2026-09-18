from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


DOMAIN_ORDER = ("webpage", "slides", "poster", "chart", "scientific_figure")
DIMENSION_ORDER = ("layout", "text", "attribute", "knowledge")


def stratified_rows(rows: list[dict], limit: int) -> list[dict]:
    buckets = defaultdict(list)
    for row in rows:
        buckets[(row.get("domain"), row.get("dimension"))].append(row)

    missing = [
        (domain, dimension)
        for domain in DOMAIN_ORDER
        for dimension in DIMENSION_ORDER
        if not buckets[(domain, dimension)]
    ]
    if missing:
        raise ValueError(f"BizGenEval is missing domain/dimension categories: {missing}")

    # The first ten selections contain two examples per domain and cover all
    # four dimensions (3 layout, 3 text, 2 attribute, 2 knowledge). Continuing
    # to twenty visits every one of the 5 x 4 domain/dimension combinations.
    category_order = []
    for slot in range(len(DIMENSION_ORDER)):
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            dimension_index = (2 * domain_index + slot) % len(DIMENSION_ORDER)
            category_order.append((domain, DIMENSION_ORDER[dimension_index]))

    selected = []
    bucket_row = 0
    while len(selected) < limit:
        added = False
        for category in category_order:
            category_rows = buckets[category]
            if bucket_row < len(category_rows):
                selected.append(category_rows[bucket_row])
                added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        bucket_row += 1
    if len(selected) != limit:
        raise ValueError(f"Requested {limit} stratified prompts, but selected only {len(selected)}")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare a BizGenEval subset for i1 inference.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--selection", choices=("first", "stratified"), default="stratified")
    args = parser.parse_args()

    if args.limit <= 0:
        raise ValueError("--limit must be positive")

    all_rows = []
    with args.input.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                all_rows.append(json.loads(line))
    if len(all_rows) < args.limit:
        raise ValueError(f"Requested {args.limit} prompts, but found only {len(all_rows)}")
    rows = all_rows[: args.limit] if args.selection == "first" else stratified_rows(all_rows, args.limit)

    prompts = []
    names = []
    for row_index, row in enumerate(rows):
        prompt = row.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"Row {row_index} has no non-empty prompt")
        domain = str(row.get("domain", "unknown")).strip() or "unknown"
        dimension = str(row.get("dimension", "unknown")).strip() or "unknown"
        item_id = row.get("id", row_index)
        name = f"{domain}_{dimension}_{item_id}.png"
        if Path(name).name != name:
            raise ValueError(f"Unsafe output name derived from row {row_index}: {name!r}")
        # prompts.txt is a human-readable convenience copy. Inference reads
        # metadata.jsonl directly so embedded line breaks remain exact.
        prompts.append(" ".join(prompt.splitlines()).strip())
        names.append(name)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "prompts.txt").write_text("\n".join(prompts) + "\n", encoding="utf-8")
    (args.output_dir / "output_names.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    with (args.output_dir / "metadata.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    for row in rows:
        print(f"id={row.get('id')} domain={row.get('domain')} dimension={row.get('dimension')}")


if __name__ == "__main__":
    main()
