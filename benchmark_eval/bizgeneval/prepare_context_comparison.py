#!/usr/bin/env python3
"""Select long BizGenEval prompts for a context-length comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from prepare_inputs import HEIGHT_KEY, WIDTH_KEY, generate_buckets, image_name, select_bucket


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=100)
    parser.add_argument("--threshold", type=int, default=1024)
    parser.add_argument("--tokenizer", default="google/t5gemma-2b-2b-ul2-it")
    parser.add_argument(
        "--selection",
        choices=("dataset-order", "longest"),
        default="dataset-order",
        help="Choose qualifying prompts in dataset order or by descending token length.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.threshold <= 0:
        raise ValueError("--threshold must be positive")

    with args.input.open("r", encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"No benchmark rows found in {args.input}")
    for line_number, row in enumerate(rows, 1):
        if not isinstance(row.get("prompt"), str) or not row["prompt"].strip():
            raise ValueError(f"Input row {line_number} has no non-empty prompt")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    encoded = tokenizer(
        [row["prompt"] for row in rows],
        truncation=False,
        padding=False,
        add_special_tokens=True,
    )["input_ids"]
    candidates = [
        (source_index, row, len(token_ids))
        for source_index, (row, token_ids) in enumerate(zip(rows, encoded))
        if len(token_ids) > args.threshold
    ]
    if len(candidates) < args.count:
        raise ValueError(
            f"Requested {args.count} prompts longer than {args.threshold} tokens, "
            f"but only found {len(candidates)}"
        )
    if args.selection == "longest":
        candidates.sort(key=lambda item: (-item[2], item[0]))
    selected = candidates[: args.count]

    buckets = generate_buckets()
    prepared: list[dict] = []
    names: list[str] = []
    for source_index, row, token_count in selected:
        name = image_name(row)
        if Path(name).name != name or not name.lower().endswith(".png"):
            raise ValueError(f"Unsafe or non-PNG output name: {name!r}")
        height, width = select_bucket(row, buckets)
        output_row = dict(row)
        output_row[HEIGHT_KEY] = height
        output_row[WIDTH_KEY] = width
        output_row["_i1_text_tokens"] = token_count
        output_row["_i1_source_index"] = source_index
        prepared.append(output_row)
        names.append(name)

    if len(set(names)) != len(names):
        raise ValueError("Selected BizGenEval output filenames are not unique")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = args.output_dir / "bizgeneval_i1.jsonl"
    names_path = args.output_dir / "output_names.txt"
    lengths_path = args.output_dir / "token_lengths.tsv"
    maximum_path = args.output_dir / "max_text_tokens.txt"

    with dataset_path.open("w", encoding="utf-8") as handle:
        for row in prepared:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    names_path.write_text("\n".join(names) + "\n", encoding="utf-8")
    with lengths_path.open("w", encoding="utf-8") as handle:
        handle.write("selected_index\tsource_index\tid\toutput_name\ttokens\n")
        for selected_index, (row, name) in enumerate(zip(prepared, names)):
            handle.write(
                f"{selected_index}\t{row['_i1_source_index']}\t{row.get('id', '')}\t"
                f"{name}\t{row['_i1_text_tokens']}\n"
            )
    maximum = max(row["_i1_text_tokens"] for row in prepared)
    maximum_path.write_text(f"{maximum}\n", encoding="utf-8")

    print(
        f"Selected {len(prepared)} of {len(candidates)} prompts longer than "
        f"{args.threshold} tokens ({args.selection}); token range "
        f"{min(row['_i1_text_tokens'] for row in prepared)}..{maximum}."
    )
    print(f"Prepared dataset: {dataset_path}")
    print(f"All-token context length: {maximum}")


if __name__ == "__main__":
    main()
