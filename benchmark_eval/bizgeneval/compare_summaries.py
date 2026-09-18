"""Compare the two BizGenEval summary CSVs produced by the pipeline."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def read_rows(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return {row["Split"]: row for row in csv.DictReader(handle)}


def compare(start_path: Path, sft_path: Path, output_path: Path) -> None:
    start_rows = read_rows(start_path)
    sft_rows = read_rows(sft_path)
    splits = [split for split in ("easy", "hard", "all") if split in start_rows or split in sft_rows]
    available_rows = [*start_rows.values(), *sft_rows.values()]
    if not available_rows:
        raise ValueError("Neither summary contains any rows")
    first = available_rows[0]
    groups = [name for name in first if name != "Split"]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["Split", "Group", "Starting checkpoint", "SFT checkpoint", "Delta"],
        )
        writer.writeheader()
        for split in splits:
            for group in groups:
                start_value = (start_rows.get(split) or {}).get(group, "")
                sft_value = (sft_rows.get(split) or {}).get(group, "")
                delta = ""
                if start_value and sft_value:
                    delta = f"{float(sft_value) - float(start_value):.4f}"
                writer.writerow(
                    {
                        "Split": split,
                        "Group": group,
                        "Starting checkpoint": start_value,
                        "SFT checkpoint": sft_value,
                        "Delta": delta,
                    }
                )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--starting", type=Path, required=True)
    parser.add_argument("--sft", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    compare(args.starting, args.sft, args.output)
    print(f"Saved comparison to {args.output}")


if __name__ == "__main__":
    main()
