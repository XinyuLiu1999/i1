"""Validate that every prepared BizGenEval item has a generated PNG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--names", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--geometry", choices=("existence", "square", "native_buckets"), default="existence")
    args = parser.parse_args()

    names = [line.strip() for line in args.names.read_text(encoding="utf-8").splitlines() if line.strip()]
    expected_sizes: list[tuple[int, int] | None] = [None] * len(names)
    if args.geometry == "square":
        expected_sizes = [(1024, 1024)] * len(names)
    elif args.geometry == "native_buckets":
        if args.data is None:
            raise SystemExit("--data is required with --geometry native_buckets")
        with args.data.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        if len(rows) != len(names):
            raise SystemExit(f"Dataset has {len(rows)} rows but names file has {len(names)} entries")
        expected_sizes = [(int(row["_i1_width"]), int(row["_i1_height"])) for row in rows]

    invalid: list[str] = []
    for name, expected_size in zip(names, expected_sizes):
        path = args.image_dir / name
        if not path.is_file():
            invalid.append(f"{name} (missing)")
            continue
        if expected_size is not None:
            try:
                with Image.open(path) as image:
                    actual_size = image.size
            except (OSError, ValueError):
                invalid.append(f"{name} (unreadable)")
                continue
            if actual_size != expected_size:
                invalid.append(f"{name} ({actual_size[0]}x{actual_size[1]}, expected {expected_size[0]}x{expected_size[1]})")

    if invalid:
        preview = "\n".join(f"  {name}" for name in invalid[:20])
        suffix = f"\n  ... and {len(invalid) - 20} more" if len(invalid) > 20 else ""
        raise SystemExit(
            f"Invalid {len(invalid)}/{len(names)} generated images in {args.image_dir}:\n"
            f"{preview}{suffix}"
        )
    print(f"Validated {len(names)} generated images in {args.image_dir} ({args.geometry})")


if __name__ == "__main__":
    main()
