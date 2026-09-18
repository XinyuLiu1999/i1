"""Validate that every selected BizGenEval judgment is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def result_name(item: dict) -> str:
    for key in ("reference_image", "reference image", "image_path", "image", "path"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value.strip()).with_suffix(".json").name
    return f"{item.get('domain', '')}_{item.get('dimension', '')}_{item.get('id')}.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    args = parser.parse_args()

    with args.data.open("r", encoding="utf-8") as handle:
        items = [json.loads(line) for line in handle if line.strip()]

    incomplete: list[str] = []
    for item in items:
        name = result_name(item)
        path = args.result_dir / name
        try:
            result = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            incomplete.append(name)
            continue
        meta = result.get("meta_info") or {}
        if result.get("accuracy") is None or len(meta) != len(item.get("questions") or []):
            incomplete.append(name)
            continue
        if any(
            not isinstance(record, dict) or record.get("reason") == "missing_from_output"
            for record in meta.values()
        ):
            incomplete.append(name)

    if incomplete:
        preview = "\n".join(f"  {name}" for name in incomplete[:20])
        suffix = f"\n  ... and {len(incomplete) - 20} more" if len(incomplete) > 20 else ""
        raise SystemExit(
            f"Incomplete {len(incomplete)}/{len(items)} evaluation results in {args.result_dir}. "
            f"Rerun run_evaluation.sh to retry them:\n{preview}{suffix}"
        )
    print(f"Validated {len(items)} complete evaluation results in {args.result_dir}")


if __name__ == "__main__":
    main()
