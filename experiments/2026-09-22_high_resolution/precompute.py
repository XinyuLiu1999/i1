#!/usr/bin/env python3
"""Build a fresh captioned RGB cache using the experiment's resolution tiers."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "torch_train"))

from configs.sft_multires_captioned import get_config
from datasets.precompute_captioned import (
    DEFAULT_MAX_SOURCE_PIXELS, DEFAULT_MAX_SOURCE_SIDE, DEFAULT_SOURCE,
    precompute_captioned,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resolution", type=int, choices=(1536, 2048), default=2048)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--records-per-part", type=int, default=2000)
    parser.add_argument("--max-shard-mib", type=int, default=256)
    parser.add_argument("--max-source-pixels", type=int, default=DEFAULT_MAX_SOURCE_PIXELS)
    parser.add_argument("--max-source-side", type=int, default=DEFAULT_MAX_SOURCE_SIDE)
    args = parser.parse_args()
    config = get_config(args.resolution)
    summary = precompute_captioned(
        args.source, args.output_dir, config.input, args.workers, config.token_len,
        "google/t5gemma-2b-2b-ul2-it", args.records_per_part,
        args.max_shard_mib * 1024**2, args.max_source_pixels, args.max_source_side,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
