#!/usr/bin/env python3
"""Download the official ODM checkpoint, verify its published SHA256, then load it."""
import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "torch_train"))

from config import get_config
from perceptual_loss import load_odm

URL = "https://huggingface.co/GD-ML/FLUX-Text/resolve/main/epoch_100.pt"
SHA256 = "a7e329c97cae19e4fd3ad1b5867036952477dd09e53b90f39b2c68b100060156"


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    config = get_config()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(config.perceptual_checkpoint))
    parser.add_argument("--fluxtext-root", default=config.fluxtext_root)
    args = parser.parse_args()
    destination = args.output.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    candidate = destination
    if not destination.exists():
        candidate = destination.with_suffix(destination.suffix + ".partial")
        # Resume interrupted downloads; only publish the fully verified file.
        subprocess.run(["curl", "-fL", "--continue-at", "-", "--retry", "3",
                        "--connect-timeout", "20", "--max-time", "1800",
                        "--output", str(candidate), URL], check=True)
    if file_hash(candidate) != SHA256:
        raise ValueError(f"ODM SHA256 mismatch: {candidate}; do not use this checkpoint")
    model = load_odm(candidate, args.fluxtext_root, "cpu")
    if candidate != destination:
        os.replace(candidate, destination)
    print(f"Verified ODM checkpoint: {destination}")
    print(f"SHA256: {SHA256}; frozen feature parameters: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()
