#!/usr/bin/env python3
"""Launch with torchrun; accepts the shared training.main CLI."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "torch_train"))

from training.main import main
from runtime import RegionFlow


if __name__ == "__main__":
    main(extension=RegionFlow())
