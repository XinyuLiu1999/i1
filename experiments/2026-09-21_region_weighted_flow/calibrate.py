#!/usr/bin/env python3
"""Measure regional/global gradients at initialization using the shared trainer."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "torch_train"))

from training.main import main
from calibration import RegionCalibration


if __name__ == "__main__":
    main(extension=RegionCalibration())
