#!/usr/bin/env python3
"""Benchmark warm PaDiM inference with one model load."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.padim_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main())
