#!/usr/bin/env python3
"""Benchmark warm PatchCore inference with one validated model load."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.padim_benchmark import main


if __name__ == "__main__":
    raise SystemExit(main(default_output_name="patchcore-warm-v1"))
