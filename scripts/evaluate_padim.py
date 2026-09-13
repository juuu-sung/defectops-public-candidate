#!/usr/bin/env python3
"""Evaluate an immutable PaDiM fit on the frozen test split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.padim_evaluation import main


if __name__ == "__main__":
    raise SystemExit(main())
