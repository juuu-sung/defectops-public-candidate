#!/usr/bin/env python3
"""Fit and calibrate the fixed exploratory PatchCore baseline."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.patchcore_experiment import main


if __name__ == "__main__":
    raise SystemExit(main())
