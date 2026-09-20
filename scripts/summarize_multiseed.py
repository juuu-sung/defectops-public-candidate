#!/usr/bin/env python3
"""Summarize the validated PaDiM/PatchCore three-seed evaluations."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.multiseed_summary import main


if __name__ == "__main__":
    raise SystemExit(main())
