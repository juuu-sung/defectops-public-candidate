#!/usr/bin/env python3
"""Summarize the frozen 3-category × 2-model × 3-seed evaluation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.three_category_summary import main


if __name__ == "__main__":
    raise SystemExit(main())
