#!/usr/bin/env python3
"""Analyze frozen exploratory PatchCore errors without changing fixed rules."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.padim_error_analysis import main


if __name__ == "__main__":
    raise SystemExit(main(default_output_name="patchcore-error-analysis-v1"))
