#!/usr/bin/env python3
"""Run the locked Real-IAD final workflow from the project checkout."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_followup_final import main


if __name__ == "__main__":
    raise SystemExit(main())
