#!/usr/bin/env python3
"""Run the post-final exploratory Real-IAD PCB development score probe."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_development_score_probe import main


if __name__ == "__main__":
    raise SystemExit(main())
