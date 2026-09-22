#!/usr/bin/env python3
"""Run or select preregistered Real-IAD follow-up development candidates."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_followup_development import main


if __name__ == "__main__":
    raise SystemExit(main())
