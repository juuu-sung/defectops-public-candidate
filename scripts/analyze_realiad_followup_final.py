"""Describe the locked Real-IAD final errors without running inference."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_followup_final_diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
