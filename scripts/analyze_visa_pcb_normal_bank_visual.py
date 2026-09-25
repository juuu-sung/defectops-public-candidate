"""Prepare read-only PCB3 v2 error cards from already-saved evaluations."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_normal_bank_visual_diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
