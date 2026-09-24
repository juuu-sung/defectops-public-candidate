"""Fit one VisA PCB normal memory bank after the external test was observed."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_normal_bank_development import main


if __name__ == "__main__":
    raise SystemExit(main())
