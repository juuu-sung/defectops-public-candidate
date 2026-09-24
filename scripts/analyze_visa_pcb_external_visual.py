"""Prepare read-only visual review cards for the verified VisA PCB v2 evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_external_visual_diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
