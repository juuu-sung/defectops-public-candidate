#!/usr/bin/env python3
"""Create compact Real-IAD consensus and seed-instability diagnostics."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_multiseed_diagnostics import main


if __name__ == "__main__":
    raise SystemExit(main())
