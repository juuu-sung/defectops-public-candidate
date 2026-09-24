"""Inspect one already-opened VisA mask without modifying artifacts or scoring images."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_mask_diagnostic import diagnose_archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(diagnose_archive(args.archive), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
