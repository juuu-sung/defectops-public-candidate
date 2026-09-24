"""Evaluate the preregistered VisA PCB external test exactly once (explicit gate)."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_external_evaluation import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--open-final", action="store_true", help="Explicitly open the official VisA test")
    args = parser.parse_args()
    if not args.open_final:
        parser.error("Official VisA test remains sealed; pass --open-final explicitly after the plan is frozen")
    try:
        result = run_evaluation(args.project_root, args.archive)
    except (OSError, ValueError, RuntimeError, KeyError,
            tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
