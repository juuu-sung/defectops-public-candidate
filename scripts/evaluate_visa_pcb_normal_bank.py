"""Evaluate the already-fitted VisA PCB3 v2 bank on the already-observed test."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_normal_bank_evaluation import run  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--exploratory-reuse-observed-test", action="store_true")
    args = parser.parse_args()
    if not args.exploratory_reuse_observed_test:
        parser.error("Pass --exploratory-reuse-observed-test to acknowledge that VisA test was already observed")
    try:
        result = run(args.project, args.archive)
    except (OSError, ValueError, RuntimeError, KeyError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
