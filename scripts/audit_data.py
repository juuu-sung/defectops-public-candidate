#!/usr/bin/env python3
"""Verify a frozen manifest and audit image integrity without tuning on the test set."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defectops.data_audit import audit_dataset, markdown_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New output directory")
    args = parser.parse_args()
    try:
        if args.output.exists():
            raise ValueError("Output directory already exists; choose a new audit run directory")
        manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
        report = audit_dataset(args.root, manifest)
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        (args.output / "summary.md").write_text(markdown_report(report), encoding="utf-8")
        print(json.dumps({key: report[key] for key in ("status", "images_decoded", "masks_decoded",
                                                       "image_specs", "mask_specs", "counts", "errors",
                                                       "warnings")}, indent=2))
        return 0 if report["status"] == "passed" else 1
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
