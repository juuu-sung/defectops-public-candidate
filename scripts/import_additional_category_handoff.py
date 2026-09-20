#!/usr/bin/env python3
"""Validate and import the Windows additional-category registration ZIP."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.additional_category_handoff import import_handoff, validate_handoff


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--import-files", action="store_true")
    args = parser.parse_args()
    try:
        result = validate_handoff(args.zip.resolve(strict=True), args.expected_sha256)
        output = {
            "status": result["status"],
            "zip_sha256": result["zip_sha256"],
            "manifest_sha256": {
                category: manifest["manifest_sha256"]
                for category, manifest in result["manifests"].items()
            },
            "counts": result["counts"],
        }
        if args.import_files:
            output["imported"] = import_handoff(
                result, project_root / "data" / "manifests", project_root / "reports"
            )
        print(json.dumps(output, indent=2, ensure_ascii=False))
        return 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
