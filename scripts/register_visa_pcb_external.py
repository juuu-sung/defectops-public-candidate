#!/usr/bin/env python3
"""Register the official VisA tar against a metadata-only PCB plan."""

from __future__ import annotations

import argparse
import json
import sys
import tarfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_external_protocol import load_config
from defectops.visa_pcb_external_registration import register


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = register(
            args.archive.resolve(strict=True),
            project_root / "data" / "manifests" / "visa-pcb-external-plan-v1.json",
            load_config(project_root / "configs" / "visa-pcb-external-v1.toml"),
            project_root / "artifacts" / "visa-pcb-external-registration-v1",
            project_root / "artifacts" / "handoff" / "visa-pcb-external-registration-v1.zip",
        )
        print(json.dumps(report, indent=2, ensure_ascii=False))
    except (OSError, ValueError, KeyError, tarfile.TarError, zipfile.BadZipFile) as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
