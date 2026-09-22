#!/usr/bin/env python3
"""Register follow-up development files and keep final holdout members sealed."""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_followup_registration import (
    DEVELOPMENT_MANIFEST_NAMES,
    FINAL_MANIFEST_NAMES,
    followup_registration_summary,
    load_followup_plan,
    register_followup_data,
    verify_followup_development_manifest,
    verify_followup_final_manifest,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    register = subparsers.add_parser("register")
    register.add_argument("--root", type=Path, required=True)
    register.add_argument("--plan", type=Path, required=True)
    register.add_argument("--pcb-zip", type=Path, required=True)
    register.add_argument("--terminalblock-zip", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--plan", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.root.resolve(strict=True)
        plan = load_followup_plan(args.plan.resolve(strict=True))
        if args.command == "register":
            payloads = register_followup_data(
                root,
                args.plan,
                {
                    "realiad_pcb": args.pcb_zip,
                    "realiad_terminalblock": args.terminalblock_zip,
                },
            )
        else:
            payloads = []
            for category in plan["categories"]:
                development = _read(root / "manifests" / DEVELOPMENT_MANIFEST_NAMES[category])
                final = _read(root / "manifests" / FINAL_MANIFEST_NAMES[category])
                verify_followup_development_manifest(root, development)
                verify_followup_final_manifest(root, final)
                payloads.extend((development, final))
        print(
            json.dumps(
                {
                    "status": f"realiad_followup_{args.command}_passed",
                    "final_images_extracted": False,
                    "manifests": [followup_registration_summary(item) for item in payloads],
                },
                indent=2,
            )
        )
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        TypeError,
        json.JSONDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

