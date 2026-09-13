#!/usr/bin/env python3
"""Create a frozen manifest once, then verify it on any machine."""

import argparse
import json
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from defectops.manifest import build_manifest, summary, verify_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    create = sub.add_parser("create")
    create.add_argument("--root", type=Path, required=True)
    create.add_argument("--config", type=Path, default=Path(__file__).resolve().parents[1] / "configs/pilot.toml")
    create.add_argument("--output", type=Path, required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("--root", type=Path, required=True)
    verify.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "create":
            config = tomllib.loads(args.config.read_text(encoding="utf-8"))
            if config["protocol"] != "A-normal-only-v1":
                raise ValueError("Unsupported protocol")
            manifest = build_manifest(args.root, config["categories"], config["split_seed"],
                                      config["calibration_fraction"])
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(manifest, stream, indent=2, ensure_ascii=True)
                stream.write("\n")
        else:
            manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
            verify_manifest(args.root, manifest)
        print(json.dumps(summary(manifest), indent=2))
        return 0
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
