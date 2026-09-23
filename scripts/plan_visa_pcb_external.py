#!/usr/bin/env python3
"""Create or verify a metadata-only VisA PCB external transfer plan."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.visa_pcb_external_protocol import build_plan, load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/visa-pcb-external-v1.toml"))
    parser.add_argument("--output", type=Path, default=Path("data/manifests/visa-pcb-external-plan-v1.json"))
    args = parser.parse_args()
    try:
        plan = build_plan(args.csv.read_bytes(), load_config(args.config))
        content = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
        if args.output.exists():
            if args.output.read_text(encoding="utf-8") != content:
                raise ValueError(f"Existing metadata plan differs and was not overwritten: {args.output}")
        else:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as stream:
                stream.write(content)
        print(json.dumps({"status": "metadata_locked", "plan_sha256": plan["plan_sha256"],
                          "counts": plan["counts"], "output": str(args.output)}, ensure_ascii=False, indent=2))
    except (OSError, ValueError, KeyError, UnicodeError) as error:
        print(f"ERROR: {type(error).__name__}: {error}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
