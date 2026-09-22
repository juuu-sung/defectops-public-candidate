#!/usr/bin/env python3
"""Seal Real-IAD follow-up development/final paths without opening image archives."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.realiad_followup_protocol import (
    build_realiad_followup_plan,
    summarize_realiad_followup_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsons-zip", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/realiad-followup-preregistration-v1.toml"),
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        output = args.output.resolve()
        plan = build_realiad_followup_plan(args.jsons_zip, args.config)
        if output.exists():
            existing = json.loads(output.read_text(encoding="utf-8"))
            if existing != plan:
                raise ValueError(f"Existing follow-up split plan does not match: {output}")
            status = "realiad_followup_split_verified"
        else:
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("x", encoding="utf-8", newline="\n") as stream:
                json.dump(plan, stream, indent=2, ensure_ascii=True)
                stream.write("\n")
            status = "realiad_followup_split_sealed"
        print(
            json.dumps(
                {
                    "status": status,
                    "output": str(output),
                    **summarize_realiad_followup_plan(plan),
                },
                indent=2,
            )
        )
        return 0
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
