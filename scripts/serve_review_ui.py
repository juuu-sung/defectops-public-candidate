#!/usr/bin/env python3
"""Launch the unauthenticated review screen on loopback only."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = PROJECT_ROOT / "artifacts" / "review" / "reviews.sqlite3"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--port", type=int, default=8501)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.db.is_file():
        print(f"Review database is not initialized: {args.db}", file=sys.stderr)
        return 2
    if not 1 <= args.port <= 65535:
        print("Port must be between 1 and 65535", file=sys.stderr)
        return 2
    environment = os.environ.copy()
    environment["DEFECTOPS_REVIEW_DB"] = str(args.db.resolve())
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(PROJECT_ROOT / "scripts" / "review_ui.py"),
        "--server.address=127.0.0.1",
        f"--server.port={args.port}",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--browser.gatherUsageStats=false",
    ]
    print(f"Local review screen: http://127.0.0.1:{args.port}", flush=True)
    try:
        return subprocess.run(command, env=environment, check=False).returncode
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
