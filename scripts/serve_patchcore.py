#!/usr/bin/env python3
"""Serve the frozen PatchCore model locally with one long-lived worker."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import uvicorn

from defectops.inspection_api import create_app


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--review-db", type=Path,
                        default=Path(__file__).resolve().parents[1]/"artifacts/review/reviews.sqlite3",
                        help="Same local SQLite database used by the review UI")
    parser.add_argument("--no-review-registration", action="store_true",
                        help="Keep the legacy file-only API workflow")
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    app = create_app(args.fit_run, args.evaluation,
                     review_database=None if args.no_review_registration else args.review_db)
    uvicorn.run(app, host="127.0.0.1", port=args.port, workers=1, reload=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
