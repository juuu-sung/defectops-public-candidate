#!/usr/bin/env python3
"""Manage the Mac-local inspection registry and append-only human review log."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.mock_sop import DEFECT_LABELS, find_mock_sop, load_mock_catalog
from defectops.review_store import HUMAN_VERDICTS, ReviewStore


def build_parser() -> argparse.ArgumentParser:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        type=Path,
        default=project_root / "artifacts" / "review" / "reviews.sqlite3",
        help="Local SQLite file; never commit or sync the live database through Git",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("init", help="Create the local database schema")
    importing = commands.add_parser("import", help="Verify and register one copied bundle")
    importing.add_argument("--bundle", type=Path, required=True)
    commands.add_parser("list", help="List registered inspections")
    show = commands.add_parser("show", help="Show the frozen record and latest review")
    show.add_argument("--inspection-id", required=True)
    verify = commands.add_parser("verify", help="Recheck every source artifact checksum")
    verify.add_argument("--inspection-id", required=True)
    review = commands.add_parser("review", help="Append a human observation")
    review.add_argument("--inspection-id", required=True)
    review.add_argument("--expected-revision", type=int, required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--verdict", choices=sorted(HUMAN_VERDICTS), required=True)
    review.add_argument("--note", required=True)
    history = commands.add_parser("history", help="Read immutable review events")
    history.add_argument("--inspection-id", required=True)
    link_sop = commands.add_parser("link-sop", help="Append one validated mock SOP source snapshot")
    link_sop.add_argument("--inspection-id", required=True)
    link_sop.add_argument("--expected-review-revision", type=int, required=True)
    link_sop.add_argument("--selected-by", required=True)
    link_sop.add_argument(
        "--defect-type",
        choices=sorted(key for key in DEFECT_LABELS if key != "UNKNOWN"),
        required=True,
    )
    link_sop.add_argument(
        "--catalog",
        type=Path,
        default=project_root / "sops" / "mock-catalog-v1.json",
    )
    link_sop.add_argument("--as-of", type=date.fromisoformat, default=date.today())
    sop_history = commands.add_parser("sop-history", help="Read immutable mock SOP source links")
    sop_history.add_argument("--inspection-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    store = ReviewStore(args.db)
    try:
        if args.command == "init":
            store.initialize()
            result = {"status": "initialized", "database": str(store.database_path)}
        elif args.command == "import":
            result = store.register(args.bundle)
        elif args.command == "list":
            result = {"inspections": store.list_inspections()}
        elif args.command == "show":
            result = store.get_inspection(args.inspection_id)
        elif args.command == "verify":
            result = store.verify(args.inspection_id)
        elif args.command == "review":
            result = store.add_review(
                args.inspection_id,
                expected_revision=args.expected_revision,
                reviewer=args.reviewer,
                human_verdict=args.verdict,
                note=args.note,
            )
        elif args.command == "history":
            result = {"history": store.review_history(args.inspection_id)}
        elif args.command == "link-sop":
            item = store.get_inspection(args.inspection_id)
            catalog = load_mock_catalog(args.catalog)
            sop_result = find_mock_sop(
                catalog,
                product=item["category"],
                defect_type=args.defect_type,
                human_verdict=(item["latest_review"] or {}).get("human_verdict"),
                as_of=args.as_of,
            )
            result = store.add_sop_evidence(
                args.inspection_id,
                expected_review_revision=args.expected_review_revision,
                selected_by=args.selected_by,
                defect_type=args.defect_type,
                result=sop_result,
            )
        else:
            result = {"history": store.sop_evidence_history(args.inspection_id)}
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
