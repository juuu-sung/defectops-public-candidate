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
from defectops.review_store import ACTION_APPROVAL_DECISIONS, HUMAN_VERDICTS, ReviewStore


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
    draft_action = commands.add_parser("draft-action", help="Create one local simulation-only action draft")
    draft_action.add_argument("--inspection-id", required=True)
    draft_action.add_argument("--expected-review-revision", type=int, required=True)
    draft_action.add_argument("--created-by", required=True)
    action_history = commands.add_parser("action-history", help="Read immutable action drafts")
    action_history.add_argument("--inspection-id", required=True)
    decide_action = commands.add_parser("decide-action", help="Approve or reject one local action draft")
    decide_action.add_argument("--inspection-id", required=True)
    decide_action.add_argument("--expected-review-revision", type=int, required=True)
    decide_action.add_argument("--action-draft-event-id", type=int, required=True)
    decide_action.add_argument("--approver", required=True)
    decide_action.add_argument("--decision", choices=sorted(ACTION_APPROVAL_DECISIONS), required=True)
    decide_action.add_argument("--note", required=True)
    approval_history = commands.add_parser("approval-history", help="Read immutable approval decisions")
    approval_history.add_argument("--inspection-id", required=True)
    issue_ticket = commands.add_parser("issue-mock-ticket", help="Create a local-only mock ticket snapshot")
    issue_ticket.add_argument("--inspection-id", required=True)
    issue_ticket.add_argument("--expected-review-revision", type=int, required=True)
    issue_ticket.add_argument("--approval-event-id", type=int, required=True)
    issue_ticket.add_argument("--issued-by", required=True)
    ticket_history = commands.add_parser("ticket-history", help="Read immutable local mock tickets")
    ticket_history.add_argument("--inspection-id", required=True)
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
        elif args.command == "sop-history":
            result = {"history": store.sop_evidence_history(args.inspection_id)}
        elif args.command == "draft-action":
            result = store.create_action_draft(
                args.inspection_id,
                expected_review_revision=args.expected_review_revision,
                created_by=args.created_by,
            )
        elif args.command == "action-history":
            result = {"history": store.action_draft_history(args.inspection_id)}
        elif args.command == "decide-action":
            result = store.decide_action_draft(
                args.inspection_id,
                expected_review_revision=args.expected_review_revision,
                action_draft_event_id=args.action_draft_event_id,
                approver=args.approver,
                decision=args.decision,
                note=args.note,
            )
        elif args.command == "approval-history":
            result = {"history": store.action_approval_history(args.inspection_id)}
        elif args.command == "issue-mock-ticket":
            result = store.issue_mock_ticket(
                args.inspection_id,
                expected_review_revision=args.expected_review_revision,
                approval_event_id=args.approval_event_id,
                issued_by=args.issued_by,
            )
        else:
            result = {"history": store.mock_ticket_history(args.inspection_id)}
    except (FileNotFoundError, KeyError, OSError, ValueError) as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
