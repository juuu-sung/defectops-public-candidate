#!/usr/bin/env python3
"""Retry one existing API result's review registration, without a GPU or inference."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from defectops.review_registration import ReviewRegistration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--inspection-id', required=True)
    parser.add_argument('--output-root', type=Path, default=ROOT/'artifacts/inspections')
    parser.add_argument('--review-db', type=Path, default=ROOT/'artifacts/review/reviews.sqlite3')
    args = parser.parse_args(argv)
    registration = ReviewRegistration(args.output_root, args.review_db)
    registration.initialize()
    result = registration.inspect(args.inspection_id, register=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result['status'] == 'REGISTERED' else 1


if __name__ == '__main__':
    raise SystemExit(main())
