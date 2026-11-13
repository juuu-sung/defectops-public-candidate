#!/usr/bin/env python3
"""Start separate MPI research input notes on localhost; never load a model."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from defectops.input_review import Evidence, Notes, create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT/'artifacts/handoff/mpi-normal-input-assessment-v1-archive.zip')
    parser.add_argument('--db', type=Path, default=ROOT/'artifacts/review/mpi-input-notes-v1.sqlite3')
    parser.add_argument('--port', type=int, default=8770)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error('port must be 1..65535')
    evidence = Evidence(args.bundle)
    notes = Notes(args.db, evidence)
    import uvicorn
    print(f'Research input review: http://127.0.0.1:{args.port}/', flush=True)
    uvicorn.run(create_app(evidence, notes), host='127.0.0.1', port=args.port)


if __name__ == '__main__':
    main()
