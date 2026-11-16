#!/usr/bin/env python3
"""Read-only CPU validation of frozen inspection bundles after history cleanup."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from defectops.padim_experiment import checksum, git_state
from defectops.padim_inspection import validate_inference_bundles


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fit-run", type=Path, required=True)
    parser.add_argument("--evaluation", type=Path, required=True)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[1]
    current = git_state(root)
    provenance = {}
    fit, config, calibration, checkpoint = validate_inference_bundles(
        args.fit_run.resolve(strict=True), args.evaluation.resolve(strict=True),
        root, provenance=provenance,
    )
    print(json.dumps({
        "status": "validated_bundles_only",
        "inspection_git": current,
        "model": config["model"],
        "fit_run_id": fit["run_id"],
        "git_provenance": provenance,
        "checkpoint_sha256": checksum(checkpoint),
        "image_threshold": calibration["image"]["threshold"],
        "checkpoint_deserialized": False,
        "gpu_inference_performed": False,
        "artifacts_modified": False,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
