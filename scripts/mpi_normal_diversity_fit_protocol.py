"""Locked C-stage inputs and normal-only Q90 calibration. No test scoring."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile

import numpy as np

from scripts import mpi_normal_diversity_protocol as selection
from scripts import mpi_score_validation_protocol as scoring

fit = selection.fit
NAME = "mpi-normal-diversity-fit-v1"
SELECTION_ZIP_SHA = "ba5abc4f137d38022c9ded9e143ef9cc9b2523d2a53a80c95479c254a575ff82"
IDS_SHA = "991667b1f10814211a75d81da3343a45866203e0db837faa1f4b02a18e0daa5c"
ROWS_SHA = "d399dfb3fda9cf0dcc21cadd7d262524d5aae951b6068fda0c8a57afecff6b4d"


def read_selection(path):
    if fit.sha256_file(path) != SELECTION_ZIP_SHA:
        raise ValueError("Selection ZIP differs from the reviewed, locked 64 photos")
    with zipfile.ZipFile(path) as z:
        names = z.namelist()
        receipt = json.loads(z.read("receipt.json"))
        if len(names) != len(set(names)) or set(names) != {*receipt["files"], "receipt.json"}:
            raise ValueError("Selection ZIP inventory differs")
        for name, sha in receipt["files"].items():
            if hashlib.sha256(z.read(name)).hexdigest() != sha:
                raise ValueError("Selection payload checksum differs")
        plan = json.loads(z.read("plan.json"))
        chosen = json.loads(z.read("selection.json"))
        environment = json.loads(z.read("report.json"))["environment"]
    if (receipt["identity"] != plan or fit.digest(plan["policy"]) != selection.POLICY_SHA256
            or fit.digest(chosen["training_photo_ids"]) != IDS_SHA
            or fit.digest(chosen["training_rows"]) != ROWS_SHA
            or chosen["status"] != "normal_diversity_selected_ready_for_review"):
        raise ValueError("Reviewed selection contract differs")
    return plan, chosen, environment


def validate_cohorts(chosen, cohorts):
    ids, rgb = set(), set()
    for label, photos in [("training", [{"photo_id": p, "regions": chosen["training_rows"][i*12:(i+1)*12]}
                                        for i, p in enumerate(chosen["training_photo_ids"])])]+list(cohorts.items()):
        if label not in {"training", "development", "calibration"}:
            raise ValueError("Only designated normal cohorts may enter C-stage")
        expected_count = {"training": 64, "development": 128, "calibration": 169}[label]
        if len(photos) != expected_count:
            raise ValueError("Normal cohort size differs")
        for photo in photos:
            pid, rows = photo["photo_id"], photo["regions"]
            if pid in ids or [r["region"] for r in rows] != list(fit.REGIONS):
                raise ValueError("Overlapping photo IDs or incorrect region order")
            ids.add(pid)
            for row in rows:
                if (row["photo_id"] != pid or row["member"] != f'extracted/{row["region"]}/train/good/{pid}.png'
                        or row["decoded_rgb_sha256"] in rgb):
                    raise ValueError("Non-normal path or exact RGB overlap")
                rgb.add(row["decoded_rgb_sha256"])


def prepare(project, handoffs):
    plan, chosen, environment = read_selection(handoffs/"mpi-normal-diversity-preparation-v1.zip")
    source = selection.spatial.prepare(project, handoffs)
    policy = json.loads((project/"configs/mpi-normal-diversity-v1.json").read_bytes())
    if fit.digest(policy) != selection.POLICY_SHA256:
        raise ValueError("Frozen policy changed")
    # Old recorded files must still match; new C-stage files may be added.
    for name, sha in plan["source_files_lf_sha256"].items():
        if hashlib.sha256((project/name).read_bytes().replace(b"\r\n", b"\n")).hexdigest() != sha:
            raise ValueError(f"Selection source changed: {name}")
    cohorts = {k: source["cohorts"][k] for k in ("development", "calibration")}
    for k, photos in cohorts.items():
        if fit.digest(photos) != policy["source_sha256"][k+"_rows"]:
            raise ValueError("Normal calibration cohort changed")
    validate_cohorts(chosen, cohorts)
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    identity = {"name": NAME, "stage": "C_fit_normal_calibration_reload_only",
                "git_commit": subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                                             check=True, capture_output=True, text=True).stdout.strip(),
                "authorization": "user approved reviewed selection on 2026-10-02; no test authorization",
                "selection_zip_sha256": SELECTION_ZIP_SHA, "training_ids_sha256": IDS_SHA,
                "training_rows_sha256": ROWS_SHA, "policy": policy, "model_config": source["config"],
                "normal_cohorts_sha256": fit.digest(cohorts),
                "selection_environment": environment,
                "source_files_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}}
    return chosen, cohorts, source["records"], identity


def validate_archive(source, chosen, cohorts):
    names = Counter(source.namelist())
    rows = chosen["training_rows"] + [r for photos in cohorts.values() for p in photos for r in p["regions"]]
    if any(names[r["member"]] != 1 for r in rows):
        raise ValueError("Missing or duplicate designated normal members")


def calibrate(development, calibration):
    for group, samples, count in (("development", development, 128), ("calibration", calibration, 169)):
        scoring.validate_samples(samples, group)
        if len(samples) != count:
            raise ValueError("Incomplete normal scores")
    if {p["photo_id"] for p in development} & {p["photo_id"] for p in calibration}:
        raise ValueError("Normal cohorts overlap")
    values = np.asarray([[r["top1"] for r in p["regions"]] for p in development], dtype=np.float64)
    center = np.median(values, axis=0)
    scale = np.quantile(values, .9, axis=0, method="higher") - center
    if not np.isfinite(scale).all() or np.any(scale <= 1e-6):
        raise ValueError("Degenerate Q90 scale; no fallback or threshold tuning")
    method = {"field": "top1", "scale_kind": "q90", "threshold": 0.,
              "regions": [{"region": r, "center": float(c), "scale": float(s)}
                          for r, c, s in zip(fit.REGIONS, center, scale, strict=True)]}
    method["threshold"] = float(np.quantile([scoring.score(p, method)["score"] for p in calibration], .95, method="higher"))
    return method


def reload_check(first, second, contract):
    maximum = 0.
    changes = 0
    for group in ("development", "calibration"):
        a, b = first[group], second[group]
        scoring.validate_samples(a, group)
        scoring.validate_samples(b, group)
        if len(a) != {"development": 128, "calibration": 169}[group] or [p["key"] for p in a] != [p["key"] for p in b]:
            raise ValueError("Reload normal inventory/order differs")
        for p, q in zip(a, b, strict=True):
            for r, s in zip(p["regions"], q["regions"], strict=True):
                if r["member"] != s["member"]:
                    raise ValueError("Reload normal path differs")
                maximum = max(maximum, *(abs(r[k]-s[k]) for k in ("top1", "native_raw")))
            changes += scoring.score(p, contract)["alert"] != scoring.score(q, contract)["alert"]
    if maximum > 1e-4 or changes:
        raise ValueError(f"Reload verification failed: raw difference {maximum}, decision changes {changes}")
    return {"passed": True, "photos": 297, "maximum_raw_score_difference": maximum,
            "decision_changes": changes, "raw_tolerance": 1e-4, "contract_refitted": False}
