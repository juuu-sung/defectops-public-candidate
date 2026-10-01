"""Normal-only, fixed-size photo selection. No model fitting or score calibration."""
from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import subprocess
from pathlib import Path
import zipfile

import numpy as np

from scripts import run_mpi_spatial_score as spatial
from scripts import prepare_mpi_pcb_normal_bank as prep

fit = spatial.fit
NAME = "mpi-normal-diversity-v1"


def rank(pid, salt="mpi-pcb-normal-bank-pilot-v1"):
    return hashlib.sha256(f"{salt}\0{pid}".encode()).hexdigest(), pid


def derive_pool(rows, protected, test_rgb, baseline_ids):
    groups = defaultdict(dict)
    for row in rows:
        pid, region = row["photo_id"], row["region"]
        if (region not in fit.REGIONS or not pid.isascii() or len(pid) != 4 or not pid.isdigit()
                or row["member"] != f"extracted/{region}/train/good/{pid}.png"
                or region in groups[pid]):
            raise ValueError("Invalid normal fingerprint row")
        groups[pid][region] = row
    if any(set(g) != set(fit.REGIONS) for g in groups.values()):
        raise ValueError("Incomplete normal photo group")
    protected_ids = [p["photo_id"] for cohort in protected.values() for p in cohort]
    if len(protected_ids) != len(set(protected_ids)):
        raise ValueError("Protected normal cohorts overlap")
    blocked = set(test_rgb)
    for cohort in protected.values():
        for photo in cohort:
            if photo["photo_id"] not in groups or photo["regions"] != [groups[photo["photo_id"]][r] for r in fit.REGIONS]:
                raise ValueError("Protected cohort fingerprint provenance differs")
            blocked.update(r["decoded_rgb_sha256"] for r in photo["regions"])
    protected_ids = set(protected_ids)
    seen, eligible, excluded = set(), [], []
    for pid in sorted(groups, key=rank):
        rgb = tuple(groups[pid][r]["decoded_rgb_sha256"] for r in fit.REGIONS)
        reason = ("protected_normal_id" if pid in protected_ids else
                  "protected_or_test_exact_rgb" if set(rgb) & blocked else
                  "identical_full_12_region_photo" if rgb in seen else None)
        if reason:
            excluded.append({"photo_id": pid, "reason": reason})
        else:
            seen.add(rgb)
            eligible.append(pid)
    ordered_rows = [groups[p][r] for p in eligible for r in fit.REGIONS]
    rgb = [r["decoded_rgb_sha256"] for r in ordered_rows]
    if len(rgb) != len(set(rgb)) or not set(baseline_ids) <= set(eligible):
        raise ValueError("Partial tile duplicates or baseline training missing from candidate pool")
    return {"ids": eligible, "rows": ordered_rows, "excluded": excluded,
            "exclusion_counts": dict(Counter(e["reason"] for e in excluded))}


def prepare(project, handoffs):
    policy = json.loads((project/"configs"/f"{NAME}.json").read_bytes())
    audit = json.loads((project/"reports/mpi-normal-diversity-v1-pool-audit.json").read_bytes())
    # Require the preregistered recipe, not an arbitrary selection-search CLI.
    if fit.digest(policy) != POLICY_SHA256:
        raise ValueError("Preregistered diversity policy changed")
    source = spatial.prepare(project, handoffs)
    names = {"normal_preparation_zip": "mpi-pcb-normal-bank-preparation-v1.zip",
             "baseline_fit_zip": "mpi-pcb-normal-bank-fit-v1.zip", "spatial_zip": "mpi-spatial-score-v1.zip",
             "external_v2_zip": "mpi-pcb-external-evaluation-v2.zip"}
    for key, name in names.items():
        if fit.sha256_file(handoffs/name) != policy["source_sha256"][key]:
            raise ValueError(f"Source bundle differs: {name}")
    with zipfile.ZipFile(handoffs/names["normal_preparation_zip"]) as z:
        rows = json.loads(z.read("normal-fingerprints.json"))
    protected = {k: source["cohorts"][k] for k in ("development", "calibration", "validation")}
    if fit.digest(rows) != policy["source_sha256"]["normal_fingerprint_rows"]:
        raise ValueError("Normal pool fingerprint hash differs")
    for k, cohort in protected.items():
        if fit.digest(cohort) != policy["source_sha256"][k+"_rows"]:
            raise ValueError("Protected normal cohort changed")
    pool = derive_pool(rows, protected, prep.read_test_fingerprints(handoffs/names["external_v2_zip"]),
                       source["preparation"]["training_photo_ids"])
    expected = policy["candidate_pool"]
    if (len(pool["ids"]) != expected["count"] or fit.digest(pool["ids"]) != expected["ids_sha256"]
            or fit.digest(pool["rows"]) != expected["rows_sha256"]
            or pool["exclusion_counts"] != audit["exclusion_counts"]):
        raise ValueError("Candidate pool differs from reviewed audit")
    # Conservative transitive source fingerprint: unrelated Python edits also require review before resume.
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py")])
    identity = {"policy": policy, "pool_ids_sha256": fit.digest(pool["ids"]),
                "git_commit": subprocess.run(["git", "-C", str(project), "rev-parse", "HEAD"],
                                             check=True, capture_output=True, text=True).stdout.strip(),
                "pool_rows_sha256": fit.digest(pool["rows"]), "source_cohorts_sha256": fit.digest(source["cohorts"]),
                "baseline_ids": source["preparation"]["training_photo_ids"],
                "source_files_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}}
    return policy, source["config"], source["records"], pool, identity


def descriptor(means):
    means = np.asarray(means)
    if means.shape != (12, 768) or means.dtype != np.float32 or not np.isfinite(means).all():
        raise ValueError("Expected finite float32 region means (12,768)")
    norm = np.sqrt(np.sum(means * means, axis=1, dtype=np.float32))
    if not np.isfinite(norm).all() or np.any(norm <= 1e-12):
        raise ValueError("Zero or invalid normal feature norm")
    return ((means/norm[:, None]).reshape(-1)/np.float32(np.sqrt(12))).astype(np.float32)


def validate_vectors(values, count):
    if values.shape != (count, 9216) or values.dtype != np.float32 or not np.isfinite(values).all():
        raise ValueError("Invalid photo descriptor array")
    norms = np.linalg.norm(values.reshape(count, 12, 768), axis=2)
    if not np.allclose(norms, 1/np.sqrt(12), atol=1e-6, rtol=0):
        raise ValueError("Descriptor region normalization differs")


def select(ids, vectors, count=64, check=lambda: None):
    if len(set(ids)) != len(ids) or not 0 < count <= len(ids):
        raise ValueError("Invalid candidate IDs or selection count")
    validate_vectors(vectors, len(ids))
    # Canonical row order makes summation and exact tie resolution invariant to caller ordering.
    order = sorted(range(len(ids)), key=lambda i: rank(ids[i]))
    ids = [ids[i] for i in order]
    x = vectors[order].astype(np.float64)
    center = x.mean(axis=0)
    distance = np.sum((x-center)**2, axis=1)
    tie = lambda i: rank(ids[i], NAME)
    first = min(np.flatnonzero(distance == distance.min()), key=tie)
    chosen, nearest = [], np.full(len(ids), np.inf)
    for _ in range(count):
        check()
        index = first if not chosen else min(np.flatnonzero(nearest == nearest.max()), key=tie)
        chosen.append(int(index))
        nearest = np.minimum(nearest, np.sum((x-x[index])**2, axis=1))
        nearest[chosen] = -np.inf
    check()
    return [ids[i] for i in chosen]


def coverage(ids, vectors, selected, check=lambda: None):
    x = vectors.astype(np.float64).reshape(len(ids), 12, 768)
    nearest = np.full(len(ids), np.inf)
    regions = np.full((len(ids), 12), np.inf)
    for pid in selected:
        check()
        d = np.sum((x-x[ids.index(pid)])**2, axis=2)
        nearest = np.minimum(nearest, d.sum(axis=1))
        regions = np.minimum(regions, d)
    return {"mean_nearest_squared_distance": float(nearest.mean()), "maximum_nearest_squared_distance": float(nearest.max()),
            "region_mean_nearest_squared_distance": regions.mean(axis=0).tolist(),
            "region_maximum_nearest_squared_distance": regions.max(axis=0).tolist(),
            "scope": "descriptor coverage only; not defect detection performance"}


POLICY_SHA256 = "124991078c5800df844471800a661f1a0274ed58e6413402b1701784d1b66a0f"
