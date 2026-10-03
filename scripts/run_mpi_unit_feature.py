"""One feature-metric ablation: normal calibration/reload then observed comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from scripts import run_mpi_spatial_score as inputs
from scripts import mpi_normal_diversity_evaluation_protocol as comparison
from scripts import evaluate_mpi_normal_diversity as evaluation
from scripts.mpi_unit_feature_runtime import UnitRuntime

fit, ev, scoring = inputs.fit, inputs.ev, comparison.c.scoring
NAME = "mpi-unit-feature-v1"
GROUPS = ("development", "calibration", "test", "validation")


def prepare(project):
    source = inputs.prepare(project, project / "artifacts/handoff")
    policy = json.loads((project / "configs" / f"{NAME}.json").read_bytes())
    if fit.digest(policy) != POLICY_SHA256:
        raise ValueError("Frozen unit-feature policy changed; use a new reviewed version")
    comparison.validate_raw(source["reference"], source["cohorts"])
    baseline = source["contract"]["candidate_top1_q90"]
    replay = comparison.contract_replay(source["reference"], baseline)
    labels = [p["label"] for p in source["reference"]["test"]]
    if comparison.counts(comparison.matrix(source["reference"]), baseline, labels) != {
        "tp": 53, "fn": 2, "fp54": 3, "tn54": 51, "fp256": 14
    }:
        raise ValueError("Reviewed control performance changed")
    paths = sorted([*project.glob("scripts/*.py"), *project.glob("src/defectops/**/*.py"),
                    project / "configs" / f"{NAME}.json", project / "docs" / f"{NAME}.md"])
    identity = {"name": NAME, "policy": policy, "source": source["identity"],
                "checkpoint_sha256": ev.CHECKPOINT_SHA256,
                "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
                "baseline_replay": replay,
                "source_files_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                    p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}}
    return source, policy, identity


def read_sample(root, group, index, signature, photo, environment):
    name = f"photo-{index:03d}"
    receipt = fit.read_phase(root / group, name, signature, ["sample.json"])
    if receipt["photo_sha256"] != fit.digest(photo) or receipt["environment"] != environment:
        raise ValueError("Cached photo identity/environment differs")
    gate_path = (root / receipt["gate_path"]).resolve()
    gate_path.relative_to(root.resolve())
    if fit.sha256_file(gate_path) != receipt["gate_sha256"]:
        raise ValueError("Cached runtime gate changed")
    gate = json.loads(gate_path.read_bytes())
    if not gate["passed"] or gate["environment"] != environment:
        raise ValueError("Cached runtime gate did not pass")
    sample = json.loads((root / group / name / "sample.json").read_bytes())
    scoring.old.validate_sample(sample, photo)
    return sample


def score_photo(archive, photo, runtime, recovered, monitor):
    runtime.synchronize()
    started = time.perf_counter()
    sample = {k: photo[k] for k in ("key", "photo_id", "label")}
    sample["regions"] = []
    for row in photo["regions"]:
        monitor.check()
        image = (ev.read_tile(archive, row, recovered) if photo["key"].startswith(("good/", "defect/"))
                 else fit.checked_image(archive, row))
        try:
            raw, values = runtime.predict(image)
            # Existing validator rejects nonfinite/negative scores; map checked before aggregation.
            import numpy as np
            values = np.asarray(values)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid candidate anomaly map")
            sample["regions"].append({"region": row["region"], "member": row["member"],
                "native_raw": float(raw), "top1": float(fit.candidate_image_score(raw, values, fit.CANDIDATE))})
        finally:
            image.close()
        monitor.check()
    runtime.synchronize()
    sample["photo_scoring_seconds"] = time.perf_counter() - started
    scoring.old.validate_sample(sample, photo)
    return sample


def execute(root, archive, checkpoint, recovery_path, source, policy, identity, handoff,
            runtime_factory=UnitRuntime, monitor_factory=evaluation.Monitor):
    signature = fit.digest(identity)
    environment = source["records"]["reload.json"]["receipt"]["runtime"]
    cohorts = source["cohorts"]
    phase_photos = {g: cohorts[g] for g in GROUPS}
    phase_photos.update({"reload-" + g: cohorts[g] for g in GROUPS[:2]})
    baseline = source["contract"]["candidate_top1_q90"]
    with fit.locked_run(root):
        if not (root / "identity").exists() and any((root / g).exists() for g in (*phase_photos, "result", "transform")):
            raise ValueError("Outputs without identity; preserved")
        inputs.v.publish(root, "identity", signature, {"plan.json": identity})
        cached = {}
        for group, photos in phase_photos.items():
            (root / group).mkdir(exist_ok=True)
            for i, photo in enumerate(photos):
                if (root / group / f"photo-{i:03d}").exists():
                    cached[group, i] = read_sample(root, group, i, signature, photo, environment)
        normal_complete = all((g, i) in cached for g in GROUPS[:2] for i in range(len(cohorts[g])))
        reload_complete = all(("reload-" + g, i) in cached for g in GROUPS[:2] for i in range(len(cohorts[g])))
        if (root / "calibrated").exists() and not normal_complete:
            raise ValueError("Calibration without all normal evidence")
        if any(g in GROUPS[2:] for g, _ in cached) and not (reload_complete and (root / "reload-check").exists()):
            raise ValueError("Observed evaluation before normal reload verification")
        total = sum(map(len, phase_photos.values()))
        # Verify source caches even on a completed rerun. No GPU work for a completed result.
        if (root / "result").exists():
            names = ["report.json", "raw-scores.json", "reload-scores.json", "contract.json", "transform.json",
                     "plan.json", "attempts.json", "predictions.json", "bootstrap.json", "runtime-gates.json",
                     "sample-receipts.json", "handoff-manifest.json"]
            fit.read_phase(root, "result", signature, names)
            if len(cached) != total:
                raise ValueError("Completed report missing score caches")
            inputs.v.probe.export(root / "result", handoff)
            return json.loads((root / "result/report.json").read_bytes())
        attempts = evaluation.attempts(root)
        if any(x.get("error") for x in attempts):
            # Resource failures cannot be silently converted into a passing cost history.
            if any(x.get("resource_violation") for x in attempts):
                raise ValueError("Prior resource violation requires review; no automatic retry")
        prior = sum(x["seconds"] for x in attempts)
        if prior >= policy["resource_plan"]["comparison_seconds"]:
            raise ValueError("Cumulative time budget exhausted")
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root / "attempts"))
        fit.write_json(attempt / "started.json", {"identity": signature, "unix_seconds": time.time()})
        started = time.perf_counter()
        runtime, monitor, failure = None, None, None
        try:
            runtime = runtime_factory(source["config"])
            with monitor_factory(runtime, prior, started, policy) as monitor:
                def load_candidate():
                    runtime.load(checkpoint, source["records"]["fit.json"]["checkpoint_context"])
                    if runtime.environment() != environment:
                        raise ValueError("GPU environment differs from verified baseline")
                    monitor.check()
                    gate = ev.normal_probe(archive, source["preparation"], source["records"]["reload.json"]["photos"], runtime)
                    if not gate["passed"]:
                        raise ValueError("Original checkpoint normal replay failed")
                    monitor.check()
                    transform = runtime.activate()
                    inputs.v.publish(root, "transform", signature, {"transform.json": transform})
                    gate.update(environment=environment, transform=transform)
                    gate_path = attempt / f"gate-{time.time_ns()}.json"
                    fit.write_json(gate_path, gate)
                    monitor.check()
                    return gate_path

                gate_path = load_candidate()

                def collect(group, recovered=None):
                    rows = []
                    for i, photo in enumerate(phase_photos[group]):
                        if (group, i) not in cached:
                            sample = score_photo(archive, photo, runtime, recovered, monitor)
                            with fit.phase_stage(root / group, f"photo-{i:03d}") as stage:
                                fit.write_json(stage / "sample.json", sample)
                                fit.seal_phase(stage, f"photo-{i:03d}", signature,
                                    {"photo_sha256": fit.digest(photo), "environment": environment,
                                     "gate_path": gate_path.relative_to(root).as_posix(),
                                     "gate_sha256": fit.sha256_file(gate_path)}, ["sample.json"])
                            cached[group, i] = read_sample(root, group, i, signature, photo, environment)
                        rows.append(cached[group, i])
                        if (i + 1) % 10 == 0 or i + 1 == len(phase_photos[group]):
                            print(f"{group} {i + 1}/{len(phase_photos[group])}", flush=True)
                    return rows

                print("1/4 Normal-only Q90 calibration; fixed unit-feature metric.", flush=True)
                raw = {g: collect(g) for g in GROUPS[:2]}
                contract = comparison.c.calibrate(**raw)
                inputs.v.publish(root, "calibrated", signature, {"contract.json": contract})
                print("2/4 Reload original checkpoint, reconstruct candidate, verify all 297 normals.", flush=True)
                if not reload_complete:
                    gate_path = load_candidate()
                reloads = {g: collect("reload-" + g) for g in GROUPS[:2]}
                reload_check = comparison.c.reload_check(raw, reloads, contract)
                inputs.v.publish(root, "reload-check", signature, {"check.json": reload_check})
                print("3/4 Previously observed test109 and normal256; no mask reads or new final test.", flush=True)
                recovered = (ev.recovered_bytes(recovery_path, source["previous"])
                             if any(("test", i) not in cached for i in range(len(cohorts["test"]))) else None)
                raw.update(test=collect("test", recovered), validation=collect("validation"))
                comparison.validate_raw(raw, cohorts)
                point, pred = comparison.compare(source["reference"], raw, {"baseline": baseline, "candidate": contract})
                print("4/4 Fixed comparison and conditional normal-resampling check.", flush=True)
                stability = (comparison.bootstrap(source["reference"], raw, policy, monitor.check) if point["passed"]
                             else {"status": "not_run_fixed_gate_failed", "passed": False})
                monitor.check()
                # The checkpoint is never rewritten, and candidate remains a derived recipe only.
                if fit.sha256_file(checkpoint) != identity["checkpoint_sha256"]:
                    raise ValueError("Original checkpoint changed during execution")
                report = {"status": "unit_feature_exploratory_comparison_completed", "point_comparison": point,
                          "reload_check": reload_check, "stability": stability.get("summary", stability),
                          "decision": "needs_cost_review" if point["passed"] and stability["passed"] else "not_adopted",
                          "new_training": False, "coreset_reselected": False, "test_masks_decoded": 0,
                          "automatic_pass_allowed": False, "service_model_changed": False,
                          "independent_performance_claim_allowed": False,
                          "warm_benchmark_status": "not_run; passing candidate still requires same-hardware paired cost benchmark"}
        except BaseException as exc:
            failure = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            try:
                if runtime is not None:
                    runtime.release()
            except BaseException as exc:
                failure = failure or f"Release failed: {type(exc).__name__}: {exc}"
                raise
            finally:
                elapsed = time.perf_counter() - started
                if failure is None and prior + elapsed > policy["resource_plan"]["comparison_seconds"]:
                    failure = "Resource cap exceeded including model release"
                fit.write_json(attempt / "finished.json", {"seconds": elapsed,
                    "error": failure, "resource_violation": bool(failure and any(s in failure for s in ("cap exceeded", "OutOfMemory", "out of memory"))),
                    "memory": monitor.snapshot() if monitor is not None else None,
                    "scope": "runtime setup, replay, scoring, cache I/O, bootstrap, release; excludes input hashing and final export",
                    "completed_cached_photos": len(cached)})
        if failure is not None:
            raise RuntimeError(failure)
        attempts = evaluation.attempts(root)
        report["attempts_with_errors"] = sum(bool(a["error"]) for a in attempts)
        report["cumulative_seconds"] = sum(a["seconds"] for a in attempts)
        payloads = {"report.json": report, "raw-scores.json": raw, "reload-scores.json": reloads,
                    "contract.json": contract, "plan.json": identity, "attempts.json": attempts,
                    "predictions.json": pred, "bootstrap.json": stability,
                    "transform.json": json.loads((root / "transform/transform.json").read_bytes())}
        receipts = {f"{g}/photo-{i:03d}": fit.read_phase(root / g, f"photo-{i:03d}", signature, ["sample.json"])
                    for g, i in cached}
        payloads["sample-receipts.json"] = receipts
        payloads["runtime-gates.json"] = {r["gate_path"]: json.loads((root / r["gate_path"]).read_bytes())
                                           for r in receipts.values()}
        payloads["handoff-manifest.json"] = {n: hashlib.sha256(fit.canonical(v)).hexdigest() for n, v in payloads.items()}
        inputs.v.publish(root, "result", signature, payloads)
        inputs.v.probe.export(root / "result", handoff)
        return report


POLICY_SHA256 = "778969ef18c1958d3290c7d91a379d78189701dedcaf75c1ab71f5a7bd869acb"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--archive", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        project = args.project.resolve()
        source, policy, identity = prepare(project)
        checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
        if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256 or fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
            raise ValueError("Original checkpoint/archive identity differs")
        with zipfile.ZipFile(args.archive) as archive:
            report = execute(project / "artifacts/analyses" / NAME, archive, checkpoint,
                project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
                source, policy, identity, project / "artifacts/handoff" / f"{NAME}.zip")
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}. Existing files preserved.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
