"""Fresh-process, resumable evaluation of the reviewed MPI target-normal bank."""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import math
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1] / "src")]

from scripts import analyze_mpi_pcb_maps as maps
from scripts import fit_mpi_pcb_normal_bank as fit
from scripts.mpi_pcb_external_protocol import compare_replay, decode_image, mask_details, model_metrics
from scripts.mpi_pcb_external_v2_protocol import MASK_MEMBER, MASK_SHA256, REPAIR_MEMBER, DAMAGED_SHA256

NAME = "mpi-pcb-normal-bank-evaluation-v1"
COMPLETE = "mpi_pcb_target_bank_evaluated_exploratory"
MODEL = "mpi_pcb_target_v1"
FIT_ZIP_SHA256 = "dd9e89e67345dc4e6076f66744d67f4772329fe0378e9619b350b6a4d6034dd5"
CHECKPOINT_SHA256 = "3d1635bdeefe89d9f8a677aa1c7fcf4966d085c4b1de9ca6d4f4382455d56a73"
DIAGNOSTICS_SHA256 = "0b0e1bacc217a2adf2274a6ddf50279b12a9408547123141cf915de45c994e3a"
THRESHOLD = 46.839376955497556
EXPECTED_COUNTS = (109, 54, 55)
PROBE = {"normal_photos": 8, "warmup_tiles": 5, "score_tolerance": 1e-4,
         "maximum_mean_tile_ms": 1000.0, "maximum_p95_tile_ms": 2000.0,
         "purpose": "detect severe runtime regression before test, NOT a service SLA"}
REPORT_FILES = ("evaluation.json", "scores.json", "cases.json", "normal-probe.json", "input-lock.json",
                "source-fit.json", "source-evaluation.json", "source-localization-summary.json")


def checked_bundle(path, expected_sha):
    if fit.sha256_file(path) != expected_sha:
        raise ValueError(f"Reviewed handoff checksum differs: {path}")
    with zipfile.ZipFile(path) as source:
        names = source.namelist()
        if len(names) != len(set(names)):
            raise ValueError("Duplicate handoff members")
        manifest = json.loads(source.read("handoff-manifest.json"))
        entries = manifest.get("members")
        expected = {r["name"]: r["sha256"] for r in entries} if entries is not None else manifest
        if (set(expected) != set(names) - {"handoff-manifest.json"}
                or (entries is not None and len(entries) != len(expected))):
            raise ValueError("Handoff manifest members differ")
        reports = {}
        for name, digest in expected.items():
            raw = source.read(name)
            if hashlib.sha256(raw).hexdigest() != digest:
                raise ValueError("Handoff payload checksum differs")
            if name.endswith(".json"):
                reports[name] = json.loads(raw)
    return reports


def load_fit(path, project, config, preparation):
    records = checked_bundle(path, FIT_ZIP_SHA256)
    report = records["fit.json"]
    if (report["status"] != fit.FINAL_STATUS or report["checkpoint_sha256"] != CHECKPOINT_SHA256
            or report["image_threshold"] != THRESHOLD or report["automatic_pass_allowed"] is not False
            or report["confirmatory_claim_allowed"] is not False
            or records["input-lock.json"]["config"] != config
            or records["input-lock.json"]["execution"] != fit.execution_identity(project, config)):
        raise ValueError("Reviewed fit, code, config or threshold differs")
    for phase, receipt in (("fit", records["fit-metadata.json"]),
                           ("calibration", records["calibration.json"]["receipt"]),
                           ("reload", records["reload.json"]["receipt"])):
        unsigned = dict(receipt)
        claimed = unsigned.pop("receipt_sha256")
        if claimed != fit.digest(unsigned) or claimed != report["phase_receipts"][phase]:
            raise ValueError("Fit phase receipt differs")
    for phase in ("calibration", "reload"):
        data = records[f"{phase}.json"]
        fit.validate_photos(data["photos"], preparation["calibration_photo_ids"])
        if (fit.digest(data["photos"]) != data["receipt"]["files"]["photos.json"]
                or fit.threshold(data["photos"]) != THRESHOLD):
            raise ValueError("Normal scores or frozen threshold differs")
    if fit.reload_differences(records["calibration.json"]["photos"], records["reload.json"]["photos"],
                             preparation["calibration_photo_ids"]) != report["reload_verification"]:
        raise ValueError("Fit reload verification differs")
    return records


def load_prior(project, handoff_dir):
    previous = maps.load_evidence(project, handoff_dir / "mpi-pcb-external-evaluation-v2.zip")
    diagnosis = checked_bundle(handoff_dir / "mpi-pcb-map-diagnostics-v1.zip", DIAGNOSTICS_SHA256)
    cases = diagnosis["cases.json"]
    if (diagnosis["source-evaluation.json"] != previous["evaluation.json"]
            or maps.summarize(cases) != diagnosis["analysis.json"]["summary"]
            or [c["key"] for c in cases] != previous["cohort-amendment.json"]["photo_keys"]):
        raise ValueError("Reviewed diagnosis and prior evaluation disagree")
    for index, case in enumerate(cases):
        if case["localization_eligible"] != (case["key"] != "defect/0006"):
            raise ValueError("Location exception changed")
        for model in maps.MODEL_KEYS:
            old = previous[f"{model}-scores.json"]["photos"][index]
            if any(case["models"][model][f] != old[f] for f in ("score", "threshold", "decision")):
                raise ValueError("Prior classification/diagnosis mismatch")
    return previous, diagnosis


def run_identity(project):
    # Hash Python implementation dependencies as well as the locked model recipe.
    paths = sorted(set(project.glob("scripts/*mpi_pcb*.py")) | set(project.glob("src/defectops/*.py")))
    return {"schema_version": 1, "fit_zip_sha256": FIT_ZIP_SHA256,
            "prior_evaluation_sha256": maps.EVALUATION_SHA256, "diagnostics_sha256": DIAGNOSTICS_SHA256,
            "checkpoint_sha256": CHECKPOINT_SHA256, "archive_sha256": fit.EXPECTED_ARCHIVE_SHA256,
            "threshold": THRESHOLD, "probe": PROBE, "automatic_pass_allowed": False,
            "source_files_lf_sha256": {p.relative_to(project).as_posix(): hashlib.sha256(
                p.read_bytes().replace(b"\r\n", b"\n")).hexdigest() for p in paths}}


def latency_summary(values):
    if not values or not all(math.isfinite(v) and v >= 0 for v in values):
        raise ValueError("Missing/nonfinite latency measurements")
    return {"count": len(values), "mean": float(np.mean(values)),
            "median": float(np.median(values)), "p95_higher": float(np.quantile(values, .95, method="higher")),
            "maximum": float(max(values))}


def normal_probe(source, preparation, expected, runtime):
    ids = preparation["calibration_photo_ids"][:PROBE["normal_photos"]]
    rows = preparation["calibration_rows"][:len(ids) * len(fit.REGIONS)]
    for row in rows[:PROBE["warmup_tiles"]]:
        with fit.checked_image(source, row) as image:
            runtime.predict(image)
    runtime.synchronize()
    runtime.reset_peak()
    started = time.perf_counter()
    photos, latencies = [], []
    # Check cost after each complete photo: the known ~9 s/tile regression must
    # not spend another 15 minutes finishing this small gate.
    for index, photo_id in enumerate(ids):
        regions = []
        for row in rows[index * len(fit.REGIONS):(index + 1) * len(fit.REGIONS)]:
            runtime.synchronize()
            before = time.perf_counter()
            with fit.checked_image(source, row) as image:
                raw, values = runtime.predict(image)
            values = np.asarray(values)
            if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
                raise ValueError("Invalid normal probe map")
            score = fit.candidate_image_score(raw, values, fit.CANDIDATE)
            runtime.synchronize()
            latencies.append((time.perf_counter() - before) * 1000)
            regions.append({"region": row["region"], "member": row["member"], "score": score})
        photos.append({"photo_id": photo_id, "score": max(r["score"] for r in regions), "regions": regions})
        latency = latency_summary(latencies)
        comparison = compare_replay(expected[:len(photos)], photos)
        if (max(comparison["maximum_absolute_region_score_difference"], comparison["maximum_absolute_photo_score_difference"]) > PROBE["score_tolerance"]
                or latency["mean"] > PROBE["maximum_mean_tile_ms"] or latency["p95_higher"] > PROBE["maximum_p95_tile_ms"]):
            break
    timing = {"seconds": time.perf_counter() - started, "memory": runtime.memory(),
              "tile_latency_ms": latency_summary(latencies),
              "scope": "checked ZIP read, PNG decode, preprocessing, GPU inference, map transfer, score; excludes warmup/model load"}
    delta = max(comparison["maximum_absolute_region_score_difference"],
                comparison["maximum_absolute_photo_score_difference"])
    latency = timing["tile_latency_ms"]
    reasons = []
    if delta > PROBE["score_tolerance"]:
        reasons.append("normal_score_drift")
    if not all(math.isfinite(latency[k]) for k in ("mean", "p95_higher")):
        reasons.append("invalid_timing")
    elif latency["mean"] > PROBE["maximum_mean_tile_ms"] or latency["p95_higher"] > PROBE["maximum_p95_tile_ms"]:
        reasons.append("severe_latency_regression")
    return {"status": "normal_probe_passed" if not reasons else "normal_probe_blocked",
            "passed": not reasons and len(photos) == len(ids), "reasons": reasons, "policy": PROBE,
            "planned_photo_ids": ids, "photo_ids": [p["photo_id"] for p in photos],
            "comparison": comparison, "timing": timing, "photos": photos,
            "threshold_recalculated": False, "frozen_threshold": THRESHOLD,
            "test_images_decoded_this_probe": 0, "test_scores_computed_this_probe": 0}


def recovered_bytes(path, previous):
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != previous["recovery.json"]["recovered_png_sha256"]:
        raise ValueError("Existing recovered tile changed; no network download/reconstruction attempted")
    with decode_image(raw) as image:
        if hashlib.sha256(image.tobytes()).hexdigest() != maps.REPAIRED_RGB_SHA256:
            raise ValueError("Recovered tile RGB differs")
    return raw


def read_tile(source, row, recovered):
    raw = source.read(row["member"])
    if row["member"] == REPAIR_MEMBER:
        if hashlib.sha256(raw).hexdigest() != DAMAGED_SHA256:
            raise ValueError("Original damaged tile changed")
        raw = recovered
    if hashlib.sha256(raw).hexdigest() != row["member_sha256"]:
        raise ValueError("Test image bytes changed")
    image = decode_image(raw)
    if hashlib.sha256(image.tobytes()).hexdigest() != row["decoded_rgb_sha256"]:
        image.close()
        raise ValueError("Test pixels changed")
    return image


def read_masks(source, photo):
    masks = []
    for row in photo["regions"]:
        if row["mask_member"] is None:
            masks.append(np.zeros((256, 256), dtype=np.bool_))
            continue
        raw = source.read(row["mask_member"])
        if hashlib.sha256(raw).hexdigest() != row["mask_sha256"]:
            raise ValueError("Test mask bytes changed")
        if row["mask_member"] == MASK_MEMBER:
            if row["mask_sha256"] != MASK_SHA256 or "mask" in row:
                raise ValueError("Known damaged-mask exception changed")
            masks.append(None)
            continue
        if mask_details(raw) != row["mask"]:
            raise ValueError("Mask semantics changed")
        with Image.open(io.BytesIO(raw)) as image:
            masks.append(np.asarray(image.convert("L")) > 0)
    return masks


def create_sample(source, photo, recovered, runtime, stage, probe_sha):
    started = time.perf_counter()
    values, rows, times = [], [], []
    for row in photo["regions"]:
        runtime.synchronize()
        before = time.perf_counter()
        with read_tile(source, row, recovered) as image:
            raw_score, anomaly_map = runtime.predict(image)
        array = np.asarray(anomaly_map, dtype=np.float32)
        if array.shape != (256, 256) or not np.isfinite(array).all() or np.any(array < 0):
            raise ValueError("Invalid anomaly map")
        score = fit.candidate_image_score(raw_score, array, fit.CANDIDATE)
        runtime.synchronize()
        times.append((time.perf_counter() - before) * 1000)
        values.append(array)
        rows.append({k: row[k] for k in ("region", "member")} | {"score": score})
    photo_scoring_ms = (time.perf_counter() - started) * 1000
    score = max(r["score"] for r in rows)
    result = {k: photo[k] for k in ("key", "photo_id", "label")}
    result.update(score=score, threshold=THRESHOLD, regions=rows, automatic_pass_allowed=False,
                  decision="REVIEW_REQUIRED" if score >= THRESHOLD else "BELOW_THRESHOLD_NOT_AUTO_PASS")
    before = time.perf_counter()
    masks = read_masks(source, photo)
    location = maps.describe_photo(result, values, masks, photo["key"] != "defect/0006")
    diagnostic_ms = (time.perf_counter() - before) * 1000
    before = time.perf_counter()
    with (stage / "maps.npz").open("xb") as stream:
        np.savez_compressed(stream, anomaly_maps=np.stack(values), regions=np.array(fit.REGIONS))
    timing = {"tile_ms": times, "photo_scoring_ms": photo_scoring_ms,
              "mask_and_localization_ms": diagnostic_ms, "map_write_ms": (time.perf_counter() - before) * 1000,
              "total_sample_ms": (time.perf_counter() - started) * 1000,
              "scope": "total includes checked image reads, inference, score, masks, localization and NPZ write; excludes model load and JSON/cache validation; not HTTP"}
    fit.write_json(stage / "sample.json", {"photo": result, "location": location, "timing": timing})
    return {"process_id": os.getpid(), "normal_probe_sha256": probe_sha, "audit_photo_sha256": fit.digest(photo)}


def read_sample(root, name, signature, photo):
    receipt = fit.read_phase(root, name, signature, ["maps.npz", "sample.json"])
    if receipt["audit_photo_sha256"] != fit.digest(photo):
        raise ValueError("Cached sample belongs to a different audited photo")
    result = json.loads((root / name / "sample.json").read_bytes())
    p = result["photo"]
    if (any(p[k] != photo[k] for k in ("key", "photo_id", "label"))
            or p["threshold"] != THRESHOLD or p["automatic_pass_allowed"] is not False
            or [(r["region"], r["member"]) for r in p["regions"]] != [(r["region"], r["member"]) for r in photo["regions"]]
            or p["score"] != max(r["score"] for r in p["regions"])
            or p["decision"] != ("REVIEW_REQUIRED" if p["score"] >= THRESHOLD else "BELOW_THRESHOLD_NOT_AUTO_PASS")):
        raise ValueError("Cached decision, identity or threshold differs")
    with np.load(root / name / "maps.npz", allow_pickle=False) as archive:
        arrays = archive["anomaly_maps"]
        if (set(archive.files) != {"anomaly_maps", "regions"} or arrays.shape != (12, 256, 256)
                or arrays.dtype != np.float32 or archive["regions"].tolist() != fit.REGIONS):
            raise ValueError("Cached map format differs")
        for array, row in zip(arrays, p["regions"], strict=True):
            maps.region_statistics(array, None, row["score"])
    return result, receipt


def paired_changes(old, new):
    if [(p["key"], p["label"]) for p in old] != [(p["key"], p["label"]) for p in new]:
        raise ValueError("Paired cohort differs")
    result = {}
    for label, name in ((1, "defect"), (0, "normal")):
        pairs = [(a, b) for a, b in zip(old, new, strict=True) if a["label"] == label]
        keys = {"both_review": [], "old_only_review": [], "new_only_review": [], "neither_review": []}
        for a, b in pairs:
            af, bf = a["score"] >= a["threshold"], b["score"] >= b["threshold"]
            key = "both_review" if af and bf else "old_only_review" if af else "new_only_review" if bf else "neither_review"
            keys[key].append(a["key"])
        result[name] = {"count": len(pairs), "counts": {k: len(v) for k, v in keys.items()}, "photo_keys": keys}
    return result


def summarize(samples, previous, diagnosis):
    photos = [r[0]["photo"] for r in samples]
    expected = previous["cohort-amendment.json"]["photo_keys"]
    if ([p["key"] for p in photos] != expected
            or (len(photos), sum(p["label"] == 0 for p in photos), sum(p["label"] == 1 for p in photos)) != EXPECTED_COUNTS):
        raise ValueError("Incomplete 54/55 test cohort")
    cases = copy.deepcopy(diagnosis["cases.json"])
    for case, (sample, _) in zip(cases, samples, strict=True):
        case["models"][MODEL] = sample["location"]
        case["prior_pair_group"] = case["group"]
        flagged = sample["photo"]["decision"] == "REVIEW_REQUIRED"
        case["group"] = ("tp" if flagged else "fn") if case["label"] else ("fp" if flagged else "tn")
        case["cards"] = []  # No new cards generated; old two-model cards do not depict this model.
    # Reuse the exact existing photo-mean/median summary logic without changing its global model list.
    alias_cases = [{**c, "models": {maps.MODEL_KEYS[0]: c["models"][MODEL],
                                   maps.MODEL_KEYS[1]: c["models"][MODEL]}} for c in cases]
    new_location = maps.summarize(alias_cases)["models"][maps.MODEL_KEYS[0]]
    all_photos = {key: previous[f"{key}-scores.json"]["photos"] for key in maps.MODEL_KEYS} | {MODEL: photos}
    metrics = {key: model_metrics(ps, ps[0]["threshold"]) for key, ps in all_photos.items()}
    reduced = {key: [p for p in ps if p["key"] != "defect/0054"] for key, ps in all_photos.items()}
    pids = sorted({r["process_id"] for _, r in samples})
    report = {"status": COMPLETE, "models": metrics, "frozen_mpi_threshold": THRESHOLD,
              "original_photos": len(photos), "normal_photos": EXPECTED_COUNTS[1], "defect_photos": EXPECTED_COUNTS[2],
              "new_model_region_scores": len(photos) * len(fit.REGIONS), "previous_models_rescored": False,
              "paired_changes_vs_previous": {key: paired_changes(ps, photos) for key, ps in all_photos.items() if key != MODEL},
              "sensitivity_without_reconstructed_photo": {
                  "original_photos": len(photos) - 1, "normal_photos": EXPECTED_COUNTS[1], "defect_photos": EXPECTED_COUNTS[2] - 1,
                  "models": {key: model_metrics(ps, ps[0]["threshold"]) for key, ps in reduced.items()},
                  "paired_changes": {key: paired_changes(ps, reduced[MODEL]) for key, ps in reduced.items() if key != MODEL}},
              "localization": {"defect_photo_denominator": EXPECTED_COUNTS[2] - 1, "excluded_photos": ["defect/0006"],
                               "models": diagnosis["analysis.json"]["summary"]["models"] | {MODEL: new_location},
                               "scope": "same tie-aware top1% attribution; per-photo mean then median, not pooled Pixel AUROC"},
              "latency_ms": {"tile_scoring": latency_summary([t for s, _ in samples for t in s["timing"]["tile_ms"]]),
                             **{key: latency_summary([s["timing"][key] for s, _ in samples]) for key in
                                ("photo_scoring_ms", "mask_and_localization_ms", "map_write_ms", "total_sample_ms")},
                             "process_ids": pids, "mixed_attempts": len({r["normal_probe_sha256"] for _, r in samples}) > 1,
                             "scope": "recorded per-photo work from successful attempts; no model load/HTTP; not a controlled warm comparison"},
              "test_previously_observed": True, "automatic_pass_allowed": False,
              "confirmatory_claim_allowed": False, "thresholds_changed": False, "training_performed": False,
              "pixel_decision_threshold_used": False,
              "limitations": ["Exploratory repeated views/manual alterations of one board; not independent production validation.",
                              "Training source, sample count and bank size differ; not a single-factor causal comparison.",
                              "109-photo primary and 108-photo sensitivity are both reported, not selected by better results."]}
    return report, photos, cases


def export_reports(output, handoff):
    payloads = {name: (output / name).read_bytes() for name in REPORT_FILES}
    payloads["handoff-manifest.json"] = fit.canonical({name: hashlib.sha256(raw).hexdigest() for name, raw in payloads.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as source:
            if set(source.namelist()) != set(payloads) or len(source.namelist()) != len(payloads) or any(source.read(k) != v for k, v in payloads.items()):
                raise ValueError("Existing evaluation handoff differs; preserved")
        return
    handoff.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=".mpi-evaluation-", suffix=".zip", dir=handoff.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with zipfile.ZipFile(temporary, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for key, raw in payloads.items():
                archive.writestr(key, raw)
        os.link(temporary, handoff)
    finally:
        temporary.unlink(missing_ok=True)


def execute(root, source, recovered_path, checkpoint, config, preparation, records, previous, diagnosis,
            identity, git, handoff, runtime_factory=fit.TorchRuntime):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        lock = root / "input-lock.json"
        if lock.exists():
            if json.loads(lock.read_bytes())["identity"] != identity:
                raise ValueError("Existing evaluation identity differs; outputs preserved")
        else:
            # Atomic publication using the already established staged-directory pattern.
            if (root / "identity").exists():
                if json.loads((root / "identity/input-lock.json").read_bytes())["identity"] != identity:
                    raise ValueError("Interrupted identity differs; preserved")
            else:
                if (root / "samples").exists() or (root / "result").exists():
                    raise ValueError("Evaluation artifacts without input lock; preserved")
                with fit.phase_stage(root, "identity") as stage:
                    fit.write_json(stage / "input-lock.json", {"identity": identity, "git": git})
            os.link(root / "identity/input-lock.json", lock)
        samples_root = root / "samples"
        samples_root.mkdir(exist_ok=True)
        audit = {p["key"]: p for p in previous["test-audit.json"]["photos"]}
        cohort = [audit[k] for k in previous["cohort-amendment.json"]["photo_keys"]]
        cached = {}
        for i, photo in enumerate(cohort, 1):
            if (samples_root / f"photo-{i:03d}").exists():
                cached[i] = read_sample(samples_root, f"photo-{i:03d}", signature, photo)
        if (root / "result").exists():
            fit.read_phase(root, "result", signature, list(REPORT_FILES))
            if len(cached) != len(cohort):
                raise ValueError("Completed report has missing cached photos")
            report = json.loads((root / "result/evaluation.json").read_bytes())
            export_reports(root / "result", handoff)
            return report
        attempt = Path(tempfile.mkdtemp(prefix="attempt-", dir=root))
        runtime = None
        test_opening_started = False
        try:
            probe = None
            if len(cached) < len(cohort):
                print("1/3 Loading the saved MPI model in this evaluation process; NO fitting...", flush=True)
                before = time.perf_counter()
                runtime = runtime_factory(config)
                runtime.load(checkpoint, records["fit.json"]["checkpoint_context"])
                environment = runtime.environment()
                expected_environment = records["reload.json"]["receipt"]["runtime"]
                if environment != expected_environment:
                    raise ValueError("GPU/runtime environment differs from reviewed reload; review before scoring")
                load_seconds = time.perf_counter() - before
                print("2/3 Checking fixed normal photos for score drift and severe latency regression...", flush=True)
                probe = normal_probe(source, preparation, records["reload.json"]["photos"], runtime)
                probe.update(process_id=os.getpid(), runtime=environment, model_setup_seconds=load_seconds)
                fit.write_json(attempt / "normal-probe.json", probe)
                if not probe["passed"]:
                    raise ValueError(f"Normal probe blocked BEFORE new test scoring: {probe['reasons']}")
                # Read the derived TEST tile only AFTER the normal gate; never download/reconstruct it again.
                test_opening_started = True
                recovered = recovered_bytes(recovered_path, previous)
                runtime.reset_peak()
                print(f"3/3 Scoring fixed test cohort; reusing {len(cached)}/{len(cohort)} verified samples...", flush=True)
                probe_sha = fit.sha256_file(attempt / "normal-probe.json")
                for i, photo in enumerate(cohort, 1):
                    if i not in cached:
                        name = f"photo-{i:03d}"
                        with fit.phase_stage(samples_root, name) as stage:
                            metadata = create_sample(source, photo, recovered, runtime, stage, probe_sha)
                            fit.seal_phase(stage, name, signature, metadata, ["maps.npz", "sample.json"])
                        cached[i] = read_sample(samples_root, name, signature, photo)
                    if i % 10 == 0 or i == len(cohort):
                        print(f"MPI target-bank test {i}/{len(cohort)} original photos", flush=True)
                fit.write_json(attempt / "runtime.json", {"process_id": os.getpid(), "test_phase_memory": runtime.memory(),
                                                          "newly_computed_samples": sum(r[1]["normal_probe_sha256"] == probe_sha for r in cached.values())})
            samples = [cached[i] for i in range(1, len(cohort) + 1)]
            report, photos, cases = summarize(samples, previous, diagnosis)
            # Retain every contributing attempt, including resumed attempts with different memory states.
            probe_hashes = {r["normal_probe_sha256"] for _, r in samples}
            probes = [json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/normal-probe.json")) if fit.sha256_file(p) in probe_hashes]
            if len(probes) != len(probe_hashes) or not all(p["passed"] for p in probes):
                raise ValueError("Cached samples missing their passing normal probe")
            report.update(git=git, execution_sha256=signature, checkpoint_sha256=CHECKPOINT_SHA256,
                          prior_evaluation_sha256=maps.EVALUATION_SHA256,
                          runtime_attempts=[json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/runtime.json"))],
                          interrupted_attempts=[json.loads(p.read_bytes()) for p in sorted(root.glob("attempt-*/failure.json"))],
                          runtime_memory_note="Completed attempts record test-phase peaks. Interrupted attempts record memory at failure when available; hard-killed attempts may have no memory record.")
            with fit.phase_stage(root, "result") as stage:
                data = {"evaluation.json": report, "scores.json": photos, "cases.json": cases,
                        "normal-probe.json": {"contributing_attempts": probes},
                        "input-lock.json": json.loads(lock.read_bytes()), "source-fit.json": records["fit.json"],
                        "source-evaluation.json": previous["evaluation.json"],
                        "source-localization-summary.json": diagnosis["analysis.json"]["summary"]}
                for name, value in data.items():
                    fit.write_json(stage / name, value)
                fit.seal_phase(stage, "result", signature, {}, list(REPORT_FILES))
            export_reports(root / "result", handoff)
            return report
        except BaseException as exc:
            failure = {"status": "incomplete_evaluation_preserved", "error": f"{type(exc).__name__}: {exc}",
                       "completed_cached_photos": len(cached), "process_id": os.getpid(),
                       "test_opening_started_this_attempt": test_opening_started,
                       "test_scores_computed_this_attempt": None if test_opening_started else 0,
                       "automatic_pass_allowed": False,
                       "note": "Partial progress may exist; no valid full comparison. Preserve data/models/attempts."}
            if runtime is not None:
                try:
                    failure["memory_at_failure"] = runtime.memory()
                except Exception:
                    failure["memory_at_failure"] = None
            fit.write_json(attempt / "failure.json", failure)
            print(f"Failure details: {attempt / 'failure.json'}", file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                runtime.release()


def run(project, archive_path):
    handoff_dir = project / "artifacts/handoff"
    config = fit.load_config(project / "configs/mpi-pcb-normal-bank-pilot-v1.toml")
    preparation = fit.load_preparation(handoff_dir / "mpi-pcb-normal-bank-preparation-v1.zip", config)
    records = load_fit(handoff_dir / "mpi-pcb-normal-bank-fit-v1.zip", project, config, preparation)
    previous, diagnosis = load_prior(project, handoff_dir)
    checkpoint = project / "artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt"
    if fit.sha256_file(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Checkpoint bytes differ from the reviewed fit")
    print("Checking source archive identity; no model or threshold changes...", flush=True)
    if fit.sha256_file(archive_path) != fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original extracted.zip checksum differs")
    git = fit.git_state(project)
    with zipfile.ZipFile(archive_path) as source:
        return execute(project / "artifacts/evaluations" / NAME, source,
                       project / "artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png",
                       checkpoint, config, preparation, records, previous, diagnosis, run_identity(project), git,
                       handoff_dir / f"{NAME}.zip")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.project.resolve(), args.archive.resolve())
        print(json.dumps({key: report[key] for key in ("status", "models", "localization", "latency_ms")},
                         ensure_ascii=False, indent=2, allow_nan=False))
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
