"""MPI-PCB v2: verified derived tile, explicit amendments, unchanged two-model evaluation."""

from __future__ import annotations

import argparse
import gc
import importlib.metadata
import json
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import evaluate_mpi_pcb_external as v1
from scripts.mpi_pcb_external_v2_protocol import (
    RecoveredSource, amended_cohort, download_source_member, load_config, reconstruct, validate_prior,
)

NAME = "mpi-pcb-external-evaluation-v2"
COMPLETE = "mpi_pcb_external_evaluated_integrity_amended_v2"
REPORT_NAMES = ("input-lock.json", "recovery.json", "test-audit.json", "cohort-amendment.json",
                "calibration-replay.json", "realiad_pcb-scores.json", "visa_pcb3_v2-scores.json",
                "evaluation.json", "failure.json")


def package_reports(output, handoff):
    files = [output / name for name in REPORT_NAMES if (output / name).is_file()]
    manifest = {"schema_version": 2, "content": "JSON only; no original/reconstructed images, maps or models",
                "members": [{"name": path.name, "sha256": v1.sha256_file(path)} for path in files]}
    handoff.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(handoff, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.name)
        archive.writestr("handoff-manifest.json", json.dumps(manifest, indent=2, allow_nan=False))


def summarize(results, config, cohort):
    expected = cohort["photo_keys"]
    for key in v1.MODEL_ORDER:
        photos = results[key]["photos"]
        if ([photo["key"] for photo in photos] != expected
                or len(photos) != 109 or any(len(photo["regions"]) != 12 for photo in photos)
                or sum(photo["label"] == 0 for photo in photos) != 54
                or sum(photo["label"] == 1 for photo in photos) != 55):
            raise ValueError("Incomplete/amended scoring denominator or order mismatch")
    sensitivity = {key: {**result, "photos": [p for p in result["photos"] if p["key"] != "defect/0054"]}
                   for key, result in results.items()}
    return {
        "status": COMPLETE, "comparison_scope": "integrity_amended_subset_with_one_derived_replacement",
        "original_photos_per_model": 109, "normal_original_photos": 54,
        "manually_altered_original_photos": 55, "test_region_scores_per_model": 1308,
        "test_region_scores_total": 2616, "test_photo_scores_total": 218,
        "cohort_accounting": cohort,
        "models": {key: {"checkpoint_sha256": result["checkpoint_sha256"], "threshold": result["threshold"],
                         "metrics": v1.model_metrics(result["photos"], result["threshold"])}
                   for key, result in results.items()},
        "paired_comparison": v1.paired_comparison(results, config),
        "reconstructed_photo_results": {key: {field: next(p for p in result["photos"] if p["key"] == "defect/0054")[field]
                                               for field in ("score", "threshold", "decision")}
                                        for key, result in results.items()},
        "sensitivity_without_reconstructed_photo": {
            "normal_photos": 54, "defect_photos": 54, "total_photos": 108,
            "models": {key: v1.model_metrics(result["photos"], result["threshold"])
                       for key, result in sensitivity.items()},
            "paired_comparison": v1.paired_comparison(sensitivity, config),
            "interpretation": "Prespecified sensitivity, not a second candidate from which to select a better result.",
        },
        "test_data_previously_opened_for_integrity": True, "pixel_metrics_computed": False,
        "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
        "limitation": "Integrity-amended stress test, not the unchanged 55/55 benchmark. Repeated views of one board and manual alterations are not production defect validation.",
    }


def execute(source, output, config, v4, calibration, gate, recipes, gpu, prior, recovery):
    print("1/3 Re-auditing all test files with exactly one verified reconstructed tile...", flush=True)
    audit = v1.audit_test(source, v1.test_inventory(source.infolist()), v4["calibration_rows"])
    v1.write_json(output / "test-audit.json", audit)
    selected, cohort = amended_cohort(audit, prior["test-audit.json"], recovery)
    v1.write_json(output / "cohort-amendment.json", cohort)
    torch = gpu[0]
    replay = {"models": {}, "thresholds_unchanged": config["thresholds"], "test_scores_computed": 0,
              "test_data_previously_opened_for_integrity": True}
    print("2/3 Replaying 169 normal photos per model; frozen thresholds are NOT recalculated...", flush=True)
    for key in v1.MODEL_ORDER:
        v1._seed_everything(42, torch)
        recipe = {"base_model": recipes[key]["base_model"]}
        model = v1._model(recipe, Path(gate["models"][key]["checkpoint_path"]), recipes[key]["checkpoint_context"], gpu)
        try:
            current = v1.score_normal_cohort(source, v4["calibration_rows"], v4["calibration_photo_ids"], model, recipe, gpu)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        comparison = v1.compare_replay(calibration["models"][key]["photo_scores"], current)
        comparison["passed"] = max(comparison["maximum_absolute_region_score_difference"],
                                   comparison["maximum_absolute_photo_score_difference"]) <= config["calibration_replay_absolute_tolerance"]
        replay["models"][key] = {**comparison, "photo_scores": current}
    replay["passed"] = all(row["passed"] for row in replay["models"].values())
    v1.write_json(output / "calibration-replay.json", replay)
    if not replay["passed"]:
        return {"status": "normal_replay_blocked_v2_no_test_scores", "test_scores_computed": 0,
                "test_data_previously_opened_for_integrity": True, "automatic_pass_allowed": False}

    print("3/3 Scoring the SAME 54 normal + 55 altered photos with both frozen models...", flush=True)
    results = {}
    for key in v1.MODEL_ORDER:
        print(f"Frozen model: {key}", flush=True)
        v1._seed_everything(42, torch)
        recipe = {"base_model": recipes[key]["base_model"]}
        model = v1._model(recipe, Path(gate["models"][key]["checkpoint_path"]), recipes[key]["checkpoint_context"], gpu)
        try:
            photos = v1.score_test_photos(source, selected, model, recipe, gpu,
                                          config["thresholds"][key], output / "maps" / key)
        finally:
            del model
            gc.collect()
            torch.cuda.empty_cache()
        results[key] = {"checkpoint_sha256": gate["models"][key]["checkpoint_sha256"],
                        "threshold": config["thresholds"][key], "photos": photos}
        v1.write_json(output / f"{key}-scores.json", results[key])
    return summarize(results, config, cohort)


def run(project: Path, archive_path: Path) -> dict:
    started = time.perf_counter()
    output = project / "artifacts" / "evaluations" / NAME
    handoff = project / "artifacts" / "handoff" / f"{NAME}.zip"
    for path in (output, handoff):
        if path.exists():
            raise FileExistsError(f"Existing v2 output preserved; share the result/error before retrying: {path}")
    git = v1.git_state(project)
    config = load_config(project / "configs" / f"{NAME}.toml")
    prior_path = project / "artifacts" / "handoff" / "mpi-pcb-external-evaluation-v1.zip"
    prior = validate_prior(prior_path)
    protocol = v1.load_protocol(project / "configs" / "mpi-pcb-model-protocol-v1.toml")
    v4 = v1.validate_v4_report(project / "artifacts" / "data-audit" / "mpi-pcb-normal-preflight-v4.json")
    gate = v1.validate_model_preflight(project / "artifacts" / "data-audit" / "mpi-pcb-model-preflight-v1.json", project, protocol)
    calibration = v1.validate_calibration(project / "artifacts" / "calibrations" / "mpi-pcb-normal-calibration-v1.json", v4, protocol, config)
    md5, digest = v1.archive_hashes(archive_path)
    if md5 != v1.SOURCE_MD5 or digest != v1.EXPECTED_ARCHIVE_SHA256:
        raise ValueError("Original extracted.zip checksum changed; do not replace its contents")
    recipes = v1.model_contexts(project, protocol)
    print("Fetching one official aligned source image using bounded HTTP ranges (about 4.3 MB)...", flush=True)
    raw, provenance = download_source_member()
    recovered, recovery = reconstruct(raw, prior["test-audit.json"])
    recovery.update(provenance)
    gpu = v1._gpu_modules()
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    output.mkdir(parents=True, exist_ok=False)
    try:
        repair_dir = output / "recovery"
        repair_dir.mkdir()
        for name, content in (("official-IMG_0056.JPG.jpg", raw), ("grid1-3-defect-0054.png", recovered)):
            with (repair_dir / name).open("xb") as stream:
                stream.write(content)
        v1.write_json(output / "recovery.json", recovery)
        v1.write_json(output / "input-lock.json", {
            "created_at": datetime.now(timezone.utc).isoformat(), "git": git,
            "evaluation_config": config, "normal_protocol": protocol, "models": gate["models"],
            "archive_sha256": digest, "prior_handoff_sha256": config["prior_handoff_sha256"],
            "python": sys.version, "versions": {name: importlib.metadata.version(name) for name in
                                                   ("torch", "torchvision", "anomalib", "numpy", "Pillow")},
            "gpu": gpu[0].cuda.get_device_name(0), "test_data_previously_opened_for_integrity": True,
        })
        with zipfile.ZipFile(archive_path) as original:
            source = RecoveredSource(original, recovered)
            report = execute(source, output, config, v4, calibration, gate, recipes, gpu, prior, recovery)
        report.update({"git": git, "evaluation_config_sha256": config["config_sha256"],
                       "calibration_report_sha256": config["calibration_report_sha256"],
                       "prior_handoff_sha256": config["prior_handoff_sha256"], "archive_sha256": digest,
                       "elapsed_seconds": time.perf_counter() - started})
        v1.write_json(output / "evaluation.json", report)
    except Exception as exc:
        v1.write_json(output / "failure.json", {
            "status": "incomplete_no_valid_v2_comparison", "error": f"{type(exc).__name__}: {exc}",
            "git": git, "test_data_previously_opened_for_integrity": True, "test_scores_computed": None,
            "note": "Unknown partial progress; preserve original data, v1 and v2 outputs. No further automatic exemptions.",
            "automatic_pass_allowed": False,
        })
        package_reports(output, handoff)
        raise
    package_reports(output, handoff)
    return {**report, "output": str(output), "handoff": str(handoff), "handoff_sha256": v1.sha256_file(handoff)}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--project", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args(argv)
    try:
        report = run(args.project.resolve(), args.archive.resolve())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False))
    return 0 if report["status"] == COMPLETE else 2


if __name__ == "__main__":
    raise SystemExit(main())
