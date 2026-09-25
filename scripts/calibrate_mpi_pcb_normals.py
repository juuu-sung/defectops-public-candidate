"""Score only the sealed MPI-PCB training normals with two frozen models."""

from __future__ import annotations

import argparse
import gc
import hashlib
import io
import json
import math
import sys
import zipfile
from pathlib import Path
from typing import Any, Callable

from PIL import Image, ImageFile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectops.padim_experiment import git_state
from defectops.realiad_followup_development import _seed_everything, candidate_image_score
from defectops.realiad_followup_final import _gpu_modules, _model, _predict
from defectops.realiad_followup_protocol import load_realiad_followup_protocol
from defectops.visa_pcb_normal_bank_development import load_execution
from scripts.audit_mpi_pcb_archive import SOURCE_MD5, archive_hashes, audit_members
from scripts.audit_mpi_pcb_selected_normals import inspect_png
from scripts.preflight_mpi_pcb_models import (
    EXPECTED_ARCHIVE_SHA256,
    EXPECTED_IDS_SHA256,
    EXPECTED_V4_SHA256,
    checked_model_paths,
    load_protocol,
    normal_threshold,
    photo_score,
    sha256_file,
    validate_v4_report,
)
from scripts.preflight_mpi_pcb_normals import normal_members


EXPECTED_MODEL_PREFLIGHT_SHA256 = "b7d328daa3b2ca0e9c6fe50bffacae4d3bfddbf0947e39f1f8c27492b6bab2bf"
MODEL_ORDER = ("realiad_pcb", "visa_pcb3_v2")
CANDIDATE = {"image_score": "mean_of_top_map_fraction", "top_fraction": 0.01}
REGIONS = [f"grid{row}-{column}_s1024" for row in range(1, 5) for column in range(1, 4)]


def validate_model_preflight(path: Path, project: Path, protocol: dict[str, Any]) -> dict[str, Any]:
    """Bind this calibration to the Windows checkpoint gate, not a new model."""
    if sha256_file(path) != EXPECTED_MODEL_PREFLIGHT_SHA256:
        raise ValueError("Reviewed MPI-PCB model preflight JSON checksum differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    paths = checked_model_paths(project, protocol)
    if not (
        report.get("status") == "mpi_pcb_model_protocol_locked_test_sealed"
        and report.get("protocol_config_sha256") == protocol["config_sha256"]
        and report.get("normal_preflight_v4_sha256") == EXPECTED_V4_SHA256
        and report.get("normal_photo_ids_sha256") == EXPECTED_IDS_SHA256
        and report.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and report.get("models") == paths
        and report.get("normal_original_photos") == 169
        and report.get("regions_per_photo") == 12
        and report.get("image_score") == "mean_of_top_1pct_anomaly_map_then_max_of_12_regions"
        and report.get("threshold") == "model_specific_95pct_higher_of_169_normal_photo_scores_not_yet_computed"
        and all(report.get(field) == 0 for field in (
            "normal_scores_computed", "test_images_decoded", "test_masks_decoded", "test_scores_computed",
        ))
        and report.get("automatic_pass_allowed") is False
    ):
        raise ValueError("MPI-PCB model preflight does not match the reviewed test-sealed gate")
    return report


def model_contexts(project: Path, protocol: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Load existing source-fit metadata and check the selected inference recipe."""
    real_protocol = load_realiad_followup_protocol(
        project / "configs" / "realiad-followup-preregistration-v1.toml"
    )
    visa_execution = load_execution(project / "configs" / "visa-pcb-normal-bank-development-v2.toml", 2)
    real_result = json.loads((
        project / "artifacts" / "realiad-followup-development-v1" / "pcb" / "category-result.json"
    ).read_text(encoding="utf-8"))
    visa_fit = json.loads((
        project / "artifacts" / "visa-pcb-normal-bank-development-v2" / "pcb3" / "fit.json"
    ).read_text(encoding="utf-8"))
    expected = protocol["models"]
    for key, source in (("realiad_pcb", real_protocol["base_model"]), ("visa_pcb3_v2", visa_execution)):
        locked = expected[key]
        if not all(source[field] == locked[field] for field in (
            "backbone", "image_size", "num_neighbors",
        )) or source["coreset_sampling_ratio"] != 0.10:
            raise ValueError(f"{key} inference recipe differs from the selected frozen model")
    if not (
        real_result["layer_groups"]["layer1-layer2"]["layers"] == expected["realiad_pcb"]["layers"]
        and visa_fit.get("layers") == expected["visa_pcb3_v2"]["layers"]
        and visa_fit.get("execution_config_sha256") == visa_execution["config_sha256"]
    ):
        raise ValueError("Source fits differ from the locked layer/configuration recipe")
    return {
        "realiad_pcb": {
            "base_model": real_protocol["base_model"],
            "checkpoint_context": {
                "category": "realiad_pcb", "seed": 42,
                "manifest_sha256": real_result["development_manifest_sha256"],
                "execution_config_sha256": real_result["execution_config_sha256"],
            },
        },
        "visa_pcb3_v2": {
            "base_model": {
                "backbone": visa_execution["backbone"],
                "image_size": visa_execution["image_size"],
                "coreset_sampling_ratio": visa_execution["coreset_sampling_ratio"],
                "num_neighbors": visa_execution["num_neighbors"],
            },
            "checkpoint_context": {
                "category": "pcb3", "seed": 42,
                "manifest_sha256": visa_fit["plan_sha256"],
                "execution_config_sha256": visa_fit["execution_config_sha256"],
            },
        },
    }


def score_normal_cohort(
    source: zipfile.ZipFile,
    rows: list[dict[str, Any]],
    photo_ids: list[str],
    model: Any,
    recipe: dict[str, Any],
    gpu: tuple[Any, ...],
    *,
    predict: Callable[..., tuple[float, Any]] = _predict,
) -> list[dict[str, Any]]:
    """Read only the 169 selected train/good IDs; return auditable photo scores."""
    expected = {(row["region"], row["photo_id"]): row for row in rows}
    if len(expected) != len(rows) or set(expected) != {
        (region, photo_id) for region in REGIONS for photo_id in photo_ids
    }:
        raise ValueError("Normal calibration rows are incomplete or repeated")
    results: list[dict[str, Any]] = []
    for index, photo_id in enumerate(photo_ids, start=1):
        region_results: list[dict[str, Any]] = []
        for region in REGIONS:
            reference = expected[(region, photo_id)]
            name = f"extracted/{region}/train/good/{photo_id}.png"
            if reference["member"] != name:
                raise ValueError(f"Non-training normal member in v4 report: {name}")
            raw = source.read(name)  # ZIP CRC is checked; never reads a test member.
            if hashlib.sha256(raw).hexdigest() != reference["member_sha256"]:
                raise ValueError(f"Normal PNG bytes changed since v4 preflight: {name}")
            details = inspect_png(raw)
            if "verify_error" in details or "load_error" in details:
                raise ValueError(f"Normal PNG integrity failed: {name}: {details}")
            with Image.open(io.BytesIO(raw)) as image:
                if image.format != "PNG" or image.mode != "RGB" or image.size != (256, 256):
                    raise ValueError(f"Unexpected normal image format/mode/size: {name}")
                image.load()
                if hashlib.sha256(image.tobytes()).hexdigest() != reference["decoded_rgb_sha256"]:
                    raise ValueError(f"Normal decoded pixels changed since v4 preflight: {name}")
                raw_score, anomaly_map = predict(model, image, recipe, gpu)
            score = candidate_image_score(raw_score, anomaly_map, CANDIDATE)
            if not math.isfinite(score):
                raise ValueError(f"Non-finite normal score: {name}")
            region_results.append({"region": region, "member": name, "score": score})
        results.append({
            "photo_id": photo_id,
            "score": photo_score([record["score"] for record in region_results]),
            "regions": region_results,
        })
        if index % 10 == 0 or index == len(photo_ids):
            print(f"normal calibration {index}/{len(photo_ids)}", flush=True)
    return results


def run(
    project: Path, archive_path: Path, v4_path: Path, model_preflight_path: Path,
    protocol_path: Path, output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    git = git_state(project)
    protocol = load_protocol(protocol_path)
    v4 = validate_v4_report(v4_path)
    model_gate = validate_model_preflight(model_preflight_path, project, protocol)
    md5, archive_sha256 = archive_hashes(archive_path)
    if md5 != SOURCE_MD5 or archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("MPI-PCB archive checksum differs from locked official source")
    recipes = model_contexts(project, protocol)
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with zipfile.ZipFile(archive_path) as source:
        metadata = audit_members(source.infolist())
        if metadata["status"] != "registered_metadata_only":
            raise ValueError("MPI-PCB ZIP member inventory changed")
        normal = normal_members(source.infolist())
        if sorted(normal) != REGIONS:
            raise ValueError("MPI-PCB region inventory changed")
        photo_ids = v4["calibration_photo_ids"]
        rows = v4["calibration_rows"]
        gpu = _gpu_modules()
        torch = gpu[0]
        results: dict[str, Any] = {}
        for key in MODEL_ORDER:
            print(f"Scoring 169 frozen normal photos × 12 regions with {key}...", flush=True)
            _seed_everything(42, torch)
            checkpoint = Path(model_gate["models"][key]["checkpoint_path"])
            recipe = {"base_model": recipes[key]["base_model"]}
            model = _model(recipe, checkpoint, recipes[key]["checkpoint_context"], gpu)
            try:
                photos = score_normal_cohort(source, rows, photo_ids, model, recipe, gpu)
            finally:
                del model
                gc.collect()
                torch.cuda.empty_cache()
            if len(photos) != 169 or any(len(photo["regions"]) != 12 for photo in photos):
                raise ValueError("Normal calibration did not score exactly 169 × 12 regions")
            threshold = normal_threshold([photo["score"] for photo in photos])
            results[key] = {
                "checkpoint_sha256": model_gate["models"][key]["checkpoint_sha256"],
                "threshold": threshold,
                "normal_original_photos_scored": len(photos),
                "normal_region_images_scored": sum(len(photo["regions"]) for photo in photos),
                "normal_photos_at_or_above_threshold": sum(photo["score"] >= threshold for photo in photos),
                "photo_scores": photos,
            }
    report = {
        "status": "mpi_pcb_normal_calibrated_test_sealed",
        "git": git,
        "protocol_config_sha256": protocol["config_sha256"],
        "model_preflight_sha256": sha256_file(model_preflight_path),
        "normal_preflight_v4_sha256": sha256_file(v4_path),
        "normal_photo_ids_sha256": v4["calibration_photo_ids_sha256"],
        "archive_sha256": archive_sha256,
        "photo_aggregation": "maximum_of_12_region_scores",
        "region_score": "mean_of_top_1pct_anomaly_map",
        "threshold_method": "95pct_higher_from_169_original_training_normal_photos_per_model",
        "models": results,
        "normal_original_photos_scored_per_model": 169,
        "normal_region_images_scored_per_model": 2028,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "test_scores_computed": 0,
        "automatic_pass_allowed": False,
        "confirmatory_claim_allowed": False,
        "note": "Normal-data adaptation only. No MPI-PCB test image or mask was opened; no external performance claim.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(report, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--v4-preflight", type=Path, required=True)
    parser.add_argument("--model-preflight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    report = run(project, args.archive, args.v4_preflight, args.model_preflight, args.config, args.output)
    print(json.dumps({
        "status": report["status"],
        "models": {key: {
            "threshold": value["threshold"],
            "normal_original_photos_scored": value["normal_original_photos_scored"],
            "normal_region_images_scored": value["normal_region_images_scored"],
        } for key, value in report["models"].items()},
        "test_images_decoded": report["test_images_decoded"],
        "test_masks_decoded": report["test_masks_decoded"],
        "test_scores_computed": report["test_scores_computed"],
        "output": str(args.output),
    }, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
