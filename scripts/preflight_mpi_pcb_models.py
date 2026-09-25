"""Lock two existing checkpoints and normal-only rules without opening MPI test."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from defectops.padim_experiment import higher_quantile
from defectops.visa_pcb_external_protocol import canonical
from scripts.audit_mpi_pcb_archive import archive_hashes
from scripts.audit_mpi_pcb_normal_pool import _ids_sha256
from scripts.preflight_mpi_pcb_normals import summarize_decoded_duplicates


EXPECTED_ARCHIVE_SHA256 = "0fa9b20fae64142a418487a1d898ee2ee824eee0bd564896a245f4d991112607"
EXPECTED_V4_SHA256 = "65605ddcb5d89b2e4f5bcbcc8d2002e2996455d6235e37e35d1faf767dd5c93d"
EXPECTED_IDS_SHA256 = "75768131852bfe1234f754c3b77054cce64e2a7c890a94db9724943b21faf11e"
EXPECTED_MODELS = {
    "realiad_pcb": {
        "source": "realiad_followup_development_pcb_seed42_layer1_layer2",
        "checkpoint_sha256": "cd0e12d3d2daadfc1f30fde6e666002287af29cc610a870a6196899468c0de13",
        "category": "realiad_pcb", "seed": 42, "backbone": "wide_resnet50_2",
        "layers": ["layer1", "layer2"], "image_size": 256, "num_neighbors": 9,
    },
    "visa_pcb3_v2": {
        "source": "visa_pcb3_normal_bank_v2_seed42",
        "checkpoint_sha256": "be8062541e2a14cb563284d34cfc9f60e3ca5e8cc1d75e0ccbd28d1287785e85",
        "category": "pcb3", "seed": 42, "backbone": "wide_resnet50_2",
        "layers": ["layer1", "layer2"], "image_size": 256, "num_neighbors": 9,
    },
}
EXPECTED_CONFIG = {
    "schema_version": 1,
    "status": "frozen_models_normal_calibration_protocol_test_sealed",
    "source_archive_sha256": EXPECTED_ARCHIVE_SHA256,
    "normal_preflight_v4_sha256": EXPECTED_V4_SHA256,
    "normal_photo_count": 169,
    "regions_per_photo": 12,
    "input_preprocessing": "stretch_256_bilinear_antialias_imagenet",
    "region_image_score": "mean_of_top_1pct_anomaly_map",
    "original_photo_aggregation": "maximum_of_12_region_scores",
    "calibration_quantile": 0.95,
    "calibration_quantile_method": "higher",
    "decision_rule": "score >= model_specific_normal_calibration_threshold",
    "normal_calibration_only": True,
    "test_image_decoding_allowed": False,
    "test_mask_decoding_allowed": False,
    "automatic_pass_allowed": False,
    "confirmatory_claim_allowed": False,
    "models": EXPECTED_MODELS,
}


def sha256_file(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def load_protocol(path: Path) -> dict:
    protocol = tomllib.loads(path.read_text(encoding="utf-8"))
    if protocol != EXPECTED_CONFIG:
        raise ValueError("MPI-PCB model/calibration protocol differs from locked values")
    return {**protocol, "config_sha256": hashlib.sha256(canonical(protocol)).hexdigest()}


def validate_v4_report(path: Path) -> dict:
    if sha256_file(path) != EXPECTED_V4_SHA256:
        raise ValueError("Reviewed MPI-PCB v4 preflight JSON checksum differs")
    report = json.loads(path.read_text(encoding="utf-8"))
    ids = report.get("calibration_photo_ids")
    rows = report.get("calibration_rows")
    regions = [f"grid{row}-{column}_s1024" for row in range(1, 5) for column in range(1, 4)]
    if not isinstance(ids, list) or not isinstance(rows, list):
        raise ValueError("v4 normal cohort is missing")
    expected_pairs = {(region, photo_id) for region in regions for photo_id in ids}
    pairs = [(row.get("region"), row.get("photo_id")) for row in rows if isinstance(row, dict)]
    hex_fields_valid = all(
        isinstance(row.get(field), str)
        and len(row[field]) == 64
        and all(character in "0123456789abcdef" for character in row[field])
        for row in rows for field in ("member_sha256", "decoded_rgb_sha256")
    )
    paths_valid = all(
        row.get("member") == f'extracted/{row["region"]}/train/good/{row["photo_id"]}.png'
        and row.get("width") == row.get("height") == 256
        for row in rows
    )
    if not (
        report.get("status") == "normal_preflight_v4_passed_test_sealed"
        and report.get("protocol_version") == 4
        and report.get("archive_sha256") == EXPECTED_ARCHIVE_SHA256
        and len(ids) == len(set(ids)) == 169
        and _ids_sha256(ids) == report.get("calibration_photo_ids_sha256") == EXPECTED_IDS_SHA256
        and len(rows) == len(pairs) == len(expected_pairs) == 2028
        and len(set(pairs)) == len(pairs)
        and set(pairs) == expected_pairs
        and hex_fields_valid and paths_valid
        and report.get("calibration_original_photos") == 169
        and report.get("calibration_region_images_decoded") == 2028
        and report.get("regions") == 12
        and report.get("dimensions") == {"256x256": 2028}
        and report.get("identical_12_region_photo_id_groups") == []
        and report.get("test_images_decoded") == 0
        and report.get("test_masks_decoded") == 0
        and report.get("model_scores_computed") == 0
        and report.get("automatic_pass_allowed") is False
    ):
        raise ValueError("MPI-PCB v4 normal preflight is not the reviewed test-sealed cohort")
    _, duplicates = summarize_decoded_duplicates(rows, ids, regions)
    if duplicates:
        raise ValueError("v4 normal photos still have identical 12-region groups")
    return report


def photo_score(region_scores: list[float]) -> float:
    """One original photo is one sample, no matter how many tiles it contains."""
    if len(region_scores) != 12 or not all(math.isfinite(score) for score in region_scores):
        raise ValueError("Exactly 12 finite region scores are required")
    return max(region_scores)


def normal_threshold(photo_scores: list[float]) -> float:
    if len(photo_scores) != 169:
        raise ValueError("Exactly 169 original-photo normal scores are required")
    return higher_quantile(photo_scores, 0.95)


def checked_model_paths(project: Path, protocol: dict) -> dict[str, dict]:
    real_dir = project / "artifacts" / "realiad-followup-development-v1" / "pcb"
    real_result = json.loads((real_dir / "category-result.json").read_text(encoding="utf-8"))
    real_layer = real_result["layer_groups"]["layer1-layer2"]
    name = real_layer["checkpoint"]
    if Path(name).name != name or not name.endswith(".pt"):
        raise ValueError("Unsafe Real-IAD checkpoint filename")
    real_path = real_dir / name
    visa_dir = project / "artifacts" / "visa-pcb-normal-bank-development-v2" / "pcb3"
    visa_result = json.loads((visa_dir / "fit.json").read_text(encoding="utf-8"))
    visa_path = visa_dir / "model.pt"
    expected_real = protocol["models"]["realiad_pcb"]["checkpoint_sha256"]
    expected_visa = protocol["models"]["visa_pcb3_v2"]["checkpoint_sha256"]
    if not (
        real_result.get("status") == "evaluated_preregistered_development"
        and real_result.get("category") == "realiad_pcb"
        and real_result.get("seed") == 42
        and real_result.get("final_holdout_evaluated") is False
        and real_result.get("automatic_pass_allowed") is False
        and real_layer.get("layers") == ["layer1", "layer2"]
        and real_layer.get("checkpoint_sha256") == expected_real
        and visa_result.get("status") == "fit_calibrated_post_test_exploratory_normal_bank"
        and visa_result.get("category") == "pcb3"
        and visa_result.get("model_seed") == 42
        and visa_result.get("checkpoint_sha256") == expected_visa
        and visa_result.get("normal_train_count") == 849
        and visa_result.get("normal_calibration_count") == 56
        and visa_result.get("official_test_scores_computed") == 0
        and visa_result.get("automatic_pass_allowed") is False
    ):
        raise ValueError("Existing frozen-model fit reports differ from the selected candidates")
    checked = {}
    for key, path, expected in (
        ("realiad_pcb", real_path, expected_real),
        ("visa_pcb3_v2", visa_path, expected_visa),
    ):
        actual = sha256_file(path)
        if actual != expected:
            raise ValueError(f"{key} checkpoint checksum differs from locked identity")
        checked[key] = {"checkpoint_path": str(path.resolve()), "checkpoint_sha256": actual}
    return checked


def run(project: Path, archive: Path, preflight: Path, config_path: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Output already exists and will not be overwritten: {output}")
    protocol = load_protocol(config_path)
    report = validate_v4_report(preflight)
    md5, archive_sha256 = archive_hashes(archive)
    if md5 != "501d11d78a58ed0f26bc4d0baa34170e" or archive_sha256 != EXPECTED_ARCHIVE_SHA256:
        raise ValueError("MPI-PCB archive differs from the reviewed official file")
    models = checked_model_paths(project, protocol)
    result = {
        "status": "mpi_pcb_model_protocol_locked_test_sealed",
        "protocol_config_sha256": protocol["config_sha256"],
        "normal_preflight_v4_sha256": sha256_file(preflight),
        "normal_photo_ids_sha256": report["calibration_photo_ids_sha256"],
        "archive_sha256": archive_sha256,
        "models": models,
        "normal_original_photos": 169,
        "regions_per_photo": 12,
        "image_score": "mean_of_top_1pct_anomaly_map_then_max_of_12_regions",
        "threshold": "model_specific_95pct_higher_of_169_normal_photo_scores_not_yet_computed",
        "normal_scores_computed": 0,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "test_scores_computed": 0,
        "automatic_pass_allowed": False,
        "note": "Identity and score-rule gate only. No GPU inference, normal threshold, or external-test result yet.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as target:
        json.dump(result, target, ensure_ascii=False, indent=2)
        target.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    project = Path(__file__).resolve().parents[1]
    print(json.dumps(run(project, args.archive, args.preflight, args.config, args.output), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
