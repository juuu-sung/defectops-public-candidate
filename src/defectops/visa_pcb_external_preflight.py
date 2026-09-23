"""Audit only VisA calibration images; keep official test image payloads sealed."""

from __future__ import annotations

import hashlib
import io
import json
import os
import tarfile
import tempfile
import tomllib
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from defectops.padim_experiment import git_state
from defectops.realiad_followup_final import validate_gate
from defectops.visa_pcb_external_protocol import canonical, load_config
from defectops.visa_pcb_external_registration import file_sha256, inspect_archive


EXPECTED_PREFLIGHT = {
    "schema_version": 1,
    "status": "registration_verified_calibration_audit_final_unopened",
    "protocol": "A-normal-only-v2",
    "registration_handoff_sha256": "52e2f24e853e60738791d02e75f937521906da51a488eb37f66e7a6b0b9ec040",
    "source_archive_sha256": "2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362",
    "model_category": "realiad_pcb",
    "model_seed": 42,
    "calibration_image_count": 224,
    "external_final_image_count": 802,
    "calibration_image_format": "JPEG",
    "calibration_image_mode": "RGB",
    "reject_exact_calibration_pixel_duplicates": True,
    "external_final_payload_decode_allowed": False,
    "external_final_model_score_allowed": False,
    "automatic_pass_allowed": False,
}


def load_preflight_config(path: Path) -> dict[str, Any]:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if config != EXPECTED_PREFLIGHT:
        raise ValueError("VisA preflight contract differs from the locked values")
    return {**config, "config_sha256": hashlib.sha256(canonical(config)).hexdigest()}


def checked_registration(
    handoff_path: Path, archive_path: Path, plan: dict[str, Any],
    protocol: dict[str, Any], preflight: dict[str, Any],
) -> dict[str, Any]:
    if file_sha256(handoff_path) != preflight["registration_handoff_sha256"]:
        raise ValueError("Registration handoff ZIP SHA-256 differs")
    with zipfile.ZipFile(handoff_path) as handoff:
        if sorted(handoff.namelist()) != ["plan.json", "registration.json"] or handoff.testzip() is not None:
            raise ValueError("Registration handoff ZIP members or CRC differ")
        attached_plan = json.loads(handoff.read("plan.json"))
        attached_report = json.loads(handoff.read("registration.json"))
    if attached_plan != plan:
        raise ValueError("Attached VisA plan differs from the repository plan")
    current_report = inspect_archive(archive_path, plan, protocol)
    if (
        attached_report != current_report
        or current_report["archive_sha256"] != preflight["source_archive_sha256"]
        or current_report["status"] != "external_archive_registered_metadata_only"
        or current_report["test_member_payloads_decoded"] is not False
        or current_report["test_scores_computed"] is not False
    ):
        raise ValueError("VisA archive or registration report differs from the locked evidence")
    return attached_report


def audit_calibration(
    archive_path: Path, records: list[dict[str, Any]],
    *, expected_count: int,
) -> dict[str, Any]:
    """Only extract calibration members; official test JPEG/PNG payloads stay unopened."""
    selected = [row for row in records if row["protocol_split"] == "calibration"]
    if len(selected) != expected_count or any(row["label"] != 0 or row["source_mask_member"] for row in selected):
        raise ValueError("VisA calibration selection differs")
    results = []
    pixels_seen: dict[str, str] = {}
    with tarfile.open(archive_path, "r:") as archive:
        for row in selected:
            name = row["source_image_member"]
            member = archive.getmember(name)
            if not member.isfile() or member.size <= 0:
                raise ValueError(f"Invalid calibration archive member: {name}")
            stream = archive.extractfile(member)
            if stream is None:
                raise ValueError(f"Calibration image could not be read: {name}")
            image_bytes = stream.read()
            if len(image_bytes) != member.size:
                raise ValueError(f"Calibration image size differs: {name}")
            with Image.open(io.BytesIO(image_bytes)) as image:
                if image.format != "JPEG" or image.mode != "RGB":
                    raise ValueError(f"Calibration image is not RGB JPEG: {name}")
                image.load()
                if min(image.size) <= 0:
                    raise ValueError(f"Invalid calibration image dimensions: {name}")
                pixel_sha = hashlib.sha256(
                    f"{image.size[0]}x{image.size[1]}:RGB:".encode() + image.tobytes()
                ).hexdigest()
                width, height = image.size
            if pixel_sha in pixels_seen:
                raise ValueError(f"Exact calibration pixel duplicate: {pixels_seen[pixel_sha]} and {name}")
            pixels_seen[pixel_sha] = name
            results.append({
                "category": row["category"], "image": name, "width": width, "height": height,
                "archive_member_bytes": member.size,
                "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
                "decoded_pixel_sha256": pixel_sha,
            })
    return {
        "schema_version": 1,
        "status": "calibration_jpeg_decoded_no_exact_pixel_duplicates",
        "count": len(results),
        "counts_by_category": dict(sorted(Counter(row["category"] for row in results).items())),
        "image_sizes": dict(sorted(Counter(f"{row['width']}x{row['height']}" for row in results).items())),
        "exact_pixel_duplicate_count": 0,
        "external_final_payloads_decoded": False,
        "records": results,
    }


def _write_json_once(path: Path, payload: dict[str, Any]) -> None:
    content = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise ValueError(f"Existing VisA preflight file differs and was not overwritten: {path}")
        return
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", prefix=f".{path.stem}-",
        suffix=".json", dir=path.parent, delete=False,
    ) as stream:
        temporary = Path(stream.name)
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def run_preflight(project_root: Path, archive_path: Path) -> dict[str, Any]:
    project_root = project_root.resolve(strict=True)
    git = git_state(project_root)
    protocol = load_config(project_root / "configs" / "visa-pcb-external-v1.toml")
    preflight = load_preflight_config(project_root / "configs" / "visa-pcb-external-preflight-v1.toml")
    plan = json.loads((project_root / "data" / "manifests" / "visa-pcb-external-plan-v1.json").read_text(encoding="utf-8"))
    unsigned = dict(plan)
    claimed = unsigned.pop("plan_sha256", None)
    if claimed != hashlib.sha256(canonical(unsigned)).hexdigest() or plan["config_sha256"] != protocol["config_sha256"]:
        raise ValueError("VisA plan checksum or protocol differs")
    if sum(row["protocol_split"] == "external_final" for row in plan["records"]) != preflight["external_final_image_count"]:
        raise ValueError("VisA official test count differs")
    registration_zip = project_root / "artifacts" / "handoff" / "visa-pcb-external-registration-v1.zip"
    registration = checked_registration(registration_zip, archive_path, plan, protocol, preflight)
    # This gate verifies the *previously fitted* Real-IAD PCB seed-42 checkpoint without loading CUDA.
    gate = validate_gate(project_root, preflight["model_category"], preflight["model_seed"])
    layer = gate["dev_result"]["layer_groups"]["layer1-layer2"]
    audit = audit_calibration(
        archive_path, plan["records"], expected_count=preflight["calibration_image_count"]
    )
    report = {
        "schema_version": 1,
        "status": "preflight_passed_external_final_unopened",
        "git": git,
        "plan_sha256": plan["plan_sha256"],
        "protocol_config_sha256": protocol["config_sha256"],
        "preflight_config_sha256": preflight["config_sha256"],
        "registration_handoff_sha256": preflight["registration_handoff_sha256"],
        "source_archive_sha256": registration["archive_sha256"],
        "realiad_checkpoint_sha256": layer["checkpoint_sha256"],
        "realiad_selection_report_sha256": gate["lock"]["selection_report_sha256"],
        "calibration_audit_sha256": hashlib.sha256(canonical(audit)).hexdigest(),
        "calibration_images_decoded": audit["count"],
        "external_final_images_decoded": 0,
        "external_final_scores_computed": 0,
        "automatic_pass_allowed": False,
        "limitations": [
            "Only calibration JPEG payloads were decoded; test JPEG and PNG payloads remain unopened.",
            "Test image/mask decodability and cross-split pixel duplicates remain unchecked until the sealed evaluation gate.",
            "No VisA calibration score, threshold, test prediction, or transfer performance was computed.",
        ],
    }
    output = project_root / "artifacts" / "visa-pcb-external-preflight-v1"
    handoff = project_root / "artifacts" / "handoff" / "visa-pcb-external-preflight-v1.zip"
    output.mkdir(parents=True, exist_ok=True)
    _write_json_once(output / "calibration-audit.json", audit)
    _write_json_once(output / "preflight.json", report)
    handoff.parent.mkdir(parents=True, exist_ok=True)
    if handoff.exists():
        with zipfile.ZipFile(handoff) as archive:
            if (
                sorted(archive.namelist()) != ["calibration-audit.json", "preflight.json"]
                or archive.testzip() is not None
                or json.loads(archive.read("calibration-audit.json")) != audit
                or json.loads(archive.read("preflight.json")) != report
            ):
                raise ValueError(f"Existing preflight handoff differs and was not overwritten: {handoff}")
    else:
        with zipfile.ZipFile(handoff, mode="x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(output / "calibration-audit.json", "calibration-audit.json")
            archive.write(output / "preflight.json", "preflight.json")
    report["handoff_zip_sha256"] = file_sha256(handoff)
    report["handoff_zip"] = str(handoff)
    return report
