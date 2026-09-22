"""Register Real-IAD follow-up development data while keeping final images sealed."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

from defectops.manifest import canonical, checksum
from defectops.realiad_followup_protocol import FOLLOWUP_CATEGORIES


DEVELOPMENT_MANIFEST_NAMES = {
    "realiad_pcb": "realiad-pcb-development-v1.json",
    "realiad_terminalblock": "realiad-terminalblock-development-v1.json",
}
FINAL_MANIFEST_NAMES = {
    "realiad_pcb": "realiad-pcb-final-sealed-v1.json",
    "realiad_terminalblock": "realiad-terminalblock-final-sealed-v1.json",
}
_HEX_SHA256 = re.compile(r"[0-9a-f]{64}")


def _safe_relative(value: object) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError("Follow-up registration contains an invalid relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or "." in path.parts or ".." in path.parts:
        raise ValueError(f"Unsafe follow-up path: {value}")
    return path.as_posix()


def _validate_plan(plan: dict[str, Any]) -> None:
    claimed = plan.get("plan_sha256")
    unsigned = dict(plan)
    unsigned.pop("plan_sha256", None)
    actual = hashlib.sha256(canonical(unsigned)).hexdigest()
    if claimed != actual:
        raise ValueError("Follow-up split plan checksum mismatch")
    if (
        plan.get("schema_version") != 1
        or plan.get("dataset") != "real_iad"
        or plan.get("protocol") != "A-normal-only-v2"
        or plan.get("categories") != ["realiad_pcb", "realiad_terminalblock"]
        or not isinstance(plan.get("records"), list)
    ):
        raise ValueError("Unexpected follow-up split plan contract")
    allowed = {"train", "calibration", "development", "final_holdout"}
    group_splits: dict[tuple[str, str], set[str]] = {}
    for row in plan["records"]:
        if (
            not isinstance(row, dict)
            or row.get("category") not in FOLLOWUP_CATEGORIES
            or row.get("source_object") != FOLLOWUP_CATEGORIES[row["category"]]
            or row.get("planned_split") not in allowed
            or row.get("view") != "C1"
        ):
            raise ValueError("Malformed follow-up split-plan record")
        _safe_relative(row.get("source_image_path"))
        if row.get("label") == 1:
            _safe_relative(row.get("source_mask_path"))
        elif row.get("label") != 0 or row.get("source_mask_path") is not None:
            raise ValueError("Follow-up split-plan label/mask contract mismatch")
        key = (row["category"], row.get("sample_group"))
        if not isinstance(key[1], str) or not key[1]:
            raise ValueError("Follow-up split-plan sample group is missing")
        group_splits.setdefault(key, set()).add(row["planned_split"])
    overlap = [key for key, splits in group_splits.items() if len(splits) != 1]
    if overlap:
        raise ValueError(f"Follow-up sample group spans splits: {overlap[0]}")


def load_followup_plan(path: Path) -> dict[str, Any]:
    plan = json.loads(path.read_text(encoding="utf-8"))
    _validate_plan(plan)
    return plan


def _member_digest(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> str:
    with archive.open(info) as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _member_info(
    archive: zipfile.ZipFile,
    by_name: dict[str, zipfile.ZipInfo],
    member: str,
) -> zipfile.ZipInfo:
    try:
        info = by_name[member]
    except KeyError as exc:
        raise ValueError(f"Real-IAD archive is missing a planned member: {member}") from exc
    if info.is_dir() or info.file_size <= 0:
        raise ValueError(f"Planned archive member is empty or not a file: {member}")
    return info


def _extract_member(
    archive: zipfile.ZipFile,
    info: zipfile.ZipInfo,
    destination: Path,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with archive.open(info) as source, destination.open("xb") as output:
        shutil.copyfileobj(source, output)


def _development_record(
    row: dict[str, Any],
    image_digest: str,
    mask_digest: str | None,
) -> dict[str, Any]:
    source_object = row["source_object"]
    image_path = f"real_iad_followup/images/{source_object}/{row['source_image_path']}"
    mask_path = None
    if row["source_mask_path"] is not None:
        mask_path = f"real_iad_followup/images/{source_object}/{row['source_mask_path']}"
    split = "test" if row["planned_split"] == "development" else row["planned_split"]
    return {
        "image_path": image_path,
        "image_sha256": image_digest,
        "category": row["category"],
        "source_object": source_object,
        "official_split": row["official_split"],
        "split": split,
        "protocol_split": row["planned_split"],
        "label": row["label"],
        "defect_type": row["defect_type"],
        "view": row["view"],
        "sample_group": row["sample_group"],
        "mask_path": mask_path,
        "mask_sha256": mask_digest,
    }


def _sealed_record(
    row: dict[str, Any],
    image_member: str,
    image_digest: str,
    mask_member: str | None,
    mask_digest: str | None,
) -> dict[str, Any]:
    return {
        "category": row["category"],
        "source_object": row["source_object"],
        "official_split": row["official_split"],
        "protocol_split": "final_holdout",
        "label": row["label"],
        "defect_type": row["defect_type"],
        "view": row["view"],
        "sample_group": row["sample_group"],
        "source_image_member": image_member,
        "image_sha256": image_digest,
        "source_mask_member": mask_member,
        "mask_sha256": mask_digest,
    }


def _development_payload(
    plan: dict[str, Any],
    category: str,
    archive_sha256: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "dataset": "real_iad_followup",
        "protocol": "A-normal-only-v2",
        "registration_status": plan["registration_status"],
        "evidence_status": plan["evidence_status"],
        "categories": [category],
        "split_seed": plan["split_seed"],
        "calibration_fraction": plan["calibration_group_fraction"],
        "development_fraction": plan["development_group_fraction"],
        "split_plan_path": "manifests/realiad-followup-split-plan-v1.json",
        "split_plan_sha256": plan["plan_sha256"],
        "source_archive_name": f"{FOLLOWUP_CATEGORIES[category]}.zip",
        "source_archive_sha256": archive_sha256,
        "final_holdout_included": False,
        "records": sorted(records, key=lambda item: item["image_path"]),
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def _final_payload(
    plan: dict[str, Any],
    category: str,
    archive_sha256: str,
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "dataset": "real_iad_followup",
        "protocol": "A-normal-only-v2",
        "registration_status": plan["registration_status"],
        "evidence_status": plan["evidence_status"],
        "categories": [category],
        "split_seed": plan["split_seed"],
        "split_plan_path": "manifests/realiad-followup-split-plan-v1.json",
        "split_plan_sha256": plan["plan_sha256"],
        "source_archive_name": f"{FOLLOWUP_CATEGORIES[category]}.zip",
        "source_archive_sha256": archive_sha256,
        "sealed": True,
        "images_extracted": False,
        "open_gate": "selected_candidate_id+selection_report_sha256+git_commit+final_manifest_sha256",
        "records": sorted(records, key=lambda item: item["source_image_member"]),
    }
    payload["manifest_sha256"] = hashlib.sha256(canonical(payload)).hexdigest()
    return payload


def register_followup_data(
    root: Path,
    plan_path: Path,
    archive_paths: dict[str, Path],
) -> list[dict[str, Any]]:
    """Extract non-final records and checksum final members without extracting them."""
    root = root.resolve(strict=True)
    plan_path = plan_path.resolve(strict=True)
    plan = load_followup_plan(plan_path)
    expected_plan_path = root / "manifests" / "realiad-followup-split-plan-v1.json"
    if plan_path != expected_plan_path.resolve(strict=True):
        raise ValueError(f"Use the committed split plan at {expected_plan_path}")
    target = root / "real_iad_followup"
    if target.exists():
        raise FileExistsError(f"Follow-up registration target already exists: {target}")
    manifests_directory = root / "manifests"
    output_paths = [
        manifests_directory / name
        for name in (*DEVELOPMENT_MANIFEST_NAMES.values(), *FINAL_MANIFEST_NAMES.values())
    ]
    existing = [path for path in output_paths if path.exists()]
    if existing:
        raise FileExistsError(f"Follow-up manifest already exists: {existing[0]}")

    normalized_archives = {
        category: archive_paths[category].resolve(strict=True)
        for category in FOLLOWUP_CATEGORIES
    }
    development_records: dict[str, list[dict[str, Any]]] = {
        category: [] for category in FOLLOWUP_CATEGORIES
    }
    final_records: dict[str, list[dict[str, Any]]] = {
        category: [] for category in FOLLOWUP_CATEGORIES
    }
    archive_hashes: dict[str, str] = {}
    seen_images: dict[str, str] = {}

    with tempfile.TemporaryDirectory(prefix=".realiad-followup-staging-", dir=root) as temporary:
        stage_root = Path(temporary) / "real_iad_followup"
        for category, source_object in FOLLOWUP_CATEGORIES.items():
            archive_path = normalized_archives[category]
            archive_hashes[category] = checksum(archive_path)
            category_rows = [row for row in plan["records"] if row["category"] == category]
            with zipfile.ZipFile(archive_path) as archive:
                bad = archive.testzip()
                if bad is not None:
                    raise ValueError(f"Corrupt ZIP member in {archive_path.name}: {bad}")
                by_name = {info.filename: info for info in archive.infolist()}
                for row in category_rows:
                    image_member = f"{source_object}/{row['source_image_path']}"
                    image_info = _member_info(archive, by_name, image_member)
                    image_digest = _member_digest(archive, image_info)
                    if image_digest in seen_images:
                        raise ValueError(
                            "Duplicate follow-up image content: "
                            f"{seen_images[image_digest]} and {image_member}"
                        )
                    seen_images[image_digest] = image_member
                    mask_member = None
                    mask_digest = None
                    mask_info = None
                    if row["source_mask_path"] is not None:
                        mask_member = f"{source_object}/{row['source_mask_path']}"
                        mask_info = _member_info(archive, by_name, mask_member)
                        mask_digest = _member_digest(archive, mask_info)
                    if row["planned_split"] == "final_holdout":
                        final_records[category].append(
                            _sealed_record(
                                row,
                                image_member,
                                image_digest,
                                mask_member,
                                mask_digest,
                            )
                        )
                        continue
                    image_destination = (
                        stage_root / "images" / source_object / PurePosixPath(row["source_image_path"])
                    )
                    _extract_member(archive, image_info, image_destination)
                    if mask_info is not None:
                        mask_destination = (
                            stage_root
                            / "images"
                            / source_object
                            / PurePosixPath(row["source_mask_path"])
                        )
                        _extract_member(archive, mask_info, mask_destination)
                    development_records[category].append(
                        _development_record(row, image_digest, mask_digest)
                    )
        shutil.move(str(stage_root), target)

    manifests_directory.mkdir(parents=True, exist_ok=True)
    payloads: list[dict[str, Any]] = []
    try:
        for category in FOLLOWUP_CATEGORIES:
            development = _development_payload(
                plan,
                category,
                archive_hashes[category],
                development_records[category],
            )
            final = _final_payload(
                plan,
                category,
                archive_hashes[category],
                final_records[category],
            )
            for path, payload in (
                (manifests_directory / DEVELOPMENT_MANIFEST_NAMES[category], development),
                (manifests_directory / FINAL_MANIFEST_NAMES[category], final),
            ):
                with path.open("x", encoding="utf-8", newline="\n") as stream:
                    json.dump(payload, stream, indent=2, ensure_ascii=True)
                    stream.write("\n")
            payloads.extend((development, final))
    except Exception as exc:
        raise RuntimeError(
            "Development images were registered but manifest writing failed; preserve the target for diagnosis"
        ) from exc
    return payloads


def _read_plan_for_manifest(root: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    relative = _safe_relative(manifest.get("split_plan_path"))
    path = (root / relative).resolve(strict=True)
    path.relative_to(root)
    plan = load_followup_plan(path)
    if manifest.get("split_plan_sha256") != plan["plan_sha256"]:
        raise ValueError("Follow-up manifest references a different split plan")
    return plan


def verify_followup_development_manifest(root: Path, manifest: dict[str, Any]) -> None:
    """Rebuild a development manifest from the committed plan and extracted bytes."""
    root = root.resolve(strict=True)
    plan = _read_plan_for_manifest(root, manifest)
    categories = manifest.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise ValueError("Follow-up development manifest must contain one category")
    category = categories[0]
    if category not in FOLLOWUP_CATEGORIES:
        raise ValueError("Unexpected follow-up category")
    archive_digest = manifest.get("source_archive_sha256")
    if not isinstance(archive_digest, str) or not _HEX_SHA256.fullmatch(archive_digest):
        raise ValueError("Follow-up source archive SHA-256 is invalid")
    expected_records = []
    for row in plan["records"]:
        if row["category"] != category or row["planned_split"] == "final_holdout":
            continue
        source_object = row["source_object"]
        image_relative = (
            f"real_iad_followup/images/{source_object}/{row['source_image_path']}"
        )
        image_path = (root / image_relative).resolve(strict=True)
        image_path.relative_to(root)
        mask_digest = None
        if row["source_mask_path"] is not None:
            mask_relative = (
                f"real_iad_followup/images/{source_object}/{row['source_mask_path']}"
            )
            mask_path = (root / mask_relative).resolve(strict=True)
            mask_path.relative_to(root)
            mask_digest = checksum(mask_path)
        expected_records.append(
            _development_record(row, checksum(image_path), mask_digest)
        )
    expected = _development_payload(plan, category, archive_digest, expected_records)
    if canonical(expected) != canonical(manifest):
        raise ValueError("Follow-up development manifest mismatch")


def verify_followup_final_manifest(
    root: Path,
    manifest: dict[str, Any],
) -> None:
    """Validate the sealed final inventory without opening any final image."""
    plan = _read_plan_for_manifest(root.resolve(strict=True), manifest)
    categories = manifest.get("categories")
    if not isinstance(categories, list) or len(categories) != 1:
        raise ValueError("Follow-up final manifest must contain one category")
    category = categories[0]
    if category not in FOLLOWUP_CATEGORIES:
        raise ValueError("Unexpected follow-up category")
    archive_digest = manifest.get("source_archive_sha256")
    if not isinstance(archive_digest, str) or not _HEX_SHA256.fullmatch(archive_digest):
        raise ValueError("Follow-up source archive SHA-256 is invalid")
    actual_records = manifest.get("records")
    if not isinstance(actual_records, list):
        raise ValueError("Follow-up final records are missing")
    by_path: dict[str, dict[str, Any]] = {}
    for row in actual_records:
        if (
            row.get("category") != category
            or row.get("protocol_split") != "final_holdout"
            or row.get("view") != "C1"
            or not isinstance(row.get("image_sha256"), str)
            or not _HEX_SHA256.fullmatch(row["image_sha256"])
        ):
            raise ValueError("Malformed sealed final record")
        path = _safe_relative(row.get("source_image_member"))
        if path in by_path:
            raise ValueError(f"Duplicate sealed final path: {path}")
        by_path[path] = row
        if row.get("label") == 1:
            _safe_relative(row.get("source_mask_member"))
            if not isinstance(row.get("mask_sha256"), str) or not _HEX_SHA256.fullmatch(
                row["mask_sha256"]
            ):
                raise ValueError("Malformed sealed final mask checksum")
        elif row.get("source_mask_member") is not None or row.get("mask_sha256") is not None:
            raise ValueError("Normal sealed final record unexpectedly has a mask")
    planned_rows = [
        row
        for row in plan["records"]
        if row["category"] == category and row["planned_split"] == "final_holdout"
    ]
    expected_paths = {
        f"{row['source_object']}/{row['source_image_path']}" for row in planned_rows
    }
    if set(by_path) != expected_paths:
        raise ValueError("Sealed final paths do not match the committed split plan")
    expected_records = []
    for planned in planned_rows:
        image_member = f"{planned['source_object']}/{planned['source_image_path']}"
        actual = by_path[image_member]
        mask_member = None
        if planned["source_mask_path"] is not None:
            mask_member = f"{planned['source_object']}/{planned['source_mask_path']}"
        if actual.get("source_mask_member") != mask_member:
            raise ValueError("Sealed final mask path does not match the committed split plan")
        expected_records.append(
            _sealed_record(
                planned,
                image_member,
                actual["image_sha256"],
                mask_member,
                actual.get("mask_sha256"),
            )
        )
    expected = _final_payload(plan, category, archive_digest, expected_records)
    if canonical(expected) != canonical(manifest):
        raise ValueError("Sealed final manifest mismatch")


def followup_registration_summary(manifest: dict[str, Any]) -> dict[str, Any]:
    records = manifest["records"]
    if manifest.get("sealed"):
        counts = Counter(
            f"final_holdout/{'anomaly' if row['label'] else 'normal'}" for row in records
        )
    else:
        counts = Counter(
            f"{row['protocol_split']}/{'anomaly' if row['label'] else 'normal'}"
            for row in records
        )
    return {
        "category": manifest["categories"][0],
        "sealed": bool(manifest.get("sealed", False)),
        "manifest_sha256": manifest["manifest_sha256"],
        "source_archive_sha256": manifest["source_archive_sha256"],
        "counts": dict(sorted(counts.items())),
    }
