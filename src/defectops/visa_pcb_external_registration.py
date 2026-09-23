"""Register official VisA archive metadata without decoding any test image."""

from __future__ import annotations

import hashlib
import json
import os
import tarfile
import tempfile
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any

from defectops.visa_pcb_external_protocol import canonical, verify_plan


CSV_MEMBER = "split_csv/1cls.csv"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_archive(archive_path: Path, plan: dict[str, Any], config: dict[str, Any]) -> dict[str, Any]:
    if archive_path.stat().st_size != config["source_archive_bytes"]:
        raise ValueError("VisA archive size differs from the preregistered source")
    archive_sha = file_sha256(archive_path)
    selected = {row["source_image_member"] for row in plan["records"]}
    selected.update(row["source_mask_member"] for row in plan["records"] if row["source_mask_member"])
    selected.add(CSV_MEMBER)
    metadata: dict[str, dict[str, Any]] = {}
    with tarfile.open(archive_path, "r:") as archive:
        for member in archive:
            name = member.name
            posix = PurePosixPath(name)
            if "\\" in name or posix.is_absolute() or ".." in posix.parts:
                raise ValueError(f"Unsafe VisA archive path: {name}")
            if name not in selected:
                continue
            if name in metadata or not member.isfile() or member.size <= 0:
                raise ValueError(f"Duplicate or invalid selected VisA member: {name}")
            metadata[name] = {"name": name, "size": member.size}
            if name == CSV_MEMBER:
                stream = archive.extractfile(member)
                if stream is None:
                    raise ValueError("Official VisA split CSV could not be read")
                csv_bytes = stream.read()
    missing = sorted(selected - metadata.keys())
    if missing:
        raise ValueError(f"Selected VisA members are missing: {missing[:5]}")
    verify_plan(plan, csv_bytes, config)
    member_metadata = [metadata[name] for name in sorted(metadata)]
    return {
        "schema_version": 1,
        "status": "external_archive_registered_metadata_only",
        "source_url": config["source_url"],
        "archive_filename": archive_path.name,
        "archive_bytes": archive_path.stat().st_size,
        "archive_sha256": archive_sha,
        "source_split_csv_sha256": hashlib.sha256(csv_bytes).hexdigest(),
        "plan_sha256": plan["plan_sha256"],
        "config_sha256": config["config_sha256"],
        "selected_member_count": len(selected),
        "selected_member_metadata_sha256": hashlib.sha256(canonical({"members": member_metadata})).hexdigest(),
        "counts": plan["counts"],
        "test_member_payloads_decoded": False,
        "test_scores_computed": False,
        "model_training_on_visa": False,
        "automatic_pass_allowed": False,
        "next_gate": "Audit image bytes and lock the frozen-model evaluator before opening official test images.",
        "limitations": [
            "Full-archive SHA-256 reads the tar bytes, but no individual test image payload is decoded or scored.",
            "Member presence and size do not prove image decodability, mask validity, or absence of near duplicates.",
        ],
    }


def register(
    archive_path: Path, plan_path: Path, config: dict[str, Any],
    output_dir: Path, handoff_path: Path,
) -> dict[str, Any]:
    if handoff_path.exists():
        raise FileExistsError(f"Existing registration handoff was not overwritten: {handoff_path}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    unsigned = dict(plan)
    claimed = unsigned.pop("plan_sha256", None)
    if claimed != hashlib.sha256(canonical(unsigned)).hexdigest():
        raise ValueError("Stored VisA plan checksum differs")
    report = inspect_archive(archive_path, plan, config)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "registration.json"
    content = json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    if report_path.exists():
        if report_path.read_text(encoding="utf-8") != content:
            raise ValueError("Existing registration report differs and was not overwritten")
    else:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", prefix=".registration-",
            suffix=".json", dir=output_dir, delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, report_path)
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(handoff_path, "x", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(report_path, "registration.json")
        archive.write(plan_path, "plan.json")
    report["handoff_zip_sha256"] = file_sha256(handoff_path)
    return report
