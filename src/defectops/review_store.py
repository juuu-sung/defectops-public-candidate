"""Mac-local, append-only human review records for transferred inspection bundles."""

from __future__ import annotations

import json
import math
import re
import sqlite3
import stat
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator

from defectops.padim_experiment import checksum, read_json


ARTIFACT_NAMES = {
    "input": "input.png",
    "heatmap": "heatmap.png",
    "overlay": "overlay.png",
    "anomaly_map": "anomaly-map.npz",
}
HUMAN_VERDICTS = frozenset({"ANOMALY_VISIBLE", "NO_ANOMALY_VISIBLE", "UNCERTAIN"})
SOP_DEFECT_TYPES = frozenset({"bent_lead", "cut_lead", "damaged_case", "misplaced"})
MAX_ARTIFACT_BYTES = 200 * 1024 * 1024
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
SOP_ID_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9_-]{1,63}\Z")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _regular_file(directory: Path, name: str) -> Path:
    """Reject symlinks, directories and names outside one transferred bundle."""
    if not name or Path(name).name != name:
        raise ValueError(f"Unsafe bundle filename: {name!r}")
    path = directory / name
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Bundle entry is not a regular file: {name}")
    if path.stat().st_size == 0 or path.stat().st_size > MAX_ARTIFACT_BYTES:
        raise ValueError(f"Bundle file has an invalid size: {name}")
    return path


def _finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{label} must be a finite number")
    return number


def validate_bundle(bundle_path: Path) -> tuple[dict[str, Any], str]:
    """Validate a transferred result and every recorded artifact checksum."""
    directory = bundle_path.resolve(strict=True)
    if not directory.is_dir():
        raise ValueError("Inspection bundle must be a directory")
    record_path = _regular_file(directory, "inspection.json")
    record = read_json(record_path)
    if record.get("schema_version") != 1 or record.get("inspection_id") != directory.name:
        raise ValueError("Inspection schema version or directory ID differs")
    if record.get("automatic_pass") is not False or record.get("automatic_pass_allowed") is not False:
        raise ValueError("Inspection bundle does not forbid automatic PASS")

    status = record.get("status")
    if status == "INSPECTION_FAILED":
        if record.get("decision") == "PASS" or record.get("decision") == "REVIEW_REQUIRED":
            raise ValueError("Failed inspection cannot have a model decision")
    elif status == "completed":
        if record.get("category") != "transistor" or record.get("model") not in {
            "padim", "patchcore"
        }:
            raise ValueError("Unexpected completed inspection category or model")
        score = _finite_number(record.get("raw_score"), "raw_score")
        threshold = _finite_number(record.get("image_threshold"), "image_threshold")
        expected_decision = "REVIEW_REQUIRED" if score >= threshold else "PASS"
        if record.get("decision") != expected_decision or record.get("decision_rule") != "score >= threshold":
            raise ValueError("Model decision differs from the frozen score rule")
        if record["model"] == "patchcore" and record.get("comparison_status") != (
            "exploratory_after_padim_test_observed"
        ):
            raise ValueError("PatchCore exploratory status is missing")
        trace = record.get("trace")
        if (
            not isinstance(trace, dict)
            or not isinstance(trace.get("fit_run_id"), str)
            or not trace["fit_run_id"]
            or not all(
                isinstance(trace.get(key), str) and SHA256_PATTERN.fullmatch(trace[key])
                for key in ("checkpoint_sha256", "manifest_sha256")
            )
        ):
            raise ValueError("Inspection provenance is incomplete")
        artifacts = record.get("artifacts")
        if not isinstance(artifacts, dict):
            raise ValueError("Inspection artifact manifest is missing")
        for key, expected_name in ARTIFACT_NAMES.items():
            if artifacts.get(key) != expected_name:
                raise ValueError(f"Unexpected {key} artifact filename")
            path = _regular_file(directory, expected_name)
            if checksum(path) != artifacts.get(f"{key}_sha256"):
                raise ValueError(f"{key} artifact checksum differs")
        input_record = record.get("input")
        if not isinstance(input_record, dict) or input_record.get("sha256") != artifacts["input_sha256"]:
            raise ValueError("Input checksum differs from the inspection record")
    else:
        raise ValueError(f"Unsupported inspection status: {status!r}")
    return record, checksum(record_path)


class ReviewStore:
    """Register immutable inspection snapshots; append human observations separately."""

    def __init__(self, database_path: Path):
        self.database_path = database_path.resolve()

    def _connect(self) -> sqlite3.Connection:
        if not self.database_path.is_file():
            raise FileNotFoundError(f"Review database is not initialized: {self.database_path}")
        connection = sqlite3.connect(self.database_path, timeout=5, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.database_path, timeout=5, isolation_level=None)
        try:
            connection.execute("PRAGMA foreign_keys = ON")
            version = connection.execute("PRAGMA user_version").fetchone()[0]
            if version not in (0, 1, 2):
                raise ValueError(f"Unsupported review database schema version: {version}")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS inspections (
                    inspection_id TEXT PRIMARY KEY,
                    bundle_path TEXT NOT NULL,
                    inspection_json_sha256 TEXT NOT NULL,
                    record_json TEXT NOT NULL,
                    status TEXT NOT NULL CHECK(status IN ('completed', 'INSPECTION_FAILED')),
                    model TEXT,
                    model_decision TEXT,
                    category TEXT,
                    imported_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS review_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL REFERENCES inspections(inspection_id),
                    revision INTEGER NOT NULL CHECK(revision > 0),
                    reviewer TEXT NOT NULL,
                    human_verdict TEXT NOT NULL CHECK(
                        human_verdict IN ('ANOMALY_VISIBLE', 'NO_ANOMALY_VISIBLE', 'UNCERTAIN')
                    ),
                    note TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    UNIQUE(inspection_id, revision)
                );
                CREATE TABLE IF NOT EXISTS sop_evidence_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    inspection_id TEXT NOT NULL,
                    review_revision INTEGER NOT NULL CHECK(review_revision > 0),
                    selected_by TEXT NOT NULL,
                    defect_type TEXT NOT NULL CHECK(
                        defect_type IN ('bent_lead', 'cut_lead', 'damaged_case', 'misplaced')
                    ),
                    document_id TEXT NOT NULL,
                    document_revision INTEGER NOT NULL CHECK(document_revision > 0),
                    catalog_sha256 TEXT NOT NULL,
                    lookup_date TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    recorded_at TEXT NOT NULL,
                    UNIQUE(inspection_id, review_revision),
                    FOREIGN KEY(inspection_id, review_revision)
                        REFERENCES review_events(inspection_id, revision)
                );
                CREATE TRIGGER IF NOT EXISTS inspections_no_update
                    BEFORE UPDATE ON inspections BEGIN SELECT RAISE(ABORT, 'immutable inspection'); END;
                CREATE TRIGGER IF NOT EXISTS inspections_no_delete
                    BEFORE DELETE ON inspections BEGIN SELECT RAISE(ABORT, 'immutable inspection'); END;
                CREATE TRIGGER IF NOT EXISTS review_events_no_update
                    BEFORE UPDATE ON review_events BEGIN SELECT RAISE(ABORT, 'append-only review'); END;
                CREATE TRIGGER IF NOT EXISTS review_events_no_delete
                    BEFORE DELETE ON review_events BEGIN SELECT RAISE(ABORT, 'append-only review'); END;
                CREATE TRIGGER IF NOT EXISTS sop_evidence_events_no_update
                    BEFORE UPDATE ON sop_evidence_events
                    BEGIN SELECT RAISE(ABORT, 'append-only SOP evidence'); END;
                CREATE TRIGGER IF NOT EXISTS sop_evidence_events_no_delete
                    BEFORE DELETE ON sop_evidence_events
                    BEGIN SELECT RAISE(ABORT, 'append-only SOP evidence'); END;
                PRAGMA user_version = 2;
                """
            )
        finally:
            connection.close()

    def register(self, bundle_path: Path) -> dict[str, Any]:
        record, record_sha256 = validate_bundle(bundle_path)
        resolved_path = str(bundle_path.resolve(strict=True))
        inspection_id = record["inspection_id"]
        with self._write() as connection:
            existing = connection.execute(
                "SELECT bundle_path, inspection_json_sha256 FROM inspections WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()
            if existing is not None:
                if existing["bundle_path"] != resolved_path or existing["inspection_json_sha256"] != record_sha256:
                    raise ValueError("Inspection ID is already registered with different content or path")
                return {"status": "already_registered", "inspection_id": inspection_id}
            connection.execute(
                """INSERT INTO inspections
                   (inspection_id, bundle_path, inspection_json_sha256, record_json,
                    status, model, model_decision, category, imported_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    inspection_id,
                    resolved_path,
                    record_sha256,
                    json.dumps(record, sort_keys=True, ensure_ascii=False),
                    record["status"],
                    record.get("model"),
                    record.get("decision"),
                    record.get("category"),
                    _utc_now(),
                ),
            )
        return {"status": "registered", "inspection_id": inspection_id}

    def list_inspections(self) -> list[dict[str, Any]]:
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT i.inspection_id, i.status, i.model, i.model_decision,
                          i.category, i.imported_at,
                          COALESCE(MAX(e.revision), 0) AS review_revision,
                          (SELECT last.human_verdict FROM review_events AS last
                           WHERE last.inspection_id = i.inspection_id
                           ORDER BY last.revision DESC LIMIT 1) AS human_verdict
                   FROM inspections AS i
                   LEFT JOIN review_events AS e ON e.inspection_id = i.inspection_id
                   GROUP BY i.inspection_id
                   ORDER BY i.imported_at DESC, i.inspection_id"""
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def get_inspection(self, inspection_id: str) -> dict[str, Any]:
        connection = self._connect()
        try:
            row = connection.execute(
                "SELECT * FROM inspections WHERE inspection_id = ?", (inspection_id,)
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown inspection ID: {inspection_id}")
            result = dict(row)
            result["record"] = json.loads(result.pop("record_json"))
            latest = connection.execute(
                """SELECT revision, reviewer, human_verdict, note, recorded_at
                   FROM review_events WHERE inspection_id = ?
                   ORDER BY revision DESC LIMIT 1""",
                (inspection_id,),
            ).fetchone()
            result["review_revision"] = latest["revision"] if latest else 0
            result["latest_review"] = dict(latest) if latest else None
            result["automatic_pass_allowed"] = False
            return result
        finally:
            connection.close()

    def verify(self, inspection_id: str) -> dict[str, Any]:
        item = self.get_inspection(inspection_id)
        _, actual_sha256 = validate_bundle(Path(item["bundle_path"]))
        if actual_sha256 != item["inspection_json_sha256"]:
            raise ValueError("Registered inspection JSON was changed after import")
        return {"status": "verified", "inspection_id": inspection_id}

    def add_review(
        self,
        inspection_id: str,
        *,
        expected_revision: int,
        reviewer: str,
        human_verdict: str,
        note: str,
    ) -> dict[str, Any]:
        if isinstance(expected_revision, bool) or not isinstance(expected_revision, int) or expected_revision < 0:
            raise ValueError("expected_revision must be a nonnegative integer")
        if not isinstance(reviewer, str) or not isinstance(note, str):
            raise ValueError("reviewer and note must be text")
        reviewer = reviewer.strip()
        note = note.strip()
        if not 1 <= len(reviewer) <= 80 or any(char in reviewer for char in "\r\n"):
            raise ValueError("reviewer must be 1–80 characters on one line")
        if human_verdict not in HUMAN_VERDICTS:
            raise ValueError("Unsupported human verdict")
        if not 1 <= len(note) <= 2000:
            raise ValueError("Review note must be 1–2000 characters")
        with self._write() as connection:
            row = connection.execute(
                "SELECT bundle_path, inspection_json_sha256, status FROM inspections WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Unknown inspection ID: {inspection_id}")
            if row["status"] != "completed":
                raise ValueError("Failed inspections cannot receive a visual review verdict")
            _, current_sha256 = validate_bundle(Path(row["bundle_path"]))
            if current_sha256 != row["inspection_json_sha256"]:
                raise ValueError("Registered inspection JSON was changed after import")
            current_revision = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM review_events WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()[0]
            if current_revision != expected_revision:
                raise ValueError(f"Review revision conflict: current revision is {current_revision}")
            event = {
                "inspection_id": inspection_id,
                "revision": current_revision + 1,
                "reviewer": reviewer,
                "human_verdict": human_verdict,
                "note": note,
                "recorded_at": _utc_now(),
                "automatic_pass_allowed": False,
            }
            connection.execute(
                """INSERT INTO review_events
                   (inspection_id, revision, reviewer, human_verdict, note, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (
                    inspection_id,
                    event["revision"],
                    reviewer,
                    human_verdict,
                    note,
                    event["recorded_at"],
                ),
            )
            return event

    def review_history(self, inspection_id: str) -> list[dict[str, Any]]:
        self.get_inspection(inspection_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT revision, reviewer, human_verdict, note, recorded_at
                   FROM review_events WHERE inspection_id = ? ORDER BY revision""",
                (inspection_id,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            connection.close()

    def add_sop_evidence(
        self,
        inspection_id: str,
        *,
        expected_review_revision: int,
        selected_by: str,
        defect_type: str,
        result: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one source snapshot to the exact current human-review revision."""
        if (
            isinstance(expected_review_revision, bool)
            or not isinstance(expected_review_revision, int)
            or expected_review_revision < 1
        ):
            raise ValueError("expected_review_revision must be a positive integer")
        if not isinstance(selected_by, str):
            raise ValueError("selected_by must be text")
        selected_by = selected_by.strip()
        if not 1 <= len(selected_by) <= 80 or any(char in selected_by for char in "\r\n"):
            raise ValueError("selected_by must be 1–80 characters on one line")
        if defect_type not in SOP_DEFECT_TYPES:
            raise ValueError("Unsupported SOP defect type")
        if not isinstance(result, dict) or result.get("status") != "found_simulation_only":
            raise ValueError("A validated simulation-only SOP result is required")
        if result.get("defect_type") != defect_type:
            raise ValueError("SOP result defect type differs")
        document_id = result.get("document_id")
        document_revision = result.get("revision")
        catalog_sha256 = result.get("catalog_sha256")
        lookup_date = result.get("as_of")
        sources = result.get("sources")
        if not isinstance(document_id, str) or not SOP_ID_PATTERN.fullmatch(document_id):
            raise ValueError("Invalid SOP document ID")
        if (
            isinstance(document_revision, bool)
            or not isinstance(document_revision, int)
            or document_revision < 1
        ):
            raise ValueError("Invalid SOP document revision")
        if not isinstance(catalog_sha256, str) or not SHA256_PATTERN.fullmatch(catalog_sha256):
            raise ValueError("Invalid SOP catalog checksum")
        try:
            parsed_date = datetime.strptime(lookup_date, "%Y-%m-%d").date()
        except (TypeError, ValueError) as exc:
            raise ValueError("Invalid SOP lookup date") from exc
        if parsed_date.isoformat() != lookup_date:
            raise ValueError("Invalid SOP lookup date")
        if not isinstance(sources, list) or not sources:
            raise ValueError("SOP source snapshot is missing")
        expected_prefix = f"{document_id}@r{document_revision}#"
        for source in sources:
            if (
                not isinstance(source, dict)
                or not isinstance(source.get("citation_id"), str)
                or not source["citation_id"].startswith(expected_prefix)
                or not isinstance(source.get("text"), str)
                or not source["text"].strip()
            ):
                raise ValueError("Invalid SOP source snapshot")
        sources_json = json.dumps(sources, sort_keys=True, ensure_ascii=False)

        with self._write() as connection:
            inspection = connection.execute(
                "SELECT bundle_path, inspection_json_sha256, status, category "
                "FROM inspections WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()
            if inspection is None:
                raise KeyError(f"Unknown inspection ID: {inspection_id}")
            if inspection["status"] != "completed":
                raise ValueError("Failed inspections cannot receive SOP evidence")
            _, current_sha256 = validate_bundle(Path(inspection["bundle_path"]))
            if current_sha256 != inspection["inspection_json_sha256"]:
                raise ValueError("Registered inspection JSON was changed after import")
            if result.get("product") != inspection["category"]:
                raise ValueError("SOP result product differs from the inspection")
            current_revision = connection.execute(
                "SELECT COALESCE(MAX(revision), 0) FROM review_events WHERE inspection_id = ?",
                (inspection_id,),
            ).fetchone()[0]
            if current_revision != expected_review_revision:
                raise ValueError(f"Review revision conflict: current revision is {current_revision}")
            review = connection.execute(
                "SELECT human_verdict FROM review_events "
                "WHERE inspection_id = ? AND revision = ?",
                (inspection_id, expected_review_revision),
            ).fetchone()
            if review is None or review["human_verdict"] != "ANOMALY_VISIBLE":
                raise ValueError("SOP evidence requires an ANOMALY_VISIBLE human review")
            existing = connection.execute(
                "SELECT * FROM sop_evidence_events "
                "WHERE inspection_id = ? AND review_revision = ?",
                (inspection_id, expected_review_revision),
            ).fetchone()
            material = {
                "defect_type": defect_type,
                "document_id": document_id,
                "document_revision": document_revision,
                "catalog_sha256": catalog_sha256,
                "lookup_date": lookup_date,
                "sources_json": sources_json,
            }
            if existing is not None:
                if all(existing[key] == value for key, value in material.items()):
                    event = dict(existing)
                    event["status"] = "already_recorded"
                    event["sources"] = json.loads(event.pop("sources_json"))
                    return event
                raise ValueError("SOP evidence already exists for this review revision")
            recorded_at = _utc_now()
            connection.execute(
                """INSERT INTO sop_evidence_events
                   (inspection_id, review_revision, selected_by, defect_type,
                    document_id, document_revision, catalog_sha256, lookup_date,
                    sources_json, recorded_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    inspection_id,
                    expected_review_revision,
                    selected_by,
                    defect_type,
                    document_id,
                    document_revision,
                    catalog_sha256,
                    lookup_date,
                    sources_json,
                    recorded_at,
                ),
            )
            return {
                "status": "recorded",
                "inspection_id": inspection_id,
                "review_revision": expected_review_revision,
                "selected_by": selected_by,
                "defect_type": defect_type,
                "document_id": document_id,
                "document_revision": document_revision,
                "catalog_sha256": catalog_sha256,
                "lookup_date": lookup_date,
                "sources": sources,
                "recorded_at": recorded_at,
            }

    def sop_evidence_history(self, inspection_id: str) -> list[dict[str, Any]]:
        self.get_inspection(inspection_id)
        connection = self._connect()
        try:
            rows = connection.execute(
                """SELECT event_id, review_revision, selected_by, defect_type,
                          document_id, document_revision, catalog_sha256, lookup_date,
                          sources_json, recorded_at
                   FROM sop_evidence_events WHERE inspection_id = ?
                   ORDER BY review_revision""",
                (inspection_id,),
            ).fetchall()
            result = []
            for row in rows:
                item = dict(row)
                item["sources"] = json.loads(item.pop("sources_json"))
                result.append(item)
            return result
        finally:
            connection.close()
