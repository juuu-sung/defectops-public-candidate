"""Deterministic lookup of explicitly simulated, versioned SOP excerpts."""

from __future__ import annotations

import hashlib
import json
import re
import stat
from datetime import date
from pathlib import Path
from typing import Any


DEFECT_LABELS = {
    "UNKNOWN": "모름 / 아직 선택하지 않음",
    "bent_lead": "금속 다리(리드) 굽힘·삽입 위치 이상",
    "cut_lead": "금속 다리(리드) 절단",
    "damaged_case": "케이스 손상",
    "misplaced": "부품 위치 이상",
}
MAX_CATALOG_BYTES = 1_000_000
ID_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9_-]{1,63}\Z")
SECTION_PATTERN = re.compile(r"[A-Z0-9][A-Z0-9_-]{1,31}\Z")


def _date(value: Any, label: str) -> date:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO date")
    try:
        result = date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be an ISO date") from exc
    if result.isoformat() != value:
        raise ValueError(f"{label} must be an ISO date")
    return result


def load_mock_catalog(path: Path) -> dict[str, Any]:
    """Load a bounded local catalog, validating every citation and revision."""
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError("Mock SOP catalog must be a regular file")
    if not 0 < path.stat().st_size <= MAX_CATALOG_BYTES:
        raise ValueError("Mock SOP catalog has an invalid size")
    raw = path.read_bytes()
    if len(raw) > MAX_CATALOG_BYTES:
        raise ValueError("Mock SOP catalog is too large")
    catalog = json.loads(raw)
    if not isinstance(catalog, dict) or catalog.get("schema_version") != 1:
        raise ValueError("Unsupported mock SOP catalog schema")
    if catalog.get("purpose") != "simulation_only":
        raise ValueError("Catalog must be marked simulation_only")
    documents = catalog.get("documents")
    if not isinstance(documents, list) or not documents:
        raise ValueError("Mock SOP documents are missing")
    seen = set()
    for document in documents:
        if not isinstance(document, dict):
            raise ValueError("Invalid mock SOP document")
        doc_id = document.get("document_id")
        revision = document.get("revision")
        if not isinstance(doc_id, str) or not ID_PATTERN.fullmatch(doc_id):
            raise ValueError("Invalid mock SOP document ID")
        if isinstance(revision, bool) or not isinstance(revision, int) or revision < 1:
            raise ValueError("Invalid mock SOP revision")
        if (doc_id, revision) in seen:
            raise ValueError("Duplicate mock SOP document revision")
        seen.add((doc_id, revision))
        if document.get("status") not in {"active", "retired"}:
            raise ValueError("Invalid mock SOP document status")
        if not isinstance(document.get("product"), str) or not document["product"]:
            raise ValueError("Invalid mock SOP product")
        if document.get("defect_type") not in DEFECT_LABELS or document["defect_type"] == "UNKNOWN":
            raise ValueError("Invalid mock SOP defect type")
        if not isinstance(document.get("title"), str) or not document["title"].strip():
            raise ValueError("Invalid mock SOP title")
        start = _date(document.get("valid_from"), "valid_from")
        if "valid_to" not in document:
            raise ValueError("Mock SOP valid_to field is missing")
        end = document.get("valid_to")
        if end is not None and _date(end, "valid_to") < start:
            raise ValueError("Mock SOP effective dates are reversed")
        sections = document.get("sections")
        if not isinstance(sections, list) or not sections:
            raise ValueError("Mock SOP sections are missing")
        section_ids = set()
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError("Invalid mock SOP section")
            section_id = section.get("section_id")
            if not isinstance(section_id, str) or not SECTION_PATTERN.fullmatch(section_id):
                raise ValueError("Invalid mock SOP section ID")
            if section_id in section_ids:
                raise ValueError("Duplicate mock SOP section ID")
            section_ids.add(section_id)
            if not isinstance(section.get("text"), str) or not section["text"].strip():
                raise ValueError("Invalid mock SOP section text")
    catalog["catalog_sha256"] = hashlib.sha256(raw).hexdigest()
    return catalog


def find_mock_sop(
    catalog: dict[str, Any],
    *,
    product: str,
    defect_type: str,
    human_verdict: str | None,
    as_of: date,
) -> dict[str, Any]:
    """Return only an exact, currently active source for an explicit human choice."""
    if human_verdict != "ANOMALY_VISIBLE":
        return {"status": "human_review_required"}
    if defect_type == "UNKNOWN":
        return {"status": "defect_type_required"}
    if defect_type not in DEFECT_LABELS:
        raise ValueError("Unknown defect type")
    if not isinstance(product, str) or not product or not isinstance(as_of, date):
        raise ValueError("Product and lookup date are required")
    candidates = [
        document
        for document in catalog["documents"]
        if document["product"] == product
        and document["defect_type"] == defect_type
        and document["status"] == "active"
        and _date(document["valid_from"], "valid_from") <= as_of
        and (document["valid_to"] is None or as_of <= _date(document["valid_to"], "valid_to"))
    ]
    if not candidates:
        return {"status": "not_found"}
    if len(candidates) != 1:
        raise ValueError("Ambiguous active mock SOP revisions; no result can be shown")
    document = candidates[0]
    return {
        "status": "found_simulation_only",
        "product": product,
        "defect_type": defect_type,
        "as_of": as_of.isoformat(),
        "document_id": document["document_id"],
        "revision": document["revision"],
        "title": document["title"],
        "valid_from": document["valid_from"],
        "valid_to": document["valid_to"],
        "catalog_sha256": catalog["catalog_sha256"],
        "sources": [
            {
                "citation_id": f"{document['document_id']}@r{document['revision']}#{section['section_id']}",
                "text": section["text"],
            }
            for section in document["sections"]
        ],
    }
