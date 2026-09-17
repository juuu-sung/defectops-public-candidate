"""Verified display data for the Mac-local human-review screen."""

from __future__ import annotations

import hashlib
import stat
from pathlib import Path
from typing import Any

from defectops.review_store import ARTIFACT_NAMES, ReviewStore


DISPLAY_ARTIFACTS = ("input", "overlay", "heatmap")
MAX_DISPLAY_BYTES = 20 * 1024 * 1024


def _read_verified_image(directory: Path, name: str, expected_sha256: str) -> bytes:
    """Read exactly the verified bytes that the UI will show, not a mutable path."""
    path = directory / name
    if path.is_symlink() or not stat.S_ISREG(path.stat().st_mode):
        raise ValueError(f"Display artifact is not a regular file: {name}")
    if not 0 < path.stat().st_size <= MAX_DISPLAY_BYTES:
        raise ValueError(f"Display artifact has an invalid size: {name}")
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected_sha256:
        raise ValueError(f"Display artifact checksum differs: {name}")
    return data


def load_review_case(store: ReviewStore, inspection_id: str) -> dict[str, Any]:
    """Verify a registered bundle before exposing any image or review controls."""
    store.verify(inspection_id)
    item = store.get_inspection(inspection_id)
    record = item["record"]
    if record["status"] != "completed":
        return {"item": item, "images": {}}
    artifacts = record["artifacts"]
    directory = Path(item["bundle_path"])
    images = {
        key: _read_verified_image(directory, ARTIFACT_NAMES[key], artifacts[f"{key}_sha256"])
        for key in DISPLAY_ARTIFACTS
    }
    return {"item": item, "images": images}
