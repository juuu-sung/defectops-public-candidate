"""Resolve only audited, identical-tree history migrations for immutable bundles.

The bundled manifest is a maintainer-reviewed trust anchor, not user input.
Never load a map from a model bundle, environment variable or CLI argument.
An artifact retains its original commit ID; this module only records the
verified relationship to a commit in the current repository.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any


_MIGRATION_FILE = Path(__file__).with_name("history_migrations") / "privacy-20261116.json"
_MIGRATION_SHA256 = "382ed81b431a6fd756f74b42f7859960cf2e5563910c707f6dd1857959b6ac95"
_REWRITTEN_TIP = "70d495d969010322c97484b0d008ebe2d6c4479c"


def _git(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # A local replace ref must not manufacture ancestry or tree equality.
    return subprocess.run(
        ["git", "--no-replace-objects", "-C", str(root), *args],
        check=False, capture_output=True, text=True,
    )


def _ancestor(root: Path, commit: str) -> bool:
    return _git(root, "merge-base", "--is-ancestor", commit, "HEAD").returncode == 0


def resolve_commit_ancestry(root: Path, commit: Any, label: str) -> dict[str, str]:
    """Require direct ancestry or a checksum-pinned, identical-tree migration."""
    if not isinstance(commit, str) or re.fullmatch(r"[0-9a-f]{40}", commit) is None:
        raise ValueError(f"{label} is missing a full Git commit")
    if _ancestor(root, commit):
        return {"method": "direct_ancestor", "original_commit": commit,
                "resolved_commit": commit}

    try:
        raw = _MIGRATION_FILE.read_bytes()
    except OSError as exc:
        raise RuntimeError("Audited history migration manifest is unavailable") from exc
    if hashlib.sha256(raw).hexdigest() != _MIGRATION_SHA256:
        raise ValueError("Audited history migration manifest checksum differs")
    manifest = json.loads(raw)
    if (manifest["schema_version"] != 1
            or manifest["policy"] != "entire_git_tree_identical"
            or manifest["rewritten_tip"] != _REWRITTEN_TIP):
        raise ValueError("Unsupported history migration contract")
    record = manifest["commits"].get(commit)
    if record is None:
        raise RuntimeError(f"{label} commit has no audited identical-tree migration")
    if not _ancestor(root, _REWRITTEN_TIP):
        raise RuntimeError("History migration anchor is not an ancestor of HEAD")
    resolved = record["rewritten_commit"]
    if not _ancestor(root, resolved):
        raise RuntimeError(f"{label} migrated commit is not an ancestor of HEAD")
    result = _git(root, "rev-parse", "--verify", f"{resolved}^{{tree}}")
    if result.returncode != 0 or result.stdout.strip() != record["tree"]:
        raise ValueError(f"{label} migrated commit tree differs from audited source")
    return {
        "method": "audited_identical_tree_migration",
        "original_commit": commit,
        "resolved_commit": resolved,
        "source_tree": record["tree"],
        "migration_id": manifest["migration_id"],
        "migration_manifest_sha256": _MIGRATION_SHA256,
    }
