"""Deliver immutable inspection bundles to the local review store, without inference.

Completed bundles are the durable source of recovery. No registration metadata is
written into inspection.json, whose hash is part of the review store contract.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import UUID

from defectops.review_store import ReviewStore, validate_bundle


class ReviewRegistration:
    def __init__(self, output_root: Path, database_path: Path):
        self.root = output_root.resolve()
        self.store = ReviewStore(database_path)

    def initialize(self) -> None:
        if self.store.database_path.exists():
            with sqlite3.connect(self.store.database_path) as connection:
                tables = {row[0] for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")}
                if tables and 'inspections' not in tables:
                    raise ValueError('Use a review database, not another application database')
        self.store.initialize()

    def _bundle(self, inspection_id: str) -> Path:
        if UUID(inspection_id).hex != inspection_id:
            raise ValueError('Non-canonical inspection ID')
        bundle = self.root / inspection_id
        if bundle.is_symlink() or not bundle.is_dir():
            raise ValueError('Inspection directory is missing or linked')
        bundle.resolve(strict=True).relative_to(self.root)
        return bundle

    def inspect(self, inspection_id: str, *, register: bool = False) -> dict:
        result = {'inspection_id': inspection_id, 'status': 'PENDING'}
        try:
            bundle = self._bundle(inspection_id)
            if register:
                self.store.register(bundle)  # Includes checksums and idempotent conflict checks.
            else:
                _, digest = validate_bundle(bundle)
                try:
                    saved = self.store.get_inspection(inspection_id)
                except KeyError:
                    return dict(result, reason='NOT_REGISTERED')
                if (saved['bundle_path'] != str(bundle.resolve())
                        or saved['inspection_json_sha256'] != digest):
                    raise ValueError('Registered content or path differs')
            return dict(result, status='REGISTERED')
        except (sqlite3.Error, PermissionError) as exc:
            # Database locks/unavailability never erase or rerun the inspection.
            return dict(result, reason='STORE_UNAVAILABLE', error_type=type(exc).__name__)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return dict(result, status='BLOCKED', reason='BUNDLE_OR_ID_CONFLICT',
                        error_type=type(exc).__name__)

    def reconcile(self) -> dict:
        """Called before accepting requests; only API-style UUID directories are eligible."""
        counts = {'REGISTERED': 0, 'PENDING': 0, 'BLOCKED': 0}
        for path in sorted(self.root.iterdir()):
            try:
                if UUID(path.name).hex != path.name:
                    continue
            except ValueError:
                continue
            state = self.inspect(path.name, register=True)
            counts[state['status']] += 1
        return counts
