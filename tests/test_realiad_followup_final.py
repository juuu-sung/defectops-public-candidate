import hashlib
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

import numpy as np

from defectops.realiad_followup_final import (
    _archive_member,
    _load_lock,
    _read_sample,
    _write_sample,
    evaluate,
)


class RealIADFollowupFinalTests(unittest.TestCase):
    def setUp(self):
        self.project = Path(__file__).resolve().parents[1]

    def test_committed_final_lock_matches_protocol(self):
        lock, protocol, execution = _load_lock(self.project)
        self.assertEqual(lock["selected_candidate_id"], "fine_features_top_1pct")
        self.assertEqual(lock["final_seeds"], [42, 43, 44])
        self.assertEqual(lock["split_plan_sha256"], execution["plan_sha256"])
        self.assertEqual(protocol["final_gate"]["run_once"], True)

    def test_sample_is_atomic_and_resume_rejects_wrong_context_or_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample-0000.npz"
            metadata = {"source_image_member": "pcb/OK/a.jpg", "label": 0, "score": 1.25}
            anomaly_map = np.arange(16, dtype=np.float32).reshape(4, 4)
            mask = np.zeros((4, 4), dtype=np.bool_)
            _write_sample(path, metadata, anomaly_map, mask)
            loaded, loaded_map, loaded_mask = _read_sample(
                path, {"source_image_member": "pcb/OK/a.jpg", "label": 0}, 4
            )
            self.assertEqual(loaded, metadata)
            np.testing.assert_array_equal(loaded_map, anomaly_map)
            np.testing.assert_array_equal(loaded_mask, mask)
            with self.assertRaisesRegex(ValueError, "not rescored"):
                _read_sample(path, {"source_image_member": "pcb/OK/other.jpg"}, 4)
            with self.assertRaisesRegex(ValueError, "not rescored"):
                _read_sample(path, {"source_image_member": "pcb/OK/a.jpg"}, 8)
            with np.load(path, allow_pickle=False) as saved:
                changed_map = np.asarray(saved["anomaly_map"]).copy()
                changed_map[0, 0] += 1
                saved_metadata = str(saved["metadata"].item())
                digest = str(saved["digest"].item())
            np.savez_compressed(path, metadata=saved_metadata, anomaly_map=changed_map, mask=mask, digest=digest)
            with self.assertRaisesRegex(ValueError, "not rescored"):
                _read_sample(path, {"source_image_member": "pcb/OK/a.jpg"}, 4)

    def test_final_archive_member_requires_exact_bytes_and_safe_path(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.zip"
            with zipfile.ZipFile(path, "w") as archive:
                archive.writestr("pcb/OK/a.jpg", b"fake-jpeg-for-hash-test")
            with zipfile.ZipFile(path) as archive:
                expected = hashlib.sha256(b"fake-jpeg-for-hash-test").hexdigest()
                self.assertEqual(_archive_member(archive, "pcb/OK/a.jpg", expected), b"fake-jpeg-for-hash-test")
                with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                    _archive_member(archive, "pcb/OK/a.jpg", "0" * 64)
                with self.assertRaisesRegex(ValueError, "Unsafe"):
                    _archive_member(archive, "../pcb/OK/a.jpg", expected)

    def test_final_cannot_open_until_every_fit_is_complete(self):
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            gate = {
                "paths": {"project": project, "source_archive": project / "missing.zip"},
                "lock": {"final_seeds": [42, 43, 44]},
            }
            with patch("defectops.realiad_followup_final._fit_record") as fit, patch(
                "defectops.realiad_followup_final.validate_gate", return_value=gate
            ) as validate:
                fit.side_effect = [{"status": "fit_calibrated_final_locked"}, RuntimeError("seed not fitted")]
                with self.assertRaisesRegex(RuntimeError, "seed not fitted"):
                    evaluate(gate, "realiad_pcb", 42)
                validate.assert_called_once()
            self.assertFalse((project / "artifacts").exists())


if __name__ == "__main__":
    unittest.main()
