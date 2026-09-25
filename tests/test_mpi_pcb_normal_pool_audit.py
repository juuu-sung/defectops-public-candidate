import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

from PIL import Image

from scripts.audit_mpi_pcb_normal_pool import propose_calibration_ids, run, scan_normal_pool
from scripts.preflight_mpi_pcb_normals import select_calibration_ids
from tests.test_mpi_pcb_selected_normal_decode_audit import corrupt_idat_crc


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


class MpiPcbNormalPoolAuditTests(unittest.TestCase):
    def test_no_invalid_ids_preserves_original_cohort(self):
        ids = {f"{index:04d}" for index in range(1687)}
        self.assertEqual(propose_calibration_ids(ids, set()), select_calibration_ids(ids))

    def test_invalid_ids_are_replaced_by_next_hash_ranked_complete_ids(self):
        ids = {f"{index:04d}" for index in range(1687)}
        original = select_calibration_ids(ids)
        bad = {original[10], original[20]}
        proposed = propose_calibration_ids(ids, bad)
        self.assertEqual(len(proposed), 169)
        self.assertNotIn(original[10], proposed)
        self.assertNotIn(original[20], proposed)
        self.assertEqual([item for item in original if item not in bad], proposed[:167])
        self.assertEqual(len(set(proposed)), 169)

    def test_insufficient_complete_ids_yields_no_candidate(self):
        ids = {"0000", "0001", "0002"}
        with patch("scripts.audit_mpi_pcb_normal_pool.EXPECTED_NORMAL_TRAIN_COUNT", 3):
            self.assertEqual(propose_calibration_ids(ids, {"0000", "0001"}, count=2), [])

    def test_all_normal_tiles_scanned_and_test_member_never_read(self):
        regions = ["grid1-1_s1024", "grid1-2_s1024"]
        photo_ids = ["0000", "0001", "0002"]
        archive_bytes = io.BytesIO()
        with zipfile.ZipFile(archive_bytes, "w") as writer:
            for region in regions:
                for photo_id in photo_ids:
                    raw = png_bytes()
                    if region == regions[1] and photo_id == "0001":
                        raw = corrupt_idat_crc(raw)
                    writer.writestr(f"extracted/{region}/train/good/{photo_id}.png", raw)
            writer.writestr("extracted/grid1-1_s1024/test/defect/0000.png", b"not image")
        with zipfile.ZipFile(io.BytesIO(archive_bytes.getvalue())) as source:
            normal = {
                region: {
                    photo_id: source.getinfo(
                        f"extracted/{region}/train/good/{photo_id}.png"
                    )
                    for photo_id in photo_ids
                }
                for region in regions
            }
            attempted, dimensions, issues_by_region, issues, invalid_ids = scan_normal_pool(
                source, normal,
            )
        self.assertEqual(attempted, 6)
        self.assertEqual(dimensions, {"4x3": 5})
        self.assertEqual(issues_by_region, {regions[1]: 1})
        self.assertEqual(len(issues), 1)
        self.assertEqual(invalid_ids, {"0001"})
        self.assertIn("bad header checksum", issues[0]["verify_error"])

    def test_run_writes_candidate_report_without_opening_test(self):
        regions = ["grid1-1_s1024", "grid1-2_s1024"]
        photo_ids = ["0000", "0001", "0002"]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            archive_path = root / "extracted.zip"
            with zipfile.ZipFile(archive_path, "w") as writer:
                for region in regions:
                    for photo_id in photo_ids:
                        raw = png_bytes()
                        if region == regions[1] and photo_id == "0001":
                            raw = corrupt_idat_crc(raw)
                        writer.writestr(f"extracted/{region}/train/good/{photo_id}.png", raw)
                writer.writestr("extracted/grid1-1_s1024/test/defect/0000.png", b"bad")
            registration = root / "registration.json"
            registration.write_text(json.dumps({
                "status": "registered_metadata_only", "region_count": 2,
                "archive_md5": "md5", "archive_sha256": "sha",
                "archive_size_bytes": archive_path.stat().st_size,
                "test_images_decoded": 0, "test_masks_decoded": 0,
                "model_scores_computed": 0, "errors": [],
            }), encoding="utf-8")
            output = root / "pool-report.json"

            def small_normals(infos):
                lookup = {info.filename: info for info in infos}
                return {region: {
                    photo_id: lookup[f"extracted/{region}/train/good/{photo_id}.png"]
                    for photo_id in photo_ids
                } for region in regions}

            with (
                patch("scripts.audit_mpi_pcb_normal_pool.SOURCE_MD5", "md5"),
                patch("scripts.audit_mpi_pcb_normal_pool.EXPECTED_ARCHIVE_SHA256", "sha"),
                patch("scripts.audit_mpi_pcb_normal_pool.EXPECTED_REGION_COUNT", 2),
                patch("scripts.audit_mpi_pcb_normal_pool.EXPECTED_NORMAL_TRAIN_COUNT", 3),
                patch("scripts.audit_mpi_pcb_normal_pool.CALIBRATION_PHOTOS", 2),
                patch("scripts.audit_mpi_pcb_normal_pool.archive_hashes", return_value=("md5", "sha")),
                patch("scripts.audit_mpi_pcb_normal_pool.audit_members", return_value={"status": "registered_metadata_only"}),
                patch("scripts.audit_mpi_pcb_normal_pool.normal_members", side_effect=small_normals),
                patch("scripts.audit_mpi_pcb_normal_pool.select_calibration_ids", return_value=["0000", "0001"]),
            ):
                report = run(archive_path, registration, output)
            self.assertEqual(report["attempted_normal_tiles"], 6)
            self.assertEqual(report["invalid_photo_ids"], ["0001"])
            self.assertEqual(report["candidate_calibration_photo_count"], 2)
            self.assertEqual(set(report["candidate_calibration_photo_ids"]), {"0000", "0002"})
            self.assertEqual(report["test_images_decoded"], 0)
            self.assertEqual(json.loads(output.read_text())["issues"], report["issues"])


if __name__ == "__main__":
    unittest.main()
