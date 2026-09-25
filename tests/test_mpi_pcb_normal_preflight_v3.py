import copy
import io
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from scripts.audit_mpi_pcb_normal_pool import _ids_sha256, propose_calibration_ids
from scripts.preflight_mpi_pcb_normals import select_calibration_ids
from scripts.preflight_mpi_pcb_normals_v3 import (
    EXPECTED_CANDIDATE_SHA256,
    EXPECTED_INVALID_PHOTO_IDS,
    scan_candidate,
    validate_pool_report,
)


REGISTRATION_SHA256 = "c055bddf6df145a1b038594b2225ae97dc44730d129b0f8dc5a06a7e3004e7e4"
ARCHIVE_SHA256 = "0fa9b20fae64142a418487a1d898ee2ee824eee0bd564896a245f4d991112607"
REGION = "grid1-1_s1024"


def reviewed_pool_report():
    all_ids = {f"{index:04d}" for index in range(1687)}
    invalid = EXPECTED_INVALID_PHOTO_IDS
    complete = sorted(all_ids - invalid)
    original = select_calibration_ids(all_ids)
    candidate = propose_calibration_ids(all_ids, invalid)
    issues = [{
        "region": REGION,
        "photo_id": photo_id,
        "member": f"extracted/{REGION}/train/good/{photo_id}.png",
        "verify_error": "SyntaxError: broken PNG file",
    } for photo_id in sorted(invalid)]
    return all_ids, {
        "status": "normal_pool_integrity_audited_test_sealed",
        "archive_sha256": ARCHIVE_SHA256,
        "registration_json_sha256": REGISTRATION_SHA256,
        "training_photo_ids": 1687,
        "regions": 12,
        "attempted_normal_tiles": 20244,
        "valid_normal_tiles": 20228,
        "invalid_normal_tiles": 16,
        "valid_dimensions": {"256x256": 20228},
        "issues_by_region": {REGION: 16},
        "issues": issues,
        "invalid_photo_ids": sorted(invalid),
        "complete_photo_ids": complete,
        "complete_photo_ids_sha256": _ids_sha256(complete),
        "complete_photo_count": 1671,
        "original_calibration_photo_ids_sha256": _ids_sha256(original),
        "candidate_protocol_version": 3,
        "candidate_selection_status": "candidate_ready_for_review",
        "candidate_calibration_photo_ids": candidate,
        "candidate_calibration_photo_ids_sha256": _ids_sha256(candidate),
        "candidate_calibration_photo_count": 169,
        "pillow_version": Image.__version__,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
        "automatic_pass_allowed": False,
    }


def png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), color).save(buffer, format="PNG")
    return buffer.getvalue()


class MpiPcbNormalPreflightV3Tests(unittest.TestCase):
    def test_reviewed_pool_report_gives_locked_candidate(self):
        all_ids, report = reviewed_pool_report()
        candidate, invalid = validate_pool_report(report, all_ids, REGISTRATION_SHA256)
        self.assertEqual(_ids_sha256(candidate), EXPECTED_CANDIDATE_SHA256)
        self.assertEqual(invalid, EXPECTED_INVALID_PHOTO_IDS)
        original = set(select_calibration_ids(all_ids))
        self.assertEqual(original - set(candidate), {"0158", "1303"})
        self.assertEqual(set(candidate) - original, {"0727", "1257"})

    def test_changed_candidate_or_test_access_is_rejected(self):
        all_ids, report = reviewed_pool_report()
        for field, value in (
            ("candidate_calibration_photo_ids", []),
            ("candidate_calibration_photo_ids_sha256", "wrong"),
            ("test_images_decoded", 1),
            ("invalid_normal_tiles", 15),
        ):
            changed = copy.deepcopy(report)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_pool_report(changed, all_ids, REGISTRATION_SHA256)

    def test_candidate_scan_reads_only_selected_training_normals(self):
        regions = [REGION, "grid1-2_s1024"]
        candidate = ["0000", "0001"]
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as writer:
            for region in regions:
                for photo_id in candidate:
                    writer.writestr(
                        f"extracted/{region}/train/good/{photo_id}.png",
                        png_bytes((int(photo_id), 20, 30)),
                    )
            writer.writestr(f"extracted/{REGION}/test/defect/0000.png", b"unreadable")
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as source:
            normal = {region: {
                photo_id: source.getinfo(f"extracted/{region}/train/good/{photo_id}.png")
                for photo_id in candidate
            } for region in regions}
            rows, dimensions = scan_candidate(source, normal, candidate)
        self.assertEqual(len(rows), 4)
        self.assertEqual(dimensions, {"4x3": 4})
        self.assertTrue(all("/train/good/" in row["member"] for row in rows))


if __name__ == "__main__":
    unittest.main()
