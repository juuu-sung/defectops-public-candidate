import copy
import hashlib
import io
import unittest
import zipfile

from PIL import Image

from scripts.audit_mpi_pcb_normal_pool import _ids_sha256, propose_calibration_ids
from scripts.preflight_mpi_pcb_normals import summarize_decoded_duplicates
from scripts.preflight_mpi_pcb_normals_v3 import EXPECTED_INVALID_PHOTO_IDS
from scripts.preflight_mpi_pcb_normals_v4 import (
    EXPECTED_V4_CANDIDATE_SHA256,
    scan_v4_candidate,
    select_v4_ids,
    validate_v3_report,
)


REGISTRATION_SHA256 = "c055bddf6df145a1b038594b2225ae97dc44730d129b0f8dc5a06a7e3004e7e4"
POOL_SHA256 = "33b58281d5840550712f253a793913754cdfd8a9a8bcecc91cd18d9720d41703"
ARCHIVE_SHA256 = "0fa9b20fae64142a418487a1d898ee2ee824eee0bd564896a245f4d991112607"
REGIONS = [f"grid{row}-{column}_s1024" for row in range(1, 5) for column in range(1, 4)]


def synthetic_v3_report():
    all_ids = {f"{index:04d}" for index in range(1687)}
    candidate = propose_calibration_ids(all_ids, EXPECTED_INVALID_PHOTO_IDS)
    rows = []
    for region in REGIONS:
        for photo_id in candidate:
            identical_id = "0879" if photo_id == "0722" else photo_id
            digest = hashlib.sha256(f"{region}/{identical_id}".encode()).hexdigest()
            rows.append({
                "region": region,
                "photo_id": photo_id,
                "member": f"extracted/{region}/train/good/{photo_id}.png",
                "width": 256,
                "height": 256,
                "member_sha256": digest,
                "decoded_rgb_sha256": digest,
            })
    tile_summary, composites = summarize_decoded_duplicates(rows, candidate, REGIONS)
    return all_ids, candidate, {
        "status": "normal_preflight_v3_needs_review_test_sealed",
        "protocol_version": 3,
        "archive_sha256": ARCHIVE_SHA256,
        "registration_json_sha256": REGISTRATION_SHA256,
        "pool_audit_json_sha256": POOL_SHA256,
        "calibration_photo_ids": candidate,
        "calibration_photo_ids_sha256": _ids_sha256(candidate),
        "calibration_original_photos": 169,
        "calibration_region_images_decoded": 2028,
        "regions": 12,
        "dimensions": {"256x256": 2028},
        "calibration_rows": rows,
        "tile_duplicate_summary": tile_summary,
        "identical_12_region_photo_id_groups": composites,
        "pillow_version": Image.__version__,
        "test_images_decoded": 0,
        "test_masks_decoded": 0,
        "model_scores_computed": 0,
        "automatic_pass_allowed": False,
    }


def png_bytes(color):
    buffer = io.BytesIO()
    Image.new("RGB", (256, 256), color).save(buffer, format="PNG")
    return buffer.getvalue()


class MpiPcbNormalPreflightV4Tests(unittest.TestCase):
    def test_reviewed_duplicate_and_locked_replacement(self):
        all_ids, candidate, report = synthetic_v3_report()
        validate_v3_report(report, candidate, REGIONS, REGISTRATION_SHA256, POOL_SHA256)
        v4 = select_v4_ids(all_ids, EXPECTED_INVALID_PHOTO_IDS, candidate)
        self.assertEqual(v4[-1], "1481")
        self.assertIn("0879", v4)
        self.assertNotIn("0722", v4)
        self.assertEqual(_ids_sha256(v4), EXPECTED_V4_CANDIDATE_SHA256)

    def test_tampered_duplicate_or_test_opening_rejected(self):
        _, candidate, report = synthetic_v3_report()
        for field, value in (
            ("test_images_decoded", 1),
            ("status", "normal_preflight_v3_passed_test_sealed"),
            ("calibration_region_images_decoded", 2027),
        ):
            changed = copy.deepcopy(report)
            changed[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_v3_report(changed, candidate, REGIONS, REGISTRATION_SHA256, POOL_SHA256)
        changed = copy.deepcopy(report)
        next(row for row in changed["calibration_rows"] if row["photo_id"] == "0722")[
            "member_sha256"
        ] = "0" * 64
        with self.assertRaises(ValueError):
            validate_v3_report(changed, candidate, REGIONS, REGISTRATION_SHA256, POOL_SHA256)
        changed = copy.deepcopy(report)
        changed["calibration_rows"].pop()
        with self.assertRaises(ValueError):
            validate_v3_report(changed, candidate, REGIONS, REGISTRATION_SHA256, POOL_SHA256)

    def test_v4_scans_only_selected_training_normals(self):
        regions = REGIONS[:2]
        candidate = ["0879", "1481"]
        archive = io.BytesIO()
        with zipfile.ZipFile(archive, "w") as writer:
            for region in regions:
                for index, photo_id in enumerate(candidate):
                    writer.writestr(
                        f"extracted/{region}/train/good/{photo_id}.png",
                        png_bytes((index * 100, 20, 30)),
                    )
            writer.writestr(f"extracted/{regions[0]}/test/defect/0000.png", b"unreadable")
        with zipfile.ZipFile(io.BytesIO(archive.getvalue())) as source:
            normal = {region: {
                photo_id: source.getinfo(f"extracted/{region}/train/good/{photo_id}.png")
                for photo_id in candidate
            } for region in regions}
            rows, dimensions = scan_v4_candidate(source, normal, candidate)
        self.assertEqual(len(rows), 4)
        self.assertEqual(dimensions, {"256x256": 4})
        self.assertTrue(all("/train/good/" in row["member"] for row in rows))


if __name__ == "__main__":
    unittest.main()
