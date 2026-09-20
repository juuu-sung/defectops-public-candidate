import hashlib
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from defectops.additional_category_handoff import (
    EXPECTED_COUNTS,
    import_handoff,
    validate_handoff,
)
from defectops.manifest import canonical


class AdditionalCategoryHandoffTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    @staticmethod
    def manifest(category):
        records = []
        index = 0
        for split, count in (
            ("train", EXPECTED_COUNTS[category]["train_good"]),
            ("calibration", EXPECTED_COUNTS[category]["calibration_good"]),
        ):
            for item in range(count):
                path = f"{category}/train/good/{split}-{item:03}.png"
                records.append(
                    {
                        "image_path": path,
                        "image_sha256": hashlib.sha256(path.encode()).hexdigest(),
                        "category": category,
                        "official_split": "train",
                        "split": split,
                        "label": 0,
                        "defect_type": "good",
                        "mask_path": None,
                        "mask_sha256": None,
                    }
                )
                index += 1
        for item in range(EXPECTED_COUNTS[category]["test_good"]):
            path = f"{category}/test/good/{item:03}.png"
            records.append(
                {
                    "image_path": path,
                    "image_sha256": hashlib.sha256(path.encode()).hexdigest(),
                    "category": category,
                    "official_split": "test",
                    "split": "test",
                    "label": 0,
                    "defect_type": "good",
                    "mask_path": None,
                    "mask_sha256": None,
                }
            )
        for item in range(EXPECTED_COUNTS[category]["test_anomaly"]):
            path = f"{category}/test/defect/{item:03}.png"
            mask = f"{category}/ground_truth/defect/{item:03}_mask.png"
            records.append(
                {
                    "image_path": path,
                    "image_sha256": hashlib.sha256(path.encode()).hexdigest(),
                    "category": category,
                    "official_split": "test",
                    "split": "test",
                    "label": 1,
                    "defect_type": "defect",
                    "mask_path": mask,
                    "mask_sha256": hashlib.sha256(mask.encode()).hexdigest(),
                }
            )
        records.sort(key=lambda row: row["image_path"])
        manifest = {
            "schema_version": 1,
            "dataset": "mvtec_ad",
            "protocol": "A-normal-only-v1",
            "categories": [category],
            "split_seed": 42,
            "calibration_fraction": 0.2,
            "split_algorithm": "sha256(seed:relative_path), ceil(n*fraction)",
            "records": records,
        }
        manifest["manifest_sha256"] = hashlib.sha256(canonical(manifest)).hexdigest()
        return manifest

    @staticmethod
    def summary(category):
        counts = EXPECTED_COUNTS[category]
        return "\n".join(
            [
                "# 데이터 준비 결과",
                "점검 상태: **passed**",
                "- 오류: 0건",
                "- 디코딩 픽셀 중복: 0쌍",
                f"| {category}/train/good | {counts['train_good']} |",
                f"| {category}/calibration/good | {counts['calibration_good']} |",
                f"| {category}/test/good | {counts['test_good']} |",
                f"| {category}/test/anomaly | {counts['test_anomaly']} |",
            ]
        )

    def make_zip(self, tamper=False):
        path = self.root / "handoff.zip"
        cable = self.manifest("cable")
        if tamper:
            cable["records"][0]["label"] = 0
        with zipfile.ZipFile(path, "w") as archive:
            archive.writestr("cable-v1.json", json.dumps(cable))
            archive.writestr("metal-nut-v1.json", json.dumps(self.manifest("metal_nut")))
            archive.writestr("summary.md", self.summary("cable"))
            archive.writestr("summary.md", self.summary("metal_nut"))
        return path

    def test_valid_handoff_is_checked_and_imported_without_overwrite(self):
        path = self.make_zip()
        expected = hashlib.sha256(path.read_bytes()).hexdigest()
        result = validate_handoff(path, expected)
        self.assertEqual(result["status"], "validated_additional_category_handoff")
        imported = import_handoff(result, self.root / "manifests", self.root / "reports")
        self.assertEqual(set(imported), {
            "cable_manifest", "metal_nut_manifest", "cable_report", "metal_nut_report"
        })
        with self.assertRaises(FileExistsError):
            import_handoff(result, self.root / "manifests", self.root / "reports")

    def test_zip_or_manifest_tampering_is_rejected(self):
        path = self.make_zip()
        with self.assertRaisesRegex(ValueError, "ZIP SHA-256"):
            validate_handoff(path, "0" * 64)
        tampered = self.make_zip(tamper=True)
        with self.assertRaisesRegex(ValueError, "manifest contract mismatch"):
            validate_handoff(tampered)


if __name__ == "__main__":
    unittest.main()
