import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from PIL import Image

from defectops.data_audit import audit_dataset
from defectops.manifest import canonical, verify_manifest
from defectops.realiad_followup_registration import (
    DEVELOPMENT_MANIFEST_NAMES,
    FINAL_MANIFEST_NAMES,
    register_followup_data,
    verify_followup_final_manifest,
)


class RealIADFollowupRegistrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "data"
        (self.root / "manifests").mkdir(parents=True)
        self.plan_path = self.root / "manifests/realiad-followup-split-plan-v1.json"
        self.plan = self.make_plan()
        self.plan_path.write_text(json.dumps(self.plan), encoding="utf-8")
        self.archives = {}
        for category, source in (
            ("realiad_pcb", "pcb"),
            ("realiad_terminalblock", "terminalblock"),
        ):
            path = Path(self.temp.name) / f"{source}.zip"
            with zipfile.ZipFile(path, "w") as archive:
                for member_index, row in enumerate(self.plan["records"]):
                    if row["category"] != category:
                        continue
                    image_member = f"{source}/{row['source_image_path']}"
                    archive.writestr(
                        image_member,
                        self.encoded_image(image_member, member_index),
                    )
                    if row["source_mask_path"] is not None:
                        mask_member = f"{source}/{row['source_mask_path']}"
                        archive.writestr(mask_member, self.encoded_mask())
            self.archives[category] = path

    def test_development_is_extracted_and_final_stays_sealed(self):
        payloads = register_followup_data(self.root, self.plan_path, self.archives)
        self.assertEqual(len(payloads), 4)
        for category, source in (
            ("realiad_pcb", "pcb"),
            ("realiad_terminalblock", "terminalblock"),
        ):
            development = json.loads(
                (self.root / "manifests" / DEVELOPMENT_MANIFEST_NAMES[category]).read_text()
            )
            final = json.loads(
                (self.root / "manifests" / FINAL_MANIFEST_NAMES[category]).read_text()
            )
            verify_manifest(self.root, development)
            verify_followup_final_manifest(self.root, final)
            self.assertTrue(all(row["protocol_split"] != "final_holdout" for row in development["records"]))
            self.assertTrue(final["sealed"])
            self.assertFalse(final["images_extracted"])
            for row in final["records"]:
                relative = row["source_image_member"].removeprefix(f"{source}/")
                self.assertFalse(
                    (self.root / "real_iad_followup/images" / source / relative).exists()
                )

    def test_tampered_development_file_is_rejected(self):
        register_followup_data(self.root, self.plan_path, self.archives)
        path = self.root / "manifests" / DEVELOPMENT_MANIFEST_NAMES["realiad_pcb"]
        manifest = json.loads(path.read_text())
        image = self.root / manifest["records"][0]["image_path"]
        image.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            verify_manifest(self.root, manifest)

    def test_tampered_final_label_is_rejected_even_with_recomputed_checksum(self):
        register_followup_data(self.root, self.plan_path, self.archives)
        path = self.root / "manifests" / FINAL_MANIFEST_NAMES["realiad_pcb"]
        manifest = json.loads(path.read_text())
        manifest["records"][0]["defect_type"] = "CHANGED"
        unsigned = dict(manifest)
        unsigned.pop("manifest_sha256")
        manifest["manifest_sha256"] = hashlib.sha256(canonical(unsigned)).hexdigest()
        with self.assertRaisesRegex(ValueError, "manifest mismatch"):
            verify_followup_final_manifest(self.root, manifest)

    def test_followup_development_audit_accepts_jpeg_and_does_not_require_final(self):
        register_followup_data(self.root, self.plan_path, self.archives)
        for category in ("realiad_pcb", "realiad_terminalblock"):
            path = self.root / "manifests" / DEVELOPMENT_MANIFEST_NAMES[category]
            manifest = json.loads(path.read_text())
            report = audit_dataset(self.root, manifest)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["errors"], [])
            self.assertEqual(set(report["image_specs"]), {"8x8/RGB"})

    @staticmethod
    def encoded_image(member, index):
        digest = hashlib.sha256(f"{index}:{member}".encode()).digest()
        image = Image.new("RGB", (8, 8), tuple(digest[:3]))
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=100, subsampling=0)
        return output.getvalue()

    @staticmethod
    def encoded_mask():
        image = Image.new("L", (8, 8), 255)
        output = io.BytesIO()
        image.save(output, format="PNG")
        return output.getvalue()

    @staticmethod
    def make_plan():
        records = []
        for category, source in (
            ("realiad_pcb", "pcb"),
            ("realiad_terminalblock", "terminalblock"),
        ):
            for index, (split, defect) in enumerate(
                (
                    ("train", "OK"),
                    ("calibration", "OK"),
                    ("development", "OK"),
                    ("development", "D1"),
                    ("final_holdout", "OK"),
                    ("final_holdout", "D1"),
                ),
                start=1,
            ):
                base = "OK" if defect == "OK" else f"NG/{defect}"
                image = f"{base}/S{index:04d}/{source}_{index:04d}_{defect}_C1_fixture.jpg"
                mask = None if defect == "OK" else image.removesuffix(".jpg") + ".png"
                records.append(
                    {
                        "category": category,
                        "source_object": source,
                        "official_split": "train" if split in {"train", "calibration"} else "test",
                        "planned_split": split,
                        "label": int(defect != "OK"),
                        "defect_type": defect,
                        "view": "C1",
                        "sample_group": f"{source}/{base}/S{index:04d}",
                        "source_image_path": image,
                        "source_mask_path": mask,
                    }
                )
        plan = {
            "schema_version": 1,
            "dataset": "real_iad",
            "protocol": "A-normal-only-v2",
            "registration_status": "annotation_metadata_observed_images_and_scores_unobserved",
            "evidence_status": "realiad_transistor1_and_plastic_nut_c1_test_observed_new_categories_unobserved",
            "annotation_variant": "realiad_jsons_sv",
            "categories": ["realiad_pcb", "realiad_terminalblock"],
            "split_seed": 20260922,
            "calibration_group_fraction": 0.2,
            "development_group_fraction": 0.5,
            "split_algorithm": "fixture",
            "annotation_zip_sha256": "0" * 64,
            "annotation_sha256": {},
            "config_sha256": "1" * 64,
            "records": records,
        }
        plan["plan_sha256"] = hashlib.sha256(canonical(plan)).hexdigest()
        return plan


if __name__ == "__main__":
    unittest.main()
