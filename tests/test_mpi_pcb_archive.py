import unittest
import zipfile

from scripts.audit_mpi_pcb_archive import audit_members


def sample_infos():
    names = []
    for region in range(12):
        root = f"extracted/grid{region}_s1024"
        names.append(f"{root}/train/good/0000.png")
        for number in range(55):
            stem = f"{number:04d}.png"
            for group in ("test/good", "test/defect", "ground_truth/defect"):
                names.append(f"{root}/{group}/{stem}")
    return [zipfile.ZipInfo(name) for name in names]


class MpiPcbArchiveTests(unittest.TestCase):
    def test_valid_metadata_does_not_decode_test(self):
        report = audit_members(sample_infos())
        self.assertEqual(report["status"], "registered_metadata_only")
        self.assertEqual(report["region_count"], 12)
        self.assertEqual(report["test_images_decoded"], 0)
        self.assertEqual(report["test_masks_decoded"], 0)

    def test_missing_mask_stops_registration(self):
        infos = sample_infos()
        infos = [info for info in infos if info.filename !=
                 "extracted/grid0_s1024/ground_truth/defect/0000.png"]
        report = audit_members(infos)
        self.assertEqual(report["status"], "failed_metadata_audit")
        self.assertTrue(any("defect image/mask stems differ" in error
                            for error in report["errors"]))

    def test_unsafe_path_stops_registration(self):
        infos = sample_infos()
        infos.append(zipfile.ZipInfo("extracted/../../other.png"))
        report = audit_members(infos)
        self.assertEqual(report["status"], "failed_metadata_audit")


if __name__ == "__main__":
    unittest.main()
