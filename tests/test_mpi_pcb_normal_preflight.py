import io
import unittest
import zipfile

from PIL import Image

from scripts.preflight_mpi_pcb_normals import (
    decode_normal,
    normal_members,
    select_calibration_ids,
)


class MpiPcbNormalPreflightTests(unittest.TestCase):
    def test_selection_is_fixed_and_has_original_photo_count(self):
        ids = {f"{index:04d}" for index in range(1687)}
        selected = select_calibration_ids(ids)
        self.assertEqual(len(selected), 169)
        self.assertEqual(selected, select_calibration_ids(set(reversed(sorted(ids)))))

    def test_normal_ids_must_match_across_regions(self):
        infos = [
            zipfile.ZipInfo(f"extracted/grid{region}_s1024/train/good/{index:04d}.png")
            for region in range(12) for index in range(1687)
        ]
        self.assertEqual(len(normal_members(infos)), 12)
        infos[-1] = zipfile.ZipInfo("extracted/grid11_s1024/train/good/9999.png")
        with self.assertRaisesRegex(ValueError, "differ between regions"):
            normal_members(infos)

    def test_decode_normal_rejects_invalid_payload(self):
        with self.assertRaisesRegex(ValueError, "Unreadable calibration normal"):
            decode_normal(b"not an image", "train/good/0000.png")
        output = io.BytesIO()
        Image.new("RGB", (3, 2), (10, 20, 30)).save(output, format="PNG")
        size, digest = decode_normal(output.getvalue(), "train/good/0000.png")
        self.assertEqual(size, (3, 2))
        self.assertEqual(len(digest), 64)


if __name__ == "__main__":
    unittest.main()
