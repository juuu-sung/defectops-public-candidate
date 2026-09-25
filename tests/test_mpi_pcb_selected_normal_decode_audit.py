import io
import unittest
import zipfile

from PIL import Image

from scripts.audit_mpi_pcb_selected_normals import inspect_png, scan_selected


def png_bytes() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (4, 3), (10, 20, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def corrupt_idat_crc(raw: bytes) -> bytes:
    damaged = bytearray(raw)
    offset = 8
    while offset < len(damaged):
        length = int.from_bytes(damaged[offset:offset + 4], "big")
        chunk = damaged[offset + 4:offset + 8]
        crc_offset = offset + 8 + length
        if chunk == b"IDAT":
            damaged[crc_offset] ^= 1
            return bytes(damaged)
        offset = crc_offset + 4
    raise AssertionError("Test PNG has no IDAT")


class MpiPcbSelectedNormalDecodeAuditTests(unittest.TestCase):
    def test_valid_png_passes_verify_and_load(self):
        details = inspect_png(png_bytes())
        self.assertEqual(details["format"], "PNG")
        self.assertEqual((details["width"], details["height"]), (4, 3))
        self.assertNotIn("verify_error", details)
        self.assertNotIn("load_error", details)

    def test_corrupt_idat_crc_is_detected_even_if_pixels_can_load(self):
        details = inspect_png(corrupt_idat_crc(png_bytes()))
        self.assertIn("bad header checksum in b'IDAT'", details["verify_error"])

    def test_selected_scan_continues_after_corrupt_png_and_never_opens_test(self):
        buffer = io.BytesIO()
        region = "grid1-1_s1024"
        with zipfile.ZipFile(buffer, "w") as archive:
            archive.writestr(f"extracted/{region}/train/good/0001.png", png_bytes())
            archive.writestr(
                f"extracted/{region}/train/good/0002.png",
                corrupt_idat_crc(png_bytes()),
            )
            archive.writestr(f"extracted/{region}/test/defect/0001.png", b"not image")
        with zipfile.ZipFile(io.BytesIO(buffer.getvalue())) as archive:
            normal = {region: {
                photo_id: archive.getinfo(f"extracted/{region}/train/good/{photo_id}.png")
                for photo_id in ("0001", "0002")
            }}
            attempted, dimensions, issues = scan_selected(archive, normal, ["0001", "0002"])
        self.assertEqual(attempted, 2)
        self.assertEqual(dimensions, {"4x3": 1})
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["photo_id"], "0002")
        self.assertIn("verify_error", issues[0])


if __name__ == "__main__":
    unittest.main()
