"""Integrity-amended MPI-PCB contract; preserve one photo through verified reconstruction."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import tomllib
import urllib.request
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile

from defectops.visa_pcb_external_protocol import canonical
from scripts.mpi_pcb_external_protocol import EXPECTED_CONFIG as V1_CONFIG, REGIONS, sha256_file


V1_HANDOFF_SHA256 = "cee3ff8d11bf73df7c0128869449e2cfd29b20230fedb0a17a51651710f6cfa1"
V1_AUDIT_SHA256 = "a2fd221d0678215844ab3b89c255589a654cb030586a6ad5409bf07e561d436f"
SOURCE_URL = "https://zenodo.org/api/records/8213098/files/original_aligned.zip/content"
SOURCE_SIZE = 5451602482
SOURCE_MEMBER = "original_aligned/test/defect/IMG_0056.JPG.jpg"
SOURCE_MEMBER_SHA256 = "126cf49bf0dedd8ba012a42a10c026240b5a8c0dd0d78ace293a50ce537a8845"
REPAIR_MEMBER = "extracted/grid1-3_s1024/test/defect/0054.png"
DAMAGED_SHA256 = "3afef0d2f99ea390dc2e6b455630128c3a026feab03bc3929d278f2aa468ad1b"
REPAIRED_RGB_SHA256 = "7ac16f5e75932c4eebe6ada691be04464272fa75125c5393393dad04bd7d54af"
MASK_MEMBER = "extracted/grid3-2_s1024/ground_truth/defect/0006.png"
MASK_SHA256 = "e4d86ef6ccee9e3ab6b2f550063904fabe5b03b35a986e53c59154914787f68c"
EXPECTED_DUPLICATES = {
    "identical_test_photo_groups": [],
    "identical_calibration_test_photos": [{"test_photo": "good/0013", "calibration_photo_id": "0866"}],
    "repeated_test_tile_groups_report_only": 0,
    "calibration_test_tile_matches_report_only": 12,
    "near_duplicates_checked": False,
}
EXPECTED_CONFIG = {
    **V1_CONFIG,
    "schema_version": 2,
    "status": "integrity_amended_external_test_before_test_scoring",
    "audit_failure_action": "only_reviewed_reconstruction_duplicate_and_mask_amendments_else_stop",
    "prior_handoff_sha256": V1_HANDOFF_SHA256,
    "included_normal_photos": 54,
    "included_defect_photos": 55,
    "excluded_test_photos": ["good/0013"],
    "reconstructed_photo": "defect/0054",
    "reconstruction_source_member_sha256": SOURCE_MEMBER_SHA256,
    "reconstruction_decoded_rgb_sha256": REPAIRED_RGB_SHA256,
    "localization_ineligible_photos": ["defect/0006"],
    "sensitivity_excludes_reconstructed_photo": True,
}


def load_config(path: Path) -> dict:
    config = tomllib.loads(path.read_text(encoding="utf-8"))
    if config != EXPECTED_CONFIG:
        raise ValueError("MPI-PCB v2 contract differs from locked amendments")
    return {**config, "config_sha256": hashlib.sha256(canonical(config)).hexdigest()}


def validate_prior(path: Path) -> dict:
    if sha256_file(path) != V1_HANDOFF_SHA256:
        raise ValueError("Reviewed blocked v1 handoff checksum differs; preserve the original ZIP")
    names = {"input-lock.json", "calibration-replay.json", "test-opening.json", "test-audit.json",
             "evaluation.json", "handoff-manifest.json"}
    with zipfile.ZipFile(path) as archive:
        if set(archive.namelist()) != names or len(archive.namelist()) != len(names):
            raise ValueError("Blocked v1 handoff members differ")
        reports = {name: json.loads(archive.read(name)) for name in names}
        for member in reports["handoff-manifest.json"]["members"]:
            if hashlib.sha256(archive.read(member["name"])).hexdigest() != member["sha256"]:
                raise ValueError("Blocked v1 report checksum differs")
        if hashlib.sha256(archive.read("test-audit.json")).hexdigest() != V1_AUDIT_SHA256:
            raise ValueError("Reviewed v1 audit checksum differs")
    result = reports["evaluation.json"]
    audit = reports["test-audit.json"]
    if (result.get("status") != "test_audit_blocked_no_scores" or result.get("test_scores_computed") != 0
            or audit.get("test_scores_computed") != 0 or len(audit.get("issues", [])) != 3
            or reports["calibration-replay.json"].get("passed") is not True):
        raise ValueError("Prior run is not the reviewed pre-score integrity stop")
    return reports


class BoundedRemoteZip(io.RawIOBase):
    """Read ONLY the official aligned ZIP ranges; refuse full-body fallbacks."""

    def __init__(self):
        self.position = 0
        self.downloaded_bytes = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError("Invalid seek mode")
        position = offset + (self.position if whence == 1 else SOURCE_SIZE if whence == 2 else 0)
        if not 0 <= position <= SOURCE_SIZE:
            raise ValueError("Invalid remote ZIP offset")
        self.position = position
        return position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = min(SOURCE_SIZE - self.position, count if count >= 0 else SOURCE_SIZE - self.position)
        if count <= 0:
            return b""
        if count > 5_000_000 or self.downloaded_bytes + count > 8_000_000:
            raise ValueError("Recovery download limit exceeded; no full archive download allowed")
        start, end = self.position, self.position + count - 1
        request = urllib.request.Request(SOURCE_URL, headers={"Range": f"bytes={start}-{end}"})
        with urllib.request.urlopen(request, timeout=45) as response:
            if (response.status != 206
                    or response.headers.get("Content-Range") != f"bytes {start}-{end}/{SOURCE_SIZE}"):
                raise ValueError("Server did not honor the exact bounded range request")
            raw = response.read(count + 1)
        if len(raw) != count:
            raise ValueError("Incomplete recovery range response")
        self.position += count
        self.downloaded_bytes += count
        return raw


def download_source_member() -> tuple[bytes, dict]:
    remote = BoundedRemoteZip()
    with zipfile.ZipFile(remote) as archive:
        if archive.namelist().count(SOURCE_MEMBER) != 1:
            raise ValueError("Official reconstruction source is missing or ambiguous")
        info = archive.getinfo(SOURCE_MEMBER)
        if info.file_size != 3989266 or info.compress_size != 3989266:
            raise ValueError("Official reconstruction member size changed")
        raw = archive.read(SOURCE_MEMBER)  # ZIP member CRC checked, but not whole-archive MD5.
    if hashlib.sha256(raw).hexdigest() != SOURCE_MEMBER_SHA256:
        raise ValueError("Official reconstruction source SHA-256 differs")
    return raw, {"source_url": SOURCE_URL, "source_member": SOURCE_MEMBER,
                 "source_member_sha256": SOURCE_MEMBER_SHA256, "source_member_crc_verified": True,
                 "whole_aligned_archive_checksum_verified": False,
                 "downloaded_bytes": remote.downloaded_bytes}


def rebuild_tiles(raw: bytes) -> dict[str, np.ndarray]:
    if hashlib.sha256(raw).hexdigest() != SOURCE_MEMBER_SHA256:
        raise ValueError("Reconstruction source bytes differ")
    ImageFile.LOAD_TRUNCATED_IMAGES = False
    with Image.open(io.BytesIO(raw)) as image:
        if image.format != "JPEG" or image.mode != "RGB" or image.size != (4096, 2816):
            raise ValueError("Reconstruction source is not the verified RGB 4096x2816 JPEG")
        image.load()
        pixels = np.asarray(image, dtype=np.uint16)
    tiles = {}
    for region in REGIONS:
        column, row = map(int, region.split("_")[0][4:].split("-"))
        x, y = (column - 1) * 1024, (row - 1) * 896
        crop = pixels[y:y + 1024, x:x + 1024]
        # Fixed 4x downscale, bilinear half-pixel centers without antialiasing.
        # uint16 accumulation and half-up rounding reproduce all 11 intact RGB tiles exactly.
        tiles[region] = ((crop[1::4, 1::4] + crop[2::4, 1::4]
                          + crop[1::4, 2::4] + crop[2::4, 2::4] + 2) // 4).astype(np.uint8)
    return tiles


def reconstruct(raw: bytes, prior_audit: dict) -> tuple[bytes, dict]:
    tiles = rebuild_tiles(raw)
    photo = next(photo for photo in prior_audit["photos"] if photo["key"] == "defect/0054")
    hashes = {region: hashlib.sha256(tile.tobytes()).hexdigest() for region, tile in tiles.items()}
    if [row["region"] for row in photo["regions"]] != REGIONS:
        raise ValueError("Reviewed reconstruction photo regions differ")
    matches = []
    for row in photo["regions"]:
        if row["member"] == REPAIR_MEMBER:
            if row.get("member_sha256") != DAMAGED_SHA256 or "decoded_rgb_sha256" in row:
                raise ValueError("Expected reviewed damaged image")
        else:
            if hashes[row["region"]] != row["decoded_rgb_sha256"]:
                raise ValueError(f'Intact reconstruction control does not match: {row["region"]}')
            matches.append(row["region"])
    if len(matches) != 11 or hashes["grid1-3_s1024"] != REPAIRED_RGB_SHA256:
        raise ValueError("Reconstruction controls or recovered pixel checksum differ")
    stream = io.BytesIO()
    Image.fromarray(tiles["grid1-3_s1024"]).save(stream, format="PNG")
    recovered = stream.getvalue()
    return recovered, {
        "status": "reconstructed_from_official_aligned_source_with_11_exact_controls",
        "repaired_member": REPAIR_MEMBER, "original_damaged_member_sha256": DAMAGED_SHA256,
        "recovered_png_sha256": hashlib.sha256(recovered).hexdigest(),
        "recovered_decoded_rgb_sha256": REPAIRED_RGB_SHA256,
        "intact_control_regions": matches, "intact_control_count": 11,
        "all_reconstructed_rgb_sha256": hashes,
        "transform": "x=(grid_x-1)*1024,y=(grid_y-1)*896,crop1024,4x_bilinear_half_pixel_no_antialias_half_up",
        "original_archive_modified": False, "ai_generated_or_inpainted_pixels": False,
        "limitation": "Derived replacement, not an independently supplied clean copy of the damaged PNG; sensitivity without this photo is reported.",
    }


class RecoveredSource:
    """Read-through overlay; never change extracted.zip or any other member."""

    def __init__(self, source, recovered: bytes):
        if hashlib.sha256(source.read(REPAIR_MEMBER)).hexdigest() != DAMAGED_SHA256:
            raise ValueError("Original damaged member no longer matches the reviewed bytes")
        self.source, self.recovered = source, recovered

    def infolist(self):
        return self.source.infolist()

    def read(self, name):
        return self.recovered if name == REPAIR_MEMBER else self.source.read(name)


def amended_cohort(audit: dict, prior_audit: dict, recovery: dict) -> tuple[list[dict], dict]:
    """Allow precisely one reconstruction, one duplicate exclusion, one mask warning."""
    expected_photos = copy.deepcopy(prior_audit["photos"])
    row = next(row for photo in expected_photos for row in photo["regions"] if row["member"] == REPAIR_MEMBER)
    row.update(member_sha256=recovery["recovered_png_sha256"], decoded_rgb_sha256=REPAIRED_RGB_SHA256)
    if audit["photos"] != expected_photos:
        raise ValueError("Fresh audit differs beyond the one approved reconstructed image")
    if (audit.get("duplicates") != EXPECTED_DUPLICATES
            or audit.get("test_image_read_attempts") != 1320 or audit.get("test_images_decoded") != 1320
            or audit.get("test_mask_read_attempts") != 660 or audit.get("test_masks_decoded") != 659
            or audit.get("test_scores_computed") != 0):
        raise ValueError("Unexpected fresh audit count/duplicate; no additional automatic exclusions")
    issues = audit.get("issues", [])
    mask_issues = [issue for issue in issues if issue.get("member") == MASK_MEMBER]
    duplicate_issues = [issue for issue in issues if issue == {
        "error": "Exact 12-region photo duplicate: test/test or calibration/test"}]
    if len(issues) != 2 or len(mask_issues) != 1 or len(duplicate_issues) != 1:
        raise ValueError("Unreviewed test issue; cannot extend the amendment automatically")
    mask_row = next(row for photo in audit["photos"] for row in photo["regions"] if row["mask_member"] == MASK_MEMBER)
    if mask_row.get("mask_sha256") != MASK_SHA256 or "mask" in mask_row:
        raise ValueError("Reviewed mask exception changed")
    selected = [photo for photo in audit["photos"] if photo["key"] != "good/0013"]
    if (len({photo["key"] for photo in selected}) != 109
            or sum(photo["label"] == 0 for photo in selected) != 54
            or sum(photo["label"] == 1 for photo in selected) != 55):
        raise ValueError("Amended cohort is not 54 normal and 55 altered unique photos")
    report = {
        "status": "integrity_amendment_passed_no_test_scores",
        "original_normal_photos": 55, "original_defect_photos": 55,
        "included_normal_photos": 54, "included_defect_photos": 55,
        "included_photos": 109, "included_fraction_of_original": 109 / 110,
        "excluded_photos": [{"key": "good/0013", "reason": "all_12_regions_equal_calibration_0866"}],
        "reconstructed_photos": ["defect/0054"], "localization_ineligible_photos": ["defect/0006"],
        "photo_keys": [photo["key"] for photo in selected],
        "photo_keys_sha256": hashlib.sha256(canonical([photo["key"] for photo in selected])).hexdigest(),
        "label_source": "official_test_folder_good_or_defect_not_mask_contents",
        "test_scores_computed": 0, "pixel_metrics_computed": False, "automatic_pass_allowed": False,
        "selection_basis": "integrity_only_before_test_scores; test pixels were already opened during v1 audit",
    }
    return selected, report
