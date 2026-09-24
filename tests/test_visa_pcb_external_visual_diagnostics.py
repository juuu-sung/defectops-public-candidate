"""Visual review uses frozen scores, source bytes and saved maps only."""

from __future__ import annotations

import hashlib
import io
import tarfile

import numpy as np
from PIL import Image

from defectops.realiad_followup_final import _write_sample
from defectops.visa_pcb_external_registration import file_sha256
from defectops.visa_pcb_external_evaluation import _pixel_sha
from defectops.visa_pcb_external_visual_diagnostics import render_card, select_cases


def _case(category: str, number: int, label: int, baseline: str, challenger: str) -> dict:
    return {
        "category": category, "image": f"{category}/Data/Images/{number:03d}.JPG",
        "label": label, "baseline_decision": baseline, "challenger_decision": challenger,
    }


def test_selection_keeps_all_pcb3_losses_and_path_sample_is_deterministic() -> None:
    rows = [_case("pcb3", i, 1, "REVIEW_REQUIRED", "PASS") for i in range(17)]
    for category in ("pcb1", "pcb2", "pcb3", "pcb4"):
        rows.extend(_case(category, i + 20, 1, "PASS", "PASS") for i in range(12))
        rows.extend(_case(category, i + 40, 0, "REVIEW_REQUIRED", "PASS") for i in range(6))
    first = select_cases(rows)
    second = select_cases(list(reversed(rows)))
    assert first == second
    assert len(first) == 65
    assert sum("pcb3_baseline_detected_challenger_missed_all" in row["review_groups"] for row in first) == 17
    try:
        select_cases(rows[1:])
    except ValueError as error:
        assert "all 17" in str(error)
    else:
        raise AssertionError("Missing PCB3 loss should fail")


def test_card_checks_original_mask_and_saved_map(tmp_path) -> None:
    original = Image.new("RGB", (320, 300), "#777777")
    jpeg_stream = io.BytesIO()
    original.save(jpeg_stream, format="JPEG")
    jpeg = jpeg_stream.getvalue()
    mask_image = Image.new("L", (320, 300), 0)
    for y in range(20, 80):
        for x in range(30, 90):
            mask_image.putpixel((x, y), 1)
    mask_stream = io.BytesIO()
    mask_image.save(mask_stream, format="PNG")
    png = mask_stream.getvalue()
    mask = np.asarray(mask_image.resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8) != 0
    image_name = "pcb3/Data/Images/Anomaly/000.JPG"
    mask_name = "pcb3/Data/Masks/Anomaly/000.png"
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, data in ((image_name, jpeg), (mask_name, png)):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    audit = {
        "image_sha256": hashlib.sha256(jpeg).hexdigest(),
        "decoded_pixel_sha256": _pixel_sha(original),
        "mask": mask_name, "mask_sha256": hashlib.sha256(png).hexdigest(),
        "mask_nonzero_256": int(mask.sum()), "width": 320, "height": 300,
    }
    evaluation = {
        "plan_sha256": "plan", "evaluation_config_sha256": "config",
        "checkpoint_sha256": "checkpoint", "source_archive_sha256": "source",
        "git": {"commit": "commit"}, "thresholds": {"pcb3": {"pixel": 50.0}},
    }
    sample_path = tmp_path / "sample-0001.npz"
    metadata = {
        "category": "pcb3", "label": 1, "image": image_name,
        "image_sha256": audit["image_sha256"], "mask_sha256": audit["mask_sha256"],
        "plan_sha256": "plan", "execution_config_sha256": "config",
        "checkpoint_sha256": "checkpoint", "source_archive_sha256": "source",
        "execution_git_commit": "commit", "score": 45.0, "challenger_score": 46.0,
    }
    _write_sample(sample_path, metadata, np.full((256, 256), 30, dtype=np.float32), mask)
    case = {
        "image": image_name, "category": "pcb3", "label": 1,
        "baseline_score": 45.0, "baseline_threshold": 50.0, "baseline_decision": "PASS",
        "challenger_score": 46.0, "challenger_threshold": 51.0, "challenger_decision": "PASS",
        "sample_sha256": file_sha256(sample_path),
    }
    with tarfile.open(archive_path, "r:") as archive:
        card = render_card(case, audit, evaluation, archive, sample_path)
        assert card.size == (768, 338)
        try:
            render_card({**case, "sample_sha256": "wrong"}, audit, evaluation, archive, sample_path)
        except ValueError as error:
            assert "Saved map checksum differs" in str(error)
        else:
            raise AssertionError("Changed saved map checksum should fail")
        try:
            render_card(case, {**audit, "mask_sha256": "wrong"}, evaluation, archive, sample_path)
        except ValueError as error:
            assert "Source mask checksum differs" in str(error)
        else:
            raise AssertionError("Changed mask checksum should fail")
