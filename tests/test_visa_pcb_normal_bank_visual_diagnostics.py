"""PCB3 v2 visual review reads frozen evidence and includes every error."""

from __future__ import annotations

import hashlib
import io
import tarfile
from collections import Counter

import numpy as np
import pytest
from PIL import Image

from defectops.realiad_followup_final import _write_sample
from defectops.visa_pcb_external_evaluation import _pixel_sha
from defectops.visa_pcb_external_registration import file_sha256
from defectops.visa_pcb_normal_bank_visual_diagnostics import render_card, select_cases


def _case(number: int, label: int, old: str, new: str) -> dict:
    kind = "Anomaly" if label else "Normal"
    return {
        "category": "pcb3", "image": f"pcb3/Data/Images/{kind}/{number:03d}.JPG",
        "label": label, "baseline_decision": old, "v2_decision": new,
    }


def test_selects_all_misses_and_false_positives_without_score_sampling() -> None:
    cases = [_case(index, 1, "REVIEW_REQUIRED", "PASS") for index in range(3)]
    cases += [_case(index + 3, 1, "PASS", "PASS") for index in range(6)]
    cases += [_case(index, 0, "PASS", "REVIEW_REQUIRED") for index in range(5)]
    cases += [_case(5, 0, "REVIEW_REQUIRED", "REVIEW_REQUIRED")]
    cases += [_case(99, 1, "PASS", "REVIEW_REQUIRED"), _case(99, 0, "PASS", "PASS")]
    first = select_cases(cases)
    assert first == select_cases(list(reversed(cases)))
    assert len(first) == 15
    assert dict(Counter(row["review_group"] for row in first)) == {
        "old_caught_v2_missed": 3, "both_missed": 6,
        "new_false_positive": 5, "both_false_positive": 1,
    }
    assert all(row["label"] == 1 for row in first[:9])
    with pytest.raises(ValueError, match="all 15"):
        select_cases(cases[1:])


def test_card_checks_source_and_both_saved_maps(tmp_path) -> None:
    source = Image.new("RGB", (320, 300), "#777777")
    image_stream = io.BytesIO()
    source.save(image_stream, format="JPEG")
    jpeg = image_stream.getvalue()
    mask_source = np.zeros((300, 320), dtype=np.uint8)
    mask_source[20:80, 30:90] = 1
    mask_image = Image.fromarray(mask_source)
    mask_stream = io.BytesIO()
    mask_image.save(mask_stream, format="PNG")
    png = mask_stream.getvalue()
    mask = np.asarray(mask_image.resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8) != 0
    image_name = "pcb3/Data/Images/Anomaly/040.JPG"
    mask_name = "pcb3/Data/Masks/Anomaly/040.png"
    archive_path = tmp_path / "source.tar"
    with tarfile.open(archive_path, "w") as archive:
        for name, data in ((image_name, jpeg), (mask_name, png)):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    audited = {
        "image_sha256": hashlib.sha256(jpeg).hexdigest(),
        "decoded_pixel_sha256": _pixel_sha(source), "width": 320, "height": 300,
        "mask_sha256": hashlib.sha256(png).hexdigest(), "mask_nonzero_256": int(mask.sum()),
    }
    record = {"label": 1, "source_mask_member": mask_name}
    prior = {
        "plan_sha256": "plan", "evaluation_config_sha256": "old-config",
        "checkpoint_sha256": "old-checkpoint", "source_archive_sha256": "source",
        "git": {"commit": "old-commit"}, "thresholds": {"pcb3": {"pixel": 50.0}},
    }
    report = {
        "source_plan_sha256": "plan", "comparison_config_sha256": "new-config",
        "fit_checkpoint_sha256": "new-checkpoint", "fit_handoff_sha256": "fit-zip",
        "baseline_evaluation_handoff_sha256": "old-zip", "source_archive_sha256": "source",
    }
    fit = {"image_threshold": 40.0, "pixel_threshold": 35.0}
    config = {"mask_threshold": 128, "mask_binarization": "nonzero_before_nearest",
              "image_score": "mean_of_top_map_fraction", "top_fraction": 0.01}
    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    _write_sample(old_path, {
        "category": "pcb3", "label": 1, "image": image_name,
        "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
        "plan_sha256": "plan", "execution_config_sha256": "old-config",
        "checkpoint_sha256": "old-checkpoint", "source_archive_sha256": "source",
        "execution_git_commit": "old-commit", "score": 45.0, "challenger_score": 46.0,
    }, np.full((256, 256), 30.0, dtype=np.float32), mask)
    _write_sample(new_path, {
        "category": "pcb3", "label": 1, "image": image_name,
        "image_sha256": audited["image_sha256"], "mask_sha256": audited["mask_sha256"],
        "plan_sha256": "plan", "comparison_config_sha256": "new-config",
        "fit_checkpoint_sha256": "new-checkpoint", "fit_handoff_sha256": "fit-zip",
        "baseline_evaluation_handoff_sha256": "old-zip", "source_archive_sha256": "source",
        "image_threshold": 40.0, "raw_score": 30.0, "score": 30.0,
    }, np.full((256, 256), 30.0, dtype=np.float32), mask)
    old = {"baseline_score": 45.0, "challenger_score": 46.0,
           "sample_sha256": file_sha256(old_path)}
    row = {
        "image": image_name, "label": 1, "review_group": "old_caught_v2_missed",
        "baseline_score": 45.0, "baseline_threshold": 40.0,
        "baseline_decision": "REVIEW_REQUIRED", "v2_score": 30.0,
        "v2_threshold": 40.0, "v2_decision": "PASS",
        "sample_sha256": file_sha256(new_path),
    }
    with tarfile.open(archive_path, "r:") as archive:
        card = render_card(row, old, record, audited, prior, report, fit, config,
                           archive, old_path, new_path)
        assert card.size == (1024, 338)
        with pytest.raises(ValueError, match="v2 saved map checksum differs"):
            render_card({**row, "sample_sha256": "wrong"}, old, record, audited, prior, report, fit,
                        config, archive, old_path, new_path)
        with pytest.raises(ValueError, match="Source JPEG checksum differs"):
            render_card(row, old, record, {**audited, "image_sha256": "wrong"}, prior, report,
                        fit, config, archive, old_path, new_path)
