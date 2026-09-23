"""Post-final diagnostics only consume frozen records and saved maps."""

from __future__ import annotations

import hashlib
import io
import zipfile

import numpy as np
from PIL import Image

from defectops.padim_experiment import checksum
from defectops.realiad_followup_final import EXPECTED_SELECTION_SHA256, _write_sample
from defectops.realiad_followup_final_diagnostics import (
    SEEDS,
    build_cases,
    render_sheets,
    summarize,
)


def test_consensus_groups_are_fixed_from_seed_votes() -> None:
    records = [
        {"source_image_member": f"pcb/{index}.jpg", "sample_group": f"pcb/{index}",
         "defect_type": "HS", "label": label}
        for index, label in enumerate((1, 1, 0, 0))
    ]
    scores = {42: (9.0, 11.0, 11.0, 9.0), 43: (9.0, 9.0, 11.0, 9.0), 44: (9.0, 9.0, 11.0, 9.0)}
    bundles = {
        seed: {"rows": [
            {"score": score, "threshold": 10.0,
             "decision": "REVIEW_REQUIRED" if score >= 10 else "PASS"}
            for score in scores[seed]
        ]}
        for seed in SEEDS
    }
    cases = build_cases(records, bundles)
    assert [case["group"] for case in cases] == [
        "consensus_false_negative", "partial_false_negative",
        "consensus_false_positive", "consensus_true_negative",
    ]
    analysis = summarize(cases, {"manifest_sha256": "abc"})
    assert analysis["group_counts"]["consensus_false_negative"] == 1
    assert analysis["automatic_pass_allowed"] is False
    assert analysis["model_changed"] is False


def test_contact_sheet_verifies_original_and_three_saved_maps(tmp_path) -> None:
    image = Image.new("RGB", (16, 16), "#777777")
    image_stream = io.BytesIO()
    image.save(image_stream, format="JPEG")
    image_bytes = image_stream.getvalue()
    source_mask = Image.new("L", (16, 16), 0)
    for x in range(3, 8):
        for y in range(3, 8):
            source_mask.putpixel((x, y), 255)
    mask_stream = io.BytesIO()
    source_mask.save(mask_stream, format="PNG")
    mask_bytes = mask_stream.getvalue()
    resized_mask = np.asarray(
        source_mask.resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8
    ) >= 128
    image_member = "pcb/NG/HS/S0001/source.jpg"
    mask_member = "pcb/NG/HS/S0001/source.png"
    normal_member = "pcb/OK/S0002/source.jpg"
    source_zip = tmp_path / "pcb.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(image_member, image_bytes)
        archive.writestr(mask_member, mask_bytes)
        archive.writestr(normal_member, image_bytes)
    record = {
        "source_image_member": image_member, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "source_mask_member": mask_member, "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
        "sample_group": "pcb/NG/HS/S0001", "defect_type": "HS", "label": 1,
    }
    normal_record = {
        "source_image_member": normal_member, "image_sha256": hashlib.sha256(image_bytes).hexdigest(),
        "sample_group": "pcb/OK/S0002", "defect_type": "good", "label": 0,
    }
    bundles = {}
    samples_root = tmp_path / "samples"
    for seed in SEEDS:
        sample_path = samples_root / f"pcb-seed{seed}" / "samples" / "sample-0000.npz"
        sample_path.parent.mkdir(parents=True)
        metadata = {
            "selection_report_sha256": EXPECTED_SELECTION_SHA256,
            "final_manifest_sha256": "manifest-sha", "checkpoint_sha256": "checkpoint-sha",
            "threshold": 10.0, "category": "realiad_pcb", "seed": seed,
            "source_image_member": image_member, "image_sha256": record["image_sha256"],
            "label": 1, "defect_type": "HS", "score": 9.0,
        }
        _write_sample(sample_path, metadata, np.full((256, 256), 5.0, dtype=np.float32), resized_mask)
        normal_path = samples_root / f"pcb-seed{seed}" / "samples" / "sample-0001.npz"
        normal_metadata = {
            **metadata, "source_image_member": normal_member,
            "label": 0, "defect_type": "good", "score": 11.0,
        }
        _write_sample(
            normal_path, normal_metadata, np.full((256, 256), 7.0, dtype=np.float32),
            np.zeros((256, 256), dtype=np.bool_),
        )
        bundles[seed] = {
            "rows": [
                {"score": 9.0, "threshold": 10.0, "decision": "PASS"},
                {"score": 11.0, "threshold": 10.0, "decision": "REVIEW_REQUIRED"},
            ],
            "evaluation": {
                "sample_artifact_sha256": [checksum(sample_path), checksum(normal_path)],
                "final_manifest_sha256": "manifest-sha", "checkpoint_sha256": "checkpoint-sha",
                "threshold": 10.0, "pixel_threshold": 8.0,
            },
        }
    records = [record, normal_record]
    cases = build_cases(records, bundles)
    output = tmp_path / "sheets"
    output.mkdir()
    sheets = render_sheets(
        cases, records, bundles, {"source_archive_sha256": checksum(source_zip)},
        source_zip, samples_root, output,
    )
    assert set(sheets) == {"good-01.png", "hs-01.png"}
    for name in sheets:
        with Image.open(output / name) as sheet:
            assert sheet.size == (1024, 296)
