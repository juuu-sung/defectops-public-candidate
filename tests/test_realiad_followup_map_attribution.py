"""Fixed, post-final map attribution calculations and tie bounds."""

from __future__ import annotations

import hashlib
import io
import json
import zipfile
from argparse import Namespace
from unittest.mock import patch

import numpy as np
from PIL import Image

from defectops.padim_experiment import checksum
from defectops.realiad_followup_final import EXPECTED_SELECTION_SHA256, _write_sample
from defectops.realiad_followup_map_attribution import map_attribution, run


def test_high_pixel_rank_can_coexist_with_zero_gt_top1pct_contribution() -> None:
    anomaly_map = np.zeros((256, 256), dtype=np.float32)
    mask = np.zeros_like(anomaly_map, dtype=np.bool_)
    mask[0, 0] = True
    anomaly_map[0, 0] = 10.0
    anomaly_map.reshape(-1)[1:657] = 11.0
    result = map_attribution(
        anomaly_map, mask, pixel_threshold=5.0, saved_image_score=11.0
    )
    assert result["top1pct_pixel_count"] == 656
    assert result["gt_pixels_in_top1pct_min"] == 0
    assert result["gt_pixels_in_top1pct_max"] == 0
    assert result["gt_score_share_of_top1pct_max"] == 0.0
    assert result["gt_vs_non_gt_rank_auc"] > 0.98
    assert result["gt_fraction_above_pixel_threshold"] == 1.0


def test_top1pct_cutoff_tie_reports_overlap_range() -> None:
    anomaly_map = np.zeros((256, 256), dtype=np.float32)
    mask = np.zeros_like(anomaly_map, dtype=np.bool_)
    mask[0, 0] = True
    anomaly_map.reshape(-1)[:657] = 5.0
    result = map_attribution(
        anomaly_map, mask, pixel_threshold=4.0, saved_image_score=5.0
    )
    assert result["top1pct_cutoff_tie_count"] == 657
    assert result["top1pct_cutoff_tie_slots"] == 656
    assert result["gt_pixels_in_top1pct_min"] == 0
    assert result["gt_pixels_in_top1pct_max"] == 1
    assert result["gt_recall_in_top1pct_min"] == 0.0
    assert result["gt_recall_in_top1pct_max"] == 1.0
    assert result["gt_score_share_of_top1pct_min"] == 0.0
    assert result["gt_score_share_of_top1pct_max"] == 1 / 656


def test_normal_image_has_no_ground_truth_metrics() -> None:
    anomaly_map = np.full((256, 256), 3.0, dtype=np.float32)
    mask = np.zeros_like(anomaly_map, dtype=np.bool_)
    result = map_attribution(
        anomaly_map, mask, pixel_threshold=2.0, saved_image_score=3.0
    )
    assert result["gt_pixel_count"] == 0
    assert result["gt_vs_non_gt_rank_auc"] is None
    assert result["gt_score_share_of_top1pct_min"] is None
    assert result["map_fraction_above_pixel_threshold"] == 1.0


def test_saved_image_score_must_match_saved_map() -> None:
    anomaly_map = np.full((256, 256), 3.0, dtype=np.float32)
    mask = np.zeros_like(anomaly_map, dtype=np.bool_)
    try:
        map_attribution(anomaly_map, mask, pixel_threshold=2.0, saved_image_score=4.0)
    except ValueError as error:
        assert "differs" in str(error)
    else:
        raise AssertionError("A changed image score must be rejected")


def test_read_only_run_reuses_saved_maps_without_inference(tmp_path) -> None:
    source_mask = Image.new("L", (1024, 1024), 0)
    for x in range(400, 500):
        for y in range(400, 500):
            source_mask.putpixel((x, y), 255)
    buffer = io.BytesIO()
    source_mask.save(buffer, format="PNG")
    mask_bytes = buffer.getvalue()
    mask_member = "pcb/NG/HS/S0001/mask.png"
    source_zip = tmp_path / "pcb.zip"
    with zipfile.ZipFile(source_zip, "w") as archive:
        archive.writestr(mask_member, mask_bytes)
    anomaly_mask = np.asarray(
        source_mask.resize((256, 256), Image.Resampling.NEAREST), dtype=np.uint8
    ) >= 128
    records = [
        {
            "source_image_member": "pcb/NG/HS/S0001/image.jpg", "image_sha256": "image-one-sha",
            "source_mask_member": mask_member, "mask_sha256": hashlib.sha256(mask_bytes).hexdigest(),
            "sample_group": "pcb/NG/HS/S0001", "defect_type": "HS", "label": 1,
        },
        {
            "source_image_member": "pcb/OK/S0002/image.jpg", "image_sha256": "image-two-sha",
            "sample_group": "pcb/OK/S0002", "defect_type": "good", "label": 0,
        },
    ]
    samples_root = tmp_path / "samples"
    bundles = {}
    for seed in (42, 43, 44):
        sample_hashes = []
        for index, record in enumerate(records):
            sample_path = samples_root / f"pcb-seed{seed}" / "samples" / f"sample-{index:04d}.npz"
            sample_path.parent.mkdir(parents=True, exist_ok=True)
            score = 5.0 if index == 0 else 7.0
            metadata = {
                "selection_report_sha256": EXPECTED_SELECTION_SHA256,
                "final_manifest_sha256": "manifest-sha", "checkpoint_sha256": "checkpoint-sha",
                "threshold": 6.0, "category": "realiad_pcb", "seed": seed,
                "source_image_member": record["source_image_member"],
                "image_sha256": record["image_sha256"], "label": record["label"],
                "defect_type": record["defect_type"], "score": score,
            }
            _write_sample(
                sample_path, metadata, np.full((256, 256), score, dtype=np.float32),
                anomaly_mask if index == 0 else np.zeros((256, 256), dtype=np.bool_),
            )
            sample_hashes.append(checksum(sample_path))
        bundles[seed] = {
            "rows": [
                {"score": 5.0, "threshold": 6.0, "decision": "PASS"},
                {"score": 7.0, "threshold": 6.0, "decision": "REVIEW_REQUIRED"},
            ],
            "evaluation": {
                "sample_artifact_sha256": sample_hashes,
                "final_manifest_sha256": "manifest-sha", "checkpoint_sha256": "checkpoint-sha",
                "threshold": 6.0, "pixel_threshold": 4.0,
            },
        }
    manifest = {"manifest_sha256": "manifest-sha", "source_archive_sha256": checksum(source_zip)}
    output = tmp_path / "analysis"
    args = Namespace(
        handoff=tmp_path / "handoff.zip", prior_diagnostics=tmp_path / "prior.zip",
        source_zip=source_zip, samples_root=samples_root, output=output,
    )
    with (
        patch("defectops.realiad_followup_map_attribution.load_final_rows", return_value=(manifest, bundles, records)),
        patch("defectops.realiad_followup_map_attribution.validate_prior_diagnostics"),
        patch("defectops.realiad_followup_map_attribution.validate_scoring_contract"),
    ):
        analysis = run(args)
    assert analysis["image_count"] == 2
    assert analysis["map_count"] == 6
    assert analysis["maximum_absolute_score_reconstruction_difference"] == 0.0
    assert analysis["by_consensus_group"]["consensus_false_negative"]["image_count"] == 1
    assert len((output / "reviewed_cases.jsonl").read_text().splitlines()) == 2
    assert json.loads((output / "analysis.json").read_text())["automatic_pass_allowed"] is False
