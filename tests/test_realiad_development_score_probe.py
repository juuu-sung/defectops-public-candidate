"""Exploratory score probe: fixed candidates and frozen-baseline replay."""

from __future__ import annotations

import numpy as np
import unittest

from defectops.padim_experiment import higher_quantile
from defectops.realiad_development_score_probe import (
    FRACTIONS,
    _score_replay,
    compare_scores,
    human_summary,
    score_map,
)


def _constant(value: float) -> np.ndarray:
    return np.full((256, 256), value, dtype=np.float32)


class DevelopmentScoreProbeTests(unittest.TestCase):
    def test_candidate_fractions_are_fixed_and_gt_free(self) -> None:
        values = _constant(0.0)
        values[0, 0] = 100.0
        scores = score_map(values)
        self.assertEqual(list(scores), list(FRACTIONS))
        self.assertGreater(scores["top_0_1pct"], scores["top_0_25pct"])
        self.assertGreater(scores["top_0_25pct"], scores["top_0_5pct"])
        self.assertGreater(scores["top_0_5pct"], scores["locked_top_1pct"])

    def test_probe_can_rescue_a_small_development_defect_without_touching_final(self) -> None:
        calibration_maps = [_constant(0.1), _constant(0.2)]
        development_maps = [_constant(0.1), _constant(0.0)]
        development_maps[1][0, 0] = 100.0
        mask = np.zeros((256, 256), dtype=np.bool_)
        mask[0, 0] = True
        calibration_records = [{"image_path": "cal1.jpg"}, {"image_path": "cal2.jpg"}]
        development_records = [
            {"image_path": "good.jpg", "label": 0, "defect_type": "OK"},
            {"image_path": "bad.jpg", "label": 1, "defect_type": "HS"},
        ]
        locked_threshold = higher_quantile(
            [score_map(values)["locked_top_1pct"] for values in calibration_maps], 0.95
        )
        cal_rows = [
            {"image_path": record["image_path"], "score": score_map(values)["locked_top_1pct"]}
            for record, values in zip(calibration_records, calibration_maps, strict=True)
        ]
        dev_rows = [
            {"image_path": record["image_path"], "score": score_map(values)["locked_top_1pct"],
             "threshold": locked_threshold, "decision": "PASS"}
            for record, values in zip(development_records, development_maps, strict=True)
        ]
        report, cases = compare_scores(
            calibration_maps, development_maps, calibration_records, development_records,
            cal_rows, dev_rows, locked_threshold=locked_threshold, pixel_threshold=0.5,
            development_masks=[np.zeros((256, 256), dtype=np.bool_), mask],
        )
        self.assertEqual(
            report["candidates"]["locked_top_1pct"]["metrics"]["fixed_threshold"]["counts"]["fn"], 1
        )
        self.assertEqual(report["candidates"]["top_0_1pct"]["previous_misses_rescued"], 1)
        self.assertEqual(cases[1]["decisions"]["top_0_1pct"], "REVIEW_REQUIRED")
        self.assertEqual(report["baseline_map_attribution"]["missed"]["median_gt_pixels"], 1)
        self.assertFalse(report["final_images_used"])
        self.assertFalse(report["automatic_pass_allowed"])
        summary = human_summary(report)
        self.assertIn("기존 1%", summary)
        self.assertIn("1/1", summary)
        self.assertIn("새 미관측 시험", summary)

    def test_replay_rejects_a_changed_original_score(self) -> None:
        values = _constant(1.0)
        with self.assertRaisesRegex(ValueError, "replay differs"):
            _score_replay(
                [{"image_path": "a.jpg"}], [values],
                [{"image_path": "a.jpg", "score": 2.0}],
            )
