import unittest
from pathlib import Path

import numpy as np

from defectops.evaluation_metrics import (
    aupro_fpr_grid,
    binary_auroc,
    binary_average_precision,
    confusion_counts,
    connected_components_4,
    rates_from_confusion,
    stratified_bootstrap_interval,
    wilson_interval,
)
from defectops.padim_evaluation import load_evaluation_config


class EvaluationMetricTests(unittest.TestCase):
    def test_frozen_evaluation_config_loads(self):
        root = Path(__file__).resolve().parents[1]
        config = load_evaluation_config(root / "configs/evaluation-v1.toml")
        self.assertEqual(config["pixel_calibration_quantile"], 0.995)
        self.assertEqual(config["aupro_fpr_limit"], 0.3)
        self.assertEqual(config["bootstrap_replicates"], 2000)
        repeat = load_evaluation_config(root / "configs/padim-evaluation-repeat-v1.toml")
        self.assertEqual(
            repeat["purpose"],
            "exploratory frozen-rule evaluation of a configured PaDiM repeat",
        )
        self.assertEqual(repeat["pixel_calibration_quantile"], 0.995)
        additional = load_evaluation_config(
            root / "configs/additional-padim-evaluation-v1.toml"
        )
        self.assertEqual(
            additional["purpose"],
            "preregistered frozen-rule evaluation of an untouched additional category",
        )
        self.assertEqual(additional["bootstrap_seed"], 20260921)

    def test_ranking_metrics_are_perfect_for_separated_scores(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        self.assertEqual(binary_auroc(labels, scores), 1.0)
        self.assertEqual(binary_average_precision(labels, scores), 1.0)

    def test_auroc_uses_average_ranks_for_ties(self):
        labels = np.asarray([0, 1])
        scores = np.asarray([0.5, 0.5])
        self.assertEqual(binary_auroc(labels, scores), 0.5)

    def test_confusion_and_rates_use_greater_than_or_equal(self):
        counts = confusion_counts(np.asarray([0, 0, 1, 1]), np.asarray([0.2, 0.5, 0.5, 0.9]), 0.5)
        self.assertEqual(counts, {"tp": 2, "fp": 1, "tn": 1, "fn": 0})
        rates = rates_from_confusion(counts)
        self.assertAlmostEqual(rates["precision"], 2 / 3)
        self.assertEqual(rates["recall"], 1.0)
        self.assertEqual(rates["escape_rate"], 0.0)
        self.assertEqual(rates["overkill_rate"], 0.5)
        self.assertEqual(rates["review_rate"], 0.75)

    def test_components_are_four_connected_and_boxes_are_exclusive(self):
        mask = np.asarray([[1, 0, 0], [0, 1, 1], [0, 1, 0]], dtype=bool)
        components = connected_components_4(mask)
        self.assertEqual([component["area"] for component in components], [3, 1])
        self.assertEqual(components[0]["bbox_xyxy_exclusive"], [1, 1, 3, 3])
        self.assertEqual(components[1]["bbox_xyxy_exclusive"], [0, 0, 1, 1])

    def test_aupro_is_one_for_perfect_map(self):
        masks = np.zeros((2, 4, 4), dtype=bool)
        masks[1, 1:3, 1:3] = True
        maps = masks.astype(np.float32)
        result = aupro_fpr_grid(maps, masks, fpr_limit=0.3, fpr_points=11)
        self.assertAlmostEqual(result["value"], 1.0)
        self.assertEqual(result["region_count"], 1)

    def test_stratified_bootstrap_is_deterministic(self):
        labels = np.asarray([0, 0, 1, 1])
        scores = np.asarray([0.1, 0.2, 0.8, 0.9])
        first = stratified_bootstrap_interval(
            labels, scores, binary_auroc, seed=42, replicates=20, confidence_level=0.95
        )
        second = stratified_bootstrap_interval(
            labels, scores, binary_auroc, seed=42, replicates=20, confidence_level=0.95
        )
        self.assertEqual(first, second)
        self.assertEqual(first, [1.0, 1.0])

    def test_wilson_interval_handles_bounds_and_empty_denominator(self):
        interval = wilson_interval(10, 10, 0.95)
        self.assertLess(interval[0], 1.0)
        self.assertEqual(interval[1], 1.0)
        self.assertIsNone(wilson_interval(0, 0, 0.95))


if __name__ == "__main__":
    unittest.main()
