import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from defectops.three_category_summary import (
    CATEGORY_CONTRACTS,
    MODELS,
    SEEDS,
    build_summary,
)


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ThreeCategorySummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def make_evaluation(self, category: str, model: str, seed: int) -> Path:
        directory = self.root / f"{model}-{category}-seed{seed}"
        directory.mkdir()
        contract = CATEGORY_CONTRACTS[category]
        normal = contract["counts"]["normal"]
        anomaly = contract["counts"]["anomaly"]
        model_gain = 0.15 if model == "patchcore" else 0.0
        seed_gain = (seed - 42) * 0.005
        recall = 0.70 + model_gain + seed_gain
        tp = round(anomaly * recall)
        fn = anomaly - tp
        fp = 2 if model == "padim" else 1
        tn = normal - fp
        precision = tp / (tp + fp)
        recall = tp / anomaly
        f1 = 2 * precision * recall / (precision + recall)
        metrics = {
            "schema_version": 1,
            "image": {
                "auroc": {"value": 0.78 + model_gain + seed_gain},
                "average_precision": {"value": 0.75 + model_gain + seed_gain},
                "fixed_threshold": {
                    "threshold": 40.0,
                    "counts": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
                    "rates": {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "escape_rate": 1.0 - recall,
                        "overkill_rate": fp / normal,
                        "review_rate": (tp + fp) / (normal + anomaly),
                    },
                },
            },
            "pixel": {
                "auroc": {"value": 0.86 + model_gain / 2 + seed_gain},
                "aupro": {"value": 0.80 + model_gain / 2 + seed_gain},
            },
        }
        metrics_path = directory / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
        checkpoint_sha = hashlib.sha256(
            f"{category}-{model}-{seed}".encode()
        ).hexdigest()
        run_id = f"{model}-{category}-seed{seed}"
        fit_run = {
            "run_id": run_id,
            "config": {"model": model, "model_seed": seed},
            "artifacts": {"checkpoint_sha256": checkpoint_sha},
        }
        (directory / "fit-run.json").write_text(
            json.dumps(fit_run, sort_keys=True), encoding="utf-8"
        )
        status = "evaluated"
        if model == "patchcore":
            status = "evaluated_exploratory" if category == "transistor" else "evaluated_preregistered"
        evaluation = {
            "schema_version": 1,
            "status": status,
            "test_evaluated": True,
            "evaluation_id": f"{run_id}-eval-v1",
            "fit": {
                "run_id": run_id,
                "model_seed": seed,
                "checkpoint_sha256": checkpoint_sha,
            },
            "manifest_sha256": contract["manifest_sha256"],
            "counts": contract["counts"],
            "test_sample_ids": [
                f"{category}/test/sample/{index:03d}.png"
                for index in range(contract["counts"]["test"])
            ],
            "latency_ms": {"p95_higher": 80.0 + model_gain + seed_gain},
            "metrics": metrics,
            "artifacts": {"metrics": "metrics.json", "metrics_sha256": _sha(metrics_path)},
        }
        (directory / "evaluation.json").write_text(
            json.dumps(evaluation, sort_keys=True), encoding="utf-8"
        )
        return directory

    def all_evaluations(self) -> list[Path]:
        return [
            self.make_evaluation(category, model, seed)
            for category in CATEGORY_CONTRACTS
            for model in MODELS
            for seed in SEEDS
        ]

    def test_eighteen_runs_are_validated_and_macro_aggregated(self):
        summary = build_summary(self.all_evaluations())
        self.assertEqual(
            summary["status"], "summarized_preregistered_three_category_mixed_evidence"
        )
        self.assertEqual(summary["selection"]["portfolio_candidate"], "patchcore")
        self.assertEqual(summary["selection"]["deciding_metric"], "escape_rate")
        self.assertFalse(summary["selection"]["automatic_pass_allowed"])
        self.assertEqual(
            summary["models"]["padim"]["macro_across_categories_by_seed"]["recall"]["count"],
            3,
        )
        self.assertEqual(
            len(summary["categories"]["cable"]["patchcore"]["runs"]), 3
        )

    def test_missing_run_status_and_manifest_tampering_are_rejected(self):
        paths = self.all_evaluations()
        with self.assertRaisesRegex(ValueError, "Exactly 18"):
            build_summary(paths[:-1])

        evaluation_path = paths[0] / "evaluation.json"
        evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
        evaluation["manifest_sha256"] = "0" * 64
        evaluation_path.write_text(json.dumps(evaluation), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            build_summary(paths)


if __name__ == "__main__":
    unittest.main()
