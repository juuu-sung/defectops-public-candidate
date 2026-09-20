import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from defectops.multiseed_summary import build_summary


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MultiseedSummaryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def make_evaluation(self, model: str, seed: int) -> Path:
        directory = self.root / f"{model}-seed{seed}"
        directory.mkdir()
        offset = (seed - 42) * 0.01 + (0.05 if model == "patchcore" else 0.0)
        counts = {"tp": 30 + seed - 42, "fp": 2, "tn": 58, "fn": 10 - seed + 42}
        precision = counts["tp"] / (counts["tp"] + counts["fp"])
        recall = counts["tp"] / (counts["tp"] + counts["fn"])
        f1 = 2 * precision * recall / (precision + recall)
        metrics = {
            "schema_version": 1,
            "image": {
                "auroc": {"value": 0.85 + offset},
                "average_precision": {"value": 0.82 + offset},
                "fixed_threshold": {
                    "threshold": 40.0 + seed,
                    "counts": counts,
                    "rates": {
                        "precision": precision,
                        "recall": recall,
                        "f1": f1,
                        "escape_rate": 1.0 - recall,
                        "overkill_rate": counts["fp"] / (counts["tn"] + counts["fp"]),
                        "review_rate": (counts["tp"] + counts["fp"]) / 100,
                    },
                },
            },
            "pixel": {
                "auroc": {"value": 0.9 + offset / 2},
                "aupro": {"value": 0.8 + offset / 2},
            },
        }
        metrics_path = directory / "metrics.json"
        metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
        checkpoint = f"{model}-{seed}-checkpoint"
        checkpoint_sha = hashlib.sha256(checkpoint.encode()).hexdigest()
        run_id = f"{model}-transistor-seed{seed}"
        fit_run = {
            "run_id": run_id,
            "config": {"model": model, "model_seed": seed},
            "artifacts": {"checkpoint_sha256": checkpoint_sha},
        }
        (directory / "fit-run.json").write_text(
            json.dumps(fit_run, sort_keys=True), encoding="utf-8"
        )
        evaluation = {
            "schema_version": 1,
            "status": "evaluated_exploratory" if model == "patchcore" else "evaluated",
            "test_evaluated": True,
            "evaluation_id": f"{run_id}-eval-v1",
            "fit": {
                "run_id": run_id,
                "model_seed": seed,
                "checkpoint_sha256": checkpoint_sha,
            },
            "manifest_sha256": "a" * 64,
            "counts": {"test": 100, "normal": 60, "anomaly": 40, "failed": 0},
            "test_sample_ids": [f"test/{index:03d}.png" for index in range(100)],
            "latency_ms": {"p95_higher": 100.0 + offset},
            "metrics": metrics,
            "artifacts": {"metrics": "metrics.json", "metrics_sha256": _sha(metrics_path)},
        }
        (directory / "evaluation.json").write_text(
            json.dumps(evaluation, sort_keys=True), encoding="utf-8"
        )
        return directory

    def all_evaluations(self) -> list[Path]:
        return [
            self.make_evaluation(model, seed)
            for model in ("padim", "patchcore")
            for seed in (42, 43, 44)
        ]

    def test_six_runs_are_validated_and_aggregated_by_model(self):
        summary = build_summary(self.all_evaluations())
        self.assertEqual(summary["status"], "summarized_exploratory_multiseed")
        self.assertEqual(summary["expected_seeds"], [42, 43, 44])
        self.assertEqual(summary["models"]["padim"]["aggregate"]["recall"]["count"], 3)
        self.assertGreater(summary["models"]["patchcore"]["aggregate"]["image_auroc"]["mean"],
                           summary["models"]["padim"]["aggregate"]["image_auroc"]["mean"])
        self.assertAlmostEqual(summary["patchcore_minus_padim_mean"]["image_auroc"], 0.05)

    def test_missing_run_and_metrics_tampering_are_rejected(self):
        paths = self.all_evaluations()
        with self.assertRaisesRegex(ValueError, "Exactly six"):
            build_summary(paths[:-1])
        metrics_path = paths[0] / "metrics.json"
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics["image"]["auroc"]["value"] = 0.01
        metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "differs"):
            build_summary(paths)


if __name__ == "__main__":
    unittest.main()
