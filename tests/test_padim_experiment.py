import json
import math
import unittest
from pathlib import Path

from defectops.padim_experiment import higher_quantile, initialize_cuda_device, load_experiment, records_for_split


class _FakeCuda:
    def __init__(self, calls):
        self.calls = calls

    def set_device(self, device):
        self.calls.append(("set_device", device))

    def synchronize(self):
        self.calls.append(("synchronize",))

    def reset_peak_memory_stats(self):
        self.calls.append(("reset_peak_memory_stats",))


class _FakeTorch:
    def __init__(self):
        self.calls = []
        self.cuda = _FakeCuda(self.calls)

    def device(self, value):
        self.calls.append(("device", value))
        return value

    def empty(self, size, *, device):
        self.calls.append(("empty", size, device))
        return object()


class PaDimExperimentTests(unittest.TestCase):
    def test_cuda_allocator_is_initialized_before_peak_reset(self):
        torch = _FakeTorch()

        device = initialize_cuda_device(torch)

        self.assertEqual(device, "cuda:0")
        self.assertEqual(
            torch.calls,
            [
                ("set_device", 0),
                ("device", "cuda:0"),
                ("empty", 1, "cuda:0"),
                ("synchronize",),
                ("reset_peak_memory_stats",),
            ],
        )

    def test_higher_quantile_matches_defined_order_statistic(self):
        self.assertEqual(higher_quantile([0.0, 1.0, 2.0, 3.0, 4.0], 0.5), 2.0)
        self.assertEqual(higher_quantile([0.0, 1.0, 2.0, 3.0], 0.5), 2.0)
        self.assertEqual(higher_quantile([3.0], 0.95), 3.0)

    def test_higher_quantile_rejects_invalid_scores(self):
        with self.assertRaises(ValueError):
            higher_quantile([], 0.95)
        with self.assertRaises(ValueError):
            higher_quantile([math.inf], 0.95)

    def test_train_and_calibration_reject_anomalies(self):
        manifest = {"records": [{"split": "train", "image_path": "bad.png", "label": 1}]}
        with self.assertRaises(ValueError):
            records_for_split(manifest, "train")

    def test_records_are_sorted_and_splits_are_explicit(self):
        manifest = {
            "records": [
                {"split": "test", "image_path": "test.png", "label": 1},
                {"split": "train", "image_path": "z.png", "label": 0},
                {"split": "train", "image_path": "a.png", "label": 0},
            ],
        }
        self.assertEqual(
            [record["image_path"] for record in records_for_split(manifest, "train")],
            ["a.png", "z.png"],
        )
        self.assertEqual(
            [record["image_path"] for record in records_for_split(manifest, "test")],
            ["test.png"],
        )

    def test_pilot_config_matches_frozen_manifest(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "data/manifests/transistor-v1.json").read_text(encoding="utf-8"))
        experiment = load_experiment(root / "configs/pilot.toml", manifest, seed=None)
        self.assertEqual(experiment["model"], "padim")
        self.assertEqual(experiment["backbone"], "resnet18")
        self.assertEqual(experiment["image_size"], 256)
        self.assertEqual(experiment["model_seed"], 42)
        self.assertEqual(experiment["batch_size"], 1)

    def test_additional_category_configs_are_preregistered_before_test(self):
        root = Path(__file__).resolve().parents[1]
        for category, config_name in (
            ("cable", "cable-padim-v1.toml"),
            ("metal_nut", "metal-nut-padim-v1.toml"),
        ):
            manifest = {
                "protocol": "A-normal-only-v1",
                "split_seed": 42,
                "categories": [category],
            }
            experiment = load_experiment(
                root / "configs" / config_name, manifest, seed=44
            )
            self.assertEqual(experiment["model_seed"], 44)
            self.assertEqual(experiment["image_size"], 256)
            self.assertEqual(experiment["backbone"], "resnet18")


if __name__ == "__main__":
    unittest.main()
