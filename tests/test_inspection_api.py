import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from defectops.inspection_api import create_app
from defectops.padim_inspection import run_inspection


class ResidentInspectionTests(unittest.TestCase):
    def test_existing_inspection_pipeline_reuses_runtime_without_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            fit_path = root / "artifacts/runs/fit"
            evaluation_path = root / "artifacts/evaluations/evaluation"
            output_path = root / "artifacts/inspections/output"
            fit_path.mkdir(parents=True)
            evaluation_path.mkdir(parents=True)
            input_path = root / "input.png"
            Image.new("RGB", (1024, 1024), (0, 0, 0)).save(input_path)
            anomaly_map = np.zeros((256, 256), dtype=np.float32)
            anomaly_map[172:256, 47:94] = 40.0
            original = np.zeros((1024, 1024, 3), dtype=np.uint8)
            prediction = {
                "original": original,
                "original_size": (1024, 1024),
                "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                "raw_score": 50.0,
                "anomaly_map": anomaly_map,
                "latency_ms": {
                    "decode_and_checksum": 1.0,
                    "preprocess": 1.0,
                    "gpu_inference_and_map_transfer": 2.0,
                },
            }
            runtime = SimpleNamespace(
                fit_path=fit_path.resolve(),
                evaluation_path=evaluation_path.resolve(),
                fit_config={"model": "patchcore", "model_seed": 42},
                fit_run={
                    "run_id": "fit",
                    "comparison_status": "exploratory_after_padim_test_observed",
                    "git": {"commit": "a" * 40},
                    "manifest_sha256": "b" * 64,
                },
                calibration_contract={
                    "image": {"threshold": 40.0},
                    "pixel": {"threshold": 33.0},
                },
                current_git={"commit": "c" * 40, "branch": "main"},
                checkpoint_sha256="d" * 64,
                setup_ms=24_000.0,
                git_provenance={"fit": {"method": "audited_identical_tree_migration",
                                        "original_commit": "a" * 40,
                                        "resolved_commit": "c" * 40}},
                device=SimpleNamespace(type="cuda"),
                versions={},
                torch=SimpleNamespace(
                    version=SimpleNamespace(cuda="12.8"),
                    cuda=SimpleNamespace(get_device_name=lambda: "test GPU"),
                ),
            )
            with patch(
                "defectops.padim_inspection.load_inference_runtime",
                side_effect=AssertionError("resident request reloaded the model"),
            ), patch(
                "defectops.padim_inspection.predict_validated_png",
                return_value=prediction,
            ), patch(
                "defectops.padim_inspection.__file__",
                str(root / "src/defectops/padim_inspection.py"),
            ):
                result = run_inspection(
                    SimpleNamespace(output=output_path, fit_run=fit_path,
                                    evaluation=evaluation_path, input=input_path),
                    expected_model="patchcore", runtime=runtime, input_source="api-upload",
                )
            self.assertEqual(result["decision"], "REVIEW_REQUIRED")
            self.assertTrue(result["model_reused"])
            self.assertEqual(result["trace"]["git_provenance"], runtime.git_provenance)
            self.assertEqual(result["trace"]["fit_git_commit"], "a" * 40)
            saved = json.loads((output_path / "inspection.json").read_text())
            self.assertEqual(saved["trace"]["git_provenance"], runtime.git_provenance)
            self.assertEqual(result["latency_ms"]["setup_and_model_load"], 0.0)
            self.assertEqual(result["input"]["source_path"], "api-upload")
            self.assertEqual(
                result["pixel"]["regions"][0]["bbox_original_xyxy_exclusive"],
                [188, 688, 376, 1024],
            )
            self.assertFalse(result["automatic_pass_allowed"])
            self.assertTrue((output_path / "overlay.png").is_file())


class InspectionApiTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name).resolve()
        self.fit_path = self.root / "fit"
        self.evaluation_path = self.root / "evaluation"
        self.output_root = self.root / "inspections"
        self.fit_path.mkdir()
        self.evaluation_path.mkdir()
        self.load_count = 0
        self.runtime = SimpleNamespace(
            fit_config={"model": "patchcore"},
            fit_run={
                "run_id": "patchcore-transistor-seed42",
                "comparison_status": "exploratory_after_padim_test_observed",
            },
            setup_ms=123.0,
            git_provenance={"fit": {"method": "audited_identical_tree_migration"}},
            device=SimpleNamespace(type="cuda"),
            torch=SimpleNamespace(
                cuda=SimpleNamespace(
                    memory_allocated=lambda: 100,
                    memory_reserved=lambda: 200,
                )
            ),
        )
        self.runner_runtimes = []

    def tearDown(self):
        self.temp.cleanup()

    def test_cpu_health_does_not_query_cuda(self):
        self.runtime.device = SimpleNamespace(type="cpu")
        self.runtime.torch = SimpleNamespace()
        with TestClient(self.make_app()) as client:
            health = client.get("/healthz").json()
            self.assertEqual(health["device"], "cpu")
            self.assertIsNone(health["gpu_memory"])
            self.assertEqual(health["model"], "patchcore")

    def load_runtime(self, project_root, fit_path, evaluation_path):
        self.load_count += 1
        self.assertEqual(fit_path, self.fit_path)
        self.assertEqual(evaluation_path, self.evaluation_path)
        return self.runtime

    def inspect(self, args, *, expected_model, runtime, input_source):
        self.assertEqual(expected_model, "patchcore")
        self.assertEqual(input_source, "api-upload")
        self.runner_runtimes.append(runtime)
        payload = args.input.read_bytes()
        if payload == b"bad":
            args.output.mkdir()
            (args.output / "inspection.json").write_text(
                json.dumps(
                    {
                        "status": "INSPECTION_FAILED",
                        "inspection_id": args.output.name,
                        "error_type": "InputValidationError",
                        "automatic_pass": False,
                        "automatic_pass_allowed": False,
                    }
                ),
                encoding="utf-8",
            )
            from defectops.padim_inspection import InputValidationError

            raise InputValidationError("bad PNG")
        args.output.mkdir()
        overlay_path = args.output / "overlay.png"
        overlay_path.write_bytes(b"fake overlay")
        result = {
            "status": "completed",
            "inspection_id": args.output.name,
            "model": "patchcore",
            "model_reused": True,
            "decision": "REVIEW_REQUIRED",
            "automatic_pass": False,
            "automatic_pass_allowed": False,
            "artifacts": {
                "overlay": "overlay.png",
                "overlay_sha256": hashlib.sha256(b"fake overlay").hexdigest(),
            },
        }
        (args.output / "inspection.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    def make_app(self):
        return create_app(
            self.fit_path,
            self.evaluation_path,
            output_root=self.output_root,
            runtime_loader=self.load_runtime,
            inspection_runner=self.inspect,
        )

    def test_startup_loads_once_and_two_requests_reuse_runtime(self):
        with TestClient(self.make_app()) as client:
            health = client.get("/healthz")
            self.assertEqual(health.json()["git_provenance"], self.runtime.git_provenance)
            self.assertEqual(health.status_code, 200)
            self.assertEqual(health.json()["fit_run_id"], "patchcore-transistor-seed42")
            self.assertEqual(health.json()["gpu_memory"]["reserved_bytes"], 200)
            first = client.post(
                "/v1/inspections", content=b"first", headers={"content-type": "image/png"}
            )
            second = client.post(
                "/v1/inspections", content=b"second", headers={"content-type": "image/png"}
            )
            self.assertEqual(first.status_code, 201)
            self.assertEqual(second.status_code, 201)
            self.assertNotEqual(first.json()["inspection_id"], second.json()["inspection_id"])
            self.assertEqual(self.load_count, 1)
            self.assertEqual(self.runner_runtimes, [self.runtime, self.runtime])
            inspection_id = first.json()["inspection_id"]
            stored = client.get(f"/v1/inspections/{inspection_id}")
            self.assertEqual(stored.json()["decision"], "REVIEW_REQUIRED")
            overlay = client.get(f"/v1/inspections/{inspection_id}/artifacts/overlay.png")
            self.assertEqual(overlay.status_code, 200)
            self.assertEqual(overlay.content, b"fake overlay")
            (self.output_root / inspection_id / "overlay.png").write_bytes(b"tampered")
            self.assertEqual(
                client.get(f"/v1/inspections/{inspection_id}/artifacts/overlay.png").status_code,
                409,
            )

    def test_bad_media_type_and_oversize_are_recorded_as_failures(self):
        with patch("defectops.inspection_api.MAX_UPLOAD_BYTES", 4):
            with TestClient(self.make_app()) as client:
                wrong_type = client.post(
                    "/v1/inspections", content=b"abc", headers={"content-type": "text/plain"}
                )
                too_large = client.post(
                    "/v1/inspections", content=b"12345", headers={"content-type": "image/png"}
                )
                self.assertEqual(wrong_type.status_code, 415)
                self.assertEqual(too_large.status_code, 413)
                for response in (wrong_type, too_large):
                    record = response.json()
                    self.assertEqual(record["status"], "INSPECTION_FAILED")
                    self.assertFalse(record["automatic_pass_allowed"])
                    self.assertEqual(
                        client.get(f"/v1/inspections/{record['inspection_id']}").json(),
                        record,
                    )
                self.assertEqual(self.runner_runtimes, [])

    def test_invalid_png_returns_failure_and_no_artifact(self):
        with TestClient(self.make_app()) as client:
            response = client.post(
                "/v1/inspections", content=b"bad", headers={"content-type": "image/png"}
            )
            self.assertEqual(response.status_code, 422)
            self.assertEqual(response.json()["status"], "INSPECTION_FAILED")
            inspection_id = response.json()["inspection_id"]
            self.assertEqual(
                client.get(f"/v1/inspections/{inspection_id}/artifacts/overlay.png").status_code,
                404,
            )
            self.assertEqual(client.get("/v1/inspections/not-an-id").status_code, 404)

    def test_wrong_model_blocks_service_startup(self):
        self.runtime.fit_config["model"] = "padim"
        with self.assertRaisesRegex(ValueError, "requires a patchcore fit bundle"):
            with TestClient(self.make_app()):
                pass


if __name__ == "__main__":
    unittest.main()
