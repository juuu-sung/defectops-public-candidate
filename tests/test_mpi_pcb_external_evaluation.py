import copy
import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from contextlib import ExitStack
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from scripts import evaluate_mpi_pcb_external as runner
from scripts import mpi_pcb_external_protocol as protocol


def png(mode="RGB", color=(10, 20, 30), size=(256, 256)):
    stream = io.BytesIO()
    Image.new(mode, size, color).save(stream, format="PNG")
    return stream.getvalue()


def fixture(*, broken=False, empty_masks=False, duplicate=False, partial_duplicate=False):
    archive = io.BytesIO()
    photos = []
    with zipfile.ZipFile(archive, "w") as writer:
        for label, group in enumerate(("good", "defect")):
            entries = []
            for index, region in enumerate(protocol.REGIONS):
                member = f"extracted/{region}/test/{group}/0000.png"
                mask = f"extracted/{region}/ground_truth/defect/0000.png" if label else None
                shared = duplicate or (partial_duplicate and index == 0)
                raw = png(color=(index, 20 + (0 if shared else label), 30))
                if broken and label and index == 1:
                    raw = b"corrupt png"
                writer.writestr(member, raw)
                if mask:
                    writer.writestr(mask, png("L", 1 if index == 0 and not empty_masks else 0))
                entries.append({"region": region, "member": member, "mask_member": mask})
            photos.append({"key": f"{group}/0000", "photo_id": "0000", "label": label, "regions": entries})
    return archive.getvalue(), photos


class ProtocolTests(unittest.TestCase):
    def test_locked_configuration(self):
        root = Path(__file__).resolve().parents[1]
        path = root / "configs" / "mpi-pcb-external-evaluation-v1.toml"
        self.assertEqual(protocol.load_evaluation_config(path)["thresholds"], protocol.EXPECTED_CONFIG["thresholds"])
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp) / "config.toml"
            changed.write_text(path.read_text().replace("67.42196179599297", "67.0"))
            with self.assertRaisesRegex(ValueError, "locked"):
                protocol.load_evaluation_config(changed)

    def test_inventory_names_only_and_missing_mask_rejected(self):
        infos = []
        for region in protocol.REGIONS:
            infos.append(zipfile.ZipInfo(f"extracted/{region}/train/good/0000.png"))
            for group in ("test/good", "test/defect", "ground_truth/defect"):
                infos.extend(zipfile.ZipInfo(f"extracted/{region}/{group}/{i:04d}.png") for i in range(55))
        photos = protocol.test_inventory(infos)
        self.assertEqual(len(photos), 110)
        self.assertEqual(sum(photo["label"] for photo in photos), 55)
        self.assertTrue(all(len(p["regions"]) == 12 for p in photos))
        with self.assertRaisesRegex(ValueError, "inventory"):
            protocol.test_inventory(infos[:-1])

    def test_mask_nonzero_encodings_and_native_size(self):
        for mode, value in (("L", 1), ("L", 255), ("RGB", (255, 255, 255))):
            self.assertEqual(protocol.mask_details(png(mode, value))["nonzero_pixels"], 65536)
        self.assertEqual(protocol.mask_details(png("L", 0))["nonzero_pixels"], 0)
        for raw in (png("RGB", (1, 0, 0)), png("RGBA", (1, 1, 1, 0)), png("L", 1, (128, 128)), b"broken"):
            with self.assertRaises(ValueError):
                protocol.mask_details(raw)

    def test_strict_test_png_crc_mode_and_size(self):
        raw = png()
        with protocol.decode_image(raw) as image:
            self.assertEqual(image.size, (256, 256))
        altered = bytearray(raw)
        offset = raw.index(b"IDAT") + 4
        altered[offset] ^= 1
        for invalid in (bytes(altered), png("L", 1), png(size=(128, 128))):
            with self.assertRaises(ValueError):
                protocol.decode_image(invalid)

    def test_empty_region_masks_allowed_whole_photo_empty_blocks(self):
        for empty in (False, True):
            raw, inventory = fixture(empty_masks=empty)
            with zipfile.ZipFile(io.BytesIO(raw)) as source:
                report = protocol.audit_test(source, inventory, [])
            self.assertEqual(report["test_images_decoded"], 24)
            self.assertEqual(report["test_masks_decoded"], 12)
            self.assertEqual(report["test_scores_computed"], 0)
            self.assertEqual(bool(report["issues"]), empty)

    def test_corruption_collected_without_excluding_sample(self):
        raw, inventory = fixture(broken=True)
        with zipfile.ZipFile(io.BytesIO(raw)) as source:
            report = protocol.audit_test(source, inventory, [])
        self.assertEqual(report["status"], "test_audit_blocked_no_scores")
        self.assertEqual(report["test_image_read_attempts"], 24)
        self.assertEqual(report["test_images_decoded"], 23)
        self.assertEqual(len(report["photos"]), 2)

    def test_full_photo_duplicate_blocks_but_partial_tile_does_not(self):
        for entire in (False, True):
            raw, inventory = fixture(duplicate=entire, partial_duplicate=not entire)
            with zipfile.ZipFile(io.BytesIO(raw)) as source:
                report = protocol.audit_test(source, inventory, [])
            self.assertEqual(bool(report["issues"]), entire)
            self.assertGreater(report["duplicates"]["repeated_test_tile_groups_report_only"], 0)

    def test_calibration_test_identical_photo_blocks(self):
        raw, inventory = fixture()
        with zipfile.ZipFile(io.BytesIO(raw)) as source:
            report = protocol.audit_test(source, inventory, [])
        rows = [{**row, "photo_id": "calibration"} for row in report["photos"][0]["regions"]]
        duplicates = protocol.duplicate_summary(report["photos"], rows)
        self.assertEqual(duplicates["identical_calibration_test_photos"],
                         [{"test_photo": "good/0000", "calibration_photo_id": "calibration"}])

    def test_replay_detects_region_difference_even_when_photo_max_same(self):
        original = [{"photo_id": "1", "score": 10., "regions": [
            {"region": r, "member": r, "score": float(i)} for i, r in enumerate(protocol.REGIONS)]}]
        changed = copy.deepcopy(original)
        changed[0]["regions"][0]["score"] += .1
        result = protocol.compare_replay(original, changed)
        self.assertEqual(result["maximum_absolute_photo_score_difference"], 0)
        self.assertEqual(result["maximum_absolute_region_score_difference"], .1)
        changed[0]["regions"][0]["score"] = float("nan")
        with self.assertRaises(ValueError):
            protocol.compare_replay(original, changed)

    def test_photo_metrics_and_paired_direction(self):
        config = copy.deepcopy(protocol.EXPECTED_CONFIG)
        config["thresholds"] = {key: 5 for key in protocol.MODEL_ORDER}
        old = [{"key": str(i), "label": int(i >= 2), "score": score}
               for i, score in enumerate((5, 1, 5, 1))]
        new = [{**p, "score": score} for p, score in zip(old, (1, 1, 5, 5))]
        self.assertEqual(protocol.model_metrics(old, 5)["fixed_threshold"]["counts"], {"tp": 1, "fp": 1, "tn": 1, "fn": 1})
        result = protocol.paired_comparison(dict(zip(protocol.MODEL_ORDER, ({"photos": old}, {"photos": new}))), config)
        self.assertEqual(result["normal"]["error_rate_delta_visa_minus_realiad"], -.5)
        self.assertEqual(result["defect"]["error_rate_delta_visa_minus_realiad"], -.5)
        self.assertEqual(result["defect"]["visa_only_review"], 1)

    def test_calibration_validation_recomputes_every_photo_and_threshold(self):
        config = copy.deepcopy(protocol.EXPECTED_CONFIG)
        ids = [f"{i:04d}" for i in range(169)]
        photos = [{"photo_id": pid, "score": float(i), "regions": [
            {"region": r, "member": f"extracted/{r}/train/good/{pid}.png", "score": float(i)}
            for r in protocol.REGIONS]} for i, pid in enumerate(ids)]
        normal_protocol = {"config_sha256": config["normal_protocol_config_sha256"],
                           "models": {key: {"checkpoint_sha256": key} for key in protocol.MODEL_ORDER}}
        report = {"status": "mpi_pcb_normal_calibrated_test_sealed",
                  "git": {"commit": "285f9aecabc17049c9aaa00cddcb2e1696a01759"},
                  "protocol_config_sha256": normal_protocol["config_sha256"],
                  "model_preflight_sha256": protocol.EXPECTED_MODEL_PREFLIGHT_SHA256,
                  "normal_preflight_v4_sha256": protocol.EXPECTED_V4_SHA256,
                  "normal_photo_ids_sha256": protocol.EXPECTED_IDS_SHA256,
                  "archive_sha256": protocol.EXPECTED_ARCHIVE_SHA256,
                  "normal_original_photos_scored_per_model": 169, "normal_region_images_scored_per_model": 2028,
                  "photo_aggregation": "maximum_of_12_region_scores", "region_score": "mean_of_top_1pct_anomaly_map",
                  "test_images_decoded": 0, "test_masks_decoded": 0, "test_scores_computed": 0,
                  "automatic_pass_allowed": False, "confirmatory_claim_allowed": False,
                  "models": {key: {"photo_scores": copy.deepcopy(photos), "threshold": 160., "checkpoint_sha256": key,
                                   "normal_original_photos_scored": 169, "normal_region_images_scored": 2028,
                                   "normal_photos_at_or_above_threshold": 9} for key in protocol.MODEL_ORDER}}
        config["thresholds"] = {key: 160. for key in protocol.MODEL_ORDER}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "calibration.json"
            for invalid in (False, True):
                if invalid:
                    report["models"]["realiad_pcb"]["photo_scores"][0]["score"] = 1.
                path.write_text(json.dumps(report))
                config["calibration_report_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
                if invalid:
                    with self.assertRaisesRegex(ValueError, "maximum"):
                        protocol.validate_calibration(path, {"calibration_photo_ids": ids}, normal_protocol, config)
                else:
                    self.assertEqual(protocol.validate_calibration(path, {"calibration_photo_ids": ids}, normal_protocol, config), report)


class RunnerTests(unittest.TestCase):
    def test_score_uses_top_map_and_saves_local_map_with_no_auto_pass(self):
        raw, inventory = fixture()
        with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(io.BytesIO(raw)) as source:
            audited = protocol.audit_test(source, inventory, [])
            def predict(_model, image, *_):
                return 999., np.full((256, 256), image.getpixel((0, 0))[0], dtype=np.float32)
            root = Path(temp)
            result = runner.score_test_photos(source, audited["photos"], None, {}, (), 11., root / "maps" / "model", predict=predict)
            self.assertEqual([p["score"] for p in result], [11., 11.])
            self.assertTrue(all(p["decision"] == "REVIEW_REQUIRED" and not p["automatic_pass_allowed"] for p in result))
            path = root / result[0]["map_file"]
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), result[0]["map_sha256"])
            with np.load(path, allow_pickle=False) as maps:
                self.assertEqual(maps["anomaly_maps"].shape, (12, 256, 256))
                self.assertEqual(maps["regions"].tolist(), protocol.REGIONS)

    def test_changed_test_bytes_fail_before_inference(self):
        raw, inventory = fixture()
        with tempfile.TemporaryDirectory() as temp, zipfile.ZipFile(io.BytesIO(raw)) as source:
            audited = protocol.audit_test(source, inventory, [])
            audited["photos"][0]["regions"][0]["member_sha256"] = "bad"
            with self.assertRaisesRegex(ValueError, "changed after audit"):
                runner.score_test_photos(source, audited["photos"], None, {}, (), 11., Path(temp) / "maps",
                                         predict=lambda *_: self.fail("No inference on changed bytes"))

    def exercise_order(self, temp, *, bad_replay=False, bad_audit=False):
        events = []
        config = copy.deepcopy(protocol.EXPECTED_CONFIG)
        photo = {"photo_id": "1", "score": 0., "regions": [{"region": r, "member": r, "score": 0.} for r in protocol.REGIONS]}
        calibration = {"models": {key: {"photo_scores": [copy.deepcopy(photo)]} for key in protocol.MODEL_ORDER}}
        gate = {"models": {key: {"checkpoint_path": key, "checkpoint_sha256": key} for key in protocol.MODEL_ORDER}}
        recipes = {key: {"base_model": {}, "checkpoint_context": {}} for key in protocol.MODEL_ORDER}
        def replay(*_):
            events.append("normal")
            result = copy.deepcopy(photo)
            result["regions"][0]["score"] = 1. if bad_replay else 0.
            return [result]
        photos = [{"key": str(i), "photo_id": str(i), "label": int(i >= 55), "regions": [{}] * 12} for i in range(110)]
        def audit(*_):
            events.append("audit")
            return {"issues": ["corrupt"] if bad_audit else [], "test_images_decoded": 1320,
                    "test_masks_decoded": 660, "photos": photos}
        def score(_source, audited, _model, _recipe, _gpu, threshold, _maps):
            events.append("test")
            self.assertIn(threshold, protocol.EXPECTED_CONFIG["thresholds"].values())
            return [{**p, "score": 100. if p["label"] else 0.} for p in audited]
        with ExitStack() as stack:
            for name, replacement in (("test_inventory", lambda *_: []), ("_seed_everything", lambda *_: None),
                                      ("_model", lambda *_: None), ("score_normal_cohort", replay),
                                      ("audit_test", audit), ("score_test_photos", score)):
                stack.enter_context(mock.patch.object(runner, name, replacement))
            result = runner.execute(mock.Mock(), Path(temp), config,
                                    {"calibration_rows": [], "calibration_photo_ids": ["1"]},
                                    calibration, gate, recipes, (mock.Mock(),))
        return result, events

    def test_replay_mismatch_never_opens_test(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(temp, bad_replay=True)
            self.assertEqual(events, ["normal", "normal"])
            self.assertEqual(result["status"], "normal_replay_blocked_test_sealed")
            self.assertFalse((Path(temp) / "test-opening.json").exists())

    def test_audit_failure_stops_both_models_without_test_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(temp, bad_audit=True)
            self.assertEqual(events, ["normal", "normal", "audit"])
            self.assertEqual(result["test_scores_computed"], 0)

    def test_complete_order_and_photo_denominators(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(temp)
            self.assertEqual(events, ["normal", "normal", "audit", "test", "test"])
            self.assertEqual(result["status"], runner.COMPLETE)
            self.assertEqual(result["models"]["realiad_pcb"]["metrics"]["fixed_threshold"]["counts"],
                             {"tp": 55, "fp": 0, "tn": 55, "fn": 0})

    def test_handoff_is_json_whitelist_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner.write_json(root / "evaluation.json", {"status": "example"})
            (root / "original.png").write_bytes(b"not shared")
            (root / "model.pt").write_bytes(b"not shared")
            handoff = root / "handoff.zip"
            runner.package_reports(root, handoff)
            with zipfile.ZipFile(handoff) as source:
                self.assertEqual(set(source.namelist()), {"evaluation.json", "handoff-manifest.json"})
                manifest = json.loads(source.read("handoff-manifest.json"))
                self.assertEqual(manifest["members"][0]["sha256"], hashlib.sha256(source.read("evaluation.json")).hexdigest())
            with self.assertRaises(FileExistsError):
                runner.package_reports(root, handoff)
            with self.assertRaises(FileExistsError):
                runner.write_json(root / "evaluation.json", {})

    def test_cli_argument_mapping_and_blocked_exit(self):
        with mock.patch.object(runner, "run", return_value={"status": runner.COMPLETE}) as run:
            self.assertEqual(runner.main(["--project", "/tmp/project", "--archive", "/tmp/source.zip"]), 0)
            run.assert_called_once_with(Path("/tmp/project").resolve(), Path("/tmp/source.zip").resolve())
        with mock.patch.object(runner, "run", return_value={"status": "test_audit_blocked_no_scores"}):
            self.assertEqual(runner.main(["--archive", "/tmp/source.zip"]), 2)

    def test_run_packages_success_and_runtime_failure_without_source_payloads(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory(prefix="MPI 한글 ") as temp, ExitStack() as stack:
                root = Path(temp)
                source = root / "source.zip"
                with zipfile.ZipFile(source, "w") as writer:
                    writer.writestr("do-not-share.png", b"source pixels")
                config = {**protocol.EXPECTED_CONFIG, "config_sha256": "config"}
                patches = {"git_state": {"commit": "test"}, "load_protocol": {}, "load_evaluation_config": config,
                           "validate_v4_report": {}, "validate_model_preflight": {"models": {}},
                           "validate_calibration": {}, "archive_hashes": (runner.SOURCE_MD5, runner.EXPECTED_ARCHIVE_SHA256),
                           "model_contexts": {}, "_gpu_modules": (mock.Mock(),)}
                for name, value in patches.items():
                    stack.enter_context(mock.patch.object(runner, name, return_value=value))
                stack.enter_context(mock.patch.object(runner.importlib.metadata, "version", return_value="test"))
                patches["_gpu_modules"][0].cuda.get_device_name.return_value = "mock GPU"
                outcome = mock.Mock(side_effect=ValueError("runtime failure")) if failed else mock.Mock(return_value={"status": runner.COMPLETE})
                stack.enter_context(mock.patch.object(runner, "execute", outcome))
                if failed:
                    with self.assertRaisesRegex(ValueError, "runtime failure"):
                        runner.run(root, source)
                else:
                    result = runner.run(root, source)
                    self.assertEqual(result["status"], runner.COMPLETE)
                handoff = root / "artifacts" / "handoff" / f"{runner.NAME}.zip"
                with zipfile.ZipFile(handoff) as archive:
                    self.assertTrue(all(name.endswith(".json") for name in archive.namelist()))
                    if failed:
                        self.assertNotIn("evaluation.json", archive.namelist())
                        failure = json.loads(archive.read("failure.json"))
                        self.assertIsNone(failure["test_scores_computed"])
                        self.assertFalse(failure["test_opening_started"])
                with self.assertRaises(FileExistsError):
                    runner.run(root, source)


if __name__ == "__main__":
    unittest.main()
