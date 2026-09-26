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
from PIL import Image, ImageDraw

from scripts import evaluate_mpi_pcb_external_v2 as runner
from scripts import mpi_pcb_external_v2_protocol as protocol


def audit_fixture():
    photos = []
    for label, group in enumerate(("good", "defect")):
        for i in range(55):
            pid = f"{i:04d}"
            rows = []
            for region in protocol.REGIONS:
                member = f"extracted/{region}/test/{group}/{pid}.png"
                row = {"region": region, "member": member,
                       "member_sha256": hashlib.sha256(member.encode()).hexdigest(),
                       "decoded_rgb_sha256": hashlib.sha256((member + "rgb").encode()).hexdigest(),
                       "mask_member": f"extracted/{region}/ground_truth/defect/{pid}.png" if label else None}
                if label:
                    row.update(mask_sha256="mask", mask={"nonzero_pixels": 1})
                if member == protocol.REPAIR_MEMBER:
                    row["member_sha256"] = protocol.DAMAGED_SHA256
                    del row["decoded_rgb_sha256"]
                if row["mask_member"] == protocol.MASK_MEMBER:
                    row["mask_sha256"] = protocol.MASK_SHA256
                    del row["mask"]
                rows.append(row)
            photos.append({"key": f"{group}/{pid}", "photo_id": pid, "label": label, "regions": rows})
    old = {"photos": photos, "duplicates": copy.deepcopy(protocol.EXPECTED_DUPLICATES),
           "test_image_read_attempts": 1320, "test_images_decoded": 1319,
           "test_mask_read_attempts": 660, "test_masks_decoded": 659, "test_scores_computed": 0,
           "issues": [{"member": protocol.MASK_MEMBER, "error": "known mask failure"},
                      {"member": protocol.REPAIR_MEMBER, "error": "known image failure"},
                      {"error": "Exact 12-region photo duplicate: test/test or calibration/test"}]}
    current = copy.deepcopy(old)
    recovery = {"recovered_png_sha256": "derived png hash"}
    repaired = next(r for p in current["photos"] for r in p["regions"] if r["member"] == protocol.REPAIR_MEMBER)
    repaired.update(member_sha256=recovery["recovered_png_sha256"], decoded_rgb_sha256=protocol.REPAIRED_RGB_SHA256)
    current["test_images_decoded"] = 1320
    current["issues"] = [old["issues"][0], old["issues"][2]]
    return old, current, recovery


class AmendmentTests(unittest.TestCase):
    def test_config_keeps_thresholds_and_refuses_edit(self):
        root = Path(__file__).resolve().parents[1]
        config = root / "configs/mpi-pcb-external-evaluation-v2.toml"
        self.assertEqual(protocol.load_config(config)["thresholds"], protocol.V1_CONFIG["thresholds"])
        with tempfile.TemporaryDirectory() as temp:
            changed = Path(temp) / "changed.toml"
            changed.write_text(config.read_text().replace("included_defect_photos = 55", "included_defect_photos = 54"))
            with self.assertRaisesRegex(ValueError, "locked"):
                protocol.load_config(changed)

    def test_exact_amendments_keep_all_defects_and_mask_only_case(self):
        old, audit, recovery = audit_fixture()
        old_copy = copy.deepcopy(old)
        selected, report = protocol.amended_cohort(audit, old, recovery)
        self.assertEqual(len(selected), 109)
        self.assertEqual(report["included_normal_photos"], 54)
        self.assertEqual(report["included_defect_photos"], 55)
        self.assertIn("defect/0054", report["photo_keys"])
        self.assertIn("defect/0006", report["photo_keys"])
        self.assertNotIn("good/0013", report["photo_keys"])
        self.assertEqual(old, old_copy)
        self.assertEqual(report["test_scores_computed"], 0)

    def test_new_error_or_duplicate_never_silently_excluded(self):
        for kind in ("error", "duplicate", "image", "mask", "count", "label", "extra_photo"):
            with self.subTest(kind=kind):
                old, audit, recovery = audit_fixture()
                if kind == "error":
                    audit["issues"].append({"member": "unexpected.png", "error": "broken"})
                elif kind == "duplicate":
                    audit["duplicates"]["identical_test_photo_groups"] = [["good/0000", "good/0001"]]
                elif kind in ("image", "mask"):
                    audit["photos"][-1]["regions"][0][kind + "_unexpected_change"] = True
                elif kind == "count":
                    audit["test_masks_decoded"] = 658
                elif kind == "label":
                    audit["photos"][0]["label"] = 1
                else:
                    audit["photos"].append(copy.deepcopy(audit["photos"][0]))
                with self.assertRaises(ValueError):
                    protocol.amended_cohort(audit, old, recovery)

    def test_overlay_never_mutates_original_or_other_members(self):
        source = mock.Mock()
        source.read.side_effect = lambda name: b"broken" if name == protocol.REPAIR_MEMBER else b"unchanged"
        with mock.patch.object(protocol, "DAMAGED_SHA256", hashlib.sha256(b"broken").hexdigest()):
            overlay = protocol.RecoveredSource(source, b"reconstructed")
        self.assertEqual(overlay.read(protocol.REPAIR_MEMBER), b"reconstructed")
        self.assertEqual(overlay.read("some-other-member"), b"unchanged")
        self.assertEqual(source.read.call_count, 2)
        with self.assertRaisesRegex(ValueError, "Original damaged"):
            protocol.RecoveredSource(source, b"reconstructed")

    def test_prior_handoff_hash_is_mandatory(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "prior.zip"
            path.write_bytes(b"not the reviewed ZIP")
            with self.assertRaisesRegex(ValueError, "checksum"):
                protocol.validate_prior(path)

    def test_sensitivity_does_not_replace_main_denominator(self):
        old, audit, recovery = audit_fixture()
        selected, cohort = protocol.amended_cohort(audit, old, recovery)
        results = {}
        for key in runner.v1.MODEL_ORDER:
            threshold = protocol.EXPECTED_CONFIG["thresholds"][key]
            results[key] = {"threshold": threshold, "checkpoint_sha256": key, "photos": [
                {**p, "score": 100. if p["label"] and p["key"] != "defect/0054" else 0.,
                 "threshold": threshold, "decision": "test"} for p in selected]}
        report = runner.summarize(results, protocol.EXPECTED_CONFIG, cohort)
        self.assertEqual(report["models"]["realiad_pcb"]["metrics"]["fixed_threshold"]["counts"],
                         {"tp": 54, "fn": 1, "fp": 0, "tn": 54})
        sensitivity = report["sensitivity_without_reconstructed_photo"]
        self.assertEqual(sensitivity["models"]["realiad_pcb"]["fixed_threshold"]["counts"],
                         {"tp": 54, "fn": 0, "fp": 0, "tn": 54})
        self.assertEqual(report["test_region_scores_total"], 2616)
        self.assertFalse(report["automatic_pass_allowed"])
        results["realiad_pcb"]["photos"].pop()
        with self.assertRaisesRegex(ValueError, "denominator"):
            runner.summarize(results, protocol.EXPECTED_CONFIG, cohort)


class ReconstructionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        image = Image.new("RGB", (4096, 2816), (251, 252, 253))
        draw = ImageDraw.Draw(image)
        for x in range(0, 4096, 29):
            draw.rectangle((x, 0, x + 2, 2815), fill=(x % 256, 23, 147))
        stream = io.BytesIO()
        image.save(stream, format="JPEG")
        cls.raw = stream.getvalue()
        cls.sha = hashlib.sha256(cls.raw).hexdigest()

    def test_transform_exact_integer_math_positions_and_no_overflow(self):
        with mock.patch.object(protocol, "SOURCE_MEMBER_SHA256", self.sha):
            tiles = protocol.rebuild_tiles(self.raw)
        with Image.open(io.BytesIO(self.raw)) as image:
            source = np.asarray(image).astype(np.uint16)
        for region in protocol.REGIONS:
            gx, gy = map(int, region.split("_")[0][4:].split("-"))
            x, y = (gx - 1) * 1024, (gy - 1) * 896
            self.assertEqual(tiles[region].shape, (256, 256, 3))
            self.assertEqual(tiles[region].dtype, np.uint8)
            for px, py in ((0, 0), (255, 255), (17, 29)):
                expected = (source[y + py * 4 + 1:y + py * 4 + 3, x + px * 4 + 1:x + px * 4 + 3].sum(axis=(0, 1)) + 2) // 4
                np.testing.assert_array_equal(tiles[region][py, px], expected)

    def test_all_11_controls_and_recovered_hash_required(self):
        with mock.patch.object(protocol, "SOURCE_MEMBER_SHA256", self.sha):
            tiles = protocol.rebuild_tiles(self.raw)
            hashes = {key: hashlib.sha256(value.tobytes()).hexdigest() for key, value in tiles.items()}
            prior, _, _ = audit_fixture()
            photo = next(p for p in prior["photos"] if p["key"] == "defect/0054")
            for row in photo["regions"]:
                if row["member"] != protocol.REPAIR_MEMBER:
                    row["decoded_rgb_sha256"] = hashes[row["region"]]
            with mock.patch.object(protocol, "REPAIRED_RGB_SHA256", hashes["grid1-3_s1024"]):
                raw, report = protocol.reconstruct(self.raw, prior)
                self.assertEqual(report["intact_control_count"], 11)
                with Image.open(io.BytesIO(raw)) as image:
                    np.testing.assert_array_equal(np.asarray(image), tiles["grid1-3_s1024"])
                photo["regions"][0]["decoded_rgb_sha256"] = "wrong"
                with self.assertRaisesRegex(ValueError, "control"):
                    protocol.reconstruct(self.raw, prior)
            with self.assertRaises(ValueError):
                protocol.reconstruct(self.raw, prior)
        with self.assertRaisesRegex(ValueError, "source bytes"):
            protocol.rebuild_tiles(self.raw + b"modified")

    def test_remote_ranges_refuse_full_download_wrong_ranges_and_truncation(self):
        for status, content_range, raw in ((200, None, b"abc"), (206, "wrong", b"abc"),
                                           (206, f"bytes 0-2/{protocol.SOURCE_SIZE}", b"ab")):
            response = mock.MagicMock()
            response.__enter__.return_value = response
            response.status, response.headers = status, {"Content-Range": content_range}
            response.read.return_value = raw
            with mock.patch.object(protocol.urllib.request, "urlopen", return_value=response):
                with self.assertRaises(ValueError):
                    protocol.BoundedRemoteZip().read(3)
        remote = protocol.BoundedRemoteZip()
        with self.assertRaisesRegex(ValueError, "limit"):
            remote.read()
        with self.assertRaises(ValueError):
            remote.seek(-1)
        with self.assertRaises(ValueError):
            remote.seek(0, 4)
        remote.seek(-3, 2)
        response = mock.MagicMock()
        response.__enter__.return_value = response
        response.status = 206
        response.headers = {"Content-Range": f"bytes {protocol.SOURCE_SIZE-3}-{protocol.SOURCE_SIZE-1}/{protocol.SOURCE_SIZE}"}
        response.read.return_value = b"abc"
        with mock.patch.object(protocol.urllib.request, "urlopen", return_value=response):
            self.assertEqual(remote.read(3), b"abc")
        self.assertEqual(remote.downloaded_bytes, 3)
        self.assertEqual(remote.read(3), b"")


class RunnerTests(unittest.TestCase):
    def exercise_order(self, root, *, invalid_audit=False, bad_replay=False, score_failure=False):
        old, audit, recovery = audit_fixture()
        if invalid_audit:
            audit["issues"].append({"error": "new issue"})
        photo = {"photo_id": "normal", "score": 0., "regions": [
            {"region": region, "member": region, "score": 0.} for region in protocol.REGIONS]}
        events = []
        def audit_call(*_):
            events.append("audit")
            return audit
        def replay(*_):
            events.append("normal")
            current = copy.deepcopy(photo)
            if bad_replay:
                current["regions"][0]["score"] = 1.
            return [current]
        def score(_source, photos, _model, _recipe, _gpu, threshold, _maps):
            events.append("test")
            if score_failure:
                raise RuntimeError("test scoring failure")
            return [{**p, "score": 100. if p["label"] else 0., "threshold": threshold, "decision": "test"}
                    for p in photos]
        calibration = {"models": {key: {"photo_scores": [photo]} for key in runner.v1.MODEL_ORDER}}
        gate = {"models": {key: {"checkpoint_path": key, "checkpoint_sha256": key} for key in runner.v1.MODEL_ORDER}}
        recipes = {key: {"base_model": {}, "checkpoint_context": {}} for key in runner.v1.MODEL_ORDER}
        with ExitStack() as stack:
            for name, replacement in (("audit_test", audit_call), ("test_inventory", lambda *_: []),
                                      ("score_normal_cohort", replay), ("score_test_photos", score),
                                      ("_model", lambda *_: None), ("_seed_everything", lambda *_: None)):
                stack.enter_context(mock.patch.object(runner.v1, name, replacement))
            try:
                report = runner.execute(mock.Mock(), root, protocol.EXPECTED_CONFIG,
                                        {"calibration_rows": [], "calibration_photo_ids": ["normal"]},
                                        calibration, gate, recipes, (mock.Mock(),), {"test-audit.json": old}, recovery)
            except Exception as exc:
                return exc, events
        return report, events

    def test_audit_failure_stops_before_models_and_scores(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(Path(temp), invalid_audit=True)
            self.assertIsInstance(result, ValueError)
            self.assertEqual(events, ["audit"])

    def test_replay_failure_blocks_both_test_models(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(Path(temp), bad_replay=True)
            self.assertEqual(events, ["audit", "normal", "normal"])
            self.assertEqual(result["test_scores_computed"], 0)
            self.assertTrue(result["test_data_previously_opened_for_integrity"])

    def test_complete_order_and_frozen_thresholds(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(Path(temp))
            self.assertEqual(events, ["audit", "normal", "normal", "test", "test"])
            self.assertEqual(result["status"], runner.COMPLETE)
            for key in runner.v1.MODEL_ORDER:
                self.assertEqual(result["models"][key]["threshold"], protocol.V1_CONFIG["thresholds"][key])
                self.assertEqual(result["models"][key]["metrics"]["fixed_threshold"]["counts"],
                                 {"tp": 55, "fn": 0, "tn": 54, "fp": 0})

    def test_partial_failure_is_not_reported_as_complete(self):
        with tempfile.TemporaryDirectory() as temp:
            result, events = self.exercise_order(Path(temp), score_failure=True)
            self.assertIsInstance(result, RuntimeError)
            self.assertEqual(events, ["audit", "normal", "normal", "test"])

    def test_json_only_handoff_excludes_recovery_pixels_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            runner.v1.write_json(root / "recovery.json", {"status": "test"})
            (root / "recovery").mkdir()
            (root / "recovery" / "source.jpg").write_bytes(b"not shared")
            (root / "image.png").write_bytes(b"not shared")
            runner.package_reports(root, root / "handoff.zip")
            with zipfile.ZipFile(root / "handoff.zip") as archive:
                self.assertEqual(set(archive.namelist()), {"recovery.json", "handoff-manifest.json"})
            with self.assertRaises(FileExistsError):
                runner.package_reports(root, root / "handoff.zip")

    def test_cli_path_mapping_and_exit_codes(self):
        with mock.patch.object(runner, "run", return_value={"status": runner.COMPLETE}) as run:
            self.assertEqual(runner.main(["--archive", "/tmp/source.zip", "--project", "/tmp/project"]), 0)
            run.assert_called_once_with(Path("/tmp/project").resolve(), Path("/tmp/source.zip").resolve())
        with mock.patch.object(runner, "run", return_value={"status": "blocked"}):
            self.assertEqual(runner.main(["--archive", "/tmp/source.zip"]), 2)
        with mock.patch.object(runner, "run", side_effect=RuntimeError("failure")):
            self.assertEqual(runner.main(["--archive", "/tmp/source.zip"]), 1)

    def test_run_preserves_v1_and_packages_success_or_partial_failure(self):
        for failed in (False, True):
            with self.subTest(failed=failed), tempfile.TemporaryDirectory(prefix="MPI 한글 ") as temp, ExitStack() as stack:
                root = Path(temp)
                archive_path = root / "source.zip"
                with zipfile.ZipFile(archive_path, "w") as source:
                    source.writestr("untouched", b"source")
                before = archive_path.read_bytes()
                prior_dir = root / "artifacts" / "evaluations" / "mpi-pcb-external-evaluation-v1"
                prior_dir.mkdir(parents=True)
                (prior_dir / "keep.json").write_bytes(b"unchanged v1")
                torch = mock.Mock()
                torch.cuda.get_device_name.return_value = "mock GPU"
                for name, value in (("git_state", {"commit": "test"}), ("load_protocol", {}),
                                    ("validate_v4_report", {}), ("validate_model_preflight", {"models": {}}),
                                    ("validate_calibration", {}), ("model_contexts", {}),
                                    ("archive_hashes", (runner.v1.SOURCE_MD5, runner.v1.EXPECTED_ARCHIVE_SHA256)),
                                    ("_gpu_modules", (torch,))):
                    stack.enter_context(mock.patch.object(runner.v1, name, return_value=value))
                for name, value in (("load_config", {**protocol.EXPECTED_CONFIG, "config_sha256": "config"}),
                                    ("validate_prior", {"test-audit.json": {}}),
                                    ("download_source_member", (b"official JPEG", {})),
                                    ("reconstruct", (b"derived PNG", {}))):
                    stack.enter_context(mock.patch.object(runner, name, return_value=value))
                stack.enter_context(mock.patch.object(runner, "RecoveredSource", side_effect=lambda source, _: source))
                stack.enter_context(mock.patch.object(runner.importlib.metadata, "version", return_value="test"))
                outcome = mock.Mock(side_effect=RuntimeError("partial failure")) if failed else mock.Mock(return_value={"status": runner.COMPLETE})
                stack.enter_context(mock.patch.object(runner, "execute", outcome))
                if failed:
                    with self.assertRaisesRegex(RuntimeError, "partial failure"):
                        runner.run(root, archive_path)
                else:
                    self.assertEqual(runner.run(root, archive_path)["status"], runner.COMPLETE)
                self.assertEqual(archive_path.read_bytes(), before)
                self.assertEqual((prior_dir / "keep.json").read_bytes(), b"unchanged v1")
                handoff = root / "artifacts" / "handoff" / f"{runner.NAME}.zip"
                with zipfile.ZipFile(handoff) as zipped:
                    self.assertTrue(all(name.endswith(".json") for name in zipped.namelist()))
                    self.assertEqual("failure.json" in zipped.namelist(), failed)
                    if failed:
                        report = json.loads(zipped.read("failure.json"))
                        self.assertIsNone(report["test_scores_computed"])
                        self.assertTrue(report["test_data_previously_opened_for_integrity"])
                with self.assertRaises(FileExistsError):
                    runner.run(root, archive_path)


if __name__ == "__main__":
    unittest.main()
