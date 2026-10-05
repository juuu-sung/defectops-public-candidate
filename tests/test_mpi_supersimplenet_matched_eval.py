"""Regress the 16-vs-8 gate defect without claiming a CUDA numerical result."""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mpi_supersimplenet_matched_eval_runtime as m
from scripts import mpi_supersimplenet_head_reset_protocol as p
from scripts import run_mpi_supersimplenet_head_reset as runner
from test_mpi_supersimplenet_varying import arrays
from test_mpi_supersimplenet_head_reset import setup_run, Monitor


class Tensor(np.ndarray):
    def abs(self): return np.abs(self)


def tensor(x): return np.asarray(x, dtype=np.float32).view(Tensor)


def test_public_gate_uses_eight_while_retaining_sixteen_item_outputs():
    seen = []
    class Model:
        def segdec(self, clean):
            seen.append(("head", clean.shape[0]))
            assert clean.shape[0] == 8
            return tensor(np.ones((8, 1, 2, 2))), tensor(np.ones(8))
        def __call__(self, x):
            seen.append(("public", x.shape[0]))
            return SimpleNamespace(anomaly_map=tensor(np.ones((8, 1, 2, 2))), pred_score=tensor(np.ones(8)))
        def anomaly_map_generator(self, raw, final_size):
            assert final_size == (256, 256)
            return raw
    maps16, scores16 = tensor(np.ones((16, 1, 2, 2))+.001), tensor(np.ones(16)+.002)
    maps_before, scores_before = maps16.copy(), scores16.copy()
    d = m.compare_public_paths(Model(), tensor(np.zeros((8, 3, 2, 2))), tensor(np.zeros((8, 6, 2, 2))),
                               maps16, scores16, 42)
    assert seen == [("head", 8), ("public", 8)]
    assert d["same_batch_map_max_abs"] == d["same_batch_score_max_abs"] == 0
    assert d["cross_batch_map_max_abs"] > .0001 and d["cross_batch_score_max_abs"] > .0001
    assert np.array_equal(maps16, maps_before) and np.array_equal(scores16, scores_before)


def valid_difference():
    return {"validation": m.PUBLIC_VALIDATION, "seed": 42, "public_batch": 8, "separated_batch": 8,
            "synthetic_batch": 16, "absolute_tolerance": 1e-4, "same_batch_map_max_abs": 0.,
            "same_batch_score_max_abs": 0., "cross_batch_map_max_abs": .001, "cross_batch_score_max_abs": .002}


@pytest.mark.parametrize("key,value", [("same_batch_map_max_abs", .00011),
    ("same_batch_score_max_abs", .00011), ("same_batch_map_max_abs", float("nan")),
    ("cross_batch_score_max_abs", float("inf")), ("separated_batch", 16), ("absolute_tolerance", .01)])
def test_no_tolerance_relaxation_or_nonfinite_bypass(key, value):
    d = valid_difference()
    m.validate_public_differences(d)
    d[key] = value
    with pytest.raises(m.PublicPathMismatch) as err:
        m.validate_public_differences(d)
    assert err.value.diagnostics is d and "seed" in str(err.value)


def previous_fixture(tmp_path):
    root = tmp_path/"v1"
    root.mkdir()
    original = {"backbone": "b", "adaptor": "a", "segmentation": "s", "classification": "c"}
    expected = {"name": p.PREVIOUS_NAME, "policy": p.POLICY, "baseline_fingerprints": {"baseline": True}}
    plan = {**expected, "runtime_environment": {"verified": True}}
    signature = p.fit.digest(plan)
    runner.publish(root, "identity", signature, plan)
    initial = {"seed": p.POLICY["head_initialization_seed"], "before": original,
               "after": {**original, "segmentation": "new", "classification": "new"},
               "batchnorm_modules": 2, "batchnorm_fresh": True, "optimizer_empty": True,
               "original_replay_max_abs_difference": 0.}
    runner.publish(root, "initialization", signature, initial)
    entries, flat = {}, {}
    for seed in p.POLICY["evaluation_seeds"]:
        a = arrays(500, seed)
        entries[str(seed)] = {**p.v.summarize(a), "checks": {"state_unchanged": True,
            "state_before_sha256": "same", "state_after_sha256": "same"}}
        flat.update({f"s{seed}__{k}": v for k, v in a.items()})
    with p.fit.phase_stage(root, "baseline-control") as stage:
        np.savez_compressed(stage/"arrays.npz", **flat)
        p.fit.write_json(stage/"progress.json", {"evaluations": entries,
            "fingerprints": expected["baseline_fingerprints"], "optimizer_updates": 0})
        p.fit.seal_phase(stage, "baseline-control", signature, {}, p.EVIDENCE_FILES)
    attempt = root/"attempt-fixture"
    attempt.mkdir()
    p.fit.write_json(attempt/"started.json", {"identity": signature})
    failed = {"status": "failed", "attempted_updates": 0, "seconds": 10.998531100107357,
              "error": p.PREVIOUS_ERROR}
    p.fit.write_json(attempt/"finished.json", failed)
    return root, expected, original, flat, failed


def test_zero_update_failure_is_read_only_and_carried_forward(tmp_path):
    root, expected, original, flat, failed = previous_fixture(tmp_path)
    before = {str(f.relative_to(root)): f.read_bytes() for f in root.rglob("*") if f.is_file()}
    prior = p.read_previous_failure(root, expected, original, flat)
    assert prior["seconds"] == failed["seconds"] and prior["attempted_updates"] == 0
    assert len(prior["files_sha256"]) == 9
    p.check_previous_files(root, prior)
    assert before == {str(f.relative_to(root)): f.read_bytes() for f in root.rglob("*") if f.is_file()}
    (root/"attempt-fixture/finished.json").write_text("tampered")
    with pytest.raises(ValueError, match="changed"): p.check_previous_files(root, prior)


@pytest.mark.parametrize("case", ["updates", "error", "status", "time", "step", "provenance", "replay", "unfinished", "multiple"])
def test_only_specific_pretraining_failure_is_accepted(tmp_path, case):
    root, expected, original, flat, failed = previous_fixture(tmp_path)
    if case == "updates": failed["attempted_updates"] = 1
    if case == "error": failed["error"] = "CUDA OOM"
    if case == "status": failed["status"] = "completed"
    if case == "time": failed["seconds"] = float("nan")
    if case == "step": (root/"step-0000").mkdir()
    if case == "provenance": expected = {**expected, "name": "other"}
    if case == "replay": flat["s42__eval_mixed_maps"] += 1
    if case == "multiple": (root/"attempt-other").mkdir()
    if case == "unfinished": (root/"attempt-fixture/started.json").unlink()
    (root/"attempt-fixture/finished.json").write_text(json.dumps(failed))
    with pytest.raises((ValueError, FileNotFoundError)):
        p.read_previous_failure(root, expected, original, flat)


def test_head_cannot_reroll_and_prior_time_exhausts_budget(tmp_path, monkeypatch):
    args, rt, seen = setup_run(tmp_path, monkeypatch)
    args[4]["previous_failure"]["initialization"]["after"]["segmentation"] = "different"
    with pytest.raises(ValueError, match="no reroll"):
        runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    assert 1 not in seen and not args[-1].exists()


def test_prior_failure_time_counts_before_loading_gpu(tmp_path, monkeypatch):
    args, rt, seen = setup_run(tmp_path, monkeypatch)
    args[4]["previous_failure"]["seconds"] = p.POLICY["resource_plan"]["comparison_seconds"]
    with pytest.raises(RuntimeError, match="time exhausted"):
        runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    assert not seen


def test_batch_differences_saved_on_failure(tmp_path, monkeypatch):
    args, rt, seen = setup_run(tmp_path, monkeypatch)
    original_eval = rt.evaluate_seed
    def evaluate(self, images, seed):
        if self.reset:
            d = valid_difference()
            d["same_batch_map_max_abs"] = .1
            raise m.PublicPathMismatch(d)
        return original_eval(self, images, seed)
    rt.evaluate_seed = evaluate
    with pytest.raises(m.PublicPathMismatch):
        runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    records = list(args[0].glob("attempt-*/finished.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_bytes())
    assert saved["attempted_updates"] == 0 and saved["diagnostics"]["same_batch_map_max_abs"] == .1
    assert saved["diagnostics"]["seed"] == 42 and 1 not in seen
