import copy
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import mpi_normal_diversity_fit_protocol as p
from scripts import fit_mpi_normal_diversity as runner


def photos(group, start, count):
    return [{"key": f"{group}/{i:04}", "photo_id": f"{i:04}", "label": 0,
             "regions": [{"photo_id": f"{i:04}", "region": region,
                          "member": f"extracted/{region}/train/good/{i:04}.png",
                          "decoded_rgb_sha256": f"rgb-{i}-{region}"} for region in p.fit.REGIONS]}
            for i in range(start, start+count)]


def scored(group, start, count):
    result = photos(group, start, count)
    for i, photo in enumerate(result):
        for j, r in enumerate(photo["regions"]):
            r.update(top1=float(i+1+j), native_raw=float(i+1+j))
    return result


def test_q90_exact_formula_and_boundary():
    dev, cal = scored("development", 64, 128), scored("calibration", 192, 169)
    method = p.calibrate(dev, cal)
    assert method["regions"][0]["center"] == 64.5
    assert method["regions"][0]["scale"] == 51.5
    assert method["threshold"] == (161-64.5)/51.5
    assert p.scoring.score(cal[160], method)["alert"]
    assert not p.scoring.score(cal[159], method)["alert"]
    data = {"development": dev, "calibration": cal}
    assert p.reload_check(data, copy.deepcopy(data), method)["maximum_raw_score_difference"] == 0


@pytest.mark.parametrize("problem", ["count", "nan", "negative", "scale", "region", "label", "overlap"])
def test_bad_calibration_rejected(problem):
    dev, cal = scored("development", 64, 128), scored("calibration", 192, 169)
    if problem == "count": dev.pop()
    if problem == "nan": dev[0]["regions"][0]["top1"] = float("nan")
    if problem == "negative": dev[0]["regions"][0]["top1"] = -1.
    if problem == "scale":
        for photo in dev: photo["regions"][0]["top1"] = 1.
    if problem == "region": dev[0]["regions"].reverse()
    if problem == "label": dev[0]["label"] = 1
    if problem == "overlap":
        cal[0]["photo_id"] = dev[0]["photo_id"]
        cal[0]["key"] = "calibration/"+dev[0]["photo_id"]
    with pytest.raises(ValueError): p.calibrate(dev, cal)


@pytest.mark.parametrize("problem", ["raw", "decision", "inventory", "path"])
def test_reload_must_use_first_contract(problem):
    first = {"development": scored("development", 64, 128), "calibration": scored("calibration", 192, 169)}
    contract = p.calibrate(**first)
    second = copy.deepcopy(first)
    if problem == "raw": second["development"][0]["regions"][0]["native_raw"] += .01
    if problem == "decision":
        # All regions tied at the threshold; tiny raw drift is allowed but a decision flip is not.
        for r in second["calibration"][160]["regions"]: r["top1"] -= .00001
    if problem == "inventory": second["development"].reverse()
    if problem == "path": second["calibration"][0]["regions"][0]["member"] = "test/forbidden.png"
    with pytest.raises(ValueError): p.reload_check(first, second, contract)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    training = photos("training", 0, 64)
    chosen = {"training_photo_ids": [x["photo_id"] for x in training],
              "training_rows": [r for x in training for r in x["regions"]]}
    cohorts = {"development": photos("development", 64, 128), "calibration": photos("calibration", 192, 169)}
    rows = chosen["training_rows"] + [r for group in cohorts.values() for photo in group for r in photo["regions"]]
    source = SimpleNamespace(namelist=lambda: [r["member"] for r in rows]+["test/never-read.png"])
    state = {"fit": 0, "load": 0, "predict": 0, "fail": None, "drift": False, "budget": False, "reads": []}
    env = {"kind": "fake_cpu_not_gpu_evidence"}
    identity = {"model_config": {"seed": 42}, "policy": {"resource_plan": {"fit_calibration_reload_seconds": 14400}},
                "selection_environment": env}
    records = {"reload.json": {"receipt": {"runtime": env}}}

    class Image:
        def __init__(self, row): self.row = row
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def close(self): pass
    def checked(z, row):
        assert z is source and "/train/good/" in row["member"]
        state["reads"].append(row["member"])
        return Image(row)
    monkeypatch.setattr(runner.fit, "checked_image", checked)
    class Runtime:
        def __init__(self, config): self.reloaded = False
        def environment(self): return env
        def fit_locked(self, images, baseline, monitor):
            assert len(list(images)) == 768
            state["fit"] += 1
            return {"memory_bank_shape": [78643, 768], "kind": "fake"}
        def save(self, path, context): path.write_bytes(b"fake checkpoint")
        def load(self, path, context):
            assert path.read_bytes() == b"fake checkpoint"
            state["load"] += 1
            self.reloaded = True
        def predict(self, image):
            state["predict"] += 1
            if state["predict"] == state["fail"]: raise RuntimeError("synthetic interruption")
            value = float(int(image.row["photo_id"])+1)
            if self.reloaded and state["drift"]: value += 1.
            return value, np.full((256, 256), value, dtype=np.float32)
        def release(self): pass
    class Monitor:
        def __init__(self, *args): pass
        def __enter__(self): return self
        def __exit__(self, *a): pass
        def check(self):
            if state["budget"]: raise RuntimeError("synthetic budget exceeded")
        def snapshot(self): return {"kind": "fake"}
    def execute():
        return runner.execute(tmp_path/"run", source, tmp_path/"baseline.pt", chosen, cohorts,
                              records, identity, tmp_path/"handoff.zip", Runtime, Monitor, lambda r: env)
    return tmp_path, chosen, cohorts, identity, state, execute


def test_full_pipeline_reload_export_and_idempotent(synthetic):
    root, chosen, cohorts, identity, state, execute = synthetic
    report = execute()
    assert state["fit"] == 1 and state["load"] == 1
    assert state["predict"] == 297*12*2
    assert len(state["reads"]) == 768+297*12*2
    assert report["reload_verification"]["passed"]
    assert report["test_scores_computed"] == 0
    assert not report["automatic_pass_allowed"]
    before = dict(state)
    assert execute() == report
    assert state == before
    with zipfile.ZipFile(root/"handoff.zip") as z:
        assert all(n.endswith(".json") for n in z.namelist())
        attempts = json.loads(z.read("attempts.json"))
        assert attempts[-1]["outcome"] == "completed"
        assert report["cumulative_attempt_seconds"] == sum(a["seconds"] for a in attempts)


def test_interruption_reuses_model_and_completed_photos(synthetic):
    root, _, _, _, state, execute = synthetic
    state["fail"] = 13
    with pytest.raises(RuntimeError, match="interruption"): execute()
    assert (root/"run/training/model.pt").exists()
    assert len(list((root/"run/first").glob("development-*"))) == 1
    state["fail"] = None
    assert execute()["reload_verification"]["passed"]
    assert state["fit"] == 1
    assert len(list((root/"run").glob("attempt-*/finished.json"))) == 2


@pytest.mark.parametrize("problem", ["score", "model", "identity", "missing_cost", "environment"])
def test_cached_tamper_stops_before_gpu(synthetic, problem):
    root, _, _, identity, state, execute = synthetic
    execute()
    if problem == "score": next((root/"run/first").glob("*/score.json")).write_text("{}")
    if problem == "model": (root/"run/training/model.pt").write_bytes(b"tampered")
    if problem == "identity": identity["changed"] = True
    if problem == "missing_cost": next((root/"run").glob("attempt-*/finished.json")).unlink()
    if problem == "environment": (root/"run/environment/value.json").write_text("{}")
    before = dict(state)
    with pytest.raises((ValueError, RuntimeError)): execute()
    assert state == before


def test_reload_drift_cannot_publish(synthetic):
    root, _, _, _, state, execute = synthetic
    state["drift"] = True
    with pytest.raises(ValueError, match="Reload verification"): execute()
    assert not (root/"handoff.zip").exists()
    assert (root/"run/training/model.pt").exists()


def test_budget_and_unknown_crash_cannot_reset(synthetic):
    root, _, _, _, state, execute = synthetic
    state["budget"] = True
    with pytest.raises(RuntimeError, match="budget"): execute()
    assert not (root/"handoff.zip").exists()
    (root/"run/attempt-unknown").mkdir()
    with pytest.raises(ValueError, match="cost receipt"): execute()


@pytest.mark.parametrize("problem", ["test", "ids", "rgb", "order", "extra"])
def test_normal_cohort_guards(synthetic, problem):
    _, chosen, cohorts, _, _, _ = synthetic
    if problem == "test": cohorts["development"][0]["regions"][0]["member"] = "test/evil.png"
    if problem == "ids": cohorts["development"][0]["photo_id"] = chosen["training_photo_ids"][0]
    if problem == "rgb": cohorts["development"][0]["regions"][0]["decoded_rgb_sha256"] = chosen["training_rows"][0]["decoded_rgb_sha256"]
    if problem == "order": cohorts["calibration"][0]["regions"].reverse()
    if problem == "extra": cohorts["validation"] = []
    with pytest.raises(ValueError): p.validate_cohorts(chosen, cohorts)


def test_cli_argument_routing(monkeypatch, tmp_path):
    seen = []
    def prepare(project, handoffs):
        seen.append((project, handoffs))
        raise ValueError("synthetic stop before GPU")
    monkeypatch.setattr(p, "prepare", prepare)
    assert runner.main(["--project", str(tmp_path), "--archive", str(tmp_path/"source.zip")]) == 1
    assert seen == [(tmp_path, tmp_path/"artifacts/handoff")]


@pytest.mark.parametrize("fail", [False, True])
def test_runtime_keeps_baseline_fit_and_restores_hooks(monkeypatch, tmp_path, fail):
    calls = []
    class Sampler:
        def update_distances(self, cluster_center): calls.append(cluster_center)
    module = "anomalib.models.components.sampling.k_center_greedy"
    monkeypatch.setitem(sys.modules, module, SimpleNamespace(KCenterGreedy=Sampler))
    original = Sampler.update_distances
    class Tensor:
        def cpu(self): return self
    tensor = Tensor()
    class Extractor:
        def load_state_dict(self, weights, strict):
            assert strict and weights == {"weight": tensor}
        def state_dict(self): return {"weight": tensor}
    def constructor(**kw):
        assert kw["pre_trained"] is False
        return SimpleNamespace(model=SimpleNamespace(feature_extractor=Extractor()))
    torch = SimpleNamespace(load=lambda *a, **kw: {"state_dict": {
        "model.feature_extractor.weight": tensor, "model.memory_bank": "must_not_reuse"}}, equal=lambda a, b: a is b)
    instance = runner.Runtime.__new__(runner.Runtime)
    instance.torch = torch
    instance.gpu = (torch, None, constructor, None, "fake_cuda")
    gpu = instance.gpu
    checks = []
    def baseline_fit(self, images):
        self.model = self.gpu[2](pre_trained=True)
        for i in range(51): Sampler().update_distances(i)
        if fail: raise RuntimeError("synthetic fit failure")
        return {"memory_bank_shape": [78643, 768]}
    monkeypatch.setattr(runner.fit.TorchRuntime, "fit", baseline_fit)
    def run():
        return instance.fit_locked([], tmp_path/"model", SimpleNamespace(check=lambda: checks.append(1)))
    if fail:
        with pytest.raises(RuntimeError, match="fit failure"): run()
    else:
        assert run()["frozen_backbone_loaded_from_baseline"]
    assert calls == list(range(51))
    assert len(checks) == (3 if fail else 4)
    assert instance.gpu == gpu and Sampler.update_distances is original


def test_c_stage_budget_does_not_use_selection_cap(monkeypatch):
    seen = []
    def init(self, runtime, policy, prior): seen.append((runtime, policy, prior))
    monkeypatch.setattr(runner.preparation.Monitor, "__init__", init)
    policy = {"resource_plan": {"selection_seconds": 3600, "selection_and_inference_reserved_GiB": 6,
                                "fit_calibration_reload_seconds": 14400, "fit_reserved_GiB": 7, "rss_GiB": 20}}
    runner.Budget("runtime", policy, 12)
    assert seen[0][1]["resource_plan"]["selection_seconds"] == 14400
    assert seen[0][1]["resource_plan"]["selection_and_inference_reserved_GiB"] == 7
    assert policy["resource_plan"]["selection_seconds"] == 3600
    assert seen[0][2] == 12
