"""CPU control-flow/evidence checks, not a substitute for lab CUDA execution."""
import hashlib
import json
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest

from scripts import mpi_supersimplenet_head_reset_protocol as p
from scripts import run_mpi_supersimplenet_head_reset as runner
from scripts.mpi_supersimplenet_head_reset_runtime import Runtime as RealRuntime
from scripts.mpi_supersimplenet_matched_eval_runtime import PUBLIC_VALIDATION
from test_mpi_supersimplenet_varying import arrays, Monitor


def setup_run(tmp_path, monkeypatch):
    for policy in (p.POLICY, p.v.POLICY):
        monkeypatch.setitem(policy, "snapshots", [0, 1, 3])
        monkeypatch.setitem(policy, "updates", 3)
    formal = tmp_path/"formal"
    (formal/"step-028800").mkdir(parents=True)
    original = formal/"step-028800/model.pt"
    original.write_bytes(b"original")
    varying = tmp_path/"varying"
    (varying/"step-0500").mkdir(parents=True)
    baseline = varying/"step-0500/model.pt"
    baseline.write_bytes(b"baseline")
    archive = tmp_path/"archive.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    def fp(index, reset=False):
        return {"backbone": "same", "adaptor": str(index),
                "segmentation": f"seg{index}-{reset}", "classification": f"cls{index}-{reset}"}
    identity = {"checkpoint_sha256": p.fit.sha256_file(original),
        "baseline_checkpoint_sha256": p.fit.sha256_file(baseline), "fit_signature": "formal",
        "baseline_signature": "varying", "baseline_fingerprints": fp(500),
        "baseline_training_targets": [str(i) for i in range(3)], "job": {"rows": [{}]*8}, "policy": p.POLICY.copy()}
    records = {"report.json": {"final_fingerprints": fp(0)}}
    identity["previous_failure"] = {"seconds": 11., "files_sha256": {}, "initialization": {
        "seed": p.POLICY["head_initialization_seed"], "before": fp(0), "after": fp(0, True),
        "batchnorm_modules": 2, "batchnorm_fresh": True, "optimizer_empty": True}}
    references = {key: {f"s{s}__{n}": a for s in p.POLICY["evaluation_seeds"]
                       for n, a in arrays(step, s).items()}
                  for key, step in (("original", 0), ("baseline", 500))}
    seen = []
    class Runtime:
        interrupt_at = None
        failure = None
        drift = False
        bad_training_targets = False
        bad_evaluation_targets = False
        bad_reset = False
        def __init__(self, *args):
            self.index, self.reset, self.control = 0, False, False
            seen.append("init")
        def fingerprints(self):
            result = fp(self.index, self.reset)
            if self.bad_reset and self.reset:
                result["adaptor"] = "changed"
            return result
        def load_baseline(self, path, signature):
            assert path == baseline and signature == "varying"
            self.index, self.control = 500, True
            seen.append("baseline")
        def reset_head(self):
            assert not self.control and not self.reset and self.index == 0
            before = self.fingerprints()
            self.reset = True
            seen.append("reset")
            return {"seed": p.POLICY["head_initialization_seed"], "before": before,
                    "after": self.fingerprints(), "batchnorm_modules": 2,
                    "batchnorm_fresh": True, "optimizer_empty": True}
        def step(self, images, index):
            assert self.reset and not self.control and len(images) == 8
            seen.append(index+1)
            if index+1 == self.interrupt_at:
                raise KeyboardInterrupt()
            if self.failure:
                raise RuntimeError(self.failure)
            self.index = index+1
            return {"step": index+1, "seed": p.v.training_seed(index), "loss": 1/(index+1),
                    "target_sha256": "wrong" if self.bad_training_targets else str(index)}
        def evaluate_seed(self, images, seed):
            changed = self.bad_evaluation_targets and self.reset
            return arrays(self.index + (10 if self.reset else 0), seed+int(changed)), {
                "state_unchanged": True, "state_before_sha256": "same", "state_after_sha256": "same",
                "public_eval_map_max_abs_difference": 0., "public_eval_score_max_abs_difference": 0.,
                "public_batch_comparison": {"validation": PUBLIC_VALIDATION, "seed": seed,
                    "public_batch": 8, "separated_batch": 8, "synthetic_batch": 16, "absolute_tolerance": 1e-4,
                    "same_batch_map_max_abs": 0., "same_batch_score_max_abs": 0.,
                    "cross_batch_map_max_abs": .0002, "cross_batch_score_max_abs": 0.}}
        def save(self, path, signature, step):
            assert self.reset
            path.write_text(json.dumps([signature, step]))
        def load(self, path, signature, step):
            assert json.loads(path.read_bytes()) == [signature, step]
            self.index, self.reset = step+int(self.drift), True
            seen.append(f"load{step}")
        def release(self):
            seen.append("release")
    monkeypatch.setattr(p.fit, "checked_image", lambda *a: Image.new("RGB", (256, 256)))
    args = (tmp_path/"run", archive, formal, varying, identity, records, references, tmp_path/"handoff.zip")
    return args, Runtime, seen


def test_policy_changes_only_initialization():
    changed = {k for k in p.v.POLICY if p.POLICY[k] != p.v.POLICY[k]}
    assert changed == {"initialization", "fixed_control"}
    assert p.POLICY["head_initialization_seed"] == 20261006
    assert p.POLICY["updates"] == 500 and not p.POLICY["automatic_model_selection"]


def test_execute_replay_reset_training_reload_export_and_noop(tmp_path, monkeypatch):
    args, rt, seen = setup_run(tmp_path, monkeypatch)
    report = runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    assert report["completed_updates"] == 3 and report["unique_training_target_hashes"] == 3
    assert report["final_reload_max_abs_difference"] == 0 and report["test_images_decoded"] == 0
    assert report["all_training_targets_match_baseline"]
    assert report["previous_failure_seconds"] == 11 and report["cumulative_compute_seconds"] >= 11
    assert seen.count("reset") == 1 and seen.count("baseline") == 1
    assert seen.index("baseline") < seen.index("reset") < seen.index(1)
    assert runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor) == report
    assert seen.count("init") == 2
    assert (args[2]/"step-028800/model.pt").read_bytes() == b"original"
    assert (args[3]/"step-0500/model.pt").read_bytes() == b"baseline"
    with zipfile.ZipFile(args[-1]) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert len(z.namelist()) == len(manifest)+1
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
        assert not any(n.endswith((".pt", ".png", ".jpg")) for n in z.namelist())
        assert "initialization.json" in manifest
    (args[0]/"step-0000/progress.json").write_bytes(b"tampered")
    with pytest.raises(ValueError):
        runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)


def test_resume_restores_without_second_reset(tmp_path, monkeypatch):
    args, rt, seen = setup_run(tmp_path, monkeypatch)
    rt.interrupt_at = 3
    with pytest.raises(KeyboardInterrupt):
        runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    rt.interrupt_at = None
    report = runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    assert seen.count("reset") == 1 and seen.count("baseline") == 1
    assert "load1" in seen and seen.count(2) == 2
    assert report["attempted_updates_including_replayed_work"] == 5


@pytest.mark.parametrize("case", ["resource", "source", "training-target", "eval-target", "reset", "replay", "reload", "budget", "hard-stop"])
def test_fail_closed(tmp_path, monkeypatch, case):
    args, rt, seen = setup_run(tmp_path, monkeypatch)
    if case == "resource": rt.failure = "resource limit"
    if case == "source": (args[3]/"step-0500/model.pt").write_bytes(b"changed")
    if case == "training-target": rt.bad_training_targets = True
    if case == "eval-target": rt.bad_evaluation_targets = True
    if case == "reset": rt.bad_reset = True
    if case == "replay": args[6]["baseline"]["s42__eval_mixed_scores"][0] += 10
    if case == "reload": rt.drift = True
    if case == "budget": monkeypatch.setitem(p.POLICY, "maximum_attempted_updates", 1)
    if case == "hard-stop":
        args[0].mkdir()
        runner.publish(args[0], "identity", p.fit.digest(args[4]), args[4])
        (args[0]/"attempt-incomplete").mkdir()
    with pytest.raises((RuntimeError, ValueError)):
        runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)
    assert not args[-1].exists()
    if case in ("replay", "source", "reset"):
        assert 1 not in seen
    if case == "resource":
        with pytest.raises(ValueError, match="review"):
            runner.execute(*args, runtime_factory=rt, monitor_factory=Monitor)


@pytest.mark.parametrize("field,value", [("seed", 7), ("batchnorm_modules", 0),
                                        ("batchnorm_fresh", False), ("optimizer_empty", False)])
def test_bad_initialization_contract(field, value):
    original = {"backbone": "b", "adaptor": "a", "segmentation": "s", "classification": "c"}
    record = {"seed": p.POLICY["head_initialization_seed"], "before": original,
              "after": {**original, "segmentation": "new", "classification": "new"},
              "batchnorm_modules": 2, "batchnorm_fresh": True, "optimizer_empty": True}
    p.validate_initialization(record, original)
    record[field] = value
    with pytest.raises(ValueError): p.validate_initialization(record, original)


def test_runtime_guards_and_cli_without_torch():
    runtime = object.__new__(RealRuntime)
    runtime.control_only = True
    with pytest.raises(ValueError, match="forbidden"): runtime.reset_head()
    with pytest.raises(ValueError, match="read-only"): runtime.step([], 0)
    with pytest.raises(ValueError): runtime.save(None, "sig", 0)
    runtime.control_only = False
    with pytest.raises(ValueError, match="initialized"): runtime.step([], 0)
    result = subprocess.run([sys.executable, str(Path(runner.__file__)), "--help"], capture_output=True, text=True)
    assert result.returncode == 0 and "--archive" in result.stdout
    result = subprocess.run([sys.executable, "-c",
        "from scripts import run_mpi_supersimplenet_head_reset; import sys; assert 'torch' not in sys.modules"],
        capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_official_head_constructor_boundary_and_rng_scope(monkeypatch):
    import numpy as np
    events = []
    class BN:
        running_mean = np.zeros(2)
        running_var = np.ones(2)
        weight = np.ones(2)
        bias = np.zeros(2)
        num_batches_tracked = 0
    class Head:
        def __init__(self, **kwargs):
            assert kwargs == {"channel_dim": 1536, "stop_grad": True}
            events.append("new official head")
        def state_dict(self): return {"fixture": np.zeros((2, 3))}
        def modules(self): return [self, BN(), BN()]
        def to(self, **kwargs):
            assert kwargs == {"device": "cuda", "dtype": "float32"}
            events.append("gpu")
            return self
        def requires_grad_(self, flag):
            assert flag is True
            return self
    old_head = Head(channel_dim=1536, stop_grad=True)
    events.clear()
    @contextmanager
    def rng(**kwargs):
        assert kwargs == {"devices": [0]}
        events.append("rng saved")
        yield
        events.append("rng restored")
    @contextmanager
    def device(name):
        assert name == "cpu"
        yield
    torch = SimpleNamespace(random=SimpleNamespace(fork_rng=rng), device=device,
        cuda=SimpleNamespace(device_count=lambda: 1), float32="float32",
        manual_seed=lambda seed: events.append(seed),
        nn=SimpleNamespace(modules=SimpleNamespace(batchnorm=SimpleNamespace(_BatchNorm=BN))))
    monkeypatch.setitem(sys.modules, "anomalib.models.image.supersimplenet.torch_model",
                        SimpleNamespace(SegmentationDetectionModule=Head))
    runtime = object.__new__(RealRuntime)
    runtime.torch = torch
    runtime.model = SimpleNamespace(segdec=old_head, eval=lambda: events.append("eval"))
    runtime.optimizer = SimpleNamespace(state={"old": True})
    runtime.new_optimizer = lambda: SimpleNamespace(state={})
    runtime.fingerprints = lambda: {"backbone": "same", "adaptor": "same",
        "segmentation": "old" if runtime.model.segdec is old_head else "fresh",
        "classification": "old" if runtime.model.segdec is old_head else "fresh"}
    record = runtime.reset_head()
    assert record["batchnorm_fresh"] and record["optimizer_empty"] and runtime.reset_applied
    assert events == ["rng saved", 20261006, "new official head", "rng restored", "gpu", "eval"]
    with pytest.raises(ValueError, match="forbidden"): runtime.reset_head()


def test_prepare_keeps_reviewed_provenance_and_pins_new_sources(tmp_path, monkeypatch):
    before = {"sources_lf_sha256": {"old.py": "unchanged"}, "policy": p.n.POLICY,
              "baseline_training_targets": [str(i) for i in range(500)], "baseline_checkpoint_sha256": "pinned"}
    monkeypatch.setattr(p.n, "prepare", lambda *a: ({"report.json": {"final_fingerprints": {}}}, before, {"baseline": True}))
    monkeypatch.setattr(p, "read_previous_failure", lambda *a: {"seconds": 11})
    records, identity, references = p.prepare(runner.ROOT, tmp_path, tmp_path, tmp_path, tmp_path)
    assert records == {"report.json": {"final_fingerprints": {}}} and references == {"baseline": True}
    assert identity["policy"] == p.POLICY and identity["baseline_checkpoint_sha256"] == "pinned"
    assert len(identity["sources_lf_sha256"]) == 7 and before["sources_lf_sha256"] == {"old.py": "unchanged"}
    before["baseline_training_targets"].pop()
    with pytest.raises(ValueError): p.prepare(runner.ROOT, tmp_path, tmp_path, tmp_path, tmp_path)


def test_main_paths(tmp_path, monkeypatch):
    archive = tmp_path/"archive.zip"
    archive.write_bytes(b"fixture")
    formal = tmp_path/"artifacts"/p.v.q.p.NAME
    fixed = tmp_path/"artifacts/analyses"/p.v.old.NAME
    varying = tmp_path/"artifacts/analyses"/p.v.NAME
    prior = tmp_path/"artifacts/analyses"/p.PREVIOUS_NAME
    for path in (formal, fixed, varying, prior): path.mkdir(parents=True)
    monkeypatch.setattr(p.fit, "EXPECTED_ARCHIVE_SHA256", p.fit.sha256_file(archive))
    env = {k: "pinned" for k in ("torch", "torchvision", "anomalib", "timm", "supersimplenet_sources")}
    def prepare(project, handoffs, fit_root, fixed_root, varying_root):
        assert (fit_root, fixed_root, varying_root) == (formal, fixed, varying)
        return {}, {"environment": env}, {}
    monkeypatch.setattr(p, "prepare", prepare)
    monkeypatch.setattr(runner, "environment", lambda: env.copy())
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *a: type("Disk", (), {"free": 20*1024**3})())
    seen = []
    monkeypatch.setattr(runner, "execute", lambda *a: seen.append(a) or {"fixture": True})
    assert runner.main(["--project", str(tmp_path), "--archive", str(archive)]) == 0
    assert seen[0][0] == tmp_path/"artifacts/analyses"/p.NAME
    assert seen[0][3] == varying and seen[0][-1] == tmp_path/"artifacts/handoff"/(p.NAME+".zip")
