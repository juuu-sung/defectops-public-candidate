"""CPU control/evidence tests; lab CUDA learning remains a separate validation."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_noise_protocol as p
from scripts import run_mpi_supersimplenet_noise as runner
from scripts.mpi_supersimplenet_noise_runtime import noise_level
from test_mpi_supersimplenet_varying import arrays, Monitor


def setup_run(tmp_path, monkeypatch):
    monkeypatch.setitem(p.POLICY, "snapshots", [0, 1, 3])
    monkeypatch.setitem(p.POLICY, "updates", 3)
    formal = tmp_path/"formal"; (formal/"step-028800").mkdir(parents=True)
    original = formal/"step-028800/model.pt"; original.write_bytes(b"original")
    varying = tmp_path/"varying"; (varying/"step-0500").mkdir(parents=True)
    baseline = varying/"step-0500/model.pt"; baseline.write_bytes(b"baseline")
    archive = tmp_path/"archive.zip"
    with zipfile.ZipFile(archive, "w"):
        pass
    identity = {"checkpoint_sha256": p.fit.sha256_file(original),
        "baseline_checkpoint_sha256": p.fit.sha256_file(baseline), "fit_signature": "formal",
        "baseline_signature": "varying", "baseline_fingerprints": {"backbone": "same", "head": 500},
        "baseline_training_targets": [str(i) for i in range(3)], "job": {"rows": [{}]*8}, "policy": p.POLICY.copy()}
    records = {"report.json": {"final_fingerprints": {"backbone": "same", "head": 0}}}
    references = {key: {f"s{s}__{n}": a for s in p.v.POLICY["evaluation_seeds"]
                       for n, a in arrays(step, s).items()}
                  for key, step in (("original", 0), ("baseline", 500))}
    seen = []
    class Runtime:
        interrupt_at = None; fail = False; drift = False; bad_targets = False
        def __init__(self, *args): self.index = 0; self.control = False; seen.append("init")
        def fingerprints(self): return {"backbone": "same", "head": self.index}
        def load_baseline(self, path, signature):
            assert path == baseline and signature == "varying"
            self.control = True; self.index = 500; seen.append("baseline")
        def step_at_strength(self, images, index, arm):
            assert not self.control and len(images) == 8
            seen.append((arm, index+1))
            if (arm, index+1) == self.interrupt_at: raise KeyboardInterrupt()
            if self.fail: raise RuntimeError("resource limit")
            self.index = index+1
            return {"step": index+1, "seed": p.v.training_seed(index), "noise_std": p.STRENGTHS[arm],
                    "loss": 1/(index+1), "target_sha256": str(index)}
        def evaluate_strength(self, images, seed, strength):
            changed = self.bad_targets and strength == "std060"
            return arrays(self.index, seed+int(changed)), {"state_unchanged": True,
                "state_before_sha256": "same", "state_after_sha256": "same", "noise_std": p.STRENGTHS[strength]}
        def save_arm(self, path, signature, step, arm): path.write_text(json.dumps([signature, step, arm]))
        def load_arm(self, path, signature, step, arm):
            assert json.loads(path.read_bytes()) == [signature, step, arm]
            self.index = step+int(self.drift); seen.append(("load", arm, step))
        def release(self): seen.append("release")
    monkeypatch.setattr(p.fit, "checked_image", lambda *a: Image.new("RGB", (256, 256)))
    args = (tmp_path/"run", archive, formal, varying, identity, records, references, tmp_path/"handoff.zip")
    return args, Runtime, seen


def test_noise_only_policy_and_common_evaluation():
    assert p.ARMS == ["std030", "std060"]
    assert p.STRENGTHS == {"std015": .015, "std030": .03, "std060": .06}
    for key in ("updates", "input_tiles", "effective_batch", "initialization", "trainable", "backbone_frozen",
                "adaptor_lr", "adaptor_weight_decay", "segdec_lr", "segdec_weight_decay", "scheduler", "amp",
                "evaluation_seeds", "heldout_seeds", "training_seed_start"):
        assert p.POLICY[key] == p.v.POLICY[key]
    assert p.POLICY["primary_evaluation_strength"] == "std015"
    assert p.POLICY["resource_plan"]["comparison_seconds"] == 2700


def test_amplitude_restored_even_on_failure():
    generator = SimpleNamespace(noise_mean=0, noise_std=.015, threshold=.6)
    with pytest.raises(RuntimeError):
        with noise_level(generator, .06):
            assert generator.noise_std == .06
            raise RuntimeError("failed trial")
    assert generator.noise_std == .015
    for invalid in (0, .1, float("nan")):
        with pytest.raises(ValueError):
            with noise_level(generator, invalid): pass
    generator.threshold = .5
    with pytest.raises(ValueError):
        with noise_level(generator, .03): pass


def test_both_arms_cross_strengths_reload_export_and_noop_resume(tmp_path, monkeypatch):
    args, runtime, seen = setup_run(tmp_path, monkeypatch)
    result = runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)
    assert result["attempted_updates_including_replayed_work"] == 6
    assert result["test_images_decoded"] == 0 and not result["automatic_model_selection"]
    assert seen.count("init") == 3 and seen.count("baseline") == 1
    for arm in p.ARMS:
        assert result["arms"][arm]["completed_updates"] == 3
        assert result["arms"][arm]["reload_max_abs_difference"] == 0
        assert set(result["arms"][arm]["trajectory"][-1]["heldout_by_evaluation_strength"]) == set(p.STRENGTHS)
        assert (arm, 1) in seen
    assert runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor) == result
    assert seen.count("init") == 3
    with zipfile.ZipFile(args[-1]) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert len(z.namelist()) == len(manifest)+1
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
        assert not any(n.endswith((".pt", ".png", ".jpg")) for n in z.namelist())
    assert (args[2]/"step-028800/model.pt").read_bytes() == b"original"
    assert (args[3]/"step-0500/model.pt").read_bytes() == b"baseline"


def test_interrupt_second_arm_keeps_first_and_replays_same_schedule(tmp_path, monkeypatch):
    args, runtime, seen = setup_run(tmp_path, monkeypatch)
    runtime.interrupt_at = ("std060", 3)
    with pytest.raises(KeyboardInterrupt):
        runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)
    runtime.interrupt_at = None
    result = runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)
    assert seen.count(("std030", 1)) == 1 and seen.count("baseline") == 1
    assert seen.count(("std060", 2)) == 2
    assert ("load", "std060", 1) in seen
    assert result["attempted_updates_including_replayed_work"] == 8


@pytest.mark.parametrize("case", ["source", "targets", "reload", "resource", "budget", "hard-stop", "replay"])
def test_fail_closed(tmp_path, monkeypatch, case):
    args, runtime, seen = setup_run(tmp_path, monkeypatch)
    if case == "source": (args[3]/"step-0500/model.pt").write_bytes(b"changed")
    if case == "targets": runtime.bad_targets = True
    if case == "reload": runtime.drift = True
    if case == "resource": runtime.fail = True
    if case == "budget": monkeypatch.setitem(p.POLICY, "maximum_attempted_updates", 1)
    if case == "replay": args[6]["baseline"]["s42__eval_mixed_maps"] += 1
    if case == "hard-stop":
        args[0].mkdir(); runner.publish(args[0], "identity", p.fit.digest(args[4]), args[4])
        (args[0]/"attempt-incomplete").mkdir()
    with pytest.raises((ValueError, RuntimeError)):
        runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)
    assert not args[-1].exists()
    if case == "resource":
        with pytest.raises(ValueError, match="review"):
            runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)


def test_completed_output_detects_tampered_evidence(tmp_path, monkeypatch):
    args, runtime, _ = setup_run(tmp_path, monkeypatch)
    runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)
    (args[0]/"std060-step-0003/arrays.npz").write_bytes(b"tamper")
    with pytest.raises(ValueError):
        runner.execute(*args, runtime_factory=runtime, monitor_factory=Monitor)


def test_baseline_forbids_updates_without_torch():
    from scripts.mpi_supersimplenet_noise_runtime import Runtime
    control = object.__new__(Runtime)
    with pytest.raises(ValueError, match="Baseline"):
        control.step_at_strength([], 0, "std015")


def test_wrong_arm_checkpoint_rejected_before_gpu_operations():
    from scripts.mpi_supersimplenet_noise_runtime import Runtime
    runtime = object.__new__(Runtime)
    payload = {"purpose": p.NAME, "identity": "same", "step": 500, "arm": "std030", "training_noise_std": .03}
    runtime.torch = SimpleNamespace(load=lambda *a, **kw: payload)
    with pytest.raises(ValueError, match="identity"):
        runtime.load_arm("unused.pt", "same", 500, "std060")


def test_clean_counterfactual_catches_score_memorization():
    entries = {str(s): p.summarize(arrays(seed=s, memorize=True)) for s in p.v.POLICY["evaluation_seeds"]}
    report = p.aggregate(entries)
    assert report["eval_classification_auroc"]["mean_over_defined_seeds"] == 1
    assert report["clean_scores_with_synthetic_labels_auroc"]["mean_over_defined_seeds"] == 1
    assert report["same_source_classification_auroc"]["mean_over_defined_seeds"] == .5


def test_cli_no_torch_import_required():
    command = subprocess.run([sys.executable, str(Path(runner.__file__)), "--help"], capture_output=True, text=True)
    assert command.returncode == 0 and "--archive" in command.stdout
    command = subprocess.run([sys.executable, "-c", "from scripts import run_mpi_supersimplenet_noise; import sys; assert 'torch' not in sys.modules"], capture_output=True, text=True)
    assert command.returncode == 0, command.stderr


def test_prepare_checks_reviewed_baseline_and_pins_new_sources(tmp_path, monkeypatch):
    handoffs = tmp_path/"handoffs"; handoffs.mkdir()
    root = tmp_path/"varying"; root.mkdir()
    prior = {"name": p.v.NAME, "sources_lf_sha256": {}, "fit_signature": "formal", "job": {}}
    signature = p.fit.digest(prior)
    runner.publish(root, "identity", signature, prior)
    flat = {f"s{s}__{n}": a for s in p.v.POLICY["evaluation_seeds"] for n, a in arrays(500, s).items()}
    evaluations = {str(s): {**p.v.summarize(arrays(500, s)), "checks": {"state_unchanged": True,
        "state_before_sha256": "same", "state_after_sha256": "same"}} for s in p.v.POLICY["evaluation_seeds"]}
    losses = [{"step": i+1, "seed": p.v.training_seed(i), "loss": .5, "target_sha256": str(i)} for i in range(500)]
    with p.fit.phase_stage(root, "step-0500") as stage:
        (stage/"model.pt").write_bytes(b"baseline")
        np.savez_compressed(stage/"arrays.npz", **flat)
        p.fit.write_json(stage/"progress.json", {"step": 500, "evaluations": evaluations, "losses": losses, "fingerprints": {"head": "fixture"}})
        receipt = p.fit.seal_phase(stage, "step-0500", signature, {}, p.v.FILES)
    payload = {"plan.json": p.fit.canonical(prior)}
    for name in ("receipt.json", "arrays.npz", "progress.json"):
        payload["step-0500/"+name] = (root/"step-0500"/name).read_bytes()
    payload["step-0000/arrays.npz"] = payload["step-0500/arrays.npz"]
    payload["handoff-manifest.json"] = p.fit.canonical({n: hashlib.sha256(b).hexdigest() for n, b in payload.items()})
    path = handoffs/(p.v.NAME+".zip")
    with zipfile.ZipFile(path, "w") as z:
        for n, b in payload.items(): z.writestr(n, b)
    monkeypatch.setattr(p, "BASELINE_ZIP_SHA", p.fit.sha256_file(path))
    monkeypatch.setattr(p.v, "prepare", lambda *a: ({}, prior, {}))
    _, identity, references = p.prepare(runner.ROOT, handoffs, tmp_path/"formal", tmp_path/"fixed", root)
    assert identity["baseline_checkpoint_sha256"] == receipt["files"]["model.pt"]
    assert len(identity["baseline_training_targets"]) == 500
    assert len(identity["sources_lf_sha256"]) == 5
    assert set(references) == {"baseline", "original"}
    (root/"step-0500/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError):
        p.prepare(runner.ROOT, handoffs, tmp_path/"formal", tmp_path/"fixed", root)


def test_main_paths(tmp_path, monkeypatch):
    archive = tmp_path/"archive.zip"; archive.write_bytes(b"fixture")
    formal = tmp_path/"artifacts"/p.v.q.p.NAME; formal.mkdir(parents=True)
    fixed = tmp_path/"artifacts/analyses"/p.v.old.NAME; fixed.mkdir(parents=True)
    varying = tmp_path/"artifacts/analyses"/p.v.NAME; varying.mkdir(parents=True)
    monkeypatch.setattr(p.fit, "EXPECTED_ARCHIVE_SHA256", p.fit.sha256_file(archive))
    env = {k: "pinned" for k in ("torch", "torchvision", "anomalib", "timm", "supersimplenet_sources")}
    def prepare(project, handoffs, fit_root, fixed_root, varying_root):
        assert (fit_root, fixed_root, varying_root) == (formal, fixed, varying)
        return {}, {"environment": env.copy()}, {}
    monkeypatch.setattr(p, "prepare", prepare)
    monkeypatch.setattr(runner, "environment", lambda: env.copy())
    monkeypatch.setattr(runner.shutil, "disk_usage", lambda *a: SimpleNamespace(free=20*1024**3))
    seen = []
    monkeypatch.setattr(runner, "execute", lambda *a: seen.append(a) or {"fixture": True})
    assert runner.main(["--project", str(tmp_path), "--archive", str(archive)]) == 0
    assert seen[0][0] == tmp_path/"artifacts/analyses"/p.NAME
    assert seen[0][-1] == tmp_path/"artifacts/handoff"/(p.NAME+".zip")
