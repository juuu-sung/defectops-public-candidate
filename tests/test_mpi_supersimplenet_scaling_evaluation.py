"""CPU control-flow and receipt tests; not a CUDA or accuracy claim."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import mpi_supersimplenet_scaling_evaluation_protocol as q
from scripts import run_mpi_supersimplenet_scaling_evaluation as r
from scripts.mpi_supersimplenet_scaling_evaluation_runtime import Monitor
from tests.test_mpi_supersimplenet_evaluation import contract, photo, Monitor as FakeMonitor, Runtime as FakeRuntime


def test_policy_never_accuracy_or_elapsed_gate_and_does_not_mutate_old():
    assert q.POLICY["accuracy_stop_gate"] is False
    assert q.POLICY["resource_plan"]["comparison_seconds"] is None
    assert q.POLICY["acceptance"]["warm_photo_p95_seconds"] is None
    assert q.POLICY["acceptance"]["later_evaluation_compute_cap_seconds"] is None
    assert "descriptive_only" in q.POLICY["acceptance"]["first_gate"]
    assert q.old.POLICY["resource_plan"]["comparison_seconds"] == 14400
    assert q.old.POLICY["acceptance"]["warm_photo_p95_seconds"] == 2
    assert q.POLICY["resource_plan"]["rss_GiB"] == 20
    assert q.POLICY["resource_plan"]["selection_and_inference_reserved_GiB"] == 6


def test_no_time_limit_memory_limit_still_enforced():
    m = SimpleNamespace(sample=lambda: None, runtime=None, error=None, rss=1, reserved=0,
                        prior=10**20, limits=q.POLICY["resource_plan"])
    Monitor.check(m)
    m.rss = 21*1024**3
    with pytest.raises(RuntimeError, match="memory limit"): Monitor.check(m)
    m.rss, m.reserved = 0, 7*1024**3
    with pytest.raises(RuntimeError, match="memory limit"): Monitor.check(m)
    m.reserved, m.error = 0, "sampler stopped"
    with pytest.raises(RuntimeError, match="monitor failed"): Monitor.check(m)


def test_invalid_resamples_kept_with_conditional_quantiles():
    good = {"valid": True, "counts": {"tp": 20, "fp54": 3, "fp256": 14}}
    trials = [{"baseline": good, "candidate": {"valid": False, "error": "Degenerate"}},
              {"baseline": good, "candidate": good}]
    summary = q.summarize(trials)
    assert summary["candidate"]["invalid"] == 1
    assert summary["candidate"]["quantiles_conditional_on_valid_draws"]["tp"] == [20, 20, 20]
    assert q.summarize(trials[:1])["candidate"]["quantiles_conditional_on_valid_draws"] is None


def test_pinned_bundle_payload_integrity(tmp_path, monkeypatch):
    path = tmp_path/"fit.zip"
    raw = b'{"foo": 1}'
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("plan.json", raw)
        z.writestr("handoff-manifest.json", json.dumps({"plan.json": hashlib.sha256(raw).hexdigest()}))
    monkeypatch.setattr(q, "FIT_SHA", q.fit.sha256_file(path))
    assert q.read_bundle(path) == {"plan.json": {"foo": 1}}
    monkeypatch.setattr(q, "FIT_SHA", "changed")
    with pytest.raises(ValueError, match="reviewed"): q.read_bundle(path)


@pytest.fixture
def run_fixture(tmp_path, monkeypatch):
    tests = [photo(f"defect/{i:04d}") for i in range(55)] + [photo(f"good/{i:04d}", 0) for i in range(54)]
    validation = [photo(f"validation/{i:04d}", 0) for i in range(256)]
    source = {"cohorts": {"test": tests, "validation": validation}, "previous": {}}
    identity = {"arm": "a64-e300", "contract": contract(), "fit_signature": "fit",
                "checkpoint_sha256": "checkpoint"}
    fit_root = tmp_path/"fit"
    fit_root.mkdir()
    archive = tmp_path/"source.zip"
    with zipfile.ZipFile(archive, "w"): pass
    accessed = []
    monkeypatch.setattr(r.engine, "normal_gate", lambda *a: {"passed": True})
    original_sha = q.fit.sha256_file
    monkeypatch.setattr(q.fit, "sha256_file", lambda f: "checkpoint" if str(f).endswith("model.pt") else original_sha(f))
    def score(archive, item, runtime, recovered, monitor, c):
        accessed.append(item["key"])
        # Small synthetic cache, not the production 256x256 tensor contract.
        values = np.zeros((12, 1), dtype=np.float32)
        return values, q.old.sample(item, values[:, 0], c) | {"photo_seconds": .1}
    monkeypatch.setattr(r.engine, "score_photo", score)
    def read(root, group, index, signature, item, c):
        folder = root/group/f"photo-{index:03d}"
        receipt = q.fit.read_phase(root/group, folder.name, signature, ["predictions.npy", "sample.json"])
        assert receipt["photo_sha256"] == q.fit.digest(item)
        if q.fit.sha256_file(root/receipt["gate_path"]) != receipt["gate_sha256"]:
            raise ValueError("gate changed")
        return json.loads((folder/"sample.json").read_bytes())
    monkeypatch.setattr(r.engine, "read_cache", read)
    monkeypatch.setattr(r.e, "compare", lambda samples, source: {"passed": False, "counts": r.e.metrics(samples["test"], samples["validation"])})
    monkeypatch.setattr(r.engine, "benchmark", lambda *a: accessed.append("WARM") or {"candidate_budget_passed": False, "p95_higher_seconds": 99})
    monkeypatch.setattr(r, "review_maps", lambda *a: accessed.append("MAPS") or {"cases": []})
    monkeypatch.setattr(r.engine, "stability", lambda *a: accessed.append("STABILITY") or {"passed": False, "trials": {}})
    args = [tmp_path, tmp_path/"output", archive, fit_root, tmp_path/"recovery",
            {"plan.json": {"policy": {"updates": 28800}}}, source, identity]
    return args, {"runtime_factory": FakeRuntime, "monitor_factory": FakeMonitor}, accessed


def test_zero_prior12_still_runs_all365_maps_warm_and_stability(run_fixture):
    args, kw, accessed = run_fixture
    report = r.execute_arm(*args, **kw)
    assert report["prior12_retained"] == 0
    assert report["counts"] == {"tp": 0, "fn": 55, "fp54": 0, "tn54": 54, "fp256": 0}
    assert report["full_comparison_run"] and report["stability_evaluated"] and report["warm_evaluated"]
    assert len(accessed) == 368 and accessed[-3:] == ["WARM", "MAPS", "STABILITY"]
    assert report["cached_photos"] == 365
    before = list(accessed)
    assert r.execute_arm(*args, **kw) == report and accessed == before


def test_ctrl_c_preserves_completed_photos_and_resumes(run_fixture, monkeypatch):
    args, kw, accessed = run_fixture
    original = r.engine.score_photo
    def interrupt(*a):
        if len(accessed) == 3: raise KeyboardInterrupt()
        return original(*a)
    monkeypatch.setattr(r.engine, "score_photo", interrupt)
    with pytest.raises(KeyboardInterrupt): r.execute_arm(*args, **kw)
    first = list(accessed)
    monkeypatch.setattr(r.engine, "score_photo", original)
    assert r.execute_arm(*args, **kw)["cached_photos"] == 365
    assert accessed[:3] == first and len(accessed) == 368
    attempts = r.attempts(args[1])
    assert [x["status"] for x in attempts].count("interrupted") == 1


def test_normal_replay_failure_opens_no_test_and_blocks_retry(run_fixture, monkeypatch):
    args, kw, accessed = run_fixture
    def fail(*a): raise ValueError("normal replay failed")
    monkeypatch.setattr(r.engine, "normal_gate", fail)
    with pytest.raises(ValueError, match="normal replay"): r.execute_arm(*args, **kw)
    assert not accessed
    with pytest.raises(ValueError, match="Prior failed"): r.execute_arm(*args, **kw)


@pytest.mark.parametrize("target", ["predictions.npy", "sample.json", "normal-gate.json"])
def test_corrupted_partial_cache_refused(run_fixture, monkeypatch, target):
    args, kw, accessed = run_fixture
    original = r.engine.score_photo
    def interrupt(*a):
        if len(accessed) == 1: raise KeyboardInterrupt()
        return original(*a)
    monkeypatch.setattr(r.engine, "score_photo", interrupt)
    with pytest.raises(KeyboardInterrupt): r.execute_arm(*args, **kw)
    path = (next(args[1].glob("attempt-*/normal-gate.json")) if target == "normal-gate.json"
            else args[1]/"test/photo-000"/target)
    path.write_bytes(b"corrupt")
    with pytest.raises(ValueError): r.execute_arm(*args, **kw)


def test_map_selection_all55_plus6_normals_not_just_misses():
    groups = {"test": [photo(f"defect/{i:04d}") for i in range(55)] + [photo(f"good/{i:04d}", 0) for i in range(54)],
              "validation": [photo(f"validation/{i:04d}", 0) for i in range(256)]}
    samples = {g: [q.old.sample(p, [float(i)]*12, contract()) for i, p in enumerate(v)] for g, v in groups.items()}
    entries = r.map_entries({"cohorts": groups}, samples)
    assert len(entries) == 61 and sum(x["card"] for x in entries) == 18
    assert entries[-1]["photo"]["key"] == "validation/0253"
    assert any(x["photo"]["key"] == "defect/0006" for x in entries)


def test_received_scaling_bundle_contract_if_available():
    path = Path.home() / "Downloads" / "mpi-supersimplenet-scaling-v1.zip"
    if not path.exists(): pytest.skip("Received evidence is not in source control")
    data = q.read_bundle(path)
    for arm in r.ARMS:
        assert data[arm+"/plan.json"] == q.scaling.arm_identity(data["plan.json"], arm)
        assert data[arm+"/reload.json"]["passed"]
        normal = data[arm+"/normal-results.json"]
        assert q.p.calibrate(**normal["tile_scores"]) == normal["contract"]


def test_saved_map_review_creates_directory_cards_and_rechecks_cache(tmp_path, monkeypatch):
    from PIL import Image
    root = tmp_path/"out"
    root.mkdir()
    bundle = {"input-lock.json": {"identity": {}}, "evaluation.json": {"execution_sha256": q.fit.digest({})}}
    monkeypatch.setattr(r.ev, "checked_bundle", lambda *a: bundle)
    monkeypatch.setattr(r.maps, "baseline_record", lambda *a: {"available": False})
    monkeypatch.setattr(r.maps, "load_baseline", lambda *a: None)
    entries = []
    samples = {"test": [], "validation": []}
    for group, key, label in (("test", "defect/0006", 1), ("validation", "validation/0001", 0)):
        item = photo(key, label)
        item["regions"] = [{"member": "normal"} for _ in range(12)]
        entries.append({"group": group, "index": 0, "role": "fixture", "photo": item, "card": True})
        directory = root/group/"photo-000"
        directory.mkdir(parents=True)
        raw = np.zeros((12, 65537), dtype=np.float32)
        np.save(directory/"predictions.npy", raw)
        samples[group].append(q.old.sample(item, raw[:, 0], contract()))
    monkeypatch.setattr(r, "map_entries", lambda *a: entries)
    monkeypatch.setattr(r.engine, "read_cache", lambda root, group, *a: samples[group][0])
    monkeypatch.setattr(r.ev, "read_masks", lambda *a: [None]*12)
    monkeypatch.setattr(r.ev, "read_tile", lambda *a: Image.new("RGB", (256, 256)))
    monkeypatch.setattr(r.fit, "checked_image", lambda *a: Image.new("RGB", (256, 256)))
    args = [tmp_path, root, "sig", None, {"previous": {}}, samples, contract(), None, FakeMonitor()]
    result = r.review_maps(*args)
    assert result["cards"] == 4 and len(list((root/"maps").glob("*/*.png"))) == 4
    assert all(not c["location_eligible"] for c in result["cases"])
    assert r.review_maps(*args) == result
    (root/"maps/case-000/card-1.png").write_bytes(b"changed")
    with pytest.raises(ValueError): r.review_maps(*args)


def test_three_arm_export_is_complete_idempotent_and_refuses_tamper(run_fixture, monkeypatch):
    args, kw, accessed = run_fixture
    root = args[0]/"suite"
    identity = {"cohorts": args[6]["cohorts"], "arms": {arm: {k: args[7][k] for k in
                ("fit_signature", "checkpoint_sha256", "contract")} for arm in r.ARMS}}
    for arm in r.ARMS:
        args[1], args[7] = root/arm, r.derived_identity(identity, arm)
        r.execute_arm(*args, **kw)
    with q.fit.locked_run(root/"q90"):
        r.publish(root/"q90", "result", q.fit.digest(identity), {"scope": "fixture"})
    handoff = args[0]/"handoff.zip"
    r.export(root, handoff, identity)
    with zipfile.ZipFile(handoff) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert len(z.namelist()) == len(manifest)+1
        assert all(hashlib.sha256(z.read(k)).hexdigest() == v for k, v in manifest.items())
        for arm in r.ARMS:
            samples = json.loads(z.read(arm+"/samples.json"))
            assert len(samples["test"]) == 109 and len(samples["validation"]) == 256
    before = q.fit.sha256_file(handoff)
    r.export(root, handoff, identity)
    assert q.fit.sha256_file(handoff) == before
    handoff.write_bytes(b"changed")
    with pytest.raises(zipfile.BadZipFile): r.export(root, handoff, identity)
