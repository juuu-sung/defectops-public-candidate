"""CPU synthetic boundary tests; never evidence of actual GPU provenance coverage."""

import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

from scripts import trace_mpi_patches as trace
from scripts import mpi_patch_trace_protocol as p
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime
from tests.test_mpi_pcb_normal_bank_fit import png_row


def test_case_selection_uses_fixed_scores_controls_and_non_GT_winner():
    def sample(key, label, value):
        return {"key": key, "photo_id": key.split('/')[1], "label": label, "score": float(value), "threshold": 10.,
                "regions": [{"region": r, "top1": float(value), "native_raw": float(value)} for r in p.fit.REGIONS]}
    photos = [sample(f"defect/{i:04d}", 1, value) for i,value in enumerate([1,2,3,4,5,11,12])]
    photos += [sample(f"good/{i:04d}",0,value) for i,value in enumerate([1,2,3,11,12])]
    normals = [sample(f"calibration/{i:04d}",0,value) for i,value in enumerate([1,5,10])]
    contract = {"normalization": [{"region": r, "median": 0., "mad": 1.} for r in p.fit.REGIONS], "thresholds": {p.A: 10.}}
    old = [dict(q, score=11. if q["key"]=='defect/0006' else 0.) for q in photos]
    b = {"calibration.json": contract, "scores.json": {p.A: photos}, "baseline-scores.json": old,
         "raw-scores.json": {"test": photos, "calibration": normals}}
    locations = [{"key": q["key"], "models": {p.ev.MODEL: {"regions": [
        {"gt": {"pixels": 0 if q['key']=='defect/0001' else 1}} for _ in p.fit.REGIONS]}}} for q in photos if q['label']]
    selected = p.select_cases(b, locations)
    assert selected['defect/0000']==['lowest_A_miss']
    assert selected['defect/0002']==['middle_A_miss']
    assert selected['defect/0004']==['nearest_cutoff_A_miss']
    assert selected['defect/0001']==['A_miss_winning_region_without_GT']
    assert selected['defect/0005']==['lowest_A_new_TP_control']
    assert selected['defect/0006']==['retained_old_TP_control']
    assert selected['calibration/0002']==['A_normal_calibration_cutoff']
    assert selected['good/0001']==['A_middle_test_normal_control']
    b['scores.json'][p.A]=list(reversed(photos))
    assert p.select_cases(b,locations)==selected


def test_query_cells_keep_tiny_GT_stable_ties_and_distinguish_control():
    mask = np.zeros((256, 256), dtype=bool)
    mask[5, 9] = True
    values = np.arange(4096, dtype=np.float32)
    queries = p.select_query_cells(values, mask)
    gt = next(q for q in queries if "GT_distance_minimum" in q["roles"])
    assert (gt["grid_y"], gt["grid_x"]) == (1, 2)
    assert len(gt["roles"]) == 3  # one tiny cell; all three GT roles deduplicated
    assert len(queries) == 3
    assert all(not any(s.startswith("GT_") for s in q["roles"]) for q in p.select_query_cells(values, None))
    assert p.select_query_cells(np.ones(4096, dtype=np.float32), mask)[0]["flat_index"] == 4095
    for bad in (values[:-1], np.full(4096, np.nan), -values):
        with pytest.raises(ValueError, match="distances"):
            p.select_query_cells(bad, mask)


def test_exact_vector_provenance_no_nearby_fallback_all_aliases_retained():
    features = np.array([[1, 2], [1, 2], [1.00001, 2]], dtype=np.float32)
    h = p.vector_hashes(features)[0]
    row = {"photo_id": "0001", "region": p.fit.REGIONS[0], "member": "normal", "member_sha256": "bytes", "decoded_rgb_sha256": "pixels"}
    origins = p.find_origins(features, [0, 2, 4], row, {h})
    assert [o["candidate_flat_index"] for o in origins] == [0, 2]
    cases = [{"key": "defect/0001", "tiles": [{"region": p.fit.REGIONS[1], "queries": [
        {"bank_index": 0, "bank_vector_sha256": h, "roles": ["GT_distance_minimum"]},
        {"bank_index": 1, "bank_vector_sha256": "missing", "roles": ["tile_distance_maximum"]}]}]}]
    result = p.summarize_origins({"0": h, "1": "missing"}, [{"origins": origins}], cases)
    assert result["resolved_bank_vectors"] == 1 and result["ambiguous_bank_vectors"] == 1
    assert result["unresolved_bank_indices"] == [1]
    assert result["cases"][0]["query_counts"] == {"cross_region_only": 1, "unresolved": 1}
    with pytest.raises(ValueError, match="Duplicate"):
        p.summarize_origins({"0": h}, [{"origins": origins*2}], [])
    with pytest.raises(ValueError, match="finite float32"):
        p.vector_hashes(np.array([[float("nan")]], dtype=np.float32))


def test_aliases_in_both_regions_report_mixed_not_cross_only():
    h = "hash"
    qs = [{"bank_index": 0, "bank_vector_sha256": h, "roles": []}]
    cases = [{"key": "x", "tiles": [{"region": "a", "queries": qs}]}]
    origins = [{"member": region, "photo_id": "0001", "candidate_flat_index": 0, "bank_vector_sha256": h, "region": region} for region in ("a", "b")]
    result = p.summarize_origins({"0": h}, [{"origins": origins}], cases)
    assert result["cases"][0]["query_counts"] == {"mixed_source_regions": 1}
    assert qs[0]["exact_source_candidates"] == 2


def test_bank_target_conflict_is_rejected():
    case = {"tiles": [{"queries": [{"bank_index": 1, "bank_vector_sha256": h} for h in ("a", "b")]}]}
    with pytest.raises(ValueError, match="inconsistent"):
        p.targets_from_cases([case])


class FakeTensor:
    def __init__(self, a):
        self.a, self.shape = a, a.shape
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.a


class FakeCore:
    def nearest_neighbors(self, embedding, n_neighbors):
        return FakeTensor(np.ones(len(embedding.a))), FakeTensor(np.zeros(len(embedding.a), dtype=int))


def test_native_capture_preserves_outputs_restores_on_error_and_skips_weighting_call():
    core, captured = FakeCore(), []
    embedding = FakeTensor(np.zeros((4096, 768), dtype=np.float32))
    with pytest.raises(RuntimeError), trace.capture_neighbors(core, captured):
        first = core.nearest_neighbors(embedding, 1)
        core.nearest_neighbors(FakeTensor(np.zeros((1, 768))), 9)
        assert len(captured) == 1 and np.array_equal(captured[0][0], first[0].a)
        raise RuntimeError("interrupt")
    assert "nearest_neighbors" not in vars(core)
    original = core.nearest_neighbors
    core.nearest_neighbors = original
    with trace.capture_neighbors(core, []):
        core.nearest_neighbors(embedding, 1)
    assert core.nearest_neighbors is original


class TraceFake(Runtime):
    def __init__(self, calls, feature_fail_after=None, missing_sources=False, bank_mutation=False, **kwargs):
        super().__init__(calls, **kwargs)
        self.feature_calls = 0
        self.feature_fail_after, self.missing_sources, self.bank_mutation = feature_fail_after, missing_sources, bank_mutation
        self.bank_reads = 0
    def bank_fingerprint(self):
        self.bank_reads += 1
        return {"shape": [3, 768], "float32_le_sha256": "changed" if self.bank_mutation and self.bank_reads > 1 else "synthetic"}
    def verify_feature_path(self, image):
        self.calls.append("verify_feature_path")
        return {"direct_replay_exact": True, "native_embedding_sha256": "synthetic"}
    def trace(self, image):
        raw, values = self.predict(image)
        self.calls.append("trace")
        return raw, values, np.arange(4096, dtype=np.float32), np.arange(4096) % 3, "synthetic"
    def bank_hashes(self, indices):
        return p.vector_hashes(np.array([[i+1.0]*768 for i in indices], dtype=np.float32))
    def candidate_features(self, image):
        self.feature_calls += 1
        self.calls.append("features")
        if self.feature_fail_after and self.feature_calls > self.feature_fail_after:
            raise RuntimeError("synthetic feature interruption")
        values = np.full((1024, 768), 4, dtype=np.float32)
        if not self.missing_sources:
            values[0] = 1
            values[1] = 2
        return values, p.fit.spatial_patch_indices(64, 2)


@pytest.fixture
def experiment(fixture):
    d = fixture
    d["config"]["feature_dimension"] = 768
    train_rows = []
    with zipfile.ZipFile(d["source_path"], "a") as z:
        for photo in ("0400", "0401"):
            for region in p.fit.REGIONS:
                raw, row = png_row(photo, region, 22)
                z.writestr(row["member"], raw)
                train_rows.append(row)
    d["preparation"].update(training_photo_ids=["0400", "0401"], training_rows=train_rows)
    normal = trace.probe.normal_photos(d["preparation"]["calibration_photo_ids"], d["preparation"]["calibration_rows"], "calibration")[1]
    photos = [normal, d["previous"]["test-audit.json"]["photos"][1], d["previous"]["test-audit.json"]["photos"][3]]
    expected = {}
    for photo, value in zip(photos, (11, 70, 70), strict=True):
        expected[photo["key"]] = {k: photo[k] for k in ("key", "photo_id", "label")} | {
            "regions": [{"region": r["region"], "member": r["member"], "top1": float(value), "native_raw": float(value)} for r in photo["regions"]]}
    contract = {"normalization": [{"region": r, "median": 0., "mad": 1.} for r in p.fit.REGIONS], "thresholds": {p.A: 60.}}
    args = {k: d[k] for k in ("root", "checkpoint", "config", "preparation", "records", "identity", "git", "handoff")}
    args.update(photos=photos, expected=expected, selection={p["key"]: ["synthetic_control"] for p in photos}, contract=contract, recovered=d["recovered_path"].read_bytes())
    return d["source_path"], args


def execute(experiment, calls, **options):
    source, args = experiment
    with zipfile.ZipFile(source) as z:
        return trace.execute(source=z, **args, runtime_factory=lambda config: TraceFake(calls, **options))


def test_full_diagnosis_cards_hashes_unresolved_sources_and_idempotent_rerun(experiment):
    calls = []
    r = execute(experiment, calls)
    assert r["status"] == "patch_trace_completed_with_unresolved_sources"
    assert r["case_count"] == 3 and r["training_candidates_checked"] == 24*1024
    assert r["maximum_absolute_score_replay_difference"] == 0
    assert r["ambiguous_bank_vectors"] > 0 and r["unresolved_bank_vectors"] > 0
    assert r["training_performed"] is r["threshold_changed"] is r["service_model_changed"] is r["automatic_pass_allowed"] is False
    assert calls.count("features") == 24 and calls.count("trace") == 36
    assert calls[-1] == "release"
    path = experiment[1]["handoff"]
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist()) == set(manifest) | {"handoff-manifest.json"}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n,h in manifest.items())
        assert len([n for n in z.namelist() if n.endswith('.jpg')]) == 9
        assert not any(n.endswith(('.pt','.npz')) for n in z.namelist())
        assert "출처 미확인" in z.read("index.html").decode()
        assert json.loads(z.read("A-calibration.json")) == experiment[1]["contract"]
    before, calls = path.read_bytes(), []
    assert execute(experiment, calls) == r
    assert calls == [] and path.read_bytes() == before


def test_unmatched_vectors_never_claim_verified_sources(experiment):
    r = execute(experiment, [], missing_sources=True)
    assert r["resolved_bank_vectors"] == 0
    assert r["unresolved_bank_vectors"] == r["selected_bank_vectors"]
    assert r["status"] == "patch_trace_completed_with_unresolved_sources"


def test_resume_source_scan_preserves_cases_and_first_complete_normal(experiment):
    calls = []
    with pytest.raises(RuntimeError, match="feature interruption"):
        execute(experiment, calls, feature_fail_after=12)
    assert not experiment[1]["handoff"].exists()
    saved = (experiment[1]["root"] / "cases/case-01/trace.json").read_bytes()
    calls = []
    r = execute(experiment, calls)
    assert calls.count("trace") == 0 and calls.count("features") == 12
    assert r["mixed_attempts"] and len(r["interrupted_attempts"]) == 1
    assert (experiment[1]["root"] / "cases/case-01/trace.json").read_bytes() == saved


@pytest.mark.parametrize("name", ["cases/case-01/trace.json", "sources/normal-01/origins.json", "result/diagnosis.json"])
def test_changed_completed_artifact_stops_before_gpu_and_preserves_file(experiment, name):
    execute(experiment, [])
    path = experiment[1]["root"] / name
    path.write_bytes(b"changed")
    calls = []
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute(experiment, calls)
    assert calls == [] and path.read_bytes() == b"changed"


def test_changed_identity_does_not_silently_reuse(experiment):
    execute(experiment, [])
    experiment[1]["identity"]["changed"] = True
    calls = []
    with pytest.raises(ValueError, match="incomplete or changed"):
        execute(experiment, calls)
    assert calls == []


def test_normal_probe_drift_stops_before_cases(experiment):
    calls = []
    with pytest.raises(ValueError, match="Normal replay gate failed"):
        execute(experiment, calls, offset=1)
    assert "trace" not in calls and "features" not in calls
    assert not experiment[1]["handoff"].exists()


def test_bank_mutation_blocks_export(experiment):
    with pytest.raises(ValueError, match="bank changed"):
        execute(experiment, [], bank_mutation=True)
    assert not experiment[1]["handoff"].exists()
    with pytest.raises(ValueError, match="Prior bank integrity"):
        execute(experiment, [])


def test_case_interruption_reuses_completed_case_but_rechecks_normal_gate(experiment):
    # Fake normal gate = 1 warmup + 24 tiles; first case = 12 tiles.
    with pytest.raises(RuntimeError, match="synthetic interruption"):
        execute(experiment, [], fail_after=37)
    calls = []
    report = execute(experiment, calls)
    assert calls.count("trace") == 24 and calls.count("features") == 24
    assert report["mixed_attempts"] and report["maximum_absolute_score_replay_difference"] == 0


def test_source_phase_targets_cannot_be_rebound_even_if_receipt_is_rehashed(experiment):
    execute(experiment, [])
    path = experiment[1]["root"] / "sources/normal-01/receipt.json"
    receipt = json.loads(path.read_bytes())
    receipt["targets_sha256"] = "different"
    receipt.pop("receipt_sha256")
    receipt["receipt_sha256"] = p.fit.digest(receipt)
    path.write_text(json.dumps(receipt))
    # Completed result should also validate the source->query connection.
    with pytest.raises(ValueError, match="targets differ"):
        execute(experiment, [])


def test_A_decision_flip_even_within_tolerance_is_blocked(experiment):
    expected = next(iter(experiment[1]["expected"].values()))
    actual = copy.deepcopy(expected)
    contract = copy.deepcopy(experiment[1]["contract"])
    contract["thresholds"][p.A] = 11.
    for row in actual["regions"]:
        row["top1"] -= 1e-6
    with pytest.raises(ValueError, match="A decision changed"):
        p.check_replay(actual, expected, contract)


def test_cli_and_powershell_entrypoint_use_real_existing_environment():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, str(root/'scripts/trace_mpi_patches.py'), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--archive' in result.stdout
    ps = (root/'scripts/run_mpi_patch_trace.ps1').read_text()
    assert '.venv\\Scripts\\python.exe' in ps and '$LASTEXITCODE' in ps
    assert 'Remove-Item' not in ps and 'pip install' not in ps
