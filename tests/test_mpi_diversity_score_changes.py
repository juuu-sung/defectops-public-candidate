import copy
import math
import time

import numpy as np
import pytest

from scripts import analyze_mpi_diversity_score_changes as p


def method(center=1., scale=2., threshold=3.):
    return {"field": "top1", "scale_kind": "q90", "threshold": threshold,
            "regions": [{"region": r, "center": center, "scale": scale} for r in p.old.REGIONS]}


def test_identical_endpoints_zero_and_boundary_alert():
    r = p.decompose(np.full(12, 7.), np.full(12, 7.), method(), method())
    assert r["before_margin"] == r["after_margin"] == 0.
    assert r["contributions"] == dict.fromkeys(p.FACTORS, 0.)
    assert r["mixed_state_ties"] == [p.old.REGIONS] * 16
    assert r["before_winning_region"] == p.old.REGIONS[0]


@pytest.mark.parametrize("factor,expected", [("raw", 1.), ("center", -.5), ("scale", -1.5), ("threshold", -1.)])
def test_single_factor_and_dummy_property(factor, expected):
    a, b = np.full(12, 7.), np.full(12, 7.)
    before, after = method(), method()
    if factor == "raw": b += 2
    if factor == "center": after = method(center=2)
    if factor == "scale": after = method(scale=4)
    if factor == "threshold": after = method(threshold=4)
    r = p.decompose(a, b, before, after)
    assert r["contributions"][factor] == expected
    assert all(v == 0 for k, v in r["contributions"].items() if k != factor)


def test_region_switch_interactions_and_independent_subset_formula():
    before, after = method(), method(center=2., scale=3., threshold=2.)
    before["regions"][0]["scale"] = 1.
    after["regions"][1]["scale"] = .5
    a, b = np.arange(12, dtype=float), np.arange(12, dtype=float)[::-1]
    r = p.decompose(a, b, before, after)
    assert r["endpoint_winning_region_changed"]
    values = r["mixed_state_margins_not_valid_models"]
    for i, name in enumerate(p.FACTORS):
        answer = 0.
        for mask in range(16):
            if mask & (1 << i): continue
            size = mask.bit_count()
            weight = math.factorial(size) * math.factorial(3-size) / math.factorial(4)
            answer += weight * (values[mask | (1 << i)]-values[mask])
        assert answer == pytest.approx(r["contributions"][name], abs=1e-12)
    assert sum(r["contributions"].values()) == pytest.approx(r["margin_change"], abs=1e-12)
    reverse = p.decompose(b, a, after, before)
    for k in p.FACTORS:
        assert reverse["contributions"][k] == pytest.approx(-r["contributions"][k], abs=1e-12)


@pytest.mark.parametrize("problem", ["nan", "inf", "zero", "negative", "tiny", "order", "shape", "threshold"])
def test_invalid_arithmetic_rejected(problem):
    a, b, m = np.ones(12), np.ones(12), method()
    if problem == "nan": a[0] = np.nan
    if problem == "inf": b[0] = np.inf
    if problem in {"zero", "negative", "tiny"}:
        m["regions"][0]["scale"] = {"zero": 0, "negative": -1, "tiny": 1e-7}[problem]
    if problem == "order": m["regions"].reverse()
    if problem == "shape": a = np.ones(11)
    if problem == "threshold": m["threshold"] = float("nan")
    with pytest.raises(ValueError): p.decompose(a, b, method(), m)


def synthetic_raw():
    raw = {}
    for group, count in p.evaluation.COUNTS.items():
        raw[group] = []
        for i in range(count):
            pid = str({"development": 0, "calibration": 128, "test": 500, "validation": 800}[group]+i)
            label = int(group == "test" and i >= 54)
            prefix = ("defect" if label else "good") if group == "test" else group
            raw[group].append({"key": prefix+"/"+pid, "photo_id": pid, "label": label,
                "regions": [{"region": r, "member": prefix+"/"+r+"/"+pid, "top1": float(i+j)}
                            for j, r in enumerate(p.old.REGIONS)]})
    return raw


@pytest.mark.parametrize("problem", ["order", "member", "label", "missing", "duplicate"])
def test_paired_inventory_rejected(problem):
    a = synthetic_raw()
    b = copy.deepcopy(a)
    if problem == "order": b["test"].reverse()
    if problem == "member": b["test"][0]["regions"][0]["member"] = "wrong"
    if problem == "label": b["test"][0]["label"] = 1
    if problem == "missing": b["calibration"].pop()
    if problem == "duplicate": b["validation"][1] = b["validation"][0]
    with pytest.raises(ValueError): p.paired_validate(a, b)


def test_group_empty_no_division_and_no_unobserved_claim():
    result = p.summarize([])
    assert result["lost_defects"]["count"] == 0
    assert result["gained_defects"]["mean_margin_change"] is None


def test_budgets_stop(monkeypatch):
    monkeypatch.setattr(p.old, "peak_rss_bytes", lambda: 2 * 1024**3 + 1)
    with pytest.raises(RuntimeError): p.budget(time.monotonic())
    monkeypatch.setattr(p.old, "peak_rss_bytes", lambda: 1)
    with pytest.raises(RuntimeError): p.budget(time.monotonic()-1801)


def test_input_pin_and_output_protection(tmp_path):
    bad = tmp_path/"bad.zip"
    bad.write_bytes(b"not reviewed input")
    with pytest.raises(ValueError, match="identity"): p.read_candidate(bad)
    out = tmp_path/"existing.json"
    out.write_text("preserve")
    with pytest.raises(FileExistsError):
        p.main(["--baseline-zip", str(bad), "--evaluation-zip", str(bad), "--validation-zip", str(bad), "--output", str(out)])
    assert out.read_text() == "preserve"


def test_original_baseline_provenance_rejected(tmp_path):
    bad = tmp_path/"bad.zip"
    bad.write_bytes(b"unverified baseline")
    with pytest.raises(ValueError, match="validation ZIP"):
        p.read_original_baseline(bad, {}, {})
