import copy
import numpy as np
import pytest
from scripts import analyze_mpi_normal_influence as a
from scripts import run_mpi_scale_shrinkage as p


def inputs():
    dev = np.tile(np.arange(10.)[:, None], (1, 12))
    arrays = {"development": dev, "calibration": dev, "test": dev, "validation": dev}
    raw = {g:[{"key":f"{g}/{i}", "label":0} for i in range(10)] for g in arrays}
    record = p.trial(arrays, np.zeros(10), np.arange(10), np.arange(10))
    return raw, arrays, record


def test_boundaries_keep_tied_regions_and_unique_keys():
    raw, arrays, record = inputs()
    trace = a.attributions(arrays, raw, record, "control")
    assert trace["calibration_boundary"] == [{"key":"calibration/9", "multiplicity":1, "regions":p.REGIONS}]
    assert trace["development_q90_support"][p.REGIONS[0]] == ["development/9"]
    assert trace["raw_region_boundaries"] == [9.]*12


def test_duplicated_sample_is_not_double_counted_as_boundary_photo():
    raw, arrays, record = inputs()
    indices = np.array([0,1,2,3,4,5,6,7,9,9])
    record = p.trial(arrays, np.zeros(10), np.arange(10), indices)
    trace = a.attributions(arrays, raw, record, "candidate")
    assert len(trace["calibration_boundary"]) == 1
    assert trace["calibration_boundary"][0]["multiplicity"] == 2


def test_group_cutoffs_empty_groups_and_constant_correlation():
    masks = a.group_mask([49,50,52,53,55])
    assert {k:int(v.sum()) for k,v in masks.items()} == {"all":5,"low":1,"middle":2,"retained":2}
    assert a.corr([1,1],[3,4]) is None
    assert a.corr([34.10068746601662]*1000, np.arange(1000)) is None
    assert a.corr([0,1,2],[2,1,0]) == pytest.approx(-1)
    assert a.describe([]) is None
    raw, arrays, record = inputs()
    trace = a.attributions(arrays,raw,record,"control")
    s = a.summarize([trace]*2, [record]*2, raw)
    assert s["groups"]["retained"]["trials"] == 0
    assert s["groups"]["retained"]["tp"] is None
    assert all(v["correlation_multiplicity_with_tp"] is None for v in s["photo_associations"]["development"])


def test_multiplicity_associations_include_all_photos():
    rows = a.associations([[0,0,1],[0,1,1],[1,1,1]], ["a","b","c"], [1,2,3])
    assert len(rows) == 3
    assert rows[0]["correlation_multiplicity_with_tp"] == pytest.approx(-1)
    assert rows[1]["correlation_multiplicity_with_tp"] == pytest.approx(1)
    assert rows[2]["correlation_multiplicity_with_tp"] is None


def test_quantile_rank_and_neighborhood_are_normal_only():
    raw, arrays, record = inputs()
    before = copy.deepcopy(raw)
    result = a.normal_order_neighborhood(raw, arrays)
    assert all(v["q90_index_zero_based"] == 9 and v["q90_value"] == 9 for v in result.values())
    assert result[p.REGIONS[0]]["near_q90"][-1]["gap_from_previous"] == 1
    assert raw == before
