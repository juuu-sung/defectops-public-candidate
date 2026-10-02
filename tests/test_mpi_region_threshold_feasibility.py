import itertools

import numpy as np
import pytest

from scripts.check_mpi_region_threshold_feasibility import feasible


def brute(t,a,b,ca,cb):
    for bound in itertools.product(*[list(np.unique(t[:,r]))+[np.inf] for r in range(t.shape[1])]):
        alert=lambda x:np.any(x>=bound,axis=1)
        if alert(t).all() and sum(alert(a))<=ca and sum(alert(b))<=cb:return True
    return False


@pytest.mark.parametrize('seed',range(20))
def test_search_matches_independent_brute_force(seed):
    rng=np.random.default_rng(seed)
    t=rng.integers(0,6,(4,3));a=rng.integers(0,6,(5,3));b=rng.integers(0,6,(6,3))
    expected=brute(t,a,b,1,2)
    r=feasible(t,a,b,1,2)
    assert (r['status']=='feasible_label_aware_witness')==expected
    assert r['status']!='unknown_resource_limit'
    assert not r['deployable'] and not r['performance_evaluation']


def test_ties_cannot_be_separated():
    r=feasible([[2,2]],[[2,2]],[[0,0]],0,0)
    assert r['status']=='infeasible_exhaustive'


def test_overlapping_normal_alerts_count_once():
    r=feasible([[2,0],[0,2]],[[2,2]],[[0,0]],1,0)
    assert r['status']=='feasible_label_aware_witness'
    assert r['witness_not_model']['fp_a']==1


@pytest.mark.parametrize('kwargs',[{'seconds':0},{'max_nodes':0},{'max_rss':0}])
def test_limit_means_unknown_not_infeasible(kwargs):
    assert feasible([[2]],[[0]],[[0]],**kwargs)['status']=='unknown_resource_limit'


def test_empty_target_and_invalid_inputs():
    assert feasible(np.zeros((0,2)),np.ones((2,2)),np.ones((2,2)))['status']=='feasible_label_aware_witness'
    for t,a,b in [([[np.nan]],[[0]],[[0]]),([[1]],[[0,1]],[[0]])]:
        with pytest.raises(ValueError):feasible(t,a,b)
