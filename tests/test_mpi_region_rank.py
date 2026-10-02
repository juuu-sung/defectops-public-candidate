import numpy as np
import pytest
from scripts import run_mpi_region_rank as r


def test_pvalue_rule_exactly_strict_max_for_297_normals():
    rng=np.random.default_rng(73);normal=rng.normal(size=(297,12))
    method=r.fit_rank(normal[:128],normal[128:]);bounds=np.array(method['strict_upper_bounds'])
    samples=np.concatenate([normal,bounds[None,:],np.nextafter(bounds,np.inf)[None,:],rng.normal(size=(200,12))*3])
    for x in samples:
        tails=np.sum(normal>=x,axis=0)
        exact=np.any(240*(1+tails)<=298)
        assert exact==bool(np.any(x>bounds))
    assert not (bounds>bounds).any()


def test_all_regions_active_same_rule_and_no_labels():
    dev=np.ones((128,12));cal=np.ones((169,12))*2
    m=r.fit_rank(dev,cal)
    x=np.ones((12,12));np.fill_diagonal(x,3)
    a={'test':x,'validation':np.ones((2,12))}
    assert r.counts(a,np.ones(12,dtype=int),m)['tp']==12
    assert r.counts(a,np.zeros(12,dtype=int),m)['fp']==12
    assert r.fit_rank(dev[:,::-1],cal[:,::-1])['strict_upper_bounds']==m['strict_upper_bounds'][::-1]


def test_invalid_normal_arrays():
    for d,c in [(np.ones((127,12)),np.ones((169,12))),(np.full((128,12),np.nan),np.ones((169,12)))]:
        with pytest.raises(ValueError):r.fit_rank(d,c)


def test_minimum_calibration_resolution():
    assert 1/170>0.05/12
    assert 1/240==0.05/12
    assert [k for k in range(298) if 240*(k+1)<=298]==[0]


def test_fixed_failure_skips_bootstrap_and_repeated_output_reuses(tmp_path,monkeypatch):
    cfg=r.q.load_config()
    a={g:np.ones((n,12)) for g,n in cfg['cohort_counts'].items()}
    raw={g:[{'key':f'{g}/{i}','label':int(g=='test' and i>=54)} for i in range(len(x))] for g,x in a.items()}
    method={'center':[0]*12,'scale':[1]*12,'threshold':0}
    monkeypatch.setattr(r.q,'load_source',lambda *args:(raw,a,None,None))
    monkeypatch.setattr(r.q.common,'replay',lambda *args:(method,{'passed':True}))
    monkeypatch.setattr(r.q.common,'trial_indices',lambda *args:pytest.fail('failed point must not bootstrap'))
    report=r.run(tmp_path/'unused',tmp_path/'output')
    assert report['decision']=='not_adopted' and report['stability'] is None
    assert report['stability_status']=='not_run_fixed_gate_failed'
    assert r.run(tmp_path/'unused',tmp_path/'output')==report
