import copy
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import run_mpi_qn_scale as q


def synthetic():
    rng = np.random.default_rng(713)
    cfg = q.load_config()
    raw = {}
    for group, n in cfg["cohort_counts"].items():
        raw[group] = []
        for i in range(n):
            label = int(group == "test" and i >= 54)
            pid = f"{group}-{i}"
            key = f"{'defect' if label else 'good'}/{pid}" if group == "test" else f"{group}/{pid}"
            raw[group].append({"key": key, "photo_id": pid, "label": label,
                "regions": [{"region": r, "top1": float(10 + rng.uniform(0, 5) + label*5)}
                            for r in q.common.REGIONS]})
    return raw, q.common.matrices(raw, cfg["cohort_counts"])


def test_hand_order_statistic_not_interpolated_quantile():
    # n=4: h=3, k=3, distances sorted 1,3,4,6,9,10.
    assert q.qn_scale([0, 1, 4, 10]) == 4
    assert q.qn_scale([0, 0, 4, 10]) == 4
    assert q.qn_scale([0, 2]) == 2


@pytest.mark.parametrize("x", [[], [1], [0]*128, [0, 0, 0, 1], [1, np.nan], [1, np.inf], [0, 1e-6]])
def test_invalid_scale_no_rescue(x):
    with pytest.raises(ValueError):
        q.qn_scale(x)


def test_partition_equals_independent_full_sort_and_equivariance():
    x = np.random.default_rng(2).uniform(0, 20, 128)
    expected = sorted(abs(float(a)-float(b)) for i,a in enumerate(x) for b in x[i+1:])[2079]
    assert q.qn_scale(x) == expected
    assert q.qn_scale(x+10) == pytest.approx(expected)
    assert q.qn_scale(x*3) == pytest.approx(expected*3)
    assert q.qn_scale(x[::-1]) == expected


def test_fit_even_median_higher_threshold_boundary_and_common_factor():
    dev = np.outer(np.arange(128), np.arange(1,13)).astype(float)
    cal = np.outer(np.arange(169), np.arange(1,13)).astype(float)
    method = q.fit_qn(dev, cal)
    assert method["center"] == (63.5*np.arange(1,13)).tolist()
    scores = q.common.normalized(cal, method).max(axis=1)
    assert method["threshold"] == sorted(scores)[160]
    assert np.sum(scores >= method["threshold"]) == 9
    changed = copy.deepcopy(method)
    changed["scale"] = (np.array(method["scale"])*2).tolist()
    changed["threshold"] /= 2
    np.testing.assert_array_equal(scores >= method["threshold"],
        q.common.normalized(cal, changed).max(axis=1) >= changed["threshold"])


def test_fit_shape_finiteness_and_region_equivariance():
    _, a = synthetic()
    dev, cal = a["development"], a["calibration"]
    m = q.fit_qn(dev, cal)
    p = q.fit_qn(dev[:, ::-1], cal[:, ::-1])
    assert p["scale"] == m["scale"][::-1] and p["threshold"] == m["threshold"]
    for d,c in [(dev[:-1], cal), (dev, cal[:-1]), (dev*np.nan, cal), (dev, cal*np.inf)]:
        with pytest.raises(ValueError):
            q.fit_qn(d,c)


def test_pairing_rng_and_invalid_bootstrap_not_dropped():
    raw,a = synthetic()
    labels = np.array([r["label"] for r in raw["test"]])
    cfg = q.load_config()
    _,di,ci = next(q.common.trial_indices("both", cfg["bootstrap"],128,169))
    rng = np.random.Generator(np.random.PCG64(20261003))
    np.testing.assert_array_equal(di, rng.integers(0,128,128))
    np.testing.assert_array_equal(ci, rng.integers(0,169,169))
    record=q.trial(a,labels,di,ci)
    assert record["methods"]["candidate"]["parameters"] == q.fit_qn(a["development"][di],a["calibration"][ci])
    invalid=q.trial(a,labels,np.zeros(128,dtype=int),ci)
    result=q.common.summarize_trials([record,invalid],cfg["development_screening"])
    assert not result["passed"]
    assert result["methods"]["candidate"]["quantiles"] is None
    assert result["methods"]["candidate"]["invalid_trials"] == 1


def test_frozen_config_rejects_changes(tmp_path):
    p=tmp_path/'cfg.json'
    p.write_text(q.CONFIG.read_text().replace('2080','2081'))
    with pytest.raises(ValueError,match='configuration'):
        q.load_config(p)


def test_source_seals_replay_and_tampering(tmp_path):
    raw,a=synthetic();cfg=q.load_config()
    m=q.common.fit_method(a['development'],a['calibration'],0)
    pred=q.common.predictions(raw,a,m)
    counts=q.common.counts(a,np.array([r['label'] for r in raw['test']]),m)
    stored={'field':'top1','threshold':m['threshold'],'regions':[
        {'region':r,'center':c,'scale':s} for r,c,s in zip(q.common.REGIONS,m['center'],m['scale'])]}
    ref={'counts':{k:counts[k] for k in ('tp','fn','fp','tn')},
        'false_negatives':[r['key'] for r in pred['test'] if r['label'] and not r['alert']],
        'false_positives':[r['key'] for r in pred['test'] if not r['label'] and r['alert']]}
    freeze={'contract':{cfg['baseline']:stored},'observed_test_reference':{cfg['baseline']:ref}}
    report={'normal_validation':{'methods':{cfg['baseline']:{'alert_keys':[r['key'] for r in pred['validation'] if r['alert']]}}}}
    blobs={n:q.common.canonical(v) for n,v in [('raw-scores.json',raw),('freeze.json',freeze),('report.json',report)]}
    path=tmp_path/'input.zip'
    with zipfile.ZipFile(path,'w') as z:
        for n,b in blobs.items():z.writestr(n,b)
        z.writestr('handoff-manifest.json',q.common.canonical({n:q.common.sha(b) for n,b in blobs.items()}))
    cfg.update(source_zip_sha256=q.common.sha(path.read_bytes()),source_raw_scores_sha256=q.common.sha(blobs['raw-scores.json']),
               source_freeze_sha256=q.common.sha(blobs['freeze.json']))
    r,arr,s,reference=q.load_source(path,cfg)
    assert q.common.replay(arr,r,s,reference,cfg)[1]['maximum_absolute_difference']==0
    for key in ['source_zip_sha256','source_raw_scores_sha256','source_freeze_sha256']:
        changed=dict(cfg);changed[key]='wrong'
        with pytest.raises(ValueError):q.load_source(path,changed)
    s['threshold']+=1
    with pytest.raises(ValueError):q.common.replay(arr,r,s,reference,cfg)


def test_checkpoint_reuse_indices_and_integrity(tmp_path,monkeypatch):
    raw,a=synthetic();labels=np.array([r['label'] for r in raw['test']])
    di,ci=np.arange(128),np.arange(169);path=tmp_path/'trial.json'
    original=q.reuse_trial(path,'id',a,labels,di,ci)
    def no_recompute(*args):raise AssertionError('completed trial recomputed')
    monkeypatch.setattr(q,'trial',no_recompute)
    assert q.reuse_trial(path,'id',a,labels,di,ci)==original
    with pytest.raises(ValueError):q.reuse_trial(path,'id',a,labels,di[::-1],ci)
    blob=json.loads(path.read_text());blob['payload']['development_indices'][0]=5
    path.write_text(json.dumps(blob))
    with pytest.raises(ValueError):q.reuse_trial(path,'id',a,labels,di,ci)


def test_attempt_accounting_unknown_cost_and_prior_budget_block(tmp_path):
    assert q.attempt_history(tmp_path,'id')==(0,0)
    q.common.publish(tmp_path/'attempt-starts/a.json','id',{'started':1})
    with pytest.raises(RuntimeError,match='unknown cost'):q.attempt_history(tmp_path,'id')
    q.common.publish(tmp_path/'attempts/a.json','id',{'elapsed_seconds':4,'peak_rss_bytes':100,'budget_exceeded':False})
    assert q.attempt_history(tmp_path,'id')==(4,100)
    q.common.publish(tmp_path/'attempt-starts/b.json','id',{'started':2})
    q.common.publish(tmp_path/'attempts/b.json','id',{'elapsed_seconds':6,'peak_rss_bytes':200,'budget_exceeded':True})
    with pytest.raises(RuntimeError,match='resource violation'):q.attempt_history(tmp_path,'id')


def test_budget_and_existing_evidence_not_overwritten(tmp_path,monkeypatch):
    cfg=q.load_config()['cpu_budget']
    monkeypatch.setattr(q.time,'monotonic',lambda:100)
    monkeypatch.setattr(q.common,'peak_rss_bytes',lambda:128)
    assert q.budget(90,5,cfg)==(15,128)
    with pytest.raises(RuntimeError):q.budget(90,1800,cfg)
    monkeypatch.setattr(q.common,'peak_rss_bytes',lambda:3*1024**3)
    with pytest.raises(RuntimeError):q.budget(90,0,cfg)
    p=tmp_path/'saved.json';q.common.publish(p,'id',{'v':1});before=p.read_bytes()
    with pytest.raises(ValueError):q.common.publish(p,'id',{'v':2})
    assert p.read_bytes()==before


def test_low_constant_detection_cannot_pass_stability():
    cfg=q.load_config()
    records=[{'methods':{n:{'valid':True,'counts':{'tp':3,'fp':0,'fp_validation':0}} for n in ['control','candidate']}}]*4
    assert not q.common.summarize_trials(records,cfg['development_screening'])['passed']


def test_point_comparison_full_changes_and_labels_not_in_fit():
    raw,a=synthetic();cfg=q.load_config()
    m=q.fit_qn(a['development'],a['calibration'])
    methods={'control':m,'candidate':m}
    point,pred=q.compare(raw,a,methods,cfg['development_screening'])
    assert point['counts']['control']==point['counts']['candidate']
    assert not any(point['changes'][g][k] for g in ['test','validation'] for k in ['added_alerts','removed_alerts'])
    for r in raw['test']:r['label']=1-r['label']
    assert q.fit_qn(a['development'],a['calibration'])==m


def test_execute_interruption_resume_and_completed_repeat(tmp_path,monkeypatch):
    raw,a=synthetic();cfg=q.load_config();cfg['bootstrap']['trials_per_scope']=2
    control=q.common.fit_method(a['development'],a['calibration'],0)
    monkeypatch.setattr(q,'load_config',lambda:cfg)
    monkeypatch.setattr(q,'source_identity',lambda c:({'test_identity':c},'synthetic-commit'))
    monkeypatch.setattr(q,'load_source',lambda *args:(raw,a,None,None))
    monkeypatch.setattr(q.common,'replay',lambda *args:(control,{'passed':True,'decision_changes':0}))
    output=tmp_path/'output'
    real_compare=q.compare
    def verify_normal_first(*args):
        assert (output/'parameters.json').exists()
        return real_compare(*args)
    monkeypatch.setattr(q,'compare',verify_normal_first)
    original=q.trial;called=[]
    def interrupted(*args):
        called.append(1)
        if len(called)==2:raise RuntimeError('synthetic interruption')
        return original(*args)
    monkeypatch.setattr(q,'trial',interrupted)
    with pytest.raises(RuntimeError,match='synthetic interruption'):q.execute(Path('unused'),output)
    saved=(output/'trials/development_only/0000.json').read_bytes()
    monkeypatch.setattr(q,'trial',original)
    report=q.execute(Path('unused'),output)
    assert report['status']=='exploratory_comparison_complete'
    assert len(list((output/'trials').glob('*/*.json')))==6
    assert (output/'trials/development_only/0000.json').read_bytes()==saved
    attempts=[json.loads(p.read_text())['payload'] for p in (output/'attempts').glob('*.json')]
    assert len(attempts)==2 and any('synthetic interruption' in (r['error'] or '') for r in attempts)
    assert report['cost']['cumulative_elapsed_seconds']==pytest.approx(sum(r['elapsed_seconds'] for r in attempts))
    monkeypatch.setattr(q,'trial',lambda *args:pytest.fail('completed repeat recomputed trials'))
    assert q.execute(Path('unused'),output)==report
    assert len(list((output/'attempts').glob('*.json')))==3


def test_identity_rejects_uncommitted_source(monkeypatch):
    original=q.subprocess.check_output
    def fake(cmd,**kwargs):
        return 'synthetic-commit\n' if cmd[1]=='rev-parse' else b'wrong bytes'
    monkeypatch.setattr(q.subprocess,'check_output',fake)
    with pytest.raises(ValueError,match='not committed'):q.source_identity(q.load_config())


def test_single_writer_lock(tmp_path):
    with q.common.locked(tmp_path):
        with pytest.raises(BlockingIOError):
            with q.common.locked(tmp_path):pass
