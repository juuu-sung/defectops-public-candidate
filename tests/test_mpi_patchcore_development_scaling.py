import copy
from contextlib import nullcontext
import hashlib
import json
import zipfile

import numpy as np
import pytest

from scripts import mpi_patchcore_development_scaling as d
from scripts import run_mpi_patchcore_development_scaling as run
from tests.test_mpi_normal_diversity_evaluation import data, method
from tests.test_mpi_patchcore_scaling import Monitor, FakeModelRuntime, fake_score


def row(pid,region):
    return {'photo_id':pid,'region':region,'member':f'extracted/{region}/train/good/{pid}.png',
            'width':256,'height':256,'member_sha256':hashlib.sha256((pid+region+'file').encode()).hexdigest(),
            'decoded_rgb_sha256':hashlib.sha256((pid+region).encode()).hexdigest()}


def selection_fixture():
    ids=[f'{i:04d}' for i in range(650)]
    rows={pid:[row(pid,r) for r in d.fit.REGIONS] for pid in ids}
    pool={'ids':ids,'rows':[r for pid in ids for r in rows[pid]]}
    group=lambda chosen:{'photo_ids':chosen,'rows':[r for pid in chosen for r in rows[pid]]}
    training=group(ids[:256])
    old=group(ids[256:320])
    prior={'training_photo_ids':old['photo_ids'],'training_rows':old['rows']}
    cohorts={'development':[{'photo_id':ids[320],'regions':rows[ids[320]]}],
             'calibration':[{'photo_id':ids[321],'regions':rows[ids[321]]}],
             'validation':[{'photo_id':ids[322],'regions':rows[ids[322]]}],
             'test':[{'photo_id':'9999','regions':[rows[ids[323]][0]]}]}
    return pool,{'training':training,'cohorts':cohorts},prior


def test_selection_excludes_roles_pixels_historical_and_keeps_order():
    pool,plan,prior=selection_fixture();before=copy.deepcopy((pool,plan,prior))
    result=d.select_additional(pool,plan,prior)
    ids=[p['photo_id'] for p in result['additional_photos']]
    assert len(ids)==128 and len(set(ids))==128
    assert result['eligible_count']==650-324
    assert not set(ids)&{f'{i:04d}' for i in range(324)}
    assert ids==sorted([f'{i:04d}' for i in range(324,650)],key=d.rank)[:128]
    assert (pool,plan,prior)==before
    assert result==d.select_additional(pool,plan,prior)


@pytest.mark.parametrize('kind',['duplicate','changed_training','historical','insufficient','bad_member'])
def test_invalid_selection_stops_without_reroll(kind):
    pool,plan,prior=selection_fixture()
    if kind=='duplicate':pool['rows'][-1]['decoded_rgb_sha256']=pool['rows'][-2]['decoded_rgb_sha256']
    if kind=='changed_training':plan['training']=copy.deepcopy(plan['training']);plan['training']['rows'][0]['member_sha256']='a'*64
    if kind=='historical':prior['training_photo_ids'][0]='9999'
    if kind=='insufficient':pool['ids']=pool['ids'][:400];pool['rows']=pool['rows'][:400*12]
    if kind=='bad_member':pool['rows'][-1]['member']='test/defect.png'
    with pytest.raises(ValueError):d.select_additional(pool,plan,prior)


def expanded_fixture():
    raw=data();additional=copy.deepcopy(raw['development'])
    for i,s in enumerate(additional):
        s['key']=f'development-added/{2000+i:04d}';s['photo_id']=f'{2000+i:04d}'
    identity={'selection':{'additional_photos':copy.deepcopy(additional)}}
    return raw,additional,identity


def test_expansion_preserves_every_other_score_and_rejects_labels():
    raw,added,identity=expanded_fixture();saved=copy.deepcopy((raw,added))
    expanded=d.validate_expanded(raw,added,identity)
    assert len(expanded['development'])==256
    for g in ['calibration','test','validation']:assert expanded[g] is raw[g]
    assert (raw,added)==saved
    bad=copy.deepcopy(added);bad[0]['label']=1
    with pytest.raises(ValueError):d.validate_expanded(raw,bad,identity)
    with pytest.raises(ValueError):d.validate_expanded(raw,added[:-1],identity)


def test_draws_full256_not_stratified_and_preserve_original_indices():
    found=list(d.draws());assert len(found)==600
    for (scope,i,di,edi,ci),(oldscope,oldi,oldi_dev,oldi_cal) in zip(found,d.p.evaluation.draws(),strict=True):
        assert (scope,i)==(oldscope,oldi)
        np.testing.assert_array_equal(di,oldi_dev);np.testing.assert_array_equal(ci,oldi_cal)
        assert edi.shape==(256,) and edi.min()>=0 and edi.max()<256
        if scope=='calibration_only':np.testing.assert_array_equal(edi,np.arange(256))
    assert any(np.sum(edi<128)!=128 for scope,_,_,edi,_ in found if scope=='both')
    for a,b in zip(found,d.draws(),strict=True):
        for x,y in zip(a[2:],b[2:]):np.testing.assert_array_equal(x,y)


def test_all_scores_evaluated_even_when_candidate_loses_every_defect():
    raw,added,identity=expanded_fixture();expanded=d.validate_expanded(raw,added,identity)
    candidate=copy.deepcopy(method(expanded));candidate['threshold']=1e6
    result=d.compare(raw,expanded,method(raw),candidate)
    assert result['counts']['candidate']['tp']==0 and result['counts']['candidate']['fn']==55
    assert not result['passed_accuracy_targets'] and not result['gates']['tp55_preserved']
    assert len(result['predictions']['candidate']['test'])==109
    assert len(result['predictions']['candidate']['validation'])==256


def test_same_values_equal_contract_and_no_test_label_in_calibration():
    raw,added,identity=expanded_fixture();expanded=d.validate_expanded(raw,added,identity)
    first=method(expanded)
    changed=copy.deepcopy(expanded)
    for sample in changed['test']:
        sample['label']=1-sample['label']
        for r in sample['regions']:r['top1']+=999
    assert method(changed)==first
    assert first==method(raw)


def test_invalid_resamples_kept_all600_and_baseline_unmodified():
    raw,added,identity=expanded_fixture();expanded=copy.deepcopy(d.validate_expanded(raw,added,identity))
    for sample in expanded['development']:
        for r in sample['regions']:r['top1']=1.
    result=d.stability(raw,expanded)
    assert not result['passed'] and sum(len(x) for x in result['trials'].values())==600
    assert all(not x['candidate']['valid'] for rows in result['trials'].values() for x in rows)
    assert result==d.stability(raw,expanded)


def test_policy_frozen_model_and_no_time_cap():
    assert d.POLICY['model_training'] is False and d.POLICY['time_limit_seconds'] is None
    assert d.POLICY['resource_plan']['comparison_seconds'] is None
    assert d.POLICY['preserve_all_55'] and d.POLICY['additional_development']==128
    assert not d.POLICY['automatic_pass_allowed'] and not d.POLICY['service_model_changed']


def test_analyze_matches_baseline_and_detects_tampered_old_trials():
    raw,added,identity=expanded_fixture();expanded=d.validate_expanded(raw,added,identity)
    previous=d.stability(raw,expanded)
    # Original baseline here is the 128-photo result, not the expanded side.
    trials={s:[{'candidate':r['baseline']} for r in rows] for s,rows in previous['trials'].items()}
    records={'contract.json':method(raw),'evaluation/stability.json':{'trials':trials}}
    result=d.analyze(raw,added,records,identity)
    assert len(result['region_changes'])==12 and result['comparison']['counts']['candidate']['tp']==55
    records['evaluation/stability.json']['trials']['both'][0]['candidate']['counts']['tp']=0
    with pytest.raises(ValueError,match='stability replay'):d.analyze(raw,added,records,identity)


def test_cpu_orchestration_interrupt_resume_export_and_cached_map_tamper(tmp_path,monkeypatch):
    raw,added,identity=expanded_fixture()
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w'):pass
    identity.update(archive_sha256=d.fit.sha256_file(archive),parent_execution_sha256='parent')
    identity['selection']['additional_rows_sha256']='test-rows'
    fitroot=tmp_path/'artifacts'/d.p.NAME;fitroot.mkdir(parents=True)
    parent={'test_fixture':True};identity['parent_execution_sha256']=d.fit.digest(parent)
    with d.fit.phase_stage(fitroot,'model') as stage:
        (stage/'model.pt').write_bytes(b'fake-model-no-torch')
        d.fit.write_json(stage/'context.json',{})
        d.fit.seal_phase(stage,'model',d.fit.digest(parent),{},['model.pt','context.json'])
    identity['checkpoint_sha256']=d.fit.sha256_file(fitroot/'model/model.pt')
    records={'plan.json':{**parent,'config':{},'cohorts':{}},'model-receipt.json':json.loads((fitroot/'model/receipt.json').read_bytes()),
             'fit-report.json':{'context':{}},'contract.json':method(raw)}
    # Model validation is independently tested below; parent fixture omits a real config.
    monkeypatch.setattr(run,'verify_model',lambda *a:fitroot)
    monkeypatch.setattr(run.engine,'Monitor',Monitor)
    monkeypatch.setattr(run.engine,'normal_gate',lambda *a,**k:{'passed':True})
    monkeypatch.setattr(run.fit,'checked_image',lambda *a:nullcontext(None))
    monkeypatch.setattr(run.engine,'score_photo',fake_score)
    prior=d.stability(raw,d.validate_expanded(raw,added,identity))
    records['evaluation/stability.json']={'trials':{s:[{'candidate':r['baseline']} for r in rows] for s,rows in prior['trials'].items()}}
    original=run.engine.collect
    def interrupt(*a,**kw):
        result=original(*a,**kw)
        raise KeyboardInterrupt()
    monkeypatch.setattr(run.engine,'collect',interrupt)
    with pytest.raises(KeyboardInterrupt):run.run(tmp_path,archive,records,raw,identity,FakeModelRuntime)
    monkeypatch.setattr(run.engine,'collect',original)
    summary,output=run.run(tmp_path,archive,records,raw,identity,FakeModelRuntime)
    assert summary['new_test_images_decoded']==0 and not summary['model_changed']
    assert summary['counts']['candidate']['tp']+summary['counts']['candidate']['fn']==55
    before=output.read_bytes()
    class Never:
        def __init__(self,*a):pytest.fail('No GPU on completed rerun')
    run.run(tmp_path,archive,records,raw,identity,Never)
    assert output.read_bytes()==before
    with zipfile.ZipFile(output) as z:
        assert 'stability.json' in z.namelist() and all(n.endswith('.json') for n in z.namelist())
    root=tmp_path/'artifacts/analyses'/d.NAME
    (root/'additional/photo-000/maps.npy').write_bytes(b'changed')
    with pytest.raises(ValueError):run.run(tmp_path,archive,records,raw,identity,Never)


def test_verify_model_refuses_changed_file(tmp_path):
    root=tmp_path/'artifacts'/d.p.NAME;root.mkdir(parents=True)
    plan={'cohorts':{}}
    with d.fit.phase_stage(root,'model') as stage:
        (stage/'model.pt').write_bytes(b'fake')
        d.fit.write_json(stage/'context.json',{})
        d.fit.seal_phase(stage,'model',d.fit.digest(plan),{},['model.pt','context.json'])
    records={'plan.json':plan,'model-receipt.json':json.loads((root/'model/receipt.json').read_bytes()),'fit-report.json':{'context':{}}}
    assert run.verify_model(tmp_path,records)==root
    (root/'model/model.pt').write_bytes(b'bad')
    with pytest.raises(ValueError):run.verify_model(tmp_path,records)
