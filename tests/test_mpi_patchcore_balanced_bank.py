"""CPU persistence/arithmetic/orchestration; real Torch test is optional, not CUDA proof."""
import copy
from contextlib import contextmanager
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import mpi_patchcore_balanced_bank as b
from scripts import mpi_patchcore_balanced_runtime as gpu
from scripts import run_mpi_patchcore_balanced_bank as run
from tests.test_mpi_patchcore_scaling import NumpyTorch, Monitor
from tests.test_mpi_normal_diversity_evaluation import data, method


def expanded():
    raw=data()
    added=copy.deepcopy(raw['development'])
    for i,s in enumerate(added):
        s['key']=f'development-added/{1500+i:04d}';s['photo_id']=f'{1500+i:04d}'
        for row in s['regions']:row['top1']+=.25;row['native_raw']+=.25
    raw['development']+=added
    return raw


def test_locked_design_and_preserved_inference():
    assert b.quotas()==b.POLICY['quotas']==[6554]*7+[6553]*5
    assert sum(b.quotas())==b.POLICY['representatives']==78643
    assert b.POLICY['normal_development']==256
    assert b.POLICY['query_search'].startswith('all 78643')
    assert b.POLICY['time_limit_seconds'] is None
    assert b.POLICY['resource_plan']==b.p.POLICY['resource_plan']
    assert gpu.Runtime.predict is gpu.Base.predict
    assert not b.POLICY['automatic_pass_allowed'] and not b.POLICY['service_model_changed']
    with pytest.raises(ValueError):b.quotas(1,2)


def test_region_mapping_covers_every_training_candidate_once():
    groups=[b.global_indices(r,4,3,5) for r in range(3)]
    np.testing.assert_array_equal(np.sort(np.concatenate(groups)),np.arange(60))
    for r,g in enumerate(groups):assert np.all(g//5%3==r)
    assert b.distribution([0,20,40],4,3,5)['regions']==[1,1,1]
    with pytest.raises(ValueError):b.distribution([0,0],4,3,5)
    with pytest.raises(ValueError):b.global_indices(3,4,3,5)


def test_projected_region_matches_parent_with_crossing_chunks(tmp_path):
    raw=np.arange(60*7,dtype=np.float32).reshape(60,7)
    paths=[]
    for i,(a,z) in enumerate(b.p.probe.slices(60,11)):
        path=tmp_path/f'{i}.npy';np.save(path,raw[a:z]);paths.append(path)
    for region in range(3):
        actual=gpu.projected_region(paths,region,4,3,5,7,11)
        np.testing.assert_array_equal(actual,raw[b.global_indices(region,4,3,5)])
    with pytest.raises(ValueError):gpu.projected_region(paths[:-1],0,4,3,5,7,11)


def select(root,features,check=lambda:None,start=3):
    return gpu.select_region(NumpyTorch,root,'sig',lambda:features,features.shape[0],17,start,check,5,7)


def reference(features,n,start):
    minimum=np.full((len(features),1),np.inf,np.float32);selected=[]
    center=start
    for _ in range(n):
        minimum=np.minimum(minimum,np.linalg.norm(features-features[center],axis=1,keepdims=True))
        center=int(np.argmax(minimum));minimum[center]=0.;selected.append(center)
    return selected,minimum


def test_real_selection_loop_matches_reference_and_resumes(tmp_path):
    features=np.random.default_rng(12).normal(size=(97,13)).astype(np.float32)
    expected,minimum=reference(features,17,3)
    ids,receipts=select(tmp_path/'full',features)
    assert ids==expected
    np.testing.assert_array_equal(np.load(tmp_path/'full/step-000017/minimum.npy'),minimum)
    root=tmp_path/'resume'
    def stop():
        if (root/'step-000010').exists():raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):select(root,features,stop)
    assert select(root,features)==(ids,receipts)
    assert gpu.select_region(NumpyTorch,root,'sig',lambda:pytest.fail('No feature read on completion'),97,17,3,lambda:None,5,7)==(ids,receipts)
    with pytest.raises(ValueError,match='initialization'):select(root,features,start=4)


def test_selection_corruption_and_identity_are_not_repaired(tmp_path):
    features=np.random.default_rng(12).normal(size=(97,13)).astype(np.float32)
    select(tmp_path,features)
    with pytest.raises(ValueError):gpu.select_region(NumpyTorch,tmp_path,'wrong',lambda:features,97,17,3,lambda:None,5,7)
    (tmp_path/'step-000005').rename(tmp_path/'preserved-five')
    with pytest.raises(ValueError,match='gap'):select(tmp_path,features)
    (tmp_path/'preserved-five').rename(tmp_path/'step-000005')
    (tmp_path/'step-000010/minimum.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError):select(tmp_path,features)
    assert (tmp_path/'step-000010/minimum.npy').read_bytes()==b'corrupt'


def test_completed_region_quotas_preserve_global_indices(tmp_path,monkeypatch):
    runtime=object.__new__(gpu.Runtime);runtime.torch=NumpyTorch
    runtime.release=lambda:None
    calls=[]
    def fixed(torch,root,signature,factory,n,maximum,start,check):
        calls.append((root.name,n,maximum,start))
        return list(range(maximum)),[]
    monkeypatch.setattr(gpu,'select_region',fixed)
    ids,_=runtime.balanced_select(tmp_path,'s',[],2814035,lambda:None)
    assert b.distribution(ids)['regions']==b.quotas()
    assert len(set(ids))==78643
    assert [c[2] for c in calls]==b.quotas()
    assert {c[1] for c in calls}=={262144}
    assert {c[3] for c in calls}=={2814035%262144}


def test_256_contract_and_comparison_do_not_use_128_only():
    raw=expanded();contract=method(raw)
    b.contract_replay(raw,contract)
    source={'cohorts':raw,'reference':raw,'contract':{'candidate_top1_q90':contract}}
    result=b.compare(source,raw,contract)
    assert result['counts']['baseline']==result['counts']['candidate']
    assert 'all55_preserved' in result['gates']
    bad=copy.deepcopy(raw)
    for s in bad['development'][128:]:s['regions'][0]['top1']+=1e5
    with pytest.raises(ValueError):b.contract_replay(bad,contract)
    missing=copy.deepcopy(raw);missing['development'].pop()
    with pytest.raises(ValueError):b.compare(source,missing,contract)


def test_stability_pairs_all256_normals_and_retains_invalid_draws():
    raw=expanded();stable=b.stability(raw,raw)
    assert sum(map(len,stable['trials'].values()))==600
    for rows in stable['trials'].values():
        for row in rows:
            assert len(row['development_indices'])==256
            assert row['baseline']==row['candidate']
    invalid=copy.deepcopy(raw)
    for s in invalid['development']:
        for r in s['regions']:r['top1']=1.
    bad=b.stability(raw,invalid)
    assert not bad['passed']
    assert sum(not x['valid'] for rows in bad['trials'].values() for x in rows)==600


def role_cohorts():
    training={'photo_ids':[f'{i:04d}' for i in range(256)],'rows':[]}
    for pid in training['photo_ids']:
        for region in b.fit.REGIONS:
            training['rows'].append({'photo_id':pid,'region':region,'decoded_rgb_sha256':pid+region})
    cohorts={}
    offset=256
    for group,n in [('development',256),('calibration',169),('validation',256),('test',109)]:
        cohorts[group]=[]
        for i in range(n):
            pid=f'{offset+i:04d}'
            cohorts[group].append({'photo_id':pid,'key':group+'/'+pid,'label':int(group=='test' and i>=54),
                'regions':[{'decoded_rgb_sha256':pid+r} for r in b.fit.REGIONS]})
        offset+=n
    return training,cohorts


def test_role_separation_checks_ids_and_pixels(monkeypatch):
    # Other source-manifest fields are separately validated by the frozen group validator.
    monkeypatch.setattr(b.development.pool_protocol,'validate_group',lambda *a:None)
    training,cohorts=role_cohorts();b.validate_roles(training,cohorts)
    bad=copy.deepcopy(cohorts);bad['development'][0]['photo_id']=training['photo_ids'][0]
    with pytest.raises(ValueError,match='roles'):b.validate_roles(training,bad)
    bad=copy.deepcopy(cohorts);bad['test'][0]['regions'][0]['decoded_rgb_sha256']=training['rows'][0]['decoded_rgb_sha256']
    with pytest.raises(ValueError,match='RGB'):b.validate_roles(training,bad)
    # Test photo numbers can coincide: they are a separate namespace.
    good=copy.deepcopy(cohorts);good['test'][0]['photo_id']=training['photo_ids'][0]
    b.validate_roles(training,good)


def test_fit_worker_flow_saves_bank_replays_all_normals_and_never_tests(tmp_path,monkeypatch):
    raw=expanded();cohorts={g:raw[g] for g in ('development','calibration')}
    identity={'config':{},'training':{},'parent_execution_sha256':'parent','parent_checkpoint_sha256':'sha',
        'baseline_contract':method(raw),'feature_receipts_sha256':'f','projection_receipts_sha256':'p','cohorts':cohorts}
    source={'parent':{'fit-report.json':{'context':{}}}}
    parent=tmp_path/'parent';root=tmp_path/'fit';root.mkdir()
    archive=tmp_path/'a.zip'
    import zipfile
    with zipfile.ZipFile(archive,'w'):pass
    monkeypatch.setattr(run,'checked_model',lambda *a:parent)
    monkeypatch.setattr(run,'checked_cache',lambda *a:([],[],3))
    monkeypatch.setattr(run.engine,'Monitor',Monitor)
    monkeypatch.setattr(run.engine,'normal_gate',lambda *a:{'passed':True})
    phases=[]
    def collect(root,phase,sig,archive,photos,runtime,monitor,checkpoint):
        phases.append(phase)
        assert phase in ('first-development','first-calibration','reload-development','reload-calibration')
        return copy.deepcopy(photos)
    monkeypatch.setattr(run.engine,'collect',collect)
    ids=[x for region,q in enumerate(b.quotas()) for x in b.global_indices(region)[:q].tolist()]
    class Fake:
        def __init__(self,*a):pass
        def load(self,*a):pass
        def release(self):pass
        def balanced_select(self,*a):return ids,{}
        def assemble(self,*a):pass
        def save(self,path,context):path.write_bytes(b'fake bank')
        def verify_bank(self,*a):return {'passed':True}
    report=run.fit_run(tmp_path,root,archive,source,identity,Fake)
    assert report['reload']['photos']==425
    assert report['test_images_decoded']==0
    assert report['contract']==method(raw)
    assert len(phases)==4
    assert (root/'model/model.pt').read_bytes()==b'fake bank'


def test_frozen_failures_block_new_attempts(tmp_path,monkeypatch):
    monkeypatch.setattr(run.engine,'Monitor',Monitor)
    root=tmp_path/'r';root.mkdir();run.engine.initialize(root,{'x':1})
    with pytest.raises(RuntimeError):
        with run.engine.attempt(root,'s',b.POLICY):raise RuntimeError('budget')
    with pytest.raises(ValueError,match='failed/resource'):
        run.engine.initialize(root,{'x':1})
    assert len(list(root.glob('attempt-*/finished.json')))==1


def test_evaluation_finishes_all_diagnostics_even_if_accuracy_fails(tmp_path,monkeypatch):
    import zipfile
    raw=expanded();raw['test'][-1]['regions'][0]['member']=run.ev.REPAIR_MEMBER
    raw['test'][-1]['regions'][0]['member_sha256']='approved'
    for sample in raw['test']:
        if sample['label']:
            for row in sample['regions']:row['top1']=1.;row['native_raw']=1.
    method_=method(raw)
    identity={'cohorts':raw,'config':{},'baseline_contract':method_}
    fit_root=tmp_path/'fit';(fit_root/'model').mkdir(parents=True)
    checkpoint=fit_root/'model/model.pt';checkpoint.write_bytes(b'model')
    sha=b.fit.sha256_file(checkpoint)
    report={'checkpoint_sha256':sha,'contract':method_,'context':{}}
    root=tmp_path/'evaluation';root.mkdir()
    archive=tmp_path/'archive.zip'
    with zipfile.ZipFile(archive,'w'):pass
    old_stability=b.stability(raw,raw)
    source={'cohorts':raw,'reference':raw,'contract':{'candidate_top1_q90':method_},
        'parent':{'fit-report.json':{'context':{}}},
        'development':{'stability.json':old_stability}}
    monkeypatch.setattr(run,'check_fit',lambda *a:(report,{g:raw[g] for g in ('development','calibration')}))
    monkeypatch.setattr(run,'checked_model',lambda *a:tmp_path/'parent')
    monkeypatch.setattr(run.engine,'Monitor',Monitor)
    monkeypatch.setattr(run.engine,'normal_gate',lambda *a:{'passed':True})
    phases=[]
    def collect(root,group,*args):phases.append(group);return raw[group]
    monkeypatch.setattr(run.engine,'collect',collect)
    monkeypatch.setattr(run.engine,'read_sample',lambda folder,name,*args:raw[folder.name][int(name.split('-')[-1])])
    calls=[]
    def warm(*args):calls.append('warm');return {'completed':True}
    def location(*args):calls.append('location');return {'completed':True}
    monkeypatch.setattr(run.b.evaluation,'warm',warm)
    monkeypatch.setattr(run,'location',location)
    class Fake:
        def __init__(self,*a):pass
        def load(self,*a):pass
        def release(self):pass
    result=run.evaluate(tmp_path,fit_root,root,archive,source,identity,Fake)
    assert not result['accuracy_targets_passed']
    assert result['all55_defects_evaluated'] and result['stability_draws']==600
    assert phases==['test','validation'] and calls==['warm','warm','location']
    # Completed rerun reads saved evidence but does not start GPU work.
    again=run.evaluate(tmp_path,fit_root,root,archive,source,identity,lambda *a:pytest.fail('No GPU'))
    assert again==result and len(phases)==2


def test_optional_torch_selection_matches_numpy(tmp_path):
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Real CUDA check requires lab GPU')
    features=np.random.default_rng(12).normal(size=(97,13)).astype(np.float32)
    expected,_=reference(features,17,3)
    actual,_=gpu.select_region(torch,tmp_path,'s',lambda:features,97,17,3,lambda:None,5,7)
    assert actual==expected


@pytest.mark.parametrize('group,index',[('calibration',0),('test',0),('development',128)])
def test_parent_maps_resolve_original_and_added_normal_receipts(tmp_path,group,index):
    raw=expanded();photo=raw[group][index]
    parent={'plan.json':{'parent':True},'fit-report.json':{'checkpoint_sha256':'sha'},
            'prediction-receipts.json':{},'evaluation/prediction-receipts.json':{}}
    dev={'plan.json':{'added':True},'prediction-receipts.json':{}}
    if group=='development':
        base=tmp_path/'artifacts/analyses'/b.development.NAME/'additional'
        local=0;signature=b.fit.digest(dev['plan.json'])
    else:
        phase='first-'+group if group=='calibration' else group
        base=tmp_path/('artifacts' if group=='calibration' else 'artifacts/analyses')/b.p.NAME/phase
        local=index;signature=b.fit.digest(parent['plan.json'])
    base.mkdir(parents=True);name=f'photo-{local:03d}'
    values=np.broadcast_to(np.array([r['top1'] for r in photo['regions']],np.float32)[:,None,None],(12,256,256)).copy()
    with b.fit.phase_stage(base,name) as stage:
        np.save(stage/'maps.npy',values,allow_pickle=False)
        b.fit.write_json(stage/'sample.json',photo)
        receipt=b.fit.seal_phase(stage,name,signature,{'photo_sha256':b.fit.digest(photo),'checkpoint_sha256':'sha'},['sample.json','maps.npy'])
    if group=='development':dev['prediction-receipts.json'][name]=receipt
    else:parent['prediction-receipts.json' if group=='calibration' else 'evaluation/prediction-receipts.json'][f'{phase}\\{name}\\receipt.json']=receipt
    source={'parent':parent,'development':dev,'reference':raw,'cohorts':raw}
    np.testing.assert_array_equal(run.parent_maps(tmp_path,source,group,index),values)
    (base/name/'maps.npy').write_bytes(b'changed')
    with pytest.raises(ValueError):run.parent_maps(tmp_path,source,group,index)


def test_location_compares_all_defects_but_keeps_mask_exception(tmp_path,monkeypatch):
    raw=expanded();raw['test'][-1]['key']='defect/0006'
    arr=np.ones((12,256,256),np.float32);mask_calls=[]
    monkeypatch.setattr(run.engine,'read_sample',lambda *a:None)
    monkeypatch.setattr(run.p,'array',lambda *a:arr)
    monkeypatch.setattr(run,'parent_maps',lambda *a:arr)
    def masks(archive,photo):mask_calls.append(photo['key']);return [None]*12
    monkeypatch.setattr(run.ev,'read_masks',masks)
    monkeypatch.setattr(run.b.evaluation.maps,'map_stats',lambda *a:{'mean':1.,'display_min':0.})
    result=run.location(tmp_path,tmp_path,tmp_path,'sig',None,{'cohorts':raw,'reference':raw},raw,method(raw),'sha',lambda:None)
    assert len(result['cases'])==67 and len(mask_calls)==54
    assert 'defect/0006' not in mask_calls
    assert sum(x['role']=='all_defects' for x in result['cases'])==55
    assert all(len(x['tiles'])==12 for x in result['cases'])
    assert all('display_min' not in t['candidate'] for x in result['cases'] for t in x['tiles'])


def test_missing_old_cache_stops_before_gpu_work(tmp_path):
    source={'parent':{'prediction-receipts.json':{'first-calibration\\photo-000\\receipt.json':{}},
                     'evaluation/prediction-receipts.json':{}},
            'development':{'prediction-receipts.json':{}}}
    with pytest.raises(ValueError,match='cache missing'):run.required_parent_maps(tmp_path,source)
