"""CPU surrogate verification; not Windows CUDA execution evidence."""
import copy
import numpy as np
import pytest

from scripts import mpi_patchcore_raw_bank as b
from scripts import mpi_patchcore_raw_runtime as gpu
from scripts import run_mpi_patchcore_raw_bank as run
from tests.test_mpi_patchcore_scaling import NumpyTorch, Monitor
from tests.test_mpi_patchcore_balanced_bank import expanded, reference
from tests.test_mpi_normal_diversity_evaluation import method


def test_locked_single_factor_and_budget():
    assert b.POLICY['representatives']==78643
    assert b.POLICY['candidates']==3145728
    assert b.POLICY['selection_dimension']==768
    assert b.POLICY['projected_dimension'] is None
    assert b.POLICY['quotas'] is None
    assert b.POLICY['normal_development']==256
    assert b.POLICY['time_limit_seconds'] is None
    assert b.POLICY['resource_plan']==b.prior.POLICY['resource_plan']
    assert b.budget()['raw_cpu_bytes']==9*1024**3
    assert b.budget()['gpu_feature_chunk_bytes']==192*1024**2
    assert b.budget()['selection_host_to_device_bytes']==9*1024**3*78643
    assert gpu.Runtime.predict is gpu.Base.predict
    assert gpu.Runtime.verify_bank is gpu.Base.verify_bank
    assert not b.POLICY['automatic_pass_allowed'] and not b.POLICY['service_model_changed']


def test_cpu_corpus_preserves_every_row_and_checks_bad_files(tmp_path):
    x=np.arange(120,dtype=np.float32).reshape(30,4)
    paths=[]
    for i in range(3):
        path=tmp_path/f'{i}.npy';np.save(path,x[i*10:(i+1)*10]);paths.append(path)
    np.testing.assert_array_equal(gpu.load_training(paths,lambda:None,10,4),x)
    np.save(paths[1],np.full((10,4),np.nan,np.float32))
    with pytest.raises(ValueError):gpu.load_training(paths,lambda:None,10,4)


@pytest.mark.parametrize('chunk',[1,7,31,65536])
def test_streamed_distance_matches_resident_reference(chunk):
    x=np.random.default_rng(42).normal(size=(97,768)).astype(np.float32)
    expected,distances=reference(x,17,3)
    minimum=np.full((97,1),np.inf,np.float32)
    ids=[];center=3
    for _ in range(17):
        center=gpu.update_streamed(NumpyTorch,x,minimum,center,chunk)
        ids.append(center)
    assert ids==expected
    np.testing.assert_array_equal(minimum,distances)


def test_tie_order_and_no_projection():
    x=np.array([[0,0],[1,0],[-1,0],[0,2]],np.float32)
    minimum=np.full((4,1),np.inf,np.float32)
    assert gpu.update_streamed(NumpyTorch,x,minimum,0,2)==3
    assert gpu.update_streamed(NumpyTorch,x,minimum,3,2)==1
    with pytest.raises(ValueError):gpu.update_streamed(NumpyTorch,x,minimum,0,0)


def select(root,x,check=lambda:None,signature='sig',start=3):
    return gpu.select_raw(NumpyTorch,root,signature,lambda:x,len(x),17,start,check,5,7)


def test_checkpoint_resume_equals_uninterrupted_and_skips_final_features(tmp_path):
    x=np.random.default_rng(5).normal(size=(97,768)).astype(np.float32)
    ids,receipts=select(tmp_path/'full',x)
    assert ids==reference(x,17,3)[0]
    root=tmp_path/'resume'
    calls=0
    def stop_mid_chunk():
        nonlocal calls
        if (root/'step-000010').exists():
            calls+=1
            if calls==4:raise KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):select(root,x,stop_mid_chunk)
    assert select(root,x)==(ids,receipts)
    assert gpu.select_raw(NumpyTorch,root,'sig',lambda:pytest.fail('No completed corpus read'),
                          97,17,3,lambda:None,5,7)==(ids,receipts)
    with pytest.raises(ValueError):select(root,x,start=4)
    with pytest.raises(ValueError):select(root,x,signature='changed')
    (root/'step-000010/minimum.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError):select(root,x)


def test_streaming_never_sends_whole_corpus_to_gpu():
    x=np.random.default_rng(5).normal(size=(97,768)).astype(np.float32)
    shapes=[]
    class Recording(NumpyTorch):
        @staticmethod
        def from_numpy(v):
            shapes.append(v.shape)
            return NumpyTorch.from_numpy(v)
    gpu.update_streamed(Recording,x,np.full((97,1),np.inf,np.float32),3,7)
    assert shapes[0]==(768,)
    assert sum(s[0] for s in shapes[1:])==97
    assert max(s[0] for s in shapes[1:])<=7


def test_contract_and_full_comparison_reused_without_quota_experiment_label():
    raw=expanded();m=method(raw)
    b.contract_replay(raw,m)
    source={'cohorts':raw,'reference':raw,'contract':{'candidate_top1_q90':m}}
    result=b.compare(source,raw,m)
    assert result['counts']['baseline']==result['counts']['candidate']
    assert 'native768D' in result['comparison']
    assert b.stability is b.prior.stability


def test_fit_uses_raw_paths_global_start_parity_and425_reload(tmp_path,monkeypatch):
    raw=expanded();cohorts={g:raw[g] for g in ('development','calibration')}
    identity={'config':{},'training':{},'parent_execution_sha256':'parent','parent_checkpoint_sha256':'sha',
        'baseline_contract':method(raw),'feature_receipts_sha256':'f','projection_receipts_sha256':'p','cohorts':cohorts}
    source={'parent':{'fit-report.json':{'context':{}}}}
    parent=tmp_path/'parent';root=tmp_path/'fit';root.mkdir()
    import zipfile
    archive=tmp_path/'a.zip'
    with zipfile.ZipFile(archive,'w'):pass
    monkeypatch.setattr(run,'checked_model',lambda *a:parent)
    monkeypatch.setattr(run,'checked_cache',lambda *a:(['raw'],['never-select-projected'],2814035))
    monkeypatch.setattr(run.engine,'Monitor',Monitor)
    monkeypatch.setattr(run.engine,'normal_gate',lambda *a:{'passed':True})
    phases=[]
    def collect(root,phase,sig,archive,photos,runtime,monitor,checkpoint):
        phases.append(phase)
        assert phase in ('first-development','first-calibration','reload-development','reload-calibration')
        return copy.deepcopy(photos)
    monkeypatch.setattr(run.engine,'collect',collect)
    called=[]
    class Fake:
        def __init__(self,*a):pass
        def load(self,*a):pass
        def release(self):pass
        def parity(self,*a):called.append('parity');return {'passed':True}
        def raw_select(self,root,sig,paths,start,check):
            assert called==['parity']
            assert paths==['raw'] and start==2814035
            return list(range(78643)),{}
        def assemble(self,*a):pass
        def save(self,path,context):path.write_bytes(b'fake bank')
        def verify_bank(self,*a):return {'passed':True}
    report=run.fit_run(tmp_path,root,archive,source,identity,Fake)
    assert report['reload']['photos']==425 and report['test_images_decoded']==0
    assert report['selection_dimension']==768 and report['contract']==method(raw)
    assert len(phases)==4


def test_timing_estimate_does_not_cap_or_change_selection(tmp_path):
    x=np.random.default_rng(5).normal(size=(97,13)).astype(np.float32)
    previews=[]
    ids,_=gpu.select_raw(NumpyTorch,tmp_path/'s','sig',lambda:x,97,30,3,lambda:None,5,7,previews.append)
    assert ids==reference(x,30,3)[0]
    assert len(previews)==1 and previews[0]['selected_at_measurement']==25
    assert previews[0]['remaining_linear_estimate_seconds']>=0


def test_duplicate_degenerate_selection_is_not_automatically_replaced(tmp_path):
    x=np.zeros((5,3),np.float32)
    with pytest.raises(ValueError,match='Repeated'):
        gpu.select_raw(NumpyTorch,tmp_path/'s','sig',lambda:x,5,3,2,lambda:None,1,2)


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


def test_optional_real_cuda_parity():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Lab CUDA check, not CPU evidence')
    pytest.importorskip('anomalib')
    runtime=object.__new__(gpu.Runtime);runtime.torch=torch
    runtime.release=lambda:torch.cuda.empty_cache()
    assert runtime.parity(lambda:None)['passed']
