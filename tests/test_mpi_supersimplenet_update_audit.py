"""CPU integrity/orchestration tests, not evidence of CUDA numerical parity."""
import hashlib
import io
import json
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_update_audit_protocol as p
from scripts import run_mpi_supersimplenet_update_audit as r


def evidence(job, mismatch=False):
    arrays = {arm+'_'+s:np.zeros((16,64,64) if s.endswith('_maps') else (16,),np.float32)
              for arm in p.ARMS for s in p.suffixes()}
    arrays.update(masks=np.zeros((16,64,64),bool),labels=np.zeros(16,bool))
    if mismatch: arrays['reference_author_post_eval_synthetic_scores'] += .1
    outputs = p.output_checks(arrays)
    checks = {pair:{kind:{'x':p.parity.compare([0.],[0.],kind)}
                    for kind in ('loss','gradients','updates','buffers')} for pair in p.PAIRS}
    result = {'job':job,'model_state_restored':True,'noise_sha256':'fixed',
              'output_checks':outputs,'parameter_checks':checks,
              'pair_passed':{pair:all(c['passed'] for c in outputs[pair].values()) for pair in p.PAIRS}}
    return result,arrays


def test_policy_time_unlimited_but_fixed_work_and_memory():
    assert p.POLICY['resource_plan']['comparison_seconds'] is None
    assert len(p.jobs())*len(p.ARMS)==8
    assert p.POLICY['resource_plan']['selection_and_inference_reserved_GiB']==6
    assert p.POLICY['saved_updated_models']==0
    assert not p.POLICY['later_phases_automatically_started']


def test_paired_differences_indices_and_values():
    a=np.zeros(100);b=a.copy();b[70]=3
    u=a.copy();v=a.copy();v[99]=4
    rows=p.selected_differences(a,b,u,v)
    by={r['flat_index']:r for r in rows}
    assert by[70]['gradient_b']==3 and by[99]['update_b']==4
    assert len(rows)<=64 and rows==p.selected_differences(a,b,u,v)
    with pytest.raises(ValueError): p.selected_differences([np.nan],[0.],[0.],[0.])


def test_output_inventory_and_recomputation():
    result,arrays=evidence(p.jobs()[0],True)
    p.verify_arrays(result,arrays)
    result['pair_passed']['author_loss']=True
    with pytest.raises(ValueError,match='verdict'):p.verify_arrays(result,arrays)
    result,arrays=evidence(p.jobs()[0]);arrays['official_a_pre_eval_clean_maps'][0,0,0]=1
    with pytest.raises(ValueError,match='metrics'):p.verify_arrays(result,arrays)


def test_monitor_no_time_limit_and_memory_limit():
    # Exercise the actual stop logic without requiring the lab-only psutil package.
    monitor=r.Monitor.__new__(r.Monitor)
    monitor.sample=lambda:None
    monitor.runtime=None;monitor.error=None;monitor.rss=0;monitor.reserved=0
    monitor.prior=1e12;monitor.started=-1e12;monitor.limits=p.POLICY['resource_plan']
    monitor.check()
    monitor.rss=21*1024**3
    with pytest.raises(RuntimeError,match='Memory cap'):monitor.check()
    monitor.rss=0;monitor.reserved=7*1024**3
    with pytest.raises(RuntimeError,match='Memory cap'):monitor.check()
    monitor.reserved=0;monitor.error='sampler failed'
    with pytest.raises(RuntimeError,match='monitor failed'):monitor.check()


class Monitor:
    def __init__(self,*args):self.runtime=None
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def check(self):pass
    def snapshot(self):return {'fake_not_cuda':True}


class Runtime:
    calls=0
    interrupt=None
    fail=None
    mismatch=False
    def __init__(self,*args):pass
    def release(self):pass
    def compare_job(self,images,job,*args):
        type(self).calls+=1
        if self.calls==self.interrupt:raise KeyboardInterrupt()
        if self.calls==self.fail:raise RuntimeError('GPU failed')
        assert len(images)==8
        return evidence(job,self.mismatch)


@pytest.fixture
def fixture(tmp_path,monkeypatch):
    Runtime.calls=0
    root,fit_root=tmp_path/'audit',tmp_path/'fit'
    checkpoints={}
    for step in (0,28800):
        f=fit_root/f'step-{step:06d}'/'model.pt';f.parent.mkdir(parents=True);f.write_bytes(b'original')
        checkpoints[str(step)]={'sha256':r.fit.sha256_file(f),'fingerprints':{}}
    (fit_root/'calibrated').mkdir();(fit_root/'calibrated/value.json').write_text(json.dumps({'contract':{'threshold':3}}))
    rows=[{'member':f'extracted/grid{i}/train/good/1073.png'} for i in range(8)]
    archive=tmp_path/'source.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for row in rows:z.writestr(row['member'],b'normal')
        z.writestr('test/defect/never.png',b'forbidden')
    prior=tmp_path/'prior.zip'
    with zipfile.ZipFile(prior,'w') as z:
        for job in p.jobs():
            result,a=evidence(job);base='jobs/'+job['name']+'/'
            z.writestr(base+'result.json',json.dumps({'noise_sha256':'fixed'}))
            buf=io.BytesIO();np.savez_compressed(buf,masks=a['masks'],labels=a['labels'],
                official_maps=a['official_a_pre_train_synthetic_maps'],official_scores=a['official_a_pre_train_synthetic_scores'])
            z.writestr(base+'arrays.npz',buf.getvalue())
    seen=[]
    def checked(z,row):
        assert '/train/good/' in row['member'];seen.append(row['member']);return Image.new('RGB',(256,256))
    monkeypatch.setattr(r.fit,'checked_image',checked)
    identity={'jobs':p.jobs(),'rows':rows,'fit_signature':'fixed','checkpoints':checkpoints,
              'contract_sha256':r.fit.digest({'threshold':3}),'prior_parity_zip_sha256':r.fit.sha256_file(prior)}
    return root,archive,fit_root,identity,tmp_path/'handoff.zip',prior,seen


def execute(fixture,runtime=Runtime):
    return r.execute(*fixture[:6],runtime_factory=runtime,monitor_factory=Monitor)


def test_complete_preserve_idempotent(fixture):
    result=execute(fixture)
    assert result['completed_disposable_updates']==8 and result['elapsed_time_cap_seconds'] is None
    assert all(result['pairs_all_jobs_matched'].values())
    assert len(fixture[6])==16
    assert execute(fixture)==result and Runtime.calls==2
    assert not list(fixture[0].rglob('model.pt'))
    with zipfile.ZipFile(fixture[4]) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert all(hashlib.sha256(z.read(k)).hexdigest()==v for k,v in manifest.items())


def test_numerical_difference_exports_without_reroll(fixture):
    class Different(Runtime):mismatch=True
    result=execute(fixture,Different)
    assert not result['pairs_all_jobs_matched']['author_loss'] and fixture[4].exists()


def test_interrupted_job_replay_fixed_attempt_cap(fixture):
    class Interrupted(Runtime):interrupt=2
    with pytest.raises(KeyboardInterrupt):execute(fixture,Interrupted)
    result=execute(fixture)
    assert result['attempted_jobs_including_interrupted']==3


def test_failed_attempt_blocks_retry(fixture):
    class Failed(Runtime):fail=1
    with pytest.raises(RuntimeError):execute(fixture,Failed)
    with pytest.raises(ValueError,match='Prior failed'):execute(fixture)


def test_input_and_cache_tamper(fixture):
    execute(fixture)
    path=fixture[0]/'jobs'/p.jobs()[0]['name']/'arrays.npz';path.write_bytes(b'changed')
    with pytest.raises(ValueError):execute(fixture)


def test_prior_noise_must_replay(fixture):
    result,arrays=evidence(p.jobs()[0]);result['noise_sha256']='changed'
    with pytest.raises(ValueError,match='noise'):r.replay_prior(fixture[5],p.jobs()[0],result,arrays)


def test_prior_forward_must_replay(fixture):
    result,arrays=evidence(p.jobs()[0]);arrays['official_a_pre_train_synthetic_scores']+=1
    with pytest.raises(ValueError,match='forward replay'):r.replay_prior(fixture[5],p.jobs()[0],result,arrays)


def test_checkpoint_and_prior_zip_preserved(fixture):
    original=fixture[2]/'step-000000/model.pt'
    original.write_bytes(b'changed')
    with pytest.raises(ValueError,match='checkpoint'):execute(fixture)
    assert original.read_bytes()==b'changed'


def test_unrecognized_output_not_overwritten(fixture):
    fixture[0].mkdir();(fixture[0]/'user.txt').write_text('keep')
    with pytest.raises(ValueError,match='without identity'):execute(fixture)
    assert (fixture[0]/'user.txt').read_text()=='keep'


def test_non_normal_input_refused(fixture):
    fixture[3]['rows'][0]['member']='test/defect/never.png'
    with pytest.raises(ValueError,match='non-normal'):execute(fixture)
    assert not fixture[6]


def test_cli_import_without_torch():
    proc=subprocess.run([sys.executable,str(r.ROOT/'scripts/run_mpi_supersimplenet_update_audit.py'),'--help'],capture_output=True,text=True)
    assert proc.returncode==0 and '--archive' in proc.stdout
