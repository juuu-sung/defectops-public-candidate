"""Gradient geometry and read-only orchestration without claiming CUDA coverage."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_gradient_protocol as p
from scripts import run_mpi_supersimplenet_gradients as runner
from scripts.mpi_supersimplenet_gradient_runtime import Runtime as ActualRuntime
from test_mpi_supersimplenet_varying import arrays, Monitor


@pytest.mark.parametrize('sign,cosine,retained',[(1,1.,1.),(-1,-1.,0.),(0,None,1.)])
def test_gradient_geometry(sign,cosine,retained):
    normal=np.array([1,2,3],np.float32); anomaly=sign*normal
    vectors=[normal,anomaly]+[np.zeros(3,np.float32) for _ in range(4)]+[normal+anomaly]
    gram=p.gram_matrix(vectors); result=p.summarize_gram(gram)
    if cosine is None: assert result['normal_anomaly_cosine'] is None
    else: assert result['normal_anomaly_cosine']==pytest.approx(cosine)
    assert result['retained_norm_fraction']==pytest.approx(retained)


def test_zero_gradient_is_undefined_and_invalid_matrices_fail():
    s=p.summarize_gram(np.zeros((7,7)))
    assert s['normal_anomaly_cosine'] is None and s['retained_norm_fraction'] is None
    with pytest.raises(ValueError):p.summarize_gram(-np.eye(7))
    with pytest.raises(ValueError):p.gram_matrix([np.array([np.nan])]*7)


def test_policy_no_training_and_fixed_jobs():
    assert len(p.jobs())==24 and len({j['name'] for j in p.jobs()})==24
    assert not p.POLICY['reset_parameters'] and p.POLICY['optimizer_updates']==0
    assert p.STRENGTHS==['std015','std060']
    runtime=object.__new__(ActualRuntime)
    for method in (runtime.step,runtime.step_at_strength):
        with pytest.raises(ValueError,match='read-only'):method([],0)


def setup_run(tmp_path,monkeypatch):
    monkeypatch.setattr(p,'SEEDS',[900001])
    paths={k:tmp_path/(k+'.pt') for k in p.STATES}
    for k,path in paths.items():path.write_bytes(k.encode())
    archive=tmp_path/'archive.zip'
    with zipfile.ZipFile(archive,'w'):pass
    identity={'checkpoint_sha256':p.fit.sha256_file(paths['original']),
        'baseline_checkpoint_sha256':p.fit.sha256_file(paths['varying015']),
        'strong_checkpoint_sha256':p.fit.sha256_file(paths['varying060']),
        'fit_signature':'formal','baseline_signature':'varying','noise_signature':'noise',
        'state_fingerprints':{s:{'state':s} for s in p.STATES},'job':{'rows':[{}]*8},'policy':p.POLICY}
    references={state:{f'{strength}__s900001__{k}':v for strength in p.STRENGTHS for k,v in arrays(seed=900001).items()} for state in p.STATES}
    seen=[]
    class Runtime:
        interrupt=None; fail=False; drift=False
        def __init__(self,*a):self.state='original';seen.append('init')
        def load_baseline(self,*a):self.state='varying015'
        def load_arm(self,*a):self.state='varying060'
        def fingerprints(self):return {'state':self.state}
        def probe(self,images,job,expected):
            assert len(images)==8 and 'train_mixed_maps' in expected
            seen.append(job['name'])
            if job['name']==self.interrupt:raise KeyboardInterrupt()
            if self.fail:raise RuntimeError('budget')
            gram=np.eye(7).tolist()
            result={'job':job,'optimizer_updates':0,'state_unchanged':True,
                'state_before_sha256':'same','state_after_sha256':'same',
                'loss_terms':{},'groups':{k:{'gram':gram,'summary':p.summarize_gram(gram)} for k in p.GROUPS}}
            if self.drift:self.state='changed'
            return result,{'fixture':np.zeros(2,np.float32)}
        def release(self):seen.append('release')
    monkeypatch.setattr(p.fit,'checked_image',lambda *a:Image.new('RGB',(256,256)))
    return (tmp_path/'run',archive,paths,identity,references,tmp_path/'handoff.zip'),Runtime,seen


def test_readonly_run_resume_export(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert report['completed_jobs']==6 and report['optimizer_updates']==0
    assert report['attempted_jobs_including_replay']==6
    assert seen.count('init')==3
    assert runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)==report
    assert seen.count('init')==3
    for k,path in args[2].items():assert path.read_bytes()==k.encode()
    with zipfile.ZipFile(args[-1]) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert len(z.namelist())==len(manifest)+1
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        assert not any(n.endswith(('.pt','.jpg','.png')) for n in z.namelist())


def test_interrupt_reuses_finished_jobs(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    rt.interrupt=p.jobs()[3]['name']
    with pytest.raises(KeyboardInterrupt):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    rt.interrupt=None
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen.count(p.jobs()[0]['name'])==1 and seen.count(p.jobs()[3]['name'])==2
    assert report['attempted_jobs_including_replay']==7


@pytest.mark.parametrize('case',['source','resource','state','budget','hard-stop'])
def test_fail_closed(tmp_path,monkeypatch,case):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    if case=='source':args[2]['varying060'].write_bytes(b'changed')
    if case=='resource':rt.fail=True
    if case=='state':rt.drift=True
    if case=='budget':monkeypatch.setitem(p.POLICY,'maximum_attempted_jobs',1)
    if case=='hard-stop':
        args[0].mkdir();runner.publish(args[0],'identity',p.fit.digest(args[3]),args[3])
        (args[0]/'attempt-incomplete').mkdir()
    with pytest.raises((ValueError,RuntimeError)):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert not args[-1].exists()
    if case=='resource':
        with pytest.raises(ValueError,match='review'):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)


def test_tampered_completed_arrays_rejected(tmp_path,monkeypatch):
    args,rt,_=setup_run(tmp_path,monkeypatch)
    runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    (args[0]/p.jobs()[0]['name']/'arrays.npz').write_bytes(b'tampered')
    with pytest.raises(ValueError):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)


def test_cli_is_importable_without_torch():
    r=subprocess.run([sys.executable,str(Path(runner.__file__)),'--help'],capture_output=True,text=True)
    assert r.returncode==0 and '--archive' in r.stdout
    r=subprocess.run([sys.executable,'-c',"from scripts import run_mpi_supersimplenet_gradients; import sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_prepare_pins_reviewed_noise_model_and_references(tmp_path,monkeypatch):
    project=runner.ROOT
    handoffs=tmp_path/'artifacts/handoff';handoffs.mkdir(parents=True)
    root=tmp_path/'noise';root.mkdir()
    prior={'name':p.old.NAME,'sources_lf_sha256':{},'fit_signature':'original'}
    signature=p.fit.digest(prior)
    runner.publish(root,'identity',signature,prior)
    b=io.BytesIO();np.savez_compressed(b,reference=np.zeros(1,np.float32))
    data=b.getvalue(); progress={'fingerprints':{'head':'fixture'}}
    with p.fit.phase_stage(root,'std060-step-0500') as stage:
        (stage/'model.pt').write_bytes(b'protected')
        (stage/'arrays.npz').write_bytes(data)
        p.fit.write_json(stage/'progress.json',progress)
        receipt=p.fit.seal_phase(stage,'std060-step-0500',signature,{},p.old.FILES)
    payload={'plan.json':p.fit.canonical(prior),'std060-step-0500/receipt.json':p.fit.canonical(receipt)}
    for phase in ('std060-step-0000','baseline','std060-step-0500'):
        payload[phase+'/arrays.npz']=data;payload[phase+'/progress.json']=p.fit.canonical(progress)
    payload['handoff-manifest.json']=p.fit.canonical({n:hashlib.sha256(v).hexdigest() for n,v in payload.items()})
    path=handoffs/(p.old.NAME+'.zip')
    with zipfile.ZipFile(path,'w') as z:
        for n,v in payload.items():z.writestr(n,v)
    monkeypatch.setattr(p,'NOISE_SHA',p.fit.sha256_file(path))
    monkeypatch.setattr(p.old,'prepare',lambda *a:({},prior,{}))
    # Source paths are from the actual checkout; handoff paths use the temporary project.
    class Project:
        def __truediv__(self,name):return tmp_path/name if name=='artifacts/handoff' else project/name
    identity,references=p.prepare(Project(),tmp_path/'formal',tmp_path/'fixed',tmp_path/'varying',root)
    assert identity['strong_checkpoint_sha256']==receipt['files']['model.pt']
    assert set(references)==set(p.STATES) and len(identity['sources_lf_sha256'])==5
    (root/'std060-step-0500/model.pt').write_bytes(b'changed')
    with pytest.raises(ValueError):p.prepare(Project(),tmp_path/'formal',tmp_path/'fixed',tmp_path/'varying',root)
