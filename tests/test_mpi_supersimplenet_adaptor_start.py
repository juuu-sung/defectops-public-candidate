"""CPU isolation/resume tests; lab CUDA learning remains unverified here."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import zipfile

from PIL import Image
import pytest

from scripts import mpi_supersimplenet_adaptor_start_protocol as p
from scripts import run_mpi_supersimplenet_adaptor_start as runner
from scripts.mpi_supersimplenet_adaptor_start_runtime import Runtime as RealRuntime
from scripts.mpi_supersimplenet_matched_eval_runtime import PUBLIC_VALIDATION
from test_mpi_supersimplenet_varying import arrays,Monitor


def test_fixed_policy_single_intervention_no_time_cap():
    for key in ('updates','snapshots','evaluation_seeds','heldout_seeds','training_seed_start',
                'adaptor_lr','adaptor_weight_decay','segdec_lr','segdec_weight_decay','trainable','synthetic'):
        assert p.POLICY[key]==p.v.POLICY[key]
    assert p.POLICY['updates']==500 and p.POLICY['resource_plan']['comparison_seconds'] is None
    assert p.POLICY['automatic_model_selection'] is False


@pytest.fixture
def setup_run(tmp_path,monkeypatch):
    for policy in (p.POLICY,p.v.POLICY):
        monkeypatch.setitem(policy,'snapshots',[0,1,3]);monkeypatch.setitem(policy,'updates',3)
    formal=tmp_path/'formal';checkpoints={}
    fp0={'backbone':'b','adaptor':'initial','segmentation':'s0','classification':'c0'}
    fpf={'backbone':'b','adaptor':'trained','segmentation':'sf','classification':'cf'}
    for step,fp in ((0,fp0),(28800,fpf)):
        f=formal/f'step-{step:06d}'/'model.pt';f.parent.mkdir(parents=True);f.write_bytes(f'original{step}'.encode())
        checkpoints[str(step)]={'sha256':p.fit.sha256_file(f),'fingerprints':fp}
    (formal/'calibrated').mkdir();(formal/'calibrated/value.json').write_text(json.dumps({'contract':{'threshold':3}}))
    rows=[{'member':f'extracted/grid{i}/train/good/1073.png'} for i in range(8)]
    archive=tmp_path/'archive.zip'
    with zipfile.ZipFile(archive,'w') as z:
        for row in rows:z.writestr(row['member'],b'normal')
        z.writestr('test/defect/never.png',b'forbidden')
    identity={'checkpoints':checkpoints,'fit_signature':'formal','contract_sha256':p.fit.digest({'threshold':3}),
              'rows':rows,'policy':copy.deepcopy(p.POLICY)}
    seen=[]
    class Runtime:
        interrupt_at=None
        failure=None
        drift=False
        bad_training=False
        bad_eval=False
        bad_initial=False
        def __init__(self,*args):self.index=0;seen.append('init')
        def initialize(self,path,identity,arm):
            self.arm=arm
            self.fp={**fp0,'adaptor':'trained' if arm==p.ARMS[0] else 'initial'}
            if self.bad_initial:self.fp['segmentation']='wrong'
            return {'arm':arm,'fingerprints':self.fingerprints(),'optimizer_empty':True}
        def fingerprints(self):return dict(self.fp) if self.index==0 else {**self.fp,'adaptor':self.fp['adaptor']+str(self.index)}
        def step(self,images,index):
            seen.append((self.arm,index+1))
            if self.interrupt_at==(self.arm,index+1):raise KeyboardInterrupt()
            if self.failure:raise RuntimeError(self.failure)
            self.index=index+1
            return {'step':index+1,'seed':p.v.training_seed(index),'loss':1/(index+1),
                    'target_sha256':'bad' if self.bad_training and self.arm==p.ARMS[1] else str(index),
                    'positive_samples':8,'positive_pixels':10}
        def evaluate_seed(self,images,seed):
            shifted=self.bad_eval and self.arm==p.ARMS[1]
            return arrays(self.index,seed+int(shifted)),{'state_unchanged':True,
                'state_before_sha256':'same','state_after_sha256':'same',
                'public_batch_comparison':{'validation':PUBLIC_VALIDATION,'seed':seed,
                    'public_batch':8,'separated_batch':8,'synthetic_batch':16,'absolute_tolerance':1e-4,
                    'same_batch_map_max_abs':0.,'same_batch_score_max_abs':0.,
                    'cross_batch_map_max_abs':.0002,'cross_batch_score_max_abs':0.}}
        def feature_statistics(self,images):return {'fixture':True}
        def save(self,path,signature,step):path.write_text(json.dumps([signature,self.arm,step]))
        def load(self,path,signature,step):
            assert json.loads(path.read_bytes())==[signature,self.arm,step]
            self.index=step+int(self.drift);seen.append((self.arm,'load',step))
        def release(self):seen.append('release')
    monkeypatch.setattr(p.fit,'checked_image',lambda *args:Image.new('RGB',(256,256)))
    args=(tmp_path/'run',archive,formal,identity,tmp_path/'handoff.zip')
    return args,Runtime,seen


def run(setup):
    args,rt,seen=setup
    return runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)


def test_two_arms_preserve_reload_export_and_noop(setup_run):
    args,rt,seen=setup_run;report=run(setup_run)
    assert report['completed_updates_total']==6 and report['completed_updates_per_arm']==3
    assert report['all_synthetic_targets_matched'] and all(d==0 for d in report['final_reload_max_abs_difference'].values())
    assert report['elapsed_time_cap_seconds'] is None
    assert run(setup_run)==report and seen.count('init')==2
    with zipfile.ZipFile(args[-1]) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert all(hashlib.sha256(z.read(n)).hexdigest()==sha for n,sha in manifest.items())
        assert not any(n.endswith(('.pt','.png','.jpg')) for n in z.namelist())
    assert (args[2]/'step-000000/model.pt').read_bytes()==b'original0'


def test_resume_second_arm_reloads_and_preserves_first(setup_run):
    args,rt,seen=setup_run;rt.interrupt_at=(p.ARMS[1],3)
    with pytest.raises(KeyboardInterrupt):run(setup_run)
    rt.interrupt_at=None
    report=run(setup_run)
    assert report['attempted_updates_including_replayed_work']==8
    assert seen.count((p.ARMS[0],1))==1 and seen.count((p.ARMS[1],2))==2


@pytest.mark.parametrize('case',['source','training','eval','initial','reload','resource','budget','non-normal','cache'])
def test_fail_closed(setup_run,monkeypatch,case):
    args,rt,seen=setup_run
    if case=='source':(args[2]/'step-000000/model.pt').write_bytes(b'changed')
    if case=='training':rt.bad_training=True
    if case=='eval':rt.bad_eval=True
    if case=='initial':rt.bad_initial=True
    if case=='reload':rt.drift=True
    if case=='resource':rt.failure='memory cap'
    if case=='budget':monkeypatch.setitem(p.POLICY,'maximum_attempted_updates',1)
    if case=='non-normal':args[3]['rows'][0]['member']='test/defect/never.png'
    if case=='cache':
        run(setup_run)
        (args[0]/p.ARMS[0]/'step-0001/arrays.npz').write_bytes(b'changed')
    with pytest.raises((RuntimeError,ValueError)):run(setup_run)
    if case=='resource':
        with pytest.raises(ValueError,match='review'):run(setup_run)


def test_initialization_detects_other_module_changes():
    zero={'backbone':'b','adaptor':'a0','segmentation':'s0','classification':'c0'}
    identity={'checkpoints':{'0':{'fingerprints':zero},'28800':{'fingerprints':{**zero,'adaptor':'af'}}}}
    for arm in p.ARMS:
        fp={**zero,'adaptor':'af' if arm==p.ARMS[0] else 'a0'}
        record={'arm':arm,'fingerprints':fp,'optimizer_empty':True}
        p.validate_initialization(record,identity,arm)
        record['fingerprints']['segmentation']='wrong'
        with pytest.raises(ValueError):p.validate_initialization(record,identity,arm)


def test_runtime_guard_and_cli_without_torch():
    rt=object.__new__(RealRuntime)
    with pytest.raises(ValueError,match='Initialize'):rt.step([],0)
    result=subprocess.run([sys.executable,str(Path(runner.__file__)),'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--archive' in result.stdout


@pytest.mark.parametrize('arm',p.ARMS)
def test_real_initializer_copies_only_registered_adaptor_with_fresh_optimizer(arm):
    # Exercise the actual initializer without pretending this is a CUDA test.
    class Tensor:
        def __init__(self,value):self.value=value
        def detach(self):return self
        def cpu(self):return self
        def clone(self):return Tensor(self.value)
    class Module:
        def __init__(self,value):self.value=value;self.trainable=True
        def state_dict(self):return {'value':Tensor(self.value)}
        def load_state_dict(self,state,strict):assert strict;self.value=state['value'].value
        def requires_grad_(self,flag):self.trainable=flag;return self
    class Model:
        def __init__(self):
            self.adaptor=Module('af');self.segdec=Module('hf');self.backbone=Module('b')
        def load_state_dict(self,state,strict):
            assert strict
            for name,value in state.items():getattr(self,name).value=value
        def requires_grad_(self,flag):
            for part in (self.adaptor,self.segdec,self.backbone):part.requires_grad_(flag)
        def eval(self):self.training=False
    rt=object.__new__(RealRuntime);rt.model=Model();rt.optimizer=SimpleNamespace(state={'old':'moments'})
    rt.new_optimizer=lambda:SimpleNamespace(state={})
    rt.fingerprints=lambda:{'backbone':rt.model.backbone.value,'adaptor':rt.model.adaptor.value,
                           'segmentation':rt.model.segdec.value,'classification':rt.model.segdec.value}
    final=rt.fingerprints();zero={**final,'adaptor':'a0','segmentation':'h0','classification':'h0'}
    identity={'fit_signature':'original','checkpoints':{'0':{'fingerprints':zero},'28800':{'fingerprints':final}}}
    payload={'purpose':p.parity.e.p.NAME,'identity':'original','step':0,'optimizer':{'state':{}},
             'model':{'adaptor':'a0','segdec':'h0','backbone':'b'}}
    def load(path,map_location,weights_only):
        assert path=='preserved' and map_location=='cpu' and weights_only
        return copy.deepcopy(payload)
    rt.torch=SimpleNamespace(load=load)
    record=rt.initialize('preserved',identity,arm)
    assert record['fingerprints']=={**zero,'adaptor':'af' if arm==p.ARMS[0] else 'a0'}
    assert rt.optimizer.state=={} and not rt.model.backbone.trainable
    assert rt.model.adaptor.trainable and rt.model.segdec.trainable and not rt.model.training
    assert payload['model']['adaptor']=='a0'
    with pytest.raises(ValueError,match='repeated'):rt.initialize('preserved',identity,arm)
