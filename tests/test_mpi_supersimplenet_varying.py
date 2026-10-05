"""CPU evidence/resume controls; no claim of CUDA learning validation."""
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

from scripts import mpi_supersimplenet_varying_protocol as p
from scripts import run_mpi_supersimplenet_varying as runner


def arrays(step=0, seed=42, memorize=False):
    masks=np.zeros((16,64,64),bool); shift=seed%10
    masks[8:,shift:shift+4,8:12]=True
    labels=masks.reshape(16,-1).any(1)
    result={"masks":masks,"labels":labels}
    for mode in p.q.POLICY["modes"]:
        positive = masks if mode!="eval_clean" or memorize else np.zeros_like(masks)
        result[mode+"_maps"]=np.where(positive,1.+step,-1.).astype(np.float32)
        positive = labels if mode!="eval_clean" or memorize else np.zeros_like(labels)
        result[mode+"_scores"]=np.where(positive,1.+step,-1.).astype(np.float32)
    return result


class Monitor:
    def __init__(self,*args): self.runtime=None
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def check(self): pass
    def snapshot(self): return {"fixture":True}


def setup_run(tmp_path,monkeypatch):
    monkeypatch.setitem(p.POLICY,"snapshots",[0,1,3]); monkeypatch.setitem(p.POLICY,"updates",3)
    fit_root=tmp_path/"fit"; (fit_root/"step-028800").mkdir(parents=True)
    original=fit_root/"step-028800/model.pt"; original.write_bytes(b"original")
    fixed_root=tmp_path/"fixed"; (fixed_root/"step-0500").mkdir(parents=True)
    fixed=fixed_root/"step-0500/model.pt"; fixed.write_bytes(b"fixed500")
    archive=tmp_path/"archive.zip"
    with zipfile.ZipFile(archive,"w"): pass
    identity={"checkpoint_sha256":p.fit.sha256_file(original),"fixed_control_checkpoint_sha256":p.fit.sha256_file(fixed),
        "fit_signature":"formal","fixed_control_signature":"fixed","fixed_control_fingerprints":{"backbone":"same","head":500},
        "job":{"rows":[{}]*8},"policy":p.POLICY.copy()}
    records={"report.json":{"final_fingerprints":{"backbone":"same","head":0}}}
    references={0:arrays(0),500:arrays(500)}; seen=[]
    class Runtime:
        interrupt_at=None; failure=None; drift=False; mutate_targets=False
        def __init__(self,*args): self.index=0; seen.append("init")
        def fingerprints(self): return {"backbone":"same","head":self.index}
        def step(self,images,index):
            assert self.index!=500 and len(images)==8
            seen.append(index+1)
            if self.failure: raise RuntimeError(self.failure)
            if index+1==self.interrupt_at: raise KeyboardInterrupt()
            self.index=index+1
            return {"step":index+1,"seed":p.training_seed(index),"loss":1/(index+1),"target_sha256":str(index)}
        def evaluate_seed(self,images,seed):
            seen.append((self.index,seed))
            a=arrays(self.index,seed+1 if self.mutate_targets and self.index==1 else seed)
            return a,{"state_unchanged":True,"state_before_sha256":"same","state_after_sha256":"same"}
        def save(self,path,signature,step): path.write_text(json.dumps([signature,step]))
        def load(self,path,signature,step):
            assert json.loads(path.read_bytes())==[signature,step]
            self.index=step+int(self.drift); seen.append(f"load{step}")
        def load_fixed_control(self,path,signature):
            assert path==fixed and signature=="fixed"
            self.index=500; seen.append("fixed-load")
        def release(self): seen.append("release")
    monkeypatch.setattr(p.fit,"checked_image",lambda *args:Image.new("RGB",(256,256)))
    args=(tmp_path/"run",archive,fit_root,fixed_root,identity,records,references,tmp_path/"out.zip")
    return args,Runtime,seen


def test_seed_schedule_is_disjoint_and_bounded():
    seeds=[p.training_seed(i) for i in range(500)]
    assert seeds==list(range(10000,10500))
    assert not set(seeds)&set(p.POLICY["evaluation_seeds"])
    assert len(p.POLICY["heldout_seeds"])==4 and 42 not in p.POLICY["heldout_seeds"]
    for i in (-1,500):
        with pytest.raises(ValueError):p.training_seed(i)


def test_counterfactual_distinguishes_location_memorization():
    memorized=p.summarize(arrays(memorize=True)); learned=p.summarize(arrays())
    assert memorized["metrics"]["modes"]["eval_mixed"]["pixel_auroc"]==1
    assert memorized["counterfactual"]["same_position_pixel_auroc"]==.5
    assert memorized["counterfactual"]["mean_map_change_at_synthetic_locations"]==0
    assert learned["counterfactual"]["same_position_pixel_auroc"]==1
    assert learned["counterfactual"]["mean_map_change_at_synthetic_locations"]==2


def test_empty_evaluation_class_is_undefined_not_rerolled():
    a=arrays(); a["masks"][:]=False; a["labels"][:]=False
    r=p.summarize(a)
    assert r["counterfactual"]["same_source_classification_auroc"] is None
    assert r["counterfactual"]["same_position_pixel_auroc"] is None


def test_aggregation_excludes_reference42():
    values={str(s):p.summarize(arrays(seed=s)) for s in p.POLICY["evaluation_seeds"]}
    values['42']['metrics']['modes']['eval_mixed']['classification_auroc']=0
    r=p.aggregate(values)
    assert r['eval_classification_auroc']=={'mean_over_defined_seeds':1.,'defined_seeds':4,'total_seeds':4}


def test_execute_control_training_reload_export_and_noop_resume(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert report['completed_updates']==3 and report['unique_training_target_hashes']==3
    assert report['test_images_decoded']==0 and report['final_reload_max_abs_difference']==0
    assert seen.count('fixed-load')==1 and seen.count('init')==2
    assert seen.index('fixed-load') < seen.index(1)
    assert runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)==report
    assert seen.count('init')==2
    assert (args[2]/'step-028800/model.pt').read_bytes()==b'original'
    assert (args[3]/'step-0500/model.pt').read_bytes()==b'fixed500'
    with zipfile.ZipFile(args[-1]) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert set(z.namelist())==set(manifest)|{'handoff-manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        assert not any(n.endswith(('.pt','.png','.jpg')) for n in z.namelist())
    (args[0]/'fixed-control/arrays.npz').write_bytes(b'changed')
    with pytest.raises(ValueError):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)


def test_interrupt_reuses_control_and_repeats_same_training_seeds(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch);rt.interrupt_at=3
    with pytest.raises(KeyboardInterrupt):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    rt.interrupt_at=None
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen.count('fixed-load')==1 and 'load1' in seen
    assert seen.count(1)==1 and seen.count(2)==2
    assert report['attempted_updates_including_replayed_work']==5


@pytest.mark.parametrize('case',['resource','hard-stop','source','targets','reload','budget'])
def test_fail_closed(tmp_path,monkeypatch,case):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    if case=='resource':rt.failure='resource cap'
    if case=='hard-stop':
        args[0].mkdir();runner.publish(args[0],'identity',p.fit.digest(args[4]),args[4])
        (args[0]/'attempt-incomplete').mkdir()
    if case=='source':(args[3]/'step-0500/model.pt').write_bytes(b'changed')
    if case=='targets':rt.mutate_targets=True
    if case=='reload':rt.drift=True
    if case=='budget':monkeypatch.setitem(p.POLICY,'maximum_attempted_updates',1)
    with pytest.raises((ValueError,RuntimeError)):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert not args[-1].exists()
    if case=='resource':
        with pytest.raises(ValueError,match='review'):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
        assert seen.count('init')==2


def test_cli_without_torch():
    r=subprocess.run([sys.executable,str(Path(runner.__file__)),'--help'],capture_output=True,text=True)
    assert r.returncode==0 and '--archive' in r.stdout
    r=subprocess.run([sys.executable,'-c',"from scripts import run_mpi_supersimplenet_varying; import sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert r.returncode==0,r.stderr


def test_control_runtime_refuses_updates_before_any_torch_operation():
    from scripts.mpi_supersimplenet_varying_runtime import Runtime
    control=object.__new__(Runtime)
    control.control_only=True
    with pytest.raises(ValueError,match='read-only'):control.step([],0)


def test_main_selects_original_fixed_and_new_output_paths(tmp_path,monkeypatch):
    archive=tmp_path/'archive.zip';archive.write_bytes(b'fixture')
    formal=tmp_path/'artifacts'/p.q.p.NAME;formal.mkdir(parents=True)
    fixed=tmp_path/'artifacts/analyses'/p.old.NAME;fixed.mkdir(parents=True)
    monkeypatch.setattr(p.fit,'EXPECTED_ARCHIVE_SHA256',p.fit.sha256_file(archive))
    env={k:'pinned' for k in ('torch','torchvision','anomalib','timm','supersimplenet_sources')}
    def prepare(project,handoffs,fit_root,fixed_root):
        assert fit_root==formal and fixed_root==fixed and handoffs==tmp_path/'artifacts/handoff'
        return {},{'environment':env.copy()},{}
    monkeypatch.setattr(p,'prepare',prepare)
    monkeypatch.setattr(runner,'environment',lambda:env.copy())
    monkeypatch.setattr(runner.shutil,'disk_usage',lambda *a:type('Disk',(),{'free':20*1024**3})())
    seen=[]
    monkeypatch.setattr(runner,'execute',lambda *a:seen.append(a) or {'fixture':True})
    assert runner.main(['--project',str(tmp_path),'--archive',str(archive)])==0
    assert seen[0][0]==tmp_path/'artifacts/analyses'/p.NAME
    assert seen[0][-1]==tmp_path/'artifacts/handoff'/(p.NAME+'.zip')


def test_policy_changes_only_synthetic_schedule_not_learning_conditions():
    for key in ('updates','snapshots','input_tiles','effective_batch','initialization','trainable',
                'backbone_frozen','adaptor_lr','adaptor_weight_decay','segdec_lr','segdec_weight_decay',
                'scheduler','amp','resource_plan'):
        assert p.POLICY[key]==p.old.POLICY[key]
    assert p.POLICY['test_images_decoded']==0 and not p.POLICY['automatic_pass_allowed']


def test_prepare_ties_fixed_control_to_reviewed_checkpoint(tmp_path,monkeypatch):
    project=runner.ROOT; handoffs=tmp_path/'handoff';handoffs.mkdir()
    root=tmp_path/'fixed';root.mkdir()
    original_identity={'policy':p.old.POLICY,'job':{'rows':[{'photo_id':'1073'}]*8},
        'fit_signature':'formal','checkpoint_sha256':'original','original_contract_sha256':'contract',
        'sources_lf_sha256':{},'environment':{}}
    previous={**original_identity,'name':p.old.NAME}; signature=p.fit.digest(previous)
    runner.publish(root,'identity',signature,previous)
    evidence={}
    with p.fit.phase_stage(root,'step-0500') as stage:
        (stage/'model.pt').write_bytes(b'fixed weights')
        np.savez_compressed(stage/'arrays.npz',**arrays(500))
        p.fit.write_json(stage/'progress.json',{'fingerprints':{'fixed':True}})
        receipt=p.fit.seal_phase(stage,'step-0500',signature,{},p.old.FILES)
    evidence['plan.json']=p.fit.canonical(previous)
    for n in ('arrays.npz','progress.json','receipt.json'):evidence['step-0500/'+n]=(root/'step-0500'/n).read_bytes()
    b=io.BytesIO();np.savez_compressed(b,**arrays());evidence['step-0000/arrays.npz']=b.getvalue()
    evidence['handoff-manifest.json']=p.fit.canonical({n:hashlib.sha256(v).hexdigest() for n,v in evidence.items()})
    path=handoffs/(p.old.NAME+'.zip')
    with zipfile.ZipFile(path,'w') as z:
        for n,v in evidence.items():z.writestr(n,v)
    monkeypatch.setattr(p,'FIXED_ZIP_SHA',p.fit.sha256_file(path))
    monkeypatch.setattr(p.old,'prepare',lambda *a:({},original_identity))
    _,identity,references=p.prepare(project,handoffs,tmp_path/'formal',root)
    assert identity['fixed_control_signature']==signature and identity['policy']==p.POLICY
    assert identity['fixed_control_checkpoint_sha256']==receipt['files']['model.pt']
    assert np.array_equal(references[0]['masks'],arrays()['masks'])
    (root/'step-0500/model.pt').write_bytes(b'tamper')
    with pytest.raises(ValueError):p.prepare(project,handoffs,tmp_path/'formal',root)
