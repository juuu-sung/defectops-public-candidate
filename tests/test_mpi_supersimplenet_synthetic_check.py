"""CPU contracts, state restoration, native signed outputs and resumable orchestration."""
import ast
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_synthetic_protocol as q
from scripts import mpi_supersimplenet_synthetic_runtime as runtime
from scripts import run_mpi_supersimplenet_synthetic_check as runner


def arrays():
    masks=np.zeros((16,64,64),dtype=bool); masks[8:,8:16,8:16]=True
    labels=masks.reshape(16,-1).any(1)
    result={"masks":masks,"labels":labels}
    for mode in q.POLICY["modes"]:
        result[mode+"_maps"]=np.where(masks,2.,-1.).astype(np.float32) if mode!="eval_clean" else np.full(masks.shape,-1.,np.float32)
        result[mode+"_scores"]=np.where(labels,2.,-1.).astype(np.float32) if mode!="eval_clean" else np.full(16,-1.,np.float32)
    return result


def test_signed_metrics_and_paired_response():
    result=q.summarize_arrays(arrays())
    assert result["positive_samples"]==8
    assert result["modes"]["train_mixed"]["pixel_auroc"]==1
    assert result["modes"]["eval_mixed"]["classification_auroc"]==1
    assert result["eval_synthetic_delta_mean"]==3
    assert "pixel_auroc" not in result["modes"]["eval_clean"]


@pytest.mark.parametrize("bad",["shape","nan","dtype","labels","extra"])
def test_invalid_arrays_fail_closed(bad):
    a=arrays()
    if bad=="shape": a["masks"]=a["masks"][:8]
    if bad=="nan": a["eval_mixed_maps"][0,0,0]=np.nan
    if bad=="dtype": a["train_mixed_scores"]=a["train_mixed_scores"].astype(np.float64)
    if bad=="labels": a["labels"][0]=True
    if bad=="extra": a["secret_features"]=np.zeros(1)
    with pytest.raises(ValueError): q.summarize_arrays(a)


def test_empty_synthetic_class_is_reported_not_rerolled():
    a=arrays(); a["masks"][:]=False; a["labels"][:]=False
    result=q.summarize_arrays(a)
    assert result["modes"]["eval_mixed"]["classification_auroc"] is None
    assert result["eval_synthetic_delta_mean"] is None


def test_jobs_are_normal_only_and_deterministic():
    plan={"cohorts":{}}
    for group in q.POLICY["groups"]:
        ids=[f"{i:04d}" for i in range(12)]
        plan["cohorts"][group]={"photo_ids":ids,"rows":[{"photo_id":pid,"region":r} for pid in ids for r in q.fit.REGIONS]}
    jobs=q.jobs(plan)
    assert len(jobs)==48 and jobs==q.jobs(plan)
    assert len({j["name"] for j in jobs})==48
    assert {j["seed"] for j in jobs}=={42,43}
    assert all(len(j["rows"])==8 for j in jobs)
    assert set(j["group"] for j in jobs)=={"training","development"}
    plan["cohorts"]["training"]["rows"][0]["region"]="changed"
    with pytest.raises(ValueError): q.jobs(plan)


class Buffer:
    def __init__(self,value): self.value=value
    def detach(self): return self
    def clone(self): return Buffer(self.value)
    def copy_(self,other): self.value=other.value


@pytest.mark.parametrize("interrupted",[False,True])
def test_modes_and_buffers_restore_even_on_exception(interrupted):
    class Model:
        training=False
        child=type("Child",(),{"training":False})()
        buffer=Buffer(3)
        def modules(self): return [self,self.child]
        def named_buffers(self): return [("running_mean",self.buffer)]
    model=Model()
    try:
        with runtime.preserved_buffers_and_modes(model):
            model.training=True; model.child.training=True; model.buffer.value=99
            if interrupted: raise KeyboardInterrupt()
    except KeyboardInterrupt: pass
    assert model.buffer.value==3 and not model.training and not model.child.training


class Monitor:
    def __init__(self,*a): self.runtime=None
    def __enter__(self): return self
    def __exit__(self,*a): pass
    def check(self): pass
    def snapshot(self): return {"fixture":True}


def setup_run(tmp_path, monkeypatch, fail=False):
    fit_root=tmp_path/"fit"; (fit_root/"step-028800").mkdir(parents=True)
    checkpoint=fit_root/"step-028800/model.pt"; checkpoint.write_bytes(b"unchanged frozen fixture")
    archive=tmp_path/"source.zip"
    with zipfile.ZipFile(archive,"w"): pass
    job={"name":"training-batch00-seed42","group":"training","row_start":0,"seed":42,"rows":[{}]*8}
    identity={"checkpoint_sha256":q.fit.sha256_file(checkpoint),"fit_signature":"fixture","jobs":[job]}
    records={"report.json":{"final_fingerprints":{"fixed":True}}}
    seen=[]
    class Runtime:
        def __init__(self,*a): seen.append("init")
        def fingerprints(self): return {"fixed":True}
        def predict(self,image): return np.zeros(65537,np.float32)
        def trial(self,images,seed):
            seen.append("trial")
            if fail: raise KeyboardInterrupt()
            return arrays(),{"state_unchanged":True}
        def release(self): seen.append("release")
    monkeypatch.setattr(q.fit,"checked_image",lambda *a:Image.new("RGB",(256,256)))
    args=(tmp_path/"run",archive,fit_root,identity,records,tmp_path/"out.zip")
    return args,Runtime,seen


def test_execute_export_resume_and_receipt_tamper(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert report["optimizer_updates"]==0 and report["test_images_decoded"]==0
    assert runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)==report
    assert seen.count("init")==1 and seen.count("trial")==1
    with zipfile.ZipFile(args[-1]) as z:
        m=json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist())==set(m)|{"handoff-manifest.json"}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in m.items())
        assert not any(n.endswith(".pt") for n in z.namelist())
    trial=args[0]/"trials/training-batch00-seed42/arrays.npz"
    trial.write_bytes(b"changed")
    with pytest.raises(ValueError): runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)


def test_interrupted_trial_is_preserved_and_retry_uses_same_job(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch,True)
    with pytest.raises(KeyboardInterrupt):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    attempts=list(args[0].glob("attempt-*/finished.json"))
    assert len(attempts)==1
    record=json.loads(attempts[0].read_bytes())
    assert record["attempted_trials"]==1 and record["status"]=="interrupted"
    assert not args[-1].exists() and seen[-1]=="release"
    monkeypatch.setattr(rt,"trial",lambda *a:(arrays(),{"state_unchanged":True}))
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert report["completed_trials"]==1 and attempts[0].exists()


def test_forced_stop_and_resource_failure_cannot_be_silently_retried(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    def fail(*a): raise RuntimeError("resource cap exceeded")
    monkeypatch.setattr(rt,"trial",fail)
    with pytest.raises(RuntimeError):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    with pytest.raises(ValueError,match="review"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen.count("init")==1


def test_checkpoint_changed_before_loading_fails(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    (args[2]/"step-028800/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Checkpoint changed"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen==[]


def test_cli_and_runtime_no_training_operations():
    result=subprocess.run([sys.executable,str(Path(runner.__file__)),"--help"],capture_output=True,text=True)
    assert result.returncode==0 and "--archive" in result.stdout
    tree=ast.parse(Path(runtime.__file__).read_text())
    calls=[n.func.attr for n in ast.walk(tree) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)]
    assert not set(calls)&{"backward","step","save"}
    result=subprocess.run([sys.executable,"-c","from scripts import run_mpi_supersimplenet_synthetic_check; import sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_main_uses_expected_normal_only_paths(tmp_path,monkeypatch):
    archive=tmp_path/"extracted.zip"; archive.write_bytes(b"fixture")
    fit_root=tmp_path/"artifacts"/q.p.NAME; fit_root.mkdir(parents=True)
    monkeypatch.setattr(q.fit,"EXPECTED_ARCHIVE_SHA256",q.fit.sha256_file(archive))
    env={k:"fixed" for k in ("torch","torchvision","anomalib","timm","supersimplenet_sources")}
    def prepare(project,handoffs,root):
        assert project==tmp_path and handoffs==tmp_path/"artifacts/handoff" and root==fit_root
        return {},{"environment":env.copy()}
    monkeypatch.setattr(q,"prepare",prepare)
    monkeypatch.setattr(runner,"environment",lambda:env.copy())
    seen=[]
    monkeypatch.setattr(runner,"execute",lambda *a:seen.append(a) or {"fixture":True})
    assert runner.main(["--project",str(tmp_path),"--archive",str(archive)])==0
    assert seen[0][0]==tmp_path/"artifacts/analyses"/q.NAME
    assert seen[0][-1]==tmp_path/"artifacts/handoff"/(q.NAME+".zip")
    assert not (fit_root/"step-028800").exists()
