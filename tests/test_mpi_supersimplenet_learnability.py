"""CPU control-plane fixtures; not an assertion that CUDA learning succeeds."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_supersimplenet_learnability_protocol as p
from scripts import run_mpi_supersimplenet_learnability as runner


def arrays(step=0):
    masks = np.zeros((16,64,64),bool); masks[8:,8:16,8:16] = True
    labels = masks.reshape(16,-1).any(1)
    result = {"masks":masks,"labels":labels}
    for mode in p.q.POLICY["modes"]:
        result[mode+"_maps"] = np.where(masks,1.+step,-1.).astype(np.float32)
        result[mode+"_scores"] = np.where(labels,1.+step,-1.).astype(np.float32)
    return result


class Monitor:
    def __init__(self,*args): self.runtime = None
    def __enter__(self): return self
    def __exit__(self,*args): pass
    def check(self): pass
    def snapshot(self): return {"fixture":True}


def setup_run(tmp_path,monkeypatch):
    monkeypatch.setitem(p.POLICY,"snapshots",[0,1,3])
    monkeypatch.setitem(p.POLICY,"updates",3)
    fit_root = tmp_path/"fit"; (fit_root/"step-028800").mkdir(parents=True)
    original = fit_root/"step-028800/model.pt"; original.write_bytes(b"frozen original")
    archive = tmp_path/"archive.zip"
    with zipfile.ZipFile(archive,"w"): pass
    identity = {"checkpoint_sha256":p.fit.sha256_file(original),"fit_signature":"fixture",
                "job":{"rows":[{}]*8},"policy":p.POLICY.copy()}
    records = {"report.json":{"final_fingerprints":{"backbone":"same","adaptor":"step0"}}}
    seen = []
    class Runtime:
        interrupt_at = None
        failure = None
        def __init__(self,*args): self.index=0; seen.append("init")
        def fingerprints(self): return {"backbone":"same","adaptor":f"step{self.index}"}
        def step(self,images,index,target):
            seen.append(index+1)
            assert len(images)==8 and target==p.target_digest(arrays())
            if self.failure: raise RuntimeError(self.failure)
            if index+1==self.interrupt_at: raise KeyboardInterrupt()
            self.index=index+1
            return {"step":self.index,"loss":1/self.index}
        def snapshot(self,images,measure=False):
            return arrays(self.index),{"state_unchanged":True,"loss_components":{"fixture":self.index}},{}
        def save(self,path,signature,step):
            assert self.index==step
            path.write_text(json.dumps({"signature":signature,"step":step}))
        def load(self,path,signature,step):
            record=json.loads(path.read_bytes())
            assert record=={"signature":signature,"step":step}
            self.index=step; seen.append(f"load{step}")
        def release(self): seen.append("release")
    monkeypatch.setattr(p.fit,"checked_image",lambda *args:Image.new("RGB",(256,256)))
    args = (tmp_path/"run",archive,fit_root,identity,records,tmp_path/"out.zip")
    return args,Runtime,seen


def test_success_replay_handoff_and_original_preserved(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert report["completed_updates"]==3 and report["test_images_decoded"]==0
    assert report["final_reload_max_abs_difference"]==0
    assert report["trainable_components_changed"]=={"adaptor":True}
    assert (args[2]/"step-028800/model.pt").read_bytes()==b"frozen original"
    assert runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)==report
    assert seen.count("init")==1 and seen.count(1)==1
    with zipfile.ZipFile(args[-1]) as z:
        manifest=json.loads(z.read("handoff-manifest.json"))
        assert set(z.namelist())==set(manifest)|{"handoff-manifest.json"}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        assert not any(n.endswith((".pt",".png",".jpg")) for n in z.namelist())
    (args[0]/"step-0001/arrays.npz").write_bytes(b"tamper")
    with pytest.raises(ValueError):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)


def test_interruption_resumes_verified_checkpoint_and_counts_replayed_work(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    rt.interrupt_at=3
    with pytest.raises(KeyboardInterrupt):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert (args[0]/"step-0001").exists() and not (args[0]/"step-0003").exists()
    rt.interrupt_at=None
    report=runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert "load1" in seen and seen.count(1)==1 and seen.count(2)==2
    assert report["attempted_updates_including_replayed_work"]==5
    assert len(list(args[0].glob("attempt-*/finished.json")))==2


@pytest.mark.parametrize("failure",["CUDA out of memory","resource budget exceeded"])
def test_failure_blocks_automatic_retry(tmp_path,monkeypatch,failure):
    args,rt,seen=setup_run(tmp_path,monkeypatch); rt.failure=failure
    with pytest.raises(RuntimeError):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert (args[0]/"step-0000").exists()
    with pytest.raises(ValueError,match="review"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen.count("init")==1 and not args[-1].exists()


def test_hard_stop_is_not_silently_rerolled(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    with p.fit.locked_run(args[0]):
        runner.publish(args[0],"identity",p.fit.digest(args[3]),args[3])
        (args[0]/"attempt-incomplete").mkdir()
    with pytest.raises(ValueError,match="Unfinished"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen==[]


def test_original_checkpoint_change_prevents_gpu_loading(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    (args[2]/"step-028800/model.pt").write_bytes(b"changed")
    with pytest.raises(ValueError,match="Original checkpoint"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert seen==[]


def test_fixed_targets_and_both_classes_required():
    a=arrays(); before=p.target_digest(a)
    a["masks"][8,8,8]=False
    assert p.target_digest(a)!=before
    a["masks"][:]=False; a["labels"][:]=False
    with pytest.raises(ValueError,match="both classes"):p.target_digest(a)


def test_reload_drift_blocks_handoff(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    original=rt.load
    def changed_load(self,*params): original(self,*params); self.index+=1
    monkeypatch.setattr(rt,"load",changed_load)
    with pytest.raises(ValueError,match="Reload"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert not args[-1].exists()


def test_attempt_budget_does_not_extend_training(tmp_path,monkeypatch):
    args,rt,seen=setup_run(tmp_path,monkeypatch)
    monkeypatch.setitem(p.POLICY,"maximum_attempted_updates",2)
    with pytest.raises(RuntimeError,match="Attempted-update budget"):runner.execute(*args,runtime_factory=rt,monitor_factory=Monitor)
    assert 3 not in seen


def test_cli_import_does_not_require_torch():
    result=subprocess.run([sys.executable,str(Path(runner.__file__)),"--help"],capture_output=True,text=True)
    assert result.returncode==0 and "--archive" in result.stdout
    result=subprocess.run([sys.executable,"-c","from scripts import run_mpi_supersimplenet_learnability; import sys; assert 'torch' not in sys.modules"],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_real_policy_is_bounded_normal_only():
    assert p.POLICY["updates"]==500 and p.POLICY["snapshots"]==[0,50,100,200,300,400,500]
    assert p.POLICY["test_images_decoded"]==0 and not p.POLICY["threshold_changed"]
    assert p.POLICY["trainable"]==["adaptor","segdec"]
    assert p.POLICY["resource_plan"]["comparison_seconds"]==1800


def test_prepare_selects_only_first_training_photo_and_eight_regions(tmp_path,monkeypatch):
    cohorts={}
    for group in ("training","development"):
        ids=[f"{i:04d}" for i in range(8)]
        cohorts[group]={"photo_ids":ids,"rows":[{"photo_id":pid,"region":r}
            for pid in ids for r in p.fit.REGIONS]}
    records={"plan.json":{"cohorts":cohorts,"environment":{}},
             "checkpoint-receipts.json":[{"files":{"model.pt":"checkpoint"}}],
             "normal-results.json":{"contract":{"frozen":True}}}
    monkeypatch.setattr(p.e,"read_fit",lambda *a:records)
    actual,identity=p.prepare(runner.ROOT,tmp_path,tmp_path)
    assert actual is records
    assert {row["photo_id"] for row in identity["job"]["rows"]}=={"0000"}
    assert [r["region"] for r in identity["job"]["rows"]]==p.fit.REGIONS[:8]
    assert identity["job"]["group"]=="training" and len(identity["job"]["rows"])==8
    assert all(len(d)==64 for d in identity["sources_lf_sha256"].values())


def test_main_uses_new_output_not_original_fit(tmp_path,monkeypatch):
    archive=tmp_path/"archive.zip"; archive.write_bytes(b"fixture")
    fit_root=tmp_path/"artifacts"/p.q.p.NAME; fit_root.mkdir(parents=True)
    monkeypatch.setattr(p.fit,"EXPECTED_ARCHIVE_SHA256",p.fit.sha256_file(archive))
    env={k:"pinned" for k in ("torch","torchvision","anomalib","timm","supersimplenet_sources")}
    monkeypatch.setattr(p,"prepare",lambda *a:({}, {"environment":env.copy()}))
    monkeypatch.setattr(runner,"environment",lambda:env.copy())
    monkeypatch.setattr(runner.shutil,"disk_usage",lambda *a:type("Disk",(),{"free":20*1024**3})())
    seen=[]
    monkeypatch.setattr(runner,"execute",lambda *a:seen.append(a) or {"fixture":True})
    assert runner.main(["--project",str(tmp_path),"--archive",str(archive)])==0
    assert seen[0][0]==tmp_path/"artifacts/analyses"/p.NAME
    assert seen[0][2]==fit_root and seen[0][-1]==tmp_path/"artifacts/handoff"/(p.NAME+".zip")


def test_feature_statistics_against_numpy_without_torch():
    from scripts.mpi_supersimplenet_learnability_runtime import Runtime
    class Tensor:
        def __init__(self,a): self.a=np.asarray(a)
        @property
        def shape(self): return self.a.shape
        def numel(self): return self.a.size
        def __iter__(self): return (Tensor(a) for a in self.a)
        def detach(self): return self
        def double(self): return Tensor(self.a.astype(np.float64))
        def sum(self): return self.a.sum()
        def square(self): return Tensor(self.a**2)
    a=np.array([[-2,0,1],[3,5,-1]],np.float32)
    result=Runtime.tensor_stats(Tensor(a))
    assert result["mean"]==pytest.approx(a.mean())
    assert result["std"]==pytest.approx(a.std())
    assert result["rms"]==pytest.approx(np.sqrt(np.mean(a*a)))
