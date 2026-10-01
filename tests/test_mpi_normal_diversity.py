import copy
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import mpi_normal_diversity_protocol as p
from scripts import prepare_mpi_normal_diversity as runner


def make_vectors(count):
    rng = np.random.default_rng(12)
    return np.stack([p.descriptor(rng.uniform(.1, 2, (12, 768)).astype(np.float32)) for _ in range(count)])


def test_selection_matches_manual_and_reorders_deterministically():
    ids = [f"{i:04}" for i in range(8)]
    x = make_vectors(8)
    actual = p.select(ids, x, 4)
    order = sorted(range(8), key=lambda i: p.rank(ids[i]))
    center = x[order].astype(float).mean(0)
    chosen = [min(ids, key=lambda pid: (float(np.sum((x[ids.index(pid)].astype(float)-center)**2)), p.rank(pid,p.NAME)))]
    while len(chosen) < 4:
        chosen.append(min((i for i in ids if i not in chosen), key=lambda pid: (
            -min(float(np.sum((x[ids.index(pid)].astype(float)-x[ids.index(j)].astype(float))**2)) for j in chosen), p.rank(pid,p.NAME))))
    assert actual == chosen
    assert p.select(ids[::-1], x[::-1], 4) == actual


def test_identical_vectors_tie_and_no_duplicate_selection():
    ids = ["0001", "0002", "0003"]
    x = np.repeat(make_vectors(1), 3, axis=0)
    assert p.select(ids,x,3) == sorted(ids,key=lambda pid:p.rank(pid,p.NAME))


@pytest.mark.parametrize("problem", ["nan", "zero", "shape", "dtype"])
def test_invalid_means(problem):
    x = np.ones((12,768),dtype=np.float32)
    if problem == "nan": x[0,0] = np.nan
    if problem == "zero": x[0] = 0
    if problem == "shape": x = x[:11]
    if problem == "dtype": x = x.astype(float)
    with pytest.raises(ValueError): p.descriptor(x)


def fingerprints(count=8):
    return [{"photo_id":f"{i:04}", "region":r, "member":f"extracted/{r}/train/good/{i:04}.png",
             "decoded_rgb_sha256":f"rgb-{i}-{r}"} for i in range(count) for r in p.fit.REGIONS]


def test_protected_id_rgb_and_full_duplicates_removed():
    rows = fingerprints()
    groups = {f"{i:04}":rows[i*12:(i+1)*12] for i in range(8)}
    protected = {"normal":[{"photo_id":"0000", "regions":copy.deepcopy(groups['0000'])}]}
    groups['0001'][0]['decoded_rgb_sha256'] = groups['0000'][0]['decoded_rgb_sha256']
    for a,b in zip(groups['0002'],groups['0003']): b['decoded_rgb_sha256'] = a['decoded_rgb_sha256']
    result = p.derive_pool(rows,protected, {groups['0004'][0]['decoded_rgb_sha256']},['0005'])
    assert result['exclusion_counts'] == {'protected_normal_id':1,'protected_or_test_exact_rgb':2,'identical_full_12_region_photo':1}
    assert set(result['ids']) >= {'0005','0006','0007'}
    assert len(result['ids']) == 4


@pytest.mark.parametrize("problem", ["partial", "missing", "path", "baseline", "protected_overlap"])
def test_pool_safety(problem):
    rows = fingerprints(3)
    protected = {}
    baseline = ['0000']
    if problem == 'partial': rows[12]['decoded_rgb_sha256']=rows[0]['decoded_rgb_sha256']
    if problem == 'missing': rows.pop()
    if problem == 'path': rows[0]['member']='extracted/test/evil.png'
    if problem == 'baseline': baseline=['9999']
    if problem == 'protected_overlap':
        photo={'photo_id':'0000','regions':copy.deepcopy(rows[:12])}
        protected={'a':[photo],'b':[photo]}
    with pytest.raises(ValueError): p.derive_pool(rows,protected,set(),baseline)


@pytest.fixture
def synthetic(tmp_path):
    policy=json.loads((Path(__file__).resolve().parents[1]/'configs/mpi-normal-diversity-v1.json').read_bytes())
    policy['selection']['photo_count']=2
    policy['selection']['preflight_photo_count']=2
    ids=sorted(['0000','0001','0002','0003'],key=p.rank)
    archive=tmp_path/'images.zip'
    rows=[]
    with zipfile.ZipFile(archive,'w') as z:
        for pid in ids:
            for i,r in enumerate(p.fit.REGIONS):
                pixels=np.full((256,256,3), [int(pid)+1,i,100],dtype=np.uint8)
                with Image.fromarray(pixels) as img:
                    b=io.BytesIO(); img.save(b,format='PNG')
                row={'photo_id':pid,'region':r,'member':f'extracted/{r}/train/good/{pid}.png',
                     'member_sha256':runner.bundle_tools.sha(b.getvalue()),'decoded_rgb_sha256':runner.bundle_tools.sha(pixels.tobytes())}
                rows.append(row); z.writestr(row['member'],b.getvalue())
        z.writestr('test/forbidden.png',b'not an image')
    pool={'ids':ids,'rows':rows,'excluded':[],'exclusion_counts':{}}
    env={'kind':'synthetic_cpu_test_not_gpu_evidence'}
    identity={'policy':policy,'baseline_ids':ids[:2], 'pool_rows_sha256':p.fit.digest(rows)}
    records={'fit.json':{'checkpoint_context':{}},'reload.json':{'receipt':{'runtime':env}}}
    state={'calls':0,'loads':0,'fail_at':None,'drift':False,'environment':env,'budget':False}
    class Runtime:
        def __init__(self,*a): pass
        def load(self,*a): state['loads']+=1
        def environment(self): return state['environment']
        def synchronize(self): pass
        def means(self,image,compare_reference=False):
            state['calls']+=1
            if state['calls']==state['fail_at']: raise RuntimeError('synthetic interruption')
            a,b,_=image.getpixel((0,0))
            x=np.random.default_rng(a*100+b).uniform(.1,2,768).astype(np.float32)
            if state['drift'] and not compare_reference: x[0]+=100
            return x,0.
        def release(self): pass
    class FakeMonitor:
        def __init__(self,*a): pass
        def __enter__(self): return self
        def check(self):
            if state['budget']: raise RuntimeError('budget exceeded')
        def snapshot(self): return {'kind':'fake_memory_not_measured'}
        def __exit__(self,*a): pass
    def execute():
        with zipfile.ZipFile(archive) as z:
            return runner.execute(tmp_path/'run',z,tmp_path/'model.pt',{},records,pool,identity,tmp_path/'handoff.zip',
                                  Runtime,FakeMonitor,lambda r:env)
    return tmp_path,archive,pool,identity,state,execute


def test_full_execution_no_test_reads_and_cached_rerun(synthetic,monkeypatch):
    root,archive,pool,identity,state,execute=synthetic
    original=zipfile.ZipFile.read
    def read(z,name,*a,**kw):
        member=name.filename if isinstance(name,zipfile.ZipInfo) else name
        if Path(z.filename)==archive: assert '/train/good/' in member
        return original(z,name,*a,**kw)
    monkeypatch.setattr(zipfile.ZipFile,'read',read)
    report=execute()
    assert report['selected_photos']==2 and report['model_scores_computed']==0
    assert report['training_performed'] is False
    calls=state['calls']; before=(root/'handoff.zip').read_bytes()
    assert execute()==report
    assert state['calls']==calls and state['loads']==1
    assert (root/'handoff.zip').read_bytes()==before
    with zipfile.ZipFile(root/'handoff.zip') as z:
        assert not any(n.endswith(('.png','.pt')) for n in z.namelist())
        assert 'descriptors.npy' in z.namelist()


def test_interrupted_resume(synthetic):
    root,_,pool,_,state,execute=synthetic
    state['fail_at']=51
    with pytest.raises(RuntimeError,match='interruption'): execute()
    first=root/'run/photos'/pool['ids'][0]/'descriptor.npy'
    before=(first.read_bytes(),first.stat().st_mtime_ns)
    state['fail_at']=None
    execute()
    assert before==(first.read_bytes(),first.stat().st_mtime_ns)
    assert len(list((root/'run').glob('attempt-*/finished.json')))==2


@pytest.mark.parametrize('problem',['drift','environment','budget'])
def test_failure_prevents_result(synthetic,problem):
    root,_,_,_,state,execute=synthetic
    if problem=='environment': state['environment']={'wrong':True}
    else: state[problem]=True
    with pytest.raises((ValueError,RuntimeError)): execute()
    assert not (root/'handoff.zip').exists()
    assert not (root/'run/result').exists()
    assert list((root/'run').glob('attempt-*/finished.json'))


def test_cache_tamper_fails_before_gpu(synthetic):
    root,_,pool,_,state,execute=synthetic
    execute()
    path=root/'run/photos'/pool['ids'][0]/'descriptor.npy'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError): execute()
    assert state['loads']==1 and path.read_bytes()==b'corrupt'


def test_changed_contract_rejected(synthetic):
    _,_,_,identity,_,execute=synthetic
    execute(); identity['changed']=True
    with pytest.raises(ValueError): execute()


def test_unchanged_selection_stops_without_reroll(synthetic):
    _,_,pool,identity,_,_=synthetic
    x=make_vectors(4)
    identity['baseline_ids']=p.select(pool['ids'],x,2)
    result=runner.selection_payload(pool,x,identity,lambda:None)
    assert result['status']=='selection_unchanged_stop'
    assert result['next_step']=='stop_no_reroll' and not result['fit_authorized']


def test_uncertain_attempt_cost_is_not_silently_discarded(tmp_path):
    (tmp_path/'attempt-unknown').mkdir()
    with pytest.raises(ValueError,match='cost receipt'): runner.read_attempts(tmp_path)


def test_cli_uses_declared_arguments(monkeypatch,tmp_path):
    calls=[]
    monkeypatch.setattr(runner,'run',lambda project,archive:calls.append((project,archive)) or {'status':'fake'})
    assert runner.main(['--project',str(tmp_path),'--archive',str(tmp_path/'a.zip')])==0
    assert calls==[(tmp_path,tmp_path/'a.zip')]


def test_monitor_caps_and_sampler_error(monkeypatch):
    import sys
    from types import SimpleNamespace
    monkeypatch.setitem(sys.modules,'psutil',SimpleNamespace(Process=lambda:SimpleNamespace(memory_info=lambda:SimpleNamespace(rss=100))))
    class Runtime:
        torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda:(800,1000)))
        def memory(self): return {'torch_cuda_peak_allocated_bytes':20,'torch_cuda_peak_reserved_bytes':30}
    policy={'resource_plan':{'rss_GiB':1,'selection_and_inference_reserved_GiB':1,'selection_seconds':100}}
    with runner.Monitor(Runtime(),policy,0) as m:
        m.check()
        assert m.snapshot()['device_used_sample_peak_bytes']==200
        policy['resource_plan']['rss_GiB']=0
        with pytest.raises(RuntimeError,match='budget'): m.check()
    with pytest.raises(RuntimeError,match='monitoring'):
        with runner.Monitor(Runtime(),policy,0) as m:
            m.error='synthetic sampler failure'


def test_feature_path_matches_forward_without_scoring_or_fitting(monkeypatch):
    from contextlib import nullcontext
    from types import SimpleNamespace
    class Tensor:
        dtype=np.float32
        def __init__(self,a): self.a=a; self.shape=a.shape
        def __sub__(self,b): return Tensor(self.a-b.a)
        def abs(self): return Tensor(np.abs(self.a))
        def max(self): return Tensor(np.array(self.a.max()))
        def item(self): return self.a.item()
        def mean(self,dim): return Tensor(self.a.mean(axis=dim))
        def __getitem__(self,i): return Tensor(self.a[i])
        def cpu(self): return self
        def numpy(self): return self.a
    values=Tensor(np.ones((1,768,64,64),dtype=np.float32))
    class Extractor:
        training=False
        def __call__(self,x): return {'layer1':values}
    class Model:
        training=False; tiler=None
        def __init__(self): self.embedding_store=[]; self.feature_extractor=Extractor()
        def feature_pooler(self,x): return x
        def generate_embedding(self,x): return x['layer1']
        def reshape_embedding(self,x): return Tensor(x.a.transpose(0,2,3,1).reshape(-1,768))
        def __call__(self,x):
            assert self.training and not self.feature_extractor.training
            value=self.reshape_embedding(values); self.embedding_store.append(value); return value
    runtime=object.__new__(runner.FeatureRuntime)
    runtime.model=SimpleNamespace(model=Model())
    runtime.config={'image_size':256}; runtime.gpu=()
    runtime.torch=SimpleNamespace(no_grad=nullcontext,float32=np.float32,isfinite=lambda x:np.isfinite(x.a))
    monkeypatch.setattr(runner.fit,'_preprocess',lambda *a:None)
    mean,difference=runtime.means(None,True)
    assert mean.shape==(768,) and difference==0
    assert not runtime.model.model.training and runtime.model.model.embedding_store==[]
