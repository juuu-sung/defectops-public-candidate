"""CPU contracts and real orchestration with fake inference, not CUDA validation."""
from contextlib import nullcontext
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import compare_mpi_patchcore_full_maps as d


@pytest.mark.parametrize('bad', [[float('nan')], [-1.], [1.1], [1., 1.]])
def test_distance_guard(bad):
    with pytest.raises(ValueError, match='Distance parity'): d.check_distances(bad, [1.], 'test')


def test_map_peak_move_and_top_aggregation():
    bank = np.ones((256, 256), np.float32)
    bank[0, 0] = 10; bank[10:20, 10:20] = 9
    full = bank.copy(); full[0, 0] = 1
    mask = np.zeros_like(bank, dtype=bool); mask[10:20, 10:20] = True
    row = d.describe({'key':'defect/x', 'label':1, 'source_row':{'region':'r'}},bank,bank,full,mask)
    assert row['maximum_moved'] and row['full_value_at_bank_peak'] == 1
    assert row['full_pool']['maximum'] == 9
    assert row['matched_bank']['top656_mean']-row['full_pool']['top656_mean'] == pytest.approx(9/656)
    assert row['photo_prediction'] is None and not row['threshold_applied']
    with pytest.raises(ValueError, match='exceeds subset'): d.describe({'source_row':{}},bank,full,bank,mask)


def test_constant_map_ties_and_zero_card():
    values = np.zeros((256,256),np.float32)
    row = d.describe({'key':'normal/a','label':0,'source_row':{'region':'r'}},values,values,values,None)
    assert row['top656_relative_drop'] is None and not row['maximum_moved']
    source=io.BytesIO();Image.new('RGB',(256,256)).save(source,format='PNG')
    with Image.open(io.BytesIO(d.card(source.getvalue(),values,values,None,0))) as image:
        assert image.size == (768,292)


def test_previous_points_replayed_not_just_maps():
    record={'key':'a','source_row':{'region':'r'}}
    features=np.ones((4096,3),np.float32); scores=np.ones(4096)
    prior={'queries':[{'key':'a','region':'r','flat_index':3,'bank_distance':1.,'full_distance':1.}]}
    assert d.point_replay([record],features,scores,scores,prior,features[[3]])['queries']==1
    with pytest.raises(ValueError,match='features changed'):
        d.point_replay([record],features,scores,scores,prior,features[[3]]+1)


def test_double_gemm_chunk_boundaries_and_ties(monkeypatch):
    class Tensor:
        def __init__(self,x): self.x=np.asarray(x)
        def __getitem__(self,s): return Tensor(self.x[s])
        def cpu(self): return self
        def numpy(self): return self.x
        def min(self,dim):
            ids=self.x.argmin(axis=dim)
            return Tensor(self.x[np.arange(len(ids)),ids]),Tensor(ids)
    def euclidean(q,b):
        assert q.x.dtype==b.x.dtype==np.float64
        return Tensor(np.sqrt(np.maximum((q.x*q.x).sum(1)[:,None]-2*q.x@b.x.T+(b.x*b.x).sum(1)[None],0)))
    runtime=object.__new__(d.Runtime)
    runtime.torch=SimpleNamespace(inference_mode=nullcontext,as_tensor=lambda x,device:Tensor(x))
    runtime.model=SimpleNamespace(model=SimpleNamespace(euclidean_dist=euclidean))
    monkeypatch.setitem(d.POLICY,'query_chunk',2);monkeypatch.setitem(d.POLICY,'candidate_chunk',2)
    q=np.array([[0,0],[1,2],[5,7]],np.float32)
    bank=np.array([[0,0],[9,9],[1,2],[0,0],[5,8]],np.float32)
    dist,ids=runtime.fast_search(q,bank,lambda:None)
    assert ids.tolist()==[0,2,4] and dist.tolist()==[0,0,1]


def fixture_run(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('coverage_fixture',Path(__file__).with_name('test_mpi_patchcore_candidate_coverage.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    archive,base,parent,sig=helper.fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(d,'POLICY',{**d.POLICY,'tiles':1,'training_photos':2,'candidates':24576,'bank':2})
    for name in d.SOURCES:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text('fixture')
    record=d.c.select_tiles()[0]
    prior={'selected_tiles':[record],'parent_execution_sha256':sig,'policy':d.c.POLICY,'sources_lf_sha256':{}}
    data={'plan.json':json.dumps(prior).encode(),'result.json':json.dumps({'queries':[
        {'key':record['key'],'region':record['source_row']['region'],'flat_index':0,'bank_distance':1.,'full_distance':1.}]}).encode(),
        'queries.npy':d.c.np_bytes(np.zeros((1,768),np.float32))}
    monkeypatch.setattr(d,'checked_input',lambda path:data)
    class Runtime(base):
        def fast_gate(self,check):return {'passed':True,'fake':True}
        def fast_search(self,q,c,check):return self.search(q,c,check)
        def synchronize(self):pass
        def maps(self,x):return np.full((256,256),x[0],np.float32)
    return archive,Runtime,base,parent,sig


def test_end_to_end_interrupt_resume_replay_and_tamper(tmp_path,monkeypatch):
    archive,runtime,base,parent,sig=fixture_run(tmp_path,monkeypatch)
    base.interrupt_at=3
    with pytest.raises(KeyboardInterrupt):d.run(tmp_path,archive,runtime)
    root=tmp_path/'artifacts/analyses'/d.NAME
    assert (root/'search-000/cost.json').exists() and not (root/'search-001').exists()
    base.interrupt_at=None
    result=d.run(tmp_path,archive,runtime)
    assert result['tiles']==1 and result['locations']==4096 and base.traced==1 and base.searches==5
    handoff=Path(result['output']);before=handoff.read_bytes()
    assert d.run(tmp_path,archive,runtime)==result and handoff.read_bytes()==before
    assert base.traced==1 and base.searches==6
    with zipfile.ZipFile(handoff) as z:
        report=json.loads(z.read('result.json'))
        assert report['scan']['subset_verified']==2
        assert not report['photo_predictions_computed']
        assert np.load(io.BytesIO(z.read('full-maps.npy')),allow_pickle=False).shape==(1,256,256)
    (root/'search-000/cost.json').write_text('{}')
    with pytest.raises(ValueError):d.run(tmp_path,archive,runtime)


def test_official_tile_replay_failure_stops_before_search(tmp_path,monkeypatch):
    archive,runtime,base,parent,sig=fixture_run(tmp_path,monkeypatch)
    monkeypatch.setattr(runtime,'maps',lambda self,x:np.full((256,256),2.,np.float32))
    with pytest.raises(ValueError,match='Official map'):d.run(tmp_path,archive,runtime)
    assert base.searches==0


def test_cuda_installed_distance_gate_if_available():
    torch=pytest.importorskip('torch')
    if not torch.cuda.is_available():pytest.skip('Real CUDA required')
    from anomalib.models.image.patchcore.torch_model import PatchcoreModel
    runtime=object.__new__(d.Runtime);runtime.torch=torch
    runtime.model=SimpleNamespace(model=SimpleNamespace(euclidean_dist=PatchcoreModel.euclidean_dist))
    assert runtime.fast_gate(lambda:None)['passed']
