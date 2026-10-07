import hashlib
import importlib.util
import json
from pathlib import Path
import zipfile

import numpy as np
import pytest

from scripts import diagnose_mpi_patchcore_map_parity as a


def test_differences_report_location_and_direction():
    out=a.difference(np.array([[1.,2.],[3.,4.]]),np.array([[1.,2.],[3.,3.5]]),.01)
    assert out['maximum_difference_index']==[1,1] and out['outside_tolerance']==1
    assert out['max_absolute_difference']==.5 and out['a_at_maximum']==4.


@pytest.mark.parametrize('value',[np.array([]),np.array([np.nan]),np.array([np.inf])])
def test_invalid_comparison_rejected(value):
    with pytest.raises(ValueError):a.difference(value,value,.01)


def test_native_error_vs_exact_same_neighbor():
    class Runtime:
        def native_neighbors(self,q):return np.ones(len(q),np.float32)*1.01,np.zeros(len(q),np.int64)
        def fast_search(self,q,b,check):return np.ones(len(q)),np.zeros(len(q),np.int64)
        search=fast_search
        def maps(self,d):return np.full((256,256),d[0],np.float32)
    q=np.zeros((2,3),np.float32);bank=np.array([[1,0,0],[2,0,0]],np.float32)
    report,arrays=a.tile_audit(Runtime(),q,np.ones(2,np.float32)*1.01,np.full((256,256),1.01,np.float32),bank,lambda:None)
    assert report['saved_native_vs_public_repeat']['max_absolute_difference']==0
    assert report['native_vs_exact_same_native_neighbor']['max_absolute_difference']==pytest.approx(.01)
    assert report['native_neighbor_exact_vs_double_minimum']['max_absolute_difference']==0
    assert report['saved_map_vs_matched_double']['outside_tolerance']==256*256
    assert report['neighbor_indices_different']==0
    assert arrays['double-map.npy'].shape==(256,256)


def test_bad_neighbors_rejected():
    with pytest.raises(ValueError):a.direct_distances(np.zeros((2,3)),np.zeros((1,3)),np.array([0,1]))


def test_actual_files_preserved_export_repeat_and_tamper(tmp_path,monkeypatch):
    spec=importlib.util.spec_from_file_location('fullmaps_fixture',Path(__file__).with_name('test_mpi_patchcore_full_maps.py'))
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    archive,base,base_state,parent,parent_sig=helper.fixture_run(tmp_path,monkeypatch)
    record=a.d.c.select_tiles()[0]
    old=tmp_path/'artifacts/analyses'/a.d.NAME
    old.mkdir(parents=True)
    identity={'name':a.d.NAME,'policy':a.d.POLICY,'tiles':[record],'parent_execution_sha256':parent_sig,
              'checkpoint_sha256':parent['model-receipt.json']['files']['model.pt'],
              'sources_lf_sha256':{n:hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in a.d.SOURCES}}
    sig=a.fit.digest(identity);a.p.publish(old,'identity',sig,identity)
    a.d.collect_tile(old,'tile-00',sig,record,a.d.c.read_overlap(None),base({}))
    attempt=old/'attempt-001';attempt.mkdir()
    (attempt/'finished.json').write_text(json.dumps({'identity':sig,'status':'failed','error':'ValueError: Matched double-precision map too far from native map'}))
    for n in a.SOURCES:(tmp_path/n).write_text('fixture')
    class Runtime(base):
        def native_neighbors(self,q):return np.ones(len(q),np.float32),np.zeros(len(q),np.int64)
    before={f.relative_to(old).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in old.rglob('*') if f.is_file()}
    result=a.run(tmp_path,Runtime)
    assert result['tiles_outside_existing_tolerance']==0
    handoff=Path(result['output']);zip_before=handoff.read_bytes()
    assert a.run(tmp_path,Runtime)==result and zip_before==handoff.read_bytes()
    assert before=={f.relative_to(old).as_posix():hashlib.sha256(f.read_bytes()).hexdigest() for f in old.rglob('*') if f.is_file()}
    with zipfile.ZipFile(handoff) as z:
        report=json.loads(z.read('report.json'))
        assert not report['permission_to_resume'] and not report['full_pool_search']
    (tmp_path/'artifacts/analyses'/a.NAME/'tile-00/report.json').write_text('{}')
    with pytest.raises(ValueError):a.run(tmp_path,Runtime)
