"""Synthetic CPU runtime only. Real CUDA reproduction requires the lab run."""
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from scripts import review_mpi_spatial_misses as r
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime, png
from tests.test_mpi_pcb_normal_bank_fit import png_row


@pytest.fixture
def review(fixture):
    d = fixture
    selected = []
    with zipfile.ZipFile(d['source_path'],'a') as z:
        for i,(pid,reason) in enumerate(r.SELECTION):
            rows = []
            for region in r.fit.REGIONS:
                raw,row = png_row(pid,region,20+i)
                z.writestr(row['member'],raw)
                mask = np.zeros((256,256),dtype=np.uint8)
                mask[10:20,10:20] = 1
                data = png(Image.fromarray(mask))
                row = row | {'mask_member':f'masks/{region}/{pid}.png',
                             'mask_sha256':hashlib.sha256(data).hexdigest(),'mask':r.ev.mask_details(data)}
                z.writestr(row['mask_member'],data)
                rows.append(row)
            photo = {'key':'defect/'+pid,'photo_id':pid,'label':1,'regions':rows}
            sample = {k:photo[k] for k in ('key','photo_id','label')} | {'regions':[
                {k:row[k] for k in ('member','region')} | {'top1':float(20+i),'native_raw':float(20+i),
                     'window26':float(20+i),'window_xyxy':[0,0,26,26]} for row in rows]}
            selected.append({'photo':photo,'expected':sample,'reason':reason})
    def method(field):
        return {'field':field,'threshold':30.,'regions':[{'region':reg,'center':0.,'scale':1.} for reg in r.fit.REGIONS]}
    old = {'control_top1_mad':method('top1'),'candidate_top1_q90':method('top1'),'candidate_native_mad':method('native_raw')}
    args = {k:d[k] for k in ('root','checkpoint','config','preparation','records','identity','handoff')}
    args.update(selected=selected,old_contract=old,methods={'control_q90':method('top1'),'window26_q90':method('window26')})
    return d['source_path'],args


def execute(review,calls,**options):
    path,args = review
    with zipfile.ZipFile(path) as z:
        return r.execute(source=z,**args,runtime_factory=lambda cfg:Runtime(calls,**options))


def test_complete_maps_masks_manifest_and_no_gpu_rerun(review):
    calls = []
    report = execute(review,calls)
    assert calls.count('predict') == 25+72 and calls[-1]=='release'
    assert report['case_count']==6 and report['maximum_absolute_replay_difference']==0
    assert not report['training_performed'] and not report['thresholds_changed']
    with zipfile.ZipFile(review[1]['handoff']) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        assert set(z.namelist()) == set(manifest)|{'handoff-manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
        assert len([n for n in manifest if n.endswith('.jpg')])==18
        cases = json.loads(z.read('cases.json'))
        w = cases[0]['regions'][0]['window']
        assert w['gt_overlap_pixels']==100 and w['gt_recall']==1
        assert w['gt_score_share']==pytest.approx(100/676)
        assert '자동 원인 분류를 하지 않습니다' in z.read('index.html').decode()
    with np.load(review[1]['root']/'cases/defect-0019/maps.npz',allow_pickle=False) as data:
        assert data['gt_masks'].dtype==np.bool_ and data['gt_masks'].shape==(12,256,256)
    before = review[1]['handoff'].read_bytes()
    calls = []
    assert execute(review,calls)==report and calls==[]
    assert review[1]['handoff'].read_bytes()==before


def test_resume_rechecks_runtime_and_preserves_completed(review):
    with pytest.raises(RuntimeError,match='interruption'):execute(review,[],fail_after=25+12)
    first=review[1]['root']/'cases/defect-0019/maps.npz'
    before=first.read_bytes(); calls=[]
    execute(review,calls)
    assert calls.count('predict')==25+60 and first.read_bytes()==before


def test_gate_failure_before_case(review):
    with pytest.raises(ValueError,match='Normal probe blocked'):execute(review,[],offset=1)
    assert not list((review[1]['root']/'cases').iterdir())


@pytest.mark.parametrize('field',['top1','window26','window_xyxy'])
def test_score_window_drift_blocks_publication(review,field):
    row=review[1]['selected'][0]['expected']['regions'][0]
    row[field]=[1,1,27,27] if field=='window_xyxy' else row[field]+1
    with pytest.raises(ValueError):execute(review,[])
    assert not (review[1]['root']/'cases/defect-0019').exists()
    assert not review[1]['handoff'].exists()


@pytest.mark.parametrize('file',['maps.npz','case.json','probe'])
def test_cache_tamper_blocks_before_gpu(review,file):
    execute(review,[])
    root=review[1]['root']; path=root/'cases/defect-0019'/file
    if file=='probe':
        receipt=json.loads((root/'cases/defect-0019/receipt.json').read_bytes())
        path=root/receipt['probe_path']
    path.write_bytes(b'changed'); calls=[]
    with pytest.raises(ValueError):execute(review,calls)
    assert calls==[]


def test_gt_only_after_replay_and_bad_mask_blocks(review,monkeypatch):
    original=r.ev.read_masks
    def missing(source,photo):
        assert len(calls)==0 or calls.count('predict')>=25+12
        return [None]*12
    monkeypatch.setattr(r.ev,'read_masks',missing)
    calls=[]
    with pytest.raises(ValueError,match='intact official masks'):execute(review,calls)
    assert not review[1]['handoff'].exists()
    monkeypatch.setattr(r.ev,'read_masks',original)


def test_attribution_zero_no_gt_and_full_gt(review):
    sample=copy.deepcopy(review[1]['selected'][0]['expected'])
    for row in sample['regions']:row.update(top1=0.,native_raw=7.,window26=0.)
    arrays=np.zeros((12,256,256),dtype=np.float32)
    masks=np.zeros_like(arrays,dtype=bool)
    case=r.describe(sample,arrays,masks,review[1]['methods'],'remaining_miss')
    assert case['regions'][0]['window']['gt_recall'] is None
    assert case['regions'][0]['window']['gt_score_share'] is None
    assert case['regions'][0]['map']['gt']['within_region_rank_auc'] is None
    masks[:]=True
    case=r.describe(sample,arrays,masks,review[1]['methods'],'remaining_miss')
    assert case['regions'][0]['window']['non_gt_mean'] is None
    assert case['regions'][0]['window']['gt_recall']==676/65536


def test_changed_mask_does_not_change_frozen_scores(review):
    sample=review[1]['selected'][0]['expected']
    arrays=np.full((12,256,256),20,dtype=np.float32)
    a=r.describe(sample,arrays,np.zeros_like(arrays,dtype=bool),review[1]['methods'],'remaining_miss')
    b=r.describe(sample,arrays,np.ones_like(arrays,dtype=bool),review[1]['methods'],'remaining_miss')
    assert a['methods']==b['methods'] and a['sample']==b['sample']
    assert a['regions'][0]['window']['gt_overlap_pixels']==0
    assert b['regions'][0]['window']['gt_overlap_pixels']==676


def test_off_origin_window_attribution_matches_direct_sum(review):
    sample=copy.deepcopy(review[1]['selected'][0]['expected'])
    arrays=np.ones((12,256,256),dtype=np.float32)
    arrays[:,60:86,100:126]=10
    masks=np.zeros_like(arrays,dtype=bool)
    masks[:,70:80,120:140]=True
    for row in sample['regions']:
        row.update(top1=10.,native_raw=10.,**r.spatial.spatial_score(arrays[0]))
    case=r.describe(sample,arrays,masks,review[1]['methods'],'remaining_miss')
    w=case['regions'][0]['window']
    assert w['window_xyxy']==[100,60,126,86]
    assert w['gt_overlap_pixels']==60 and w['gt_recall']==.3
    assert w['gt_score_share']==pytest.approx(60/676)
    assert w['non_gt_mean']==10 and w['gt_mean']==10


def test_tiny_spatial_drift_cannot_flip_decision(review):
    expected=copy.deepcopy(review[1]['selected'][0]['expected'])
    sample=copy.deepcopy(expected)
    methods=copy.deepcopy(review[1]['methods'])
    methods['window26_q90']['threshold']=20.000005
    sample['regions'][0]['window26']+=.00001
    with pytest.raises(ValueError,match='Frozen decision changed'):
        r.replay(sample,expected,review[1]['old_contract'],methods)


def test_cli_help():
    proc=subprocess.run([sys.executable,str(Path(r.__file__)),'--help'],capture_output=True,text=True)
    assert proc.returncode==0 and '--archive' in proc.stdout
