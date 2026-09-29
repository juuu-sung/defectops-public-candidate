"""One frozen spatial ablation; synthetic CPU execution, never real GPU evidence."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np
import pytest

from scripts import run_mpi_spatial_score as r
from scripts import mpi_spatial_score_protocol as s
from tests.test_mpi_score_validation import validation_run
from tests.test_mpi_score_probe import experiment, scored
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime


def test_integral_matches_naive_windows_and_bottom_right_not_cropped():
    rng = np.random.default_rng(42)
    values = rng.uniform(0,20,(256,256)).astype(np.float32)
    result = s.spatial_score(values)
    windows = np.lib.stride_tricks.sliding_window_view(values,(26,26))
    naive = windows.mean(axis=(-2,-1),dtype=np.float64)
    y,x = np.unravel_index(np.argmax(naive),naive.shape)
    assert result['window_xyxy'] == [x,y,x+26,y+26]
    assert result[s.FIELD] == pytest.approx(naive.max(),abs=1e-10)
    values[:] = 0; values[-26:,-26:] = 100
    assert s.spatial_score(values) == {s.FIELD:100.,'window_xyxy':[230,230,256,256]}


def test_constant_ties_and_scattered_vs_connected():
    assert s.spatial_score(np.ones((256,256),np.float32)) == {s.FIELD:1.,'window_xyxy':[0,0,26,26]}
    connected = np.zeros((256,256),np.float32); connected[:26,:26] = 10
    scattered = np.zeros_like(connected)
    scattered.flat[np.arange(676)*97] = 10
    assert np.count_nonzero(scattered) == np.count_nonzero(connected) == 676
    assert s.spatial_score(connected)[s.FIELD] > s.spatial_score(scattered)[s.FIELD]
    for invalid in (np.zeros((20,20)),np.full((256,256),np.nan),-np.ones((256,256))):
        with pytest.raises(ValueError): s.spatial_score(invalid)


def spatial_sample(prefix,pid,value):
    sample = scored(prefix,pid,value)
    for row in sample['regions']:
        row.update(window26=float(value),window_xyxy=[0,0,26,26])
    return sample


def test_normal_only_recipe_exact_q90_and_higher_quantile():
    dev = [spatial_sample('development',str(i),v) for i,v in enumerate([0,1,2,3,20])]
    cal = [spatial_sample('calibration',str(10+i),v) for i,v in enumerate([1,2,5,8])]
    method = s.fit_contract(dev,cal)
    assert method['regions'][0]['center'] == 2 and method['regions'][0]['scale'] == 18
    assert method['threshold'] == pytest.approx(1/3)
    assert r.p.score(cal[-1],method)['alert']
    changed = copy.deepcopy(cal); changed[0]['label']=1
    with pytest.raises(ValueError,match='designated normal'): s.fit_contract(dev,changed)
    changed=copy.deepcopy(cal); changed[0].update(photo_id='0',key='calibration/0')
    with pytest.raises(ValueError,match='overlap'): s.fit_contract(dev,changed)
    with pytest.raises(ValueError,match='Degenerate'): s.fit_contract(dev[:1],cal)


@pytest.fixture
def spatial_run(validation_run):
    archive,original = validation_run
    args = {k:v for k,v in original.items() if k not in ('git','frozen')}
    args['reference'] = dict(args['reference'])
    photo = args['cohorts']['validation'][0]
    args['reference']['validation'] = [{k:photo[k] for k in ('key','photo_id','label')} | {'regions':[
        {k:row[k] for k in ('member','region')} | {'top1':9.,'native_raw':9.} for row in photo['regions']]}]
    args['contract'] = original['frozen']['contract']
    args['policy'] = {'maximum_added_scoring_fraction':.1}
    return archive,args


def execute(data,calls,**options):
    path,args=data
    with zipfile.ZipFile(path) as source:
        return r.execute(source=source,**args,runtime_factory=lambda cfg:Runtime(calls,**options))


def test_complete_replay_freeze_export_resume_without_gpu(spatial_run,monkeypatch):
    def mask_forbidden(*a,**kw): raise AssertionError('No mask-based scoring')
    monkeypatch.setattr(r.ev,'read_masks',mask_forbidden)
    calls=[]; result=execute(spatial_run,calls)
    assert calls.count('predict') == 25+10*12 and calls[-1]=='release'
    assert result['maximum_absolute_replay_difference']==0 and not result['training_performed']
    assert not result['comparison']['screening_passed']  # same maps, no new detection
    root=spatial_run[1]['root']
    assert json.loads((root/'calibrated/calibration.json').read_bytes())['evaluation_scores_used']==0
    with zipfile.ZipFile(spatial_run[1]['handoff']) as z:
        manifest=json.loads(z.read('handoff-manifest.json'))
        assert set(z.namelist())==set(manifest)|{'handoff-manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest()==h for n,h in manifest.items())
    before=spatial_run[1]['handoff'].read_bytes();calls=[]
    assert execute(spatial_run,calls)==result and calls==[]
    assert spatial_run[1]['handoff'].read_bytes()==before


def test_resume_before_and_after_calibration(spatial_run):
    with pytest.raises(RuntimeError,match='interruption'): execute(spatial_run,[],fail_after=25+12)
    root=spatial_run[1]['root'];first=(root/'development/photo-001/sample.json').read_bytes()
    assert not (root/'calibrated').exists()
    with pytest.raises(RuntimeError,match='interruption'): execute(spatial_run,[],fail_after=25+4*12+12)
    assert (root/'calibrated').exists() and (root/'test/photo-001').exists()
    calls=[];execute(spatial_run,calls)
    assert calls.count('predict')==25+4*12
    assert (root/'development/photo-001/sample.json').read_bytes()==first


def test_control_replay_failure_blocks_calibration_and_evaluation(spatial_run):
    spatial_run[1]['reference']['development'][0]['regions'][0]['top1'] += 1
    with pytest.raises(ValueError,match='replay drift'): execute(spatial_run,[])
    root=spatial_run[1]['root']
    assert not (root/'calibrated').exists() and not list((root/'test').glob('photo-*'))


def test_runtime_failure_blocks_all_cohorts(spatial_run):
    with pytest.raises(ValueError,match='Normal probe blocked'): execute(spatial_run,[],offset=1)
    assert not list((spatial_run[1]['root']/'development').glob('photo-*'))


def test_invalid_calibration_never_scores_observed_test(spatial_run,monkeypatch):
    def fail(*args): raise ValueError('Degenerate spatial Q90 scale')
    monkeypatch.setattr(s,'fit_contract',fail)
    calls=[]
    with pytest.raises(ValueError,match='Degenerate'): execute(spatial_run,calls)
    assert calls.count('predict') == 25+5*12
    assert not list((spatial_run[1]['root']/'test').glob('photo-*'))


@pytest.mark.parametrize('problem',[None,'lost_tp','test_fp','normal_fp','cost'])
def test_screening_requires_added_tp_no_losses_no_fp_increase_and_cost(problem):
    def item(key,label,baseline,candidate):
        row=spatial_sample(key.split('/')[0],key.split('/')[1],baseline)
        row['label']=label
        for region in row['regions']:
            region['top1']=baseline;region['window26']=candidate
        row['timing']={'photo_scoring_ms':100.,'spatial_ms':[.1]*12}
        return row
    test=[item('defect/1',1,6,6),item('defect/2',1,4,6),item('good/3',0,4,4)]
    normal=[item('validation/4',0,4,4)]
    if problem=='lost_tp':
        for row in test[0]['regions']: row['window26']=4
    if problem=='test_fp':
        for row in test[2]['regions']: row['window26']=6
    if problem=='normal_fp':
        for row in normal[0]['regions']: row['window26']=6
    if problem=='cost':
        for row in test+normal: row['timing']['spatial_ms']=[5.]*12
    control={'field':'top1','threshold':5.,'regions':[{'region':reg,'center':0.,'scale':1.} for reg in r.fit.REGIONS]}
    candidate=dict(control,field='window26')
    result=s.compare({'test':test,'validation':normal},control,candidate)
    assert result['screening_passed'] is (problem is None)
    assert not result['automatic_promotion']


def test_cache_and_calibration_tampering_block(spatial_run):
    execute(spatial_run,[])
    (spatial_run[1]['root']/'calibrated/calibration.json').write_text('{}')
    calls=[]
    with pytest.raises(ValueError): execute(spatial_run,calls)
    assert calls==[]


def test_policy_hash_and_cli():
    project=Path(r.__file__).resolve().parents[1]
    content=(project/'configs/mpi-spatial-score-v1.json').read_bytes().replace(b'\r\n',b'\n')
    assert hashlib.sha256(content).hexdigest()==r.POLICY_LF_SHA256
    result=subprocess.run([sys.executable,r.__file__,'--help'],capture_output=True,text=True)
    assert result.returncode==0 and '--archive' in result.stdout
