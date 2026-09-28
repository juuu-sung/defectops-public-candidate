"""CPU-only synthetic maps: integrity, paired attribution, exclusions and restart."""
import copy
import hashlib
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts import review_mpi_region_maps as run
from scripts import mpi_region_map_review_protocol as q
from scripts import mpi_region_map_review_cards as cards
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, execute as baseline_execute
from tests.test_mpi_region_search import experiment, RegionFake


class ChangedFake(RegionFake):
    def predict_region(self, image, indices):
        _, array = super().predict_region(image, indices)
        value = image.getpixel((0, 0))[0]
        if value in (10, 11):
            array[:] = 1001 if value == 10 else 1000
        return float(array[0, 0]), array


@pytest.fixture
def paired(experiment, fixture, monkeypatch):
    source_path, args = experiment
    baseline_execute(fixture, [])
    base_root = fixture['root']
    base = {n: json.loads((base_root/'result'/n).read_bytes()) for n in ('input-lock.json', 'scores.json', 'evaluation.json')}
    args = args | {'root': base_root.parent/'region', 'handoff': base_root.parent/'region.zip'}
    with zipfile.ZipFile(source_path) as source:
        # Create synthetic INPUT caches only, not diagnostic inference.
        run.region.execute(source=source, **args, runtime_factory=lambda cfg: ChangedFake([]))
    data = {p.name: json.loads(p.read_bytes()) for p in (args['root']/'result').glob('*.json')}
    selected = q.select(data['scores.json'], controls={})
    assert {v['group'] for v in selected} == {'new_fn', 'new_tp', 'new_fp', 'cleared_fp'}
    reference_data = copy.deepcopy(data)
    for s in reference_data['raw-scores.json']['calibration']:
        if s['arms'][q.p.ARMS[0]]['photo_id'] == '0100':
            for arm in q.p.ARMS:
                s['arms'][arm]['key'] = 'calibration/0411'
    reference = q.calibration_reference(reference_data)
    monkeypatch.setattr(q, 'calibration_reference', lambda d: reference)
    fixture['checkpoint'].unlink()  # A saved-map review must not need the model file.
    return dict(root=base_root.parent/'review', baseline_root=base_root, candidate_root=args['root'],
                recovered=fixture['recovered_path'].read_bytes(), data=data, baseline=base,
                previous=fixture['previous'], selection=selected, identity={'synthetic': 'read-only'},
                handoff=base_root.parent/'review.zip', reference_images={}, source_path=source_path)


def execute(d):
    with zipfile.ZipFile(d['source_path']) as source:
        return run.execute(source=source, **{k: v for k, v in d.items() if k != 'source_path'})


def input_hashes(d):
    return {str(p): q.fit.sha256_file(p) for name in ('baseline_root', 'candidate_root')
            for p in d[name].rglob('*') if p.is_file()}


def test_read_only_complete_and_repeat(paired, monkeypatch):
    def forbidden(*a, **kw):
        raise AssertionError('No model access in saved-map diagnostics')
    monkeypatch.setattr(run.region, 'RegionRuntime', forbidden)
    before = input_hashes(paired)
    report = execute(paired)
    assert input_hashes(paired) == before
    assert report['status'] == 'saved_region_maps_reviewed_exploratory'
    assert report['new_inference_calls'] == report['normal_candidate_maps_created'] == 0
    assert report['calibration_reference']['paired_normal_maps_available'] is False
    cases = json.loads((paired['root']/'result/cases.json').read_bytes())
    damaged = next(c for c in cases if c['key'] == 'defect/0006')
    assert damaged['localization_eligible'] is False
    assert all(r['map']['gt'] is None for a in damaged['arms'].values() for r in a['regions'])
    assert next(c for c in cases if c['key'] == 'defect/0054')['reconstructed_image_used'] is True
    with zipfile.ZipFile(paired['handoff']) as z:
        assert z.testzip() is None
        manifest = json.loads(z.read('handoff-manifest.json'))
        assert set(manifest) == set(z.namelist())-{'handoff-manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n, h in manifest.items())
        assert not any(n.endswith(('.pt', '.npz', '.png')) for n in z.namelist())
        assert len([n for n in z.namelist() if n.endswith('.jpg')]) == 8
        assert '후보 정상 지도는 저장되지 않아' in z.read('index.html').decode()
    zip_before = paired['handoff'].read_bytes()
    assert execute(paired) == report
    assert paired['handoff'].read_bytes() == zip_before and input_hashes(paired) == before


@pytest.mark.parametrize('which,name', [('baseline_root','samples/photo-001/maps.npz'),
    ('candidate_root','test/photo-001/maps.npz'), ('candidate_root','test/photo-001/sample.json'),
    ('candidate_root','test/photo-001/receipt.json')])
def test_corrupt_input_preserved_and_no_result(paired, which, name):
    target = paired[which]/name
    target.write_bytes(b'changed')
    with pytest.raises(ValueError, match='incomplete or changed'):
        execute(paired)
    assert target.read_bytes() == b'changed' and not (paired['root']/'result').exists()


@pytest.mark.parametrize('kind', ['image', 'mask'])
def test_modified_source_and_mask_do_not_publish(paired, kind):
    with zipfile.ZipFile(paired['source_path']) as z:
        payload = {n: z.read(n) for n in z.namelist()}
    # The diagnostic rechecks original images even when the map cache is intact.
    first = (paired['previous']['test-audit.json']['photos'][0]['regions'][0]['member'] if kind == 'image'
             else paired['previous']['test-audit.json']['photos'][3]['regions'][0]['mask_member'])
    payload[first] = b'changed source'
    with zipfile.ZipFile(paired['source_path'], 'w') as z:
        for n, b in payload.items(): z.writestr(n, b)
    with pytest.raises(ValueError, match='bytes changed'):
        execute(paired)
    assert not (paired['root']/'result').exists()


def test_resume_reuses_complete_cards(paired, monkeypatch):
    original = cards.render
    calls = []
    def interrupt(*a, **kw):
        calls.append(1)
        if len(calls) == 3: raise RuntimeError('synthetic review interruption')
        return original(*a, **kw)
    monkeypatch.setattr(cards, 'render', interrupt)
    with pytest.raises(RuntimeError, match='review interruption'):
        execute(paired)
    first = paired['root']/'cases/photo-001/card-1.jpg'
    raw = first.read_bytes()
    assert not (paired['root']/'result').exists()
    monkeypatch.setattr(cards, 'render', original)
    execute(paired)
    assert first.read_bytes() == raw


def test_completed_output_and_changed_gate_stop_without_overwrite(paired):
    execute(paired)
    target = paired['root']/'result/cards/photo-001-1.jpg'
    target.write_bytes(b'changed review')
    before = paired['handoff'].read_bytes()
    with pytest.raises(ValueError, match='incomplete or changed'):
        execute(paired)
    assert target.read_bytes() == b'changed review' and paired['handoff'].read_bytes() == before


def test_candidate_gate_tampering_stops_publication(paired):
    path = next(paired['candidate_root'].glob('attempt-*/candidate-gate.json'))
    path.write_bytes(b'changed gate')
    with pytest.raises(ValueError, match='Missing contributing candidate'):
        execute(paired)
    assert not (paired['root']/'result').exists() and path.read_bytes() == b'changed gate'


def test_selection_keeps_all_changes_and_controls():
    def row(key, label, score):
        return dict(key=key, label=label, score=score, threshold=5,
                    decision='REVIEW_REQUIRED' if score>=5 else 'BELOW_THRESHOLD_NOT_AUTO_PASS')
    old = [row('defect/1',1,6), row('defect/2',1,2), row('good/1',0,1), row('good/2',0,1)]
    new = [row('defect/1',1,2), row('defect/2',1,6), row('good/1',0,6), row('good/2',0,1)]
    selected = q.select(dict(zip(q.p.ARMS,(old,new))), {'good/2':'retained_tn'})
    assert [s['group'] for s in selected] == ['new_fn','new_tp','new_fp','retained_tn']
    with pytest.raises(ValueError, match='missing or changed'):
        q.select(dict(zip(q.p.ARMS,(old,new))), {'defect/1':'persistent_fn'})


def test_raw_scores_rise_but_boundary_can_rise_more_and_gt_is_local():
    photo = {'key':'defect/1','photo_id':'1','label':1,
             'regions':[dict(region=r,member=r) for r in q.fit.REGIONS]}
    samples = {a:{**photo,'regions':[r | {'top1':v} for r in photo['regions']]}
               for a,v in zip(q.p.ARMS,(10.,12.))}
    arrays = {a:np.full((12,256,256),v,dtype=np.float32) for a,v in zip(q.p.ARMS,(10.,12.))}
    cs = {a:{'normalization':[dict(region=r,median=0,mad=1) for r in q.fit.REGIONS],'threshold':v}
          for a,v in zip(q.p.ARMS,(9.,15.))}
    masks = [np.zeros((256,256),dtype=bool) for _ in range(12)]
    masks[0][2:4,2:4] = True
    c = q.describe(photo,samples,arrays,masks,cs)
    assert [c['arms'][a]['decision'] for a in q.p.ARMS] == ['REVIEW_REQUIRED','BELOW_THRESHOLD_NOT_AUTO_PASS']
    assert c['region_deltas'][0]['raw_top1_change'] == 2
    assert c['region_deltas'][0]['raw_boundary_change'] == 6
    gt = c['arms'][q.p.ARMS[0]]['regions'][0]['map']['gt']
    assert gt['within_region_rank_auc'] == .5 and gt['top1pct_overlap_pixels_min'] == 0
    assert gt['top1pct_overlap_pixels_max'] == 4
    arrays[q.p.ARMS[1]][0,0,0] = 0
    with pytest.raises(ValueError, match='monotonicity'):
        q.describe(photo,samples,arrays,masks,cs)


def test_fixed_scale_not_per_image_and_cpu_import():
    image = Image.new('RGB',(256,256),(90,90,90))
    values = np.full((256,256),10,dtype=np.float32)
    a = cards.pixels(image,values,None)[1]
    values[0,0] = 10000
    b = cards.pixels(image,values,None)[1]
    assert np.array_equal(a[1,1],b[1,1])
    result = subprocess.run([sys.executable,'-c',
        'import sys; from scripts import review_mpi_region_maps; assert "torch" not in sys.modules'], capture_output=True,text=True)
    assert result.returncode == 0, result.stderr
