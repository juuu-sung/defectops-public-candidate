"""Synthetic CPU checks only; not evidence of real GPU reproduction."""
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

from scripts import review_mpi_score_cases as r
from tests.test_mpi_pcb_normal_bank_evaluation import fixture, Runtime, png
from tests.test_mpi_pcb_normal_bank_fit import png_row


@pytest.fixture
def review(fixture):
    d = fixture
    selected = []
    with zipfile.ZipFile(d['source_path'], 'a') as z:
        for index, (key, reason) in enumerate(r.SELECTION):
            group, photo_id = key.split('/')
            rows = []
            for region in r.fit.REGIONS:
                raw, row = png_row(photo_id, region, 20 + index)
                z.writestr(row['member'], raw)
                row = dict(row, mask_member=None)
                if group == 'defect':
                    mask = np.zeros((256, 256), dtype=np.uint8)
                    mask[10:20, 10:20] = 1
                    data = png(Image.fromarray(mask))
                    row.update(mask_member=f'masks/{region}/{photo_id}.png',
                               mask_sha256=hashlib.sha256(data).hexdigest(), mask=r.ev.mask_details(data))
                    z.writestr(row['mask_member'], data)
                rows.append(row)
            photo = dict(key=key, photo_id=photo_id, label=int(group=='defect'), regions=rows)
            sample = {k:photo[k] for k in ('key', 'photo_id', 'label')} | {'regions':[
                {k:row[k] for k in ('member','region')} | {'top1':float(20+index), 'native_raw':float(20+index)} for row in rows]}
            selected.append(dict(photo=photo, expected=sample, reason=reason))
    contract = {name: {'field':field, 'threshold':30., 'regions':[
        {'region':reg, 'center':0., 'scale':1.} for reg in r.fit.REGIONS]}
        for name, field in [('control_top1_mad','top1'),('candidate_top1_q90','top1'),('candidate_native_mad','native_raw')]}
    args = {k:d[k] for k in ('root','checkpoint','config','preparation','records','identity','handoff')}
    args.update(selected=selected, contract=contract)
    return d['source_path'], args


def execute(review, calls, **options):
    path, args = review
    with zipfile.ZipFile(path) as source:
        return r.execute(source=source, **args, runtime_factory=lambda cfg: Runtime(calls, **options))


def test_complete_hashes_maps_cards_and_no_gpu_rerun(review):
    calls = []
    report = execute(review, calls)
    assert calls.count('predict') == 25 + 60
    assert calls[-1] == 'release'
    assert report['case_count'] == 5 and report['maximum_absolute_replay_difference'] == 0
    assert not report['training_performed'] and not report['thresholds_changed']
    with zipfile.ZipFile(review[1]['handoff']) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        assert set(z.namelist()) == set(manifest) | {'handoff-manifest.json'}
        assert all(hashlib.sha256(z.read(n)).hexdigest() == h for n,h in manifest.items())
        assert len([n for n in manifest if n.endswith('.jpg')]) == 15
        assert len([n for n in manifest if n.endswith('.npz')]) == 5
        cases = json.loads(z.read('cases.json'))
        assert cases[-1]['regions'][0]['map']['gt']['pixels'] == 100
        assert 'Q90' in z.read('index.html').decode()
    before = review[1]['handoff'].read_bytes()
    calls = []
    assert execute(review, calls) == report
    assert calls == [] and review[1]['handoff'].read_bytes() == before


def test_resume_keeps_completed_case_and_rechecks_runtime(review):
    with pytest.raises(RuntimeError, match='interruption'):
        execute(review, [], fail_after=25+12)
    first = review[1]['root'] / 'cases/validation-0970/maps.npz'
    before = first.read_bytes()
    calls = []
    execute(review, calls)
    assert calls.count('predict') == 25+48
    assert first.read_bytes() == before


def test_runtime_drift_stops_before_cases(review):
    with pytest.raises(ValueError, match='Normal probe blocked'):
        execute(review, [], offset=1)
    assert not list((review[1]['root'] / 'cases').glob('validation-*'))


def test_case_score_drift_not_published(review):
    review[1]['selected'][0]['expected']['regions'][0]['top1'] += 1
    with pytest.raises(ValueError, match='Raw score replay drift'):
        execute(review, [])
    assert not (review[1]['root'] / 'cases/validation-0970').exists()
    assert not review[1]['handoff'].exists()


@pytest.mark.parametrize('target', ['maps.npz','probe'])
def test_cache_tampering_stops_before_gpu(review, target):
    execute(review, [])
    root = review[1]['root']
    path = root / 'cases/validation-0970/maps.npz'
    if target == 'probe':
        receipt = json.loads((path.parent / 'receipt.json').read_bytes())
        path = root / receipt['probe_path']
    path.write_bytes(b'changed')
    calls = []
    with pytest.raises(ValueError):
        execute(review, calls)
    assert calls == []


def test_describe_native_not_map_max_and_empty_map_ties(review):
    sample = copy.deepcopy(review[1]['selected'][0]['expected'])
    for row in sample['regions']:
        row.update(top1=0., native_raw=7.)
    arrays = np.zeros((12,256,256),dtype=np.float32)
    masks = np.ones_like(arrays,dtype=bool)
    case = r.describe(sample, arrays, masks, review[1]['contract'], 'remaining_miss')
    assert case['methods']['candidate_native_mad']['score'] == 7
    assert case['regions'][0]['map']['gt']['top1pct_score_share_min'] is None
    assert '기여율 정의 불가' in r.cards.index([case])


def test_cli_help():
    result = subprocess.run([sys.executable, str(Path(r.__file__)), '--help'], capture_output=True, text=True)
    assert result.returncode == 0 and '--archive' in result.stdout
