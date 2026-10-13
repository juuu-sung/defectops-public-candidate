import copy
import hashlib
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import assess_mpi_normal_inputs as a


@pytest.mark.parametrize('area,index,reason', [(0, 2, None), (1, 1, 'CONNECTED_ZERO_PRESENT'),
                                               (65535, 1, 'CONNECTED_ZERO_PRESENT'),
                                               (65536, 0, 'ALL_ZERO_REQUIRED_TILE')])
def test_tile_rules(area, index, reason):
    assert a.tile_state({'connected_black_pixels': area}) == (a.STATES[index], reason)


@pytest.mark.parametrize('area', [-1, 65537, float('nan'), 1.2, True, None])
def test_invalid_statistics(area):
    with pytest.raises(ValueError): a.tile_state({'connected_black_pixels': area})


def test_photo_required_regions_and_priority():
    regions = [{'region': r, 'state': a.STATES[2]} for r in a.fit.REGIONS]
    assert a.photo_state(regions) == a.STATES[2]
    regions[0]['state'] = a.STATES[1]; regions[-1]['state'] = a.STATES[0]
    assert a.photo_state(regions) == a.STATES[0]
    for bad in [regions[:-1], regions[:-1]+[regions[0]],
                [{'region': r, 'state': 'NOT_ASSESSED'} for r in a.fit.REGIONS]]:
        with pytest.raises(ValueError): a.photo_state(bad)


def fixture():
    groups = {'training': [], 'validation': []}; lookup = {}; images = {}
    for g, ids in [('training', ['0001']), ('validation', ['0002', '0003', '0004'])]:
        for pid in ids:
            rows = []
            for index, region in enumerate(a.fit.REGIONS):
                rgb = np.full((256, 256, 3), 1, np.uint8)
                if pid == '0002' and index == 0: rgb[:] = 0
                if pid == '0003' and index == 0: rgb[0, 0] = 0
                raw = a.census.png_bytes(Image.fromarray(rgb))
                source = {'photo_id': pid, 'region': region, 'member': f'extracted/{region}/train/good/{pid}.png',
                          'member_sha256': hashlib.sha256(raw).hexdigest(),
                          'decoded_rgb_sha256': hashlib.sha256(rgb.tobytes()).hexdigest()}
                rows.append(source); images[g, pid, region] = raw
                lookup[g, pid, region] = {'source': source, 'statistics': a.census.black_statistics(rgb)}
            groups[g].append({'photo_id': pid, 'label': 0, 'regions': rows})
    models = [{'photo_id': pid, 'score': 4 if pid != '0003' else 2, 'alert': pid != '0003'}
              for pid in ['0002', '0003', '0004']]
    return groups, lookup, images, models


def test_assessment_preserves_model_null_and_union_counts():
    groups, lookup, images, models = fixture(); original = copy.deepcopy(models)
    photos, summary = a.assess(groups, lookup, set(images), models)
    assert models == original
    assert photos[0]['model_result'] is None
    assert [p['state'] for p in photos[1:]] == list(a.STATES)
    assert photos[1]['model_result']['alert'] is True
    assert summary['validation']['input_followup_photos'] == 2
    assert summary['validation']['input_and_model_alert'] == 1
    assert summary['validation']['input_or_model_alert'] == 3
    assert sum(sum(v.values()) for v in summary['validation']['cross_table'].values()) == 3
    assert summary['training']['cross_table'][a.STATES[2]]['not_scored'] == 1
    assert all(p['automatic_pass_allowed'] is False and p['human_assessment'] == 'NOT_RECORDED' for p in photos)
    with pytest.raises(ValueError): a.assess(groups, lookup, set(), models[:-1])


def test_near_black_is_not_called_suitable():
    rgb = np.ones((256, 256, 3), np.uint8)
    assert a.tile_state(a.census.black_statistics(rgb))[0] == 'NO_LISTED_ISSUE_DETECTED'
    assert any('Near-black' in x for x in a.POLICY['limits'])


def test_prediction_replay_and_fixed_threshold():
    row = {'key': 'x', 'photo_id': '0001', 'label': 0,
           'regions': [{'region': r, 'top1': 2.0} for r in a.fit.REGIONS]}
    contract = {'threshold': 1.0, 'regions': [{'region': r, 'center': 1., 'scale': 1.} for r in a.fit.REGIONS]}
    result = a.model_predictions({'test': [row], 'validation': [row]}, contract)
    assert result['test'][0]['score'] == 1 and result['test'][0]['alert'] is True
    row['regions'][0]['top1'] = float('nan')
    with pytest.raises(ValueError): a.model_predictions({'test': [row], 'validation': []}, contract)


def test_pixels_and_archive_only_frozen_normals(tmp_path, monkeypatch):
    groups, lookup, images, _ = fixture(); archive = tmp_path/'images.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for key, row in lookup.items(): z.writestr(row['source']['member'], images[key])
        z.writestr('extracted/test/defect/DO_NOT_READ.png', b'bad')
    accessed = []; read = zipfile.ZipFile.read
    def track(self, name, *args, **kwargs):
        accessed.append(name); return read(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'read', track)
    verified = set(); collected = {}
    a.verify_archive(archive, a.fit.sha256_file(archive), lookup, collected, verified)
    assert len(verified) == 48 and len(collected) == 2
    assert not any('/test/' in str(n) for n in accessed)
    with pytest.raises(ValueError): a.verify_archive(archive, 'bad', lookup, {}, set())
    with pytest.raises(ValueError): a.check_pixels(b'corrupt', next(iter(lookup.values())))


def test_run_repeated_same_output_and_preserved_tamper(tmp_path, monkeypatch):
    groups, lookup, images, models = fixture()
    def fake(*args):
        return groups, {}, lookup, images.copy(), set(images), models, {'fp256': 2}, {'threshold': 3.0}
    monkeypatch.setattr(a, 'load_evidence', fake)
    for name in a.SOURCES:
        p = tmp_path/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_text('fixed\n')
    first = a.run(tmp_path, None, None); output = a.Path(first['output'])
    digest = a.fit.sha256_file(output)
    assert a.run(tmp_path, None, None) == first and a.fit.sha256_file(output) == digest
    with zipfile.ZipFile(output) as z:
        report = json.loads(z.read('input-assessment.json'))
        assert report['baseline_counts_unchanged']['fp256'] == 2
        for n, h in json.loads(z.read('handoff-manifest.json')).items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h
        assert '기존 모델 경고' not in z.read('index.html').decode()
        assert '기존 모델 경고' not in z.read('cases/validation-0002.html').decode()
        assert '기존 모델 경고' in z.read('model-comparison.html').decode()
    target = a.Path(first['index']); target.write_bytes(b'changed')
    with pytest.raises(ValueError): a.run(tmp_path, None, None)
    assert target.read_bytes() == b'changed' and a.fit.sha256_file(output) == digest
    failure = list(target.parent.parent.glob('failure-*.json'))
    assert len(failure) == 1 and json.loads(failure[0].read_bytes())['assessment_completed'] is False


def test_cli_archive_is_optional(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(a, 'run', lambda *args: seen.append(args) or {})
    assert a.main(['--project', str(tmp_path), '--census', 'c.zip', '--baseline', 'b.zip']) == 0
    assert seen[0][-1] is None
    assert a.main(['--project', str(tmp_path), '--archive', 'a.zip']) == 0
    assert seen[-1][-1] == a.Path('a.zip')


def test_frozen_census_mismatch_refused(tmp_path):
    p = tmp_path/'bad.zip'; p.write_bytes(b'bad')
    with pytest.raises(ValueError, match='Frozen census'): a.load_evidence(tmp_path, p, None)
