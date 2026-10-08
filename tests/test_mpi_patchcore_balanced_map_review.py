import copy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts import review_mpi_patchcore_balanced_maps as r


def synthetic_selection():
    predictions = {m: {'test': [], 'validation': []} for m in ('baseline', 'candidate')}
    for m in predictions:
        for group, keys in r.EXPECTED_NEW.items():
            predictions[m][group] = [{'key': k, 'winning_region': r.fit.REGIONS[0]} for k in keys]
    cases = []
    for key in r.EXPECTED_DEFECTS:
        tiles = []
        for region in r.LOWER_REGIONS:
            worst = (region == r.LOWER_REGIONS[0] and key in ('defect/0033', 'defect/0028')) or (
                region == r.LOWER_REGIONS[1] and key in ('defect/0025', 'defect/0008'))
            tiles.append({'region': region, 'baseline': {'location': {'within_tile_auroc': .95}},
                          'candidate': {'location': {'within_tile_auroc': .65 if worst else .9}}})
        cases.append({'key': key, 'location_eligible': True, 'tiles': tiles})
    return {'evaluation/comparison.json': {'new_false_positives': copy.deepcopy(r.EXPECTED_NEW),
        'resolved_false_positives': {'test': ['good/0018'], 'validation': []}, 'predictions': predictions},
        'evaluation/location.json': {'cases': cases}}


def test_selection_is_fixed_all_eight_plus_control_and_top_declines():
    data = synthetic_selection(); before = copy.deepcopy(data)
    result = r.selection(data)
    assert len(result) == 14 and data == before
    assert sum('new_false_positive' in x['reasons'] for x in result) == 8
    assert sorted(x['key'] for x in result if x['key'].startswith('defect/')) == r.EXPECTED_DEFECTS
    data['evaluation/location.json']['cases'].reverse()
    assert r.selection(data) == result
    assert 'grid3-2_s1024' in next(x for x in result if x['key'] == 'defect/0024')['focus_regions']


@pytest.mark.parametrize('change', ['new', 'control', 'decline'])
def test_selection_drift_fails_no_substitution(change):
    data = synthetic_selection()
    if change == 'new': data['evaluation/comparison.json']['new_false_positives']['test'].pop()
    elif change == 'control': data['evaluation/comparison.json']['resolved_false_positives']['test'] = []
    else: data['evaluation/location.json']['cases'][0]['key'] = 'defect/9999'
    with pytest.raises(ValueError): r.selection(data)


def method(center=10., scale=2., threshold=3.):
    return {'threshold': threshold, 'regions': [{'region': x, 'center': center, 'scale': scale} for x in r.fit.REGIONS]}


def test_arithmetic_lower_raw_still_alerts_without_refit():
    old = {'regions': [{'region': x, 'top1': 15.9} for x in r.fit.REGIONS]}
    new = {'regions': [{'region': x, 'top1': 15.5} for x in r.fit.REGIONS]}
    assert not r.arithmetic(old, method())['alert']
    assert r.arithmetic(new, method(threshold=2.5))['alert']
    assert not r.arithmetic(new, method())['alert']
    assert r.arithmetic(old, method(threshold=2.5))['alert']
    assert r.arithmetic(old, method())['winning_region'] == r.fit.REGIONS[0]
    assert r.arithmetic(new, method(threshold=2.75))['alert']  # >= at boundary
    assert r.arithmetic(new, method())['tiles'][0]['raw_boundary'] == 16.


@pytest.mark.parametrize('scale', [0, -1, float('nan'), float('inf')])
def test_invalid_normal_contract_rejected(scale):
    sample = {'regions': [{'region': x, 'top1': 15.} for x in r.fit.REGIONS]}
    with pytest.raises(ValueError): r.arithmetic(sample, method(scale=scale))


def png(mode='RGB', value=(40, 70, 90)):
    out = io.BytesIO(); Image.new(mode, (256,256), value).save(out, format='PNG'); return out.getvalue()


def fixture(project):
    payload, photos, baseline, candidate = {}, [], [], []
    arrays = {}
    for i, key in enumerate(('good/0021', 'defect/0024')):
        photo = {'key': key, 'photo_id': key.split('/')[1], 'label': i, 'regions': []}
        for j, region in enumerate(r.fit.REGIONS):
            member = f'{key}/{j:02d}.png'; raw = png(value=(40,70,j))
            payload[member] = raw
            row = {'region': region, 'member': member, 'member_sha256': hashlib.sha256(raw).hexdigest(),
                   'decoded_rgb_sha256': hashlib.sha256(Image.open(io.BytesIO(raw)).tobytes()).hexdigest()}
            if i:
                row['mask_member'] = None
                if j == 7:
                    mask_raw = png('L', 255); mask_member = member + '.mask.png'; payload[mask_member] = mask_raw
                    row.update(mask_member=mask_member, mask_sha256=hashlib.sha256(mask_raw).hexdigest(),
                               mask=r.engine.ev.mask_details(mask_raw))
            photo['regions'].append(row)
        photos.append(photo)
        for model, target in [('baseline', baseline), ('candidate', candidate)]:
            offset = 1 if model == 'candidate' else 0
            values = np.stack([np.full((256,256), 15. + i + j/10 + offset, np.float32) for j in range(12)])
            rows = [{k: row[k] for k in ('region', 'member')} | {
                'native_raw': float(v.max()), 'top1': float(r.fit.candidate_image_score(float(v.max()), v, r.fit.CANDIDATE))}
                for row, v in zip(photo['regions'], values, strict=True)]
            sample = {k: photo[k] for k in ('key', 'photo_id', 'label')} | {'regions': rows, 'photo_seconds': .2}
            target.append(sample); arrays[model, i] = values
    archive = project/'original.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for name, raw in payload.items(): z.writestr(name, raw)
    parent_plan = {'cohorts': {'test': photos}, 'archive_sha256': r.fit.sha256_file(archive)}
    parent_checkpoint = 'a'*64; checkpoint = 'b'*64
    candidate_plan = {**parent_plan, 'parent_checkpoint_sha256': parent_checkpoint, 'baseline_contract': method()}
    parent_receipts, candidate_receipts = {}, {}
    for model, plan, sha, records, samples, name_root in [
        ('baseline', parent_plan, parent_checkpoint, parent_receipts, baseline, r.p.NAME),
        ('candidate', candidate_plan, checkpoint, candidate_receipts, candidate, r.b.NAME)]:
        for i, photo in enumerate(photos):
            name = f'photo-{i:03d}'; folder = project/'artifacts/analyses'/name_root/'test'/name
            folder.mkdir(parents=True)
            r.fit.write_json(folder/'sample.json', samples[i])
            np.save(folder/'maps.npy', arrays[model,i], allow_pickle=False)
            r.fit.seal_phase(folder, name, r.fit.digest(plan), {'photo_sha256': r.fit.digest(photo), 'checkpoint_sha256': sha}, ['sample.json','maps.npy'])
            records[f'test/{name}/receipt.json'] = json.loads((folder/'receipt.json').read_bytes())
    source = {'parent': {'plan.json': parent_plan, 'fit-report.json': {'checkpoint_sha256': parent_checkpoint},
                        'evaluation/prediction-receipts.json': parent_receipts},
              'development': {}, 'cohorts': {'test': photos}, 'reference': {'test': baseline}}
    data = {'plan.json': candidate_plan, 'fit-report.json': {'checkpoint_sha256': checkpoint},
            'contract.json': method(threshold=2.), 'evaluation/prediction-receipts.json': candidate_receipts,
            'evaluation/location.json': {'cases': []}}
    cases = []
    for i, photo in enumerate(photos):
        samples = {'baseline': baseline[i], 'candidate': candidate[i]}
        methods = {'baseline': method(), 'candidate': data['contract.json']}
        cases.append({'group':'test', 'key': photo['key'], 'label': photo['label'], 'reasons':['fixture'],
            'focus_regions':['grid3-2_s1024'], 'models': {m:r.arithmetic(samples[m],methods[m]) for m in methods},
            'crossed_scores_and_contracts': {s:{m:r.arithmetic(samples[s],methods[m]) for m in methods} for s in samples}})
    for relative in ('scripts/review_mpi_patchcore_balanced_maps.py','scripts/run_mpi_patchcore_balanced_map_review.ps1','docs/mpi-patchcore-balanced-map-review-v1.md'):
        path = project/relative; path.parent.mkdir(exist_ok=True); path.touch()
    return data, source, {'test':candidate}, {'cases':cases}, archive


def test_read_pair_checks_both_receipts_and_scores(tmp_path):
    data,source,candidate,report,archive = fixture(tmp_path)
    pair,_ = r.read_pair(tmp_path,data,source,candidate,'test',0)
    assert pair[0].dtype == pair[1].dtype == np.float32
    candidate['test'][0]['regions'][0]['top1'] += 1
    with pytest.raises(ValueError,match='pairing'): r.read_pair(tmp_path,data,source,candidate,'test',0)


@pytest.mark.parametrize('model', ['baseline','candidate'])
def test_corrupt_map_is_not_recomputed(tmp_path,model):
    data,source,candidate,report,archive = fixture(tmp_path)
    name = r.p.NAME if model == 'baseline' else r.b.NAME
    path = tmp_path/'artifacts/analyses'/name/'test/photo-000/maps.npy'
    path.write_bytes(b'corrupt')
    with pytest.raises(ValueError): r.read_pair(tmp_path,data,source,candidate,'test',0)
    assert path.read_bytes() == b'corrupt'


def test_cpu_export_all_regions_empty_masks_manifest_and_no_overwrite(tmp_path):
    data,source,candidate,report,archive = fixture(tmp_path)
    before = {p:r.fit.sha256_file(p) for p in tmp_path.rglob('*') if p.is_file()}
    output = r.export(tmp_path,archive,data,source,candidate,report)
    original = output.read_bytes()
    with zipfile.ZipFile(output) as z:
        review = json.loads(z.read('review.json'))
        assert review['photos'] == 2 and review['tiles'] == 24
        assert not review['new_inference'] and not review['threshold_changed']
        assert len([x for x in z.namelist() if x.endswith('/card.png')]) == 24
        assert len([x for x in z.namelist() if x.endswith('/mask.npy')]) == 12
        assert len([x for x in z.namelist() if x.endswith('/mask-source.png')]) == 1
        assert review['records'][19]['stats']['candidate']['location']['gt_pixels'] == 65536
        assert '점수 / 보정계약' in z.read('index.html').decode()
        for name,sha in json.loads(z.read('handoff-manifest.json')).items():
            assert hashlib.sha256(z.read(name)).hexdigest() == sha
    assert all(r.fit.sha256_file(path) == sha for path,sha in before.items())
    assert r.export(tmp_path,archive,data,source,candidate,report) == output
    assert output.read_bytes() == original
    report['changed'] = True
    with pytest.raises(ValueError,match='Existing review ZIP differs'):
        r.export(tmp_path,archive,data,source,candidate,report)
    assert output.read_bytes() == original


def test_shared_palette_no_per_model_normalization_and_finite_constant():
    raw = np.full((256,256), 5., np.float32)
    assert np.array_equal(r.heat(raw,0,10), r.heat(raw.copy(),0,10))
    assert not np.array_equal(r.heat(raw,0,10),r.heat(raw,0,20))
    assert np.isfinite(r.heat(raw,5,5)).all()
    mask = np.zeros((256,256),bool); mask[100:110,100:110] = True
    numbers = [dict(raw=5.,raw_boundary=6.,score=1.,threshold=2.)]*2
    image,limits = r.card(png(),[raw,raw+5],mask,'defect/0024',r.fit.REGIONS[0],numbers)
    assert limits == [5.,10.] and Image.open(io.BytesIO(image)).size == (1152,552)


def test_html_escapes_and_has_no_external_assets(tmp_path):
    data,source,candidate,report,archive=fixture(tmp_path)
    report['cases'][0]['key']='<script>alert(1)</script>'
    text=r.index_html(report).decode()
    assert '<script>' not in text and '&lt;script&gt;' in text
    assert 'https://' not in text and 'http://' not in text


def test_cli_requires_archive_before_loading(monkeypatch):
    def forbidden(*args): raise AssertionError('Should not load without archive')
    monkeypatch.setattr(r,'load',forbidden)
    with pytest.raises(SystemExit): r.main([])


def test_stat_comparison():
    assert r.close_stats({'a': [1.,None]}, {'a':[1.+1e-8,None]})
    assert not r.close_stats({'a':[1.,None]}, {'a':[1.1,None]})
