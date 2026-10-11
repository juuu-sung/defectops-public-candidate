import copy
import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import review_mpi_patchcore_original_edges as a


def test_edge_stats_and_gt():
    m = np.ones((256, 256), np.float32); m[:4] = 4
    gt = np.zeros_like(m, dtype=bool); gt[:2, :10] = True
    s = a.map_stats(m, gt)
    assert s['top1'] == 4 and s['top656_outer32_fraction'] == 1
    assert s['gt']['pixels'] == s['gt']['outer32_pixels'] == 20
    assert s['peak_xy'] == [0, 0] and s['peak_in_outer32']
    assert s['top656_outer32_tie_bounds'] == [1., 1.]


def test_constant_ties_are_explicit_not_evidence_of_border_response():
    s = a.map_stats(np.ones((256, 256), np.float32))
    assert s['cutoff_tie_pixels'] == 65536
    assert s['top656_outer32_tie_bounds'] == [0, 1]


def test_interior_and_empty_gt():
    m = np.zeros((256, 256), np.float32); m[100:130, 100:130] = 2
    s = a.map_stats(m, np.zeros(m.shape, bool))
    assert s['top656_outer32_fraction'] == 0
    assert s['gt']['outer32_fraction'] is None and s['gt']['pixels'] == 0
    assert not s['peak_in_outer32']


@pytest.mark.parametrize('kind', ['shape', 'dtype', 'nan', 'negative', 'mask_shape', 'mask_dtype'])
def test_invalid_arrays_fail(kind):
    m = np.ones((256, 256), np.float32); mask = None
    if kind == 'shape': m = m[:128]
    if kind == 'dtype': m = m.astype('float64')
    if kind == 'nan': m[0, 0] = np.nan
    if kind == 'negative': m[0, 0] = -1
    if kind == 'mask_shape': mask = np.zeros((2, 2), bool)
    if kind == 'mask_dtype': mask = np.zeros((256, 256), np.uint8)
    with pytest.raises(ValueError): a.map_stats(m, mask)


def fixture():
    pixels = np.full((256, 256, 3), 110, np.uint8)
    png = a.png_bytes(Image.fromarray(pixels))
    photo = {'key': 'validation/0147', 'photo_id': '0147', 'label': 0,
             'regions': [{'region': r, 'member': f'extracted/{r}/train/good/0147.png',
                          'member_sha256': hashlib.sha256(png).hexdigest(),
                          'decoded_rgb_sha256': hashlib.sha256(pixels.tobytes()).hexdigest()}
                         for r in a.fit.REGIONS]}
    maps = np.ones((12, 256, 256), np.float32)
    sample = {k: photo[k] for k in ('key', 'photo_id', 'label')}
    sample.update(regions=[{'region': r['region'], 'member': r['member'], 'top1': 1., 'native_raw': 1.}
                           for r in photo['regions']], photo_seconds=1.)
    contract = {'threshold': .5, 'field': 'top1', 'scale_kind': 'q90',
                'regions': [{'region': r, 'center': 0., 'scale': 1.} for r in a.fit.REGIONS]}
    return photo, sample, maps, png, contract


def test_read_saved_receipt_and_tamper(tmp_path):
    photo, sample, maps, png, contract = fixture()
    root = tmp_path/'artifacts/analyses'/a.parent.raw_bank.NAME/'validation'; root.mkdir(parents=True)
    with a.fit.phase_stage(root, 'photo-000') as stage:
        a.fit.write_json(stage/'sample.json', sample); np.save(stage/'maps.npy', maps)
        receipt = a.fit.seal_phase(stage, 'photo-000', a.parent.EXECUTION_SHA,
                    {'photo_sha256': a.fit.digest(photo), 'checkpoint_sha256': 'checkpoint'}, ['sample.json', 'maps.npy'])
    data = {'plan.json': {'cohorts': {'validation': [photo]}},
            'fit-report.json': {'checkpoint_sha256': 'checkpoint'},
            'evaluation/prediction-receipts.json': {'validation/photo-000/receipt.json': receipt}}
    assert a.read_saved(tmp_path, data, {'validation': [sample]}, 'validation', 0)[1] == sample
    (root/'photo-000/maps.npy').write_bytes(b'corrupt')
    with pytest.raises(ValueError): a.read_saved(tmp_path, data, {'validation': [sample]}, 'validation', 0)


def test_roundtrip_no_model_and_output_tamper(tmp_path, monkeypatch):
    photo, sample, maps, png, contract = fixture()
    archive = tmp_path/'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for row in photo['regions']: z.writestr(row['member'], png)
    identity = {'archive_sha256': a.fit.sha256_file(archive), 'focus_regions': [a.fit.REGIONS[0]]}
    data = {'contract.json': contract}; raw = {'validation': [sample], 'test': []}
    monkeypatch.setattr(a, 'prepare', lambda *args: (data, raw, identity))
    monkeypatch.setattr(a, 'read_saved', lambda *args: (photo, sample, maps, {}))
    monkeypatch.setattr(a.engine, 'Runtime', lambda *args: pytest.fail('No runtime/model'))
    result = a.run(tmp_path, None, archive)
    output = tmp_path/'artifacts/handoff'/f'{a.NAME}.zip'; h = a.fit.sha256_file(output)
    assert result['model_inference_calls'] == 0 and result['cards'] == 1
    assert a.run(tmp_path, None, archive) == result and a.fit.sha256_file(output) == h
    with zipfile.ZipFile(output) as z:
        report = json.loads(z.read('review.json'))
        assert report['summary']['normal_groups']['non_alert_control']['count'] == 0
        for n, sha in json.loads(z.read('handoff-manifest.json')).items():
            assert hashlib.sha256(z.read(n)).hexdigest() == sha
        assert Image.open(io.BytesIO(z.read('validation/0147/00/card.png'))).size == (1088, 380)
    target = tmp_path/'artifacts/analyses'/a.NAME/'review/review.json'; target.write_bytes(b'corrupt')
    with pytest.raises(ValueError): a.run(tmp_path, None, archive)


def test_summary_exclusion_and_region_matching():
    _, _, maps, _, _ = fixture()
    gt = np.zeros((256, 256), bool); gt[0, 0] = True
    tile = {'region': a.fit.REGIONS[0], **a.map_stats(maps[0], gt)}
    case = {'group': 'test', 'key': 'defect/0001', 'location_eligible': True, 'tiles': [tile]}
    excluded = {**case, 'key': 'defect/0006', 'location_eligible': False}
    s = a.summary([case, excluded], [a.fit.REGIONS[0]])
    assert s['all_gt_positive']['tiles'] == s['all_gt_positive']['photos'] == 1
    assert s['fp_winning_regions_gt_positive']['outer32_gt_pixels'] == 1
    assert s['outer32_area_fraction'] == .4375


def test_cli_mapping(monkeypatch, tmp_path):
    seen = []; monkeypatch.setattr(a, 'run', lambda *args: seen.append(args) or {})
    assert a.main(['--project', str(tmp_path), '--input', 'in.zip', '--archive', 'a.zip']) == 0
    assert seen[0] == (tmp_path, a.Path('in.zip'), a.Path('a.zip'))


def test_prepare_preserves_exact_cases_and_predictions(monkeypatch, tmp_path):
    _, sample, _, _, contract = fixture()
    normal = []
    for i in range(256):
        s = copy.deepcopy(sample)
        s['photo_id'] = a.EXPECTED_FP[i] if i < 15 else f'control{i}'
        s['key'] = 'validation/'+s['photo_id']
        if i >= 15:
            for r in s['regions']: r['top1'] = 0.
        normal.append(s)
    defects = []
    for i in range(55):
        s = copy.deepcopy(sample); s.update(label=1, key=f'defect/{i:04d}', photo_id=f'{i:04d}')
        defects.append(s)
    raw = {'validation': normal, 'test': defects, 'development': [], 'calibration': []}
    predictions = {g: [{**{k: s[k] for k in ('key', 'label', 'photo_id')}, **a.decision(s, contract)}
                      for s in samples] for g, samples in raw.items()}
    data = {'contract.json': contract, 'evaluation/comparison.json': {'predictions': {'candidate': predictions}},
            'evaluation/location.json': {'location_excluded': ['defect/0006']},
            'evaluation/result.json': {'status': 'raw_bank_observed_comparison_completed'},
            'fit-report.json': {'checkpoint_sha256': 'x'}, 'plan.json': {'archive_sha256': 'y'}}
    for n in a.SOURCES:
        target = tmp_path/n; target.parent.mkdir(parents=True, exist_ok=True); target.write_text('test')
    monkeypatch.setattr(a.parent, 'checked_input', lambda *args: (data, raw))
    assert a.prepare(tmp_path, None)[2]['false_positive_ids'] == a.EXPECTED_FP
    data['evaluation/comparison.json']['predictions']['candidate']['validation'][0]['score'] += 1
    with pytest.raises(ValueError, match='predictions'): a.prepare(tmp_path, None)


def test_missing_or_wrong_model_receipt_is_not_replaced_by_inference(tmp_path):
    with pytest.raises((FileNotFoundError, ValueError)):
        a.read_saved(tmp_path, {'plan.json': {'cohorts': {'validation': [{}]}},
            'evaluation/prediction-receipts.json': {}}, {'validation': [{}]}, 'validation', 0)


def test_index_links_and_escaped_names():
    text = a.index_html([{'key': 'validation/0147', 'region': 'grid1-1', 'prefix': 'validation/0147/00'},
                        {'key': 'defect/<x>', 'region': 'grid1-2', 'prefix': 'defect/test/01'}]).decode()
    assert 'validation/0147/00/card.png' in text and 'defect/&lt;x&gt;' in text


def test_defect_export_keeps_all_positive_matched_tiles_and_exclusion(tmp_path, monkeypatch):
    photo, sample, maps, png, contract = fixture()
    photos = []; samples = []; gt = np.zeros((256, 256), bool); gt[80:90, 80:90] = True
    mask_raw = a.png_bytes(Image.fromarray(gt.astype(np.uint8)*255))
    for pid in ('0001', '0006'):
        ph = copy.deepcopy(photo); ph.update(key='defect/'+pid, photo_id=pid, label=1)
        sa = copy.deepcopy(sample); sa.update(key=ph['key'], photo_id=pid, label=1)
        for j, row in enumerate(ph['regions']):
            row.update(member=f'original/{pid}/{j}.png', mask_member=f'mask/{pid}/{j}.png')
            sa['regions'][j]['member'] = row['member']
        photos.append(ph); samples.append(sa)
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for ph in photos:
            for row in ph['regions']:
                z.writestr(row['member'], png); z.writestr(row['mask_member'], mask_raw)
    data = {'contract.json': contract}; raw = {'validation': [], 'test': samples}
    identity = {'archive_sha256': a.fit.sha256_file(archive), 'focus_regions': a.fit.REGIONS[:2]}
    monkeypatch.setattr(a, 'prepare', lambda *args: (data, raw, identity))
    monkeypatch.setattr(a, 'read_saved', lambda project, data, raw, group, i: (photos[i], samples[i], maps, {}))
    calls = []
    def masks(archive, ph):
        calls.append(ph['key']); return [gt]*12
    monkeypatch.setattr(a.ev, 'read_masks', masks)
    result = a.run(tmp_path, None, archive)
    assert calls == ['defect/0001'] and result['cards'] == 2
    with zipfile.ZipFile(result['output']) as z:
        report = json.loads(z.read('review.json'))
        assert report['summary']['all_gt_positive']['tiles'] == 12
        assert report['summary']['fp_winning_regions_gt_positive']['tiles'] == 2
        assert report['cases'][1]['location_eligible'] is False
        assert 'defect/0001/00/mask-original.png' in z.namelist()
        assert not any(n.startswith('defect/0006/') for n in z.namelist())
