import copy
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from scripts import analyze_mpi_patchcore_scaling_sensitivity as s
from scripts import export_mpi_patchcore_scaling_review as export


def test_shapley_assigns_interaction_equally_and_conserves():
    v = [2*(bool(i & 1))+3*(bool(i & 2))+4*(bool(i & 4))+12*(i == 7) for i in range(8)]
    assert s.allocation(v) == pytest.approx({"center": 6., "scale": 7., "threshold": 8.})
    assert s.allocation([5]*8) == dict.fromkeys(s.FACTORS, 0.)
    with pytest.raises(ValueError): s.allocation([1, 2])
    with pytest.raises(ValueError): s.allocation([np.nan]*8)


def test_mixing_preserves_inputs_and_factor_order():
    a = {"threshold": 3, "regions": [{"center": 1, "scale": 2}]}
    b = {"threshold": 30, "regions": [{"center": 10, "scale": 20}]}
    saved = copy.deepcopy((a, b))
    assert s.mixed(a, b, 0) == a and s.mixed(a, b, 7) == b
    assert s.mixed(a, b, 2) == {"threshold": 3, "regions": [{"center": 1, "scale": 20}]}
    assert (a, b) == saved
    with pytest.raises(ValueError): s.mixed(a, b, 8)


def test_higher_quantile_owners_keep_ties_and_multiplicity():
    x = np.zeros((3, 12)); x[1, :2] = 5; x[2, 0] = 5
    method = {"regions": [{"center": 0, "scale": 1}]*12, "threshold": 5.}
    rows = s.boundary_owners(x, [{"key": str(i)} for i in range(3)], [0, 1, 1, 2], method)
    assert rows[0]["multiplicity"] == 2 and len(rows[0]["regions"]) == 2
    assert rows[1]["key"] == '2' and len(rows) == 2
    with pytest.raises(ValueError): s.boundary_owners(x, [{"key": str(i)} for i in range(3)], [0, 1], dict(method, threshold=4.))


def test_order_pivots_even_median_and_higher():
    values = np.repeat(np.array([1, 2, 3, 4])[:, None], 12, axis=1)
    p = s.order_pivots(values, [{"key": str(i)} for i in range(4)], [3, 0, 2, 1], s.p.fit.REGIONS[0])
    assert [p[k]["value"] for k in ['median_lower', 'median_upper', 'q90_higher']] == [2, 3, 4]


def test_output_never_overwrites_different_report(tmp_path):
    path = tmp_path/'report.json'
    s.write_new(path, {"x": 1}); s.write_new(path, {"x": 1})
    with pytest.raises(ValueError): s.write_new(path, {"x": 2})
    assert json.loads(path.read_text()) == {"x": 1}


def test_checked_zip_rejects_changed_payload(tmp_path):
    path = tmp_path/'a.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('x.json', '{}')
        z.writestr('handoff-manifest.json', json.dumps({'x.json': '0'*64}))
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match='Payload'): s.checked_bundle(path, sha)
    with pytest.raises(ValueError, match='ZIP fingerprint'): s.checked_bundle(path, 'changed')


def test_review_selection_deduplicates_key_region_and_preserves_reasons():
    report = {'normal_cases': [{'key': 'good/0018', 'candidate_winning_region': 'r'}],
              'summaries': {'both': {'bad_draw_boundary_owners': [{'key': 'good/0018', 'region': 'r'}],
                                   'worst_draw_development_pivots': {'q90_higher': {'key': 'development/a'}}}},
              'fixed_development_pivots': {'q90_higher': {'key': 'development/a'}}, 'dominant_defect_region': 'r'}
    rows = export.selection(report)
    assert len(rows) == 2 and all(len(r['reasons']) == 2 for r in rows)


def test_export_png_checks_file_and_decoded_hashes(tmp_path):
    image = Image.new('RGB', (256, 256), 'red'); buf = io.BytesIO(); image.save(buf, format='PNG')
    raw = buf.getvalue(); path = tmp_path/'image.zip'
    with zipfile.ZipFile(path, 'w') as z: z.writestr('normal.png', raw)
    row = {'member': 'normal.png', 'member_sha256': hashlib.sha256(raw).hexdigest(),
           'decoded_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()}
    with zipfile.ZipFile(path) as z:
        assert export.checked_png(z, row) == raw
        with pytest.raises(ValueError, match='pixels'): export.checked_png(z, dict(row, decoded_rgb_sha256='bad'))
        with pytest.raises(ValueError, match='bytes'): export.checked_png(z, dict(row, member_sha256='bad'))


def test_export_bundle_rerun_checks_each_payload(tmp_path):
    path = tmp_path/'out.zip'; payload = {'review.json': b'{}', 'tiles/0/map.npy': b'fake'}
    export.write_bundle(path, payload)
    before = path.read_bytes(); export.write_bundle(path, payload)
    assert path.read_bytes() == before
    with pytest.raises(ValueError): export.write_bundle(path, {'review.json': b'changed'})
    with zipfile.ZipFile(path) as z: contents = {n: z.read(n) for n in z.namelist()}
    contents['tiles/0/map.npy'] = b'corrupt'
    with zipfile.ZipFile(path, 'w') as z:
        for n, raw in contents.items(): z.writestr(n, raw)
    with pytest.raises(ValueError): export.write_bundle(path, payload)


def test_trace_draw_detects_tampered_counts():
    rng = np.random.default_rng(1)
    arrays = {g: rng.random((n, 12)) for g, n in [('development', 128), ('calibration', 169), ('test', 109), ('validation', 256)]}
    raw = {'test': [{'label': int(i < 55)} for i in range(109)],
           'calibration': [{'key': str(i)} for i in range(169)]}
    fixed = s.p.comparison.fit_method(arrays['development'], arrays['calibration'])
    trial = {'development_indices': list(range(128)), 'calibration_indices': list(range(169)),
             'candidate': {'valid': True, 'counts': {}, 'threshold': fixed['threshold']}}
    with pytest.raises(ValueError, match='does not reproduce'): s.trace_draw(raw, arrays, fixed, trial)


def test_export_end_to_end_saved_cache_only_and_corruption_stop(tmp_path, monkeypatch):
    archive = tmp_path/'source.zip'
    buf = io.BytesIO(); image = Image.new('RGB', (256, 256), 'red'); image.save(buf, format='PNG')
    png = buf.getvalue()
    cohorts = {}
    with zipfile.ZipFile(archive, 'w') as z:
        for group, key in [('development', 'development/0001'), ('test', 'good/0002')]:
            rows = []
            for region in s.p.fit.REGIONS:
                member = f'{group}/{region}.png'; z.writestr(member, png)
                rows.append({'region': region, 'member': member, 'member_sha256': hashlib.sha256(png).hexdigest(),
                             'decoded_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest()})
            cohorts[group] = [{'key': key, 'label': 0, 'photo_id': key.split('/')[1], 'regions': rows}]
    plan = {'cohorts': cohorts, 'archive_sha256': s.p.fit.sha256_file(archive)}
    sig = s.p.fit.digest(plan)
    data = {'plan.json': plan, 'contract.json': {}, 'fit-report.json': {'checkpoint_sha256': 'checkpoint'},
            'prediction-receipts.json': {}, 'evaluation/prediction-receipts.json': {}}
    paths = []
    for group, photos in cohorts.items():
        photo = photos[0]; stage = 'first-development' if group == 'development' else 'test'
        root = tmp_path/'artifacts'/s.p.NAME if group == 'development' else tmp_path/'artifacts/analyses'/s.p.NAME
        folder = root/stage/'photo-000'; folder.mkdir(parents=True)
        paths.append(folder)
        maps = np.ones((12, 256, 256), dtype=np.float32)
        np.save(folder/'maps.npy', maps)
        sample = {k: photo[k] for k in ['key', 'photo_id', 'label']}
        sample['regions'] = [{k: r[k] for k in ['region', 'member']} | {'top1': 1., 'native_raw': 1.} for r in photo['regions']]
        s.p.fit.write_json(folder/'sample.json', sample)
        receipt = s.p.fit.seal_phase(folder, 'photo-000', sig,
            {'checkpoint_sha256': 'checkpoint', 'photo_sha256': s.p.fit.digest(photo)}, ['sample.json', 'maps.npy'])
        name = 'prediction-receipts.json' if group == 'development' else 'evaluation/prediction-receipts.json'
        data[name][f'{stage}\\photo-000\\receipt.json'] = receipt
    report = {'normal_cases': [{'key': ps[0]['key'], 'candidate_winning_region': s.p.fit.REGIONS[0]} for ps in cohorts.values()],
              'summaries': {}, 'dominant_defect_region': s.p.fit.REGIONS[0],
              'fixed_development_pivots': {'q90_higher': {'key': 'development/0001'}}}
    monkeypatch.setattr(s, 'checked_bundle', lambda *args: data)
    monkeypatch.setattr(s, 'analyze', lambda *args: report)
    result = export.run(tmp_path, archive)
    assert result['tiles'] == 2
    with zipfile.ZipFile(result['output']) as z:
        assert len(z.namelist()) == 7
        assert z.read('tiles/000/source.png') == png
        np.testing.assert_array_equal(np.load(io.BytesIO(z.read('tiles/001/map.npy')), allow_pickle=False), maps[0])
    assert export.run(tmp_path, archive) == result
    np.save(paths[0]/'maps.npy', maps*2)
    with pytest.raises(ValueError, match='incomplete or changed'): export.run(tmp_path, archive)
