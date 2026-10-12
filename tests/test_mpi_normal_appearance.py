import hashlib
import io
import json
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import review_mpi_normal_appearance as a


def test_black_geometry_and_interior_disconnected():
    rgb = np.full((256, 256, 3), 100, np.uint8)
    rgb[:10, :20] = 0; rgb[100:102, 100:102] = 0
    s = a.black_statistics(rgb)
    assert s['exact_black_pixels'] == 204
    assert s['connected_black_pixels'] == 200 and s['disconnected_black_pixels'] == 4
    assert s['bbox_xyxy'] == [0, 0, 19, 9] and s['centroid_xy'] == [9.5, 4.5]
    assert s['interface_edges'] == 30
    assert s['adjacent_nonblack_2px'] == 64
    assert s['interior192_black_pixels'] == 0
    assert sum(map(sum, s['grid4x4_pixels'])) == 200


def test_black_connectivity_is_four_not_eight_and_not_near_black():
    rgb = np.full((256, 256, 3), 1, np.uint8)
    rgb[0, 0] = 0; rgb[1, 1] = 0
    s = a.black_statistics(rgb)
    assert s['connected_black_pixels'] == 1 and s['disconnected_black_pixels'] == 1
    assert s['interface_edges'] == 2


@pytest.mark.parametrize('value,area', [(0, 65536), (1, 0), (255, 0)])
def test_full_or_empty(value, area):
    s = a.black_statistics(np.full((256, 256, 3), value, np.uint8))
    assert s['connected_black_pixels'] == area and s['interface_edges'] == 0
    assert s['adjacent_nonblack_2px'] == 0
    if not area: assert s['bbox_xyxy'] is None and s['centroid_xy'] is None


@pytest.mark.parametrize('rgb', [np.zeros((256, 256), np.uint8), np.zeros((256, 256, 3)), np.zeros((32, 32, 3), np.uint8)])
def test_invalid_image(rgb):
    with pytest.raises(ValueError): a.black_statistics(rgb)


def make_photo(pid):
    return {'photo_id': pid, 'label': 0, 'regions': [
        {'photo_id': pid, 'region': r, 'member': f'extracted/{r}/train/good/{pid}.png'}
        for r in a.fit.REGIONS]}


def plan():
    training = [make_photo(f'{i:04d}') for i in range(256)]
    return {'training': {'photo_ids': [p['photo_id'] for p in training],
                         'rows': [r for p in training for r in p['regions']]},
            'cohorts': {'validation': [make_photo(f'{i:04d}') for i in range(256, 512)]}}


def test_groups_preserve_exact_roles_and_id_order():
    p = plan(); p['training']['rows'].reverse(); p['cohorts']['validation'].reverse()
    g = a.normal_groups(p)
    assert [x['photo_id'] for x in g['training']] == [f'{i:04d}' for i in range(256)]
    assert g['validation'][0]['photo_id'] == '0256'


@pytest.mark.parametrize('kind', ['overlap', 'label', 'test_path', 'region', 'ids', 'count', 'row_id'])
def test_bad_roles_refused(kind):
    p = plan()
    if kind == 'overlap': p['cohorts']['validation'][0] = make_photo('0000')
    if kind == 'label': p['cohorts']['validation'][0]['label'] = 1
    if kind == 'test_path': p['training']['rows'][0]['member'] = 'extracted/test/defect/0000.png'
    if kind == 'region': p['cohorts']['validation'][0]['regions'].pop()
    if kind == 'ids': p['training']['photo_ids'][0] = '9999'
    if kind == 'count': p['cohorts']['validation'].pop()
    if kind == 'row_id': p['cohorts']['validation'][0]['regions'][0]['photo_id'] = '9999'
    with pytest.raises(ValueError): a.normal_groups(p)


def fixture(tmp_path):
    rgb = np.full((256, 256, 3), 120, np.uint8); rgb[:8, :9] = 0
    raw = a.png_bytes(Image.fromarray(rgb)); g = {'training': [make_photo('0001')], 'validation': [make_photo('0002')]}
    for photos in g.values():
        for photo in photos:
            for r in photo['regions']:
                r.update(member_sha256=hashlib.sha256(raw).hexdigest(), decoded_rgb_sha256=hashlib.sha256(rgb.tobytes()).hexdigest())
    archive = tmp_path/'archive.zip'
    with zipfile.ZipFile(archive, 'w') as z:
        for photos in g.values():
            for photo in photos:
                for row in photo['regions']: z.writestr(row['member'], raw)
        z.writestr('extracted/test/defect/DO_NOT_READ.png', b'invalid png')
    return g, {'archive_sha256': a.fit.sha256_file(archive)}, archive


def test_roundtrip_no_test_reads_and_tamper(tmp_path, monkeypatch):
    groups, identity, archive = fixture(tmp_path)
    monkeypatch.setattr(a, 'prepare', lambda *args: (groups, identity))
    read = zipfile.ZipFile.read; accessed = []
    def track(self, name, *args, **kwargs):
        accessed.append(name)
        return read(self, name, *args, **kwargs)
    monkeypatch.setattr(zipfile.ZipFile, 'read', track)
    result = a.run(tmp_path, None, archive)
    assert result['normal_photos'] == 2 and result['normal_tiles'] == 24
    assert result['sheets'] == 2 and result['inference_calls'] == 0
    assert not any('/test/' in str(n) for n in accessed)
    output = a.Path(result['output']); sha = a.fit.sha256_file(output)
    assert a.run(tmp_path, None, archive) == result and a.fit.sha256_file(output) == sha
    with zipfile.ZipFile(output) as z:
        for n, h in json.loads(z.read('handoff-manifest.json')).items():
            assert hashlib.sha256(z.read(n)).hexdigest() == h
        d = json.loads(z.read('review.json'))
        assert d['summary']['training']['photos_with_connected_black'] == 1
        assert Image.open(io.BytesIO(z.read('training/sheets/page-01.png'))).size == (1072, 1184)
    target = tmp_path/'artifacts/analyses'/a.NAME/'review/review.json'
    target.write_bytes(b'corrupt')
    with pytest.raises(ValueError): a.run(tmp_path, None, archive)
    assert target.read_bytes() == b'corrupt'


def test_archive_changed_refused_before_output(tmp_path, monkeypatch):
    groups, identity, archive = fixture(tmp_path); identity['archive_sha256'] = 'bad'
    monkeypatch.setattr(a, 'prepare', lambda *args: (groups, identity))
    with pytest.raises(ValueError, match='archive differs'): a.run(tmp_path, None, archive)
    assert not (tmp_path/'artifacts').exists()


def test_contact_sheet_preserves_pixels_and_partial_page():
    rgb = np.full((256, 256, 3), [12, 34, 56], np.uint8)
    entries = [(f'{i:04d}', a.png_bytes(Image.fromarray(rgb))) for i in range(16)]
    sheet = np.asarray(Image.open(io.BytesIO(a.contact_sheet('training', entries, 1))))
    np.testing.assert_array_equal(sheet[80:336, 12:268], rgb)
    np.testing.assert_array_equal(sheet[920:1176, 804:1060], rgb)


def test_cli_argument_mapping(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(a, 'run', lambda *args: seen.append(args) or {})
    assert a.main(['--project', str(tmp_path), '--input', 'i.zip', '--archive', 'a.zip']) == 0
    assert seen[0] == (tmp_path, a.Path('i.zip'), a.Path('a.zip'))


def test_prepare_source_lock_and_actual_group_contract(tmp_path, monkeypatch):
    p = plan(); p['archive_sha256'] = 'archive'
    data = {'plan.json': p, 'fit-report.json': {'checkpoint_sha256': 'checkpoint'}}
    monkeypatch.setattr(a.parent, 'checked_input', lambda *args: (data, {}))
    for name in a.SOURCES:
        file = tmp_path/name; file.parent.mkdir(parents=True, exist_ok=True); file.write_bytes(b'line\r\n')
    groups, identity = a.prepare(tmp_path, None)
    assert all(len(photos) == 256 for photos in groups.values())
    assert identity['policy']['test_images_decoded'] == 0
    assert identity['sources_lf_sha256'][a.SOURCES[0]] == hashlib.sha256(b'line\n').hexdigest()


def test_summary_counts_photos_not_independent_tiles():
    rgb = np.full((256, 256, 3), 100, np.uint8); rgb[:2, :3] = 0
    stats = a.black_statistics(rgb)
    rows = [{'group': 'training', 'photo_id': '0001', 'region': r, 'statistics': stats} for r in a.fit.REGIONS]
    s = a.summarize(rows)['training']
    assert s['photos_with_connected_black'] == 1 and s['tiles_with_connected_black'] == 12
    assert s['mean_grid4x4_black_pixels'][0][0] == 6
    assert s['tiles_touching_frame'] == {'top': 12, 'bottom': 0, 'left': 12, 'right': 0}


def test_png_identity_failure_and_duplicate_archive(tmp_path, monkeypatch):
    groups, identity, archive = fixture(tmp_path)
    monkeypatch.setattr(a, 'prepare', lambda *args: (groups, identity))
    row = groups['training'][0]['regions'][0]; original_hash = row['member_sha256']
    row['member_sha256'] = 'bad'
    with pytest.raises(ValueError, match='bytes differ'): a.run(tmp_path, None, archive)
    row['member_sha256'] = original_hash
    with zipfile.ZipFile(archive, 'a') as z:
        with pytest.warns(UserWarning): z.writestr(row['member'], b'bad')
    identity['archive_sha256'] = a.fit.sha256_file(archive)
    with pytest.raises(ValueError, match='Duplicate'): a.run(tmp_path, None, archive)


def test_matches_existing_connected_black_rule():
    from scripts.mpi_normal_map_cards import statistics
    rng = np.random.default_rng(4)
    rgb = rng.integers(1, 256, (256, 256, 3), dtype=np.uint8)
    yy, xx = np.indices((256, 256)); rgb[xx+yy < 130] = 0; rgb[150:155, 150:155] = 0
    new = a.black_statistics(rgb); old = statistics(Image.fromarray(rgb), np.ones((256, 256), np.float32))['partitions']
    assert new['connected_black_pixels'] == old['edge_connected_black']['area_pixels']
    assert new['adjacent_nonblack_2px'] == old['adjacent_nonblack_2px']['area_pixels']
