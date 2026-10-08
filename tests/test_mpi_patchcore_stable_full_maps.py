"""Independent map arithmetic plus real v2 orchestration with fake GPU runtime."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
import pytest

from scripts import compare_mpi_patchcore_full_maps_v2 as v
from scripts import mpi_patchcore_stable_maps as m


def test_reference_nearest_reflect_and_asymmetric_cross_correlation():
    rng = np.random.default_rng(5)
    values = rng.uniform(0, 60, 4096)
    kernel = rng.uniform(0, 1, (33, 33)); kernel /= kernel.sum()
    result = m.numpy_reference(values, kernel)
    image = np.pad(np.repeat(np.repeat(values.reshape(64, 64), 4, 0), 4, 1), 16, mode='reflect')
    for y, x in [(0, 0), (255, 255), (0, 128), (144, 97), (16, 16)]:
        assert result[y, x] == pytest.approx(np.sum(image[y:y+33, x:x+33]*kernel), abs=1e-11)
    assert np.allclose(m.numpy_reference(np.ones(4096)*17, kernel), 17, atol=1e-11)
    assert not m.numpy_reference(np.zeros(4096), kernel).any()


@pytest.mark.parametrize('bad', [np.zeros(3), np.full(4096, np.nan), -np.ones(4096)])
def test_bad_distances_rejected(bad):
    with pytest.raises(ValueError): m.numpy_reference(bad, np.ones((33, 33))/1089)


def test_reference_contract_rejects_wrong_interpolation_or_kernel():
    with pytest.raises(ValueError): m.kernel_array(np.ones((33, 33)))
    with pytest.raises(ValueError): m.compare(np.ones((256, 256)), np.ones((256, 256))+.0001, enforce=True)
    out = m.compare(np.ones((256, 256)), np.ones((256, 256))+.01)
    assert out['outside_float64_tolerance'] == 65536  # diagnostic alone does not silently pass equivalence


def test_tf32_restored_on_failure_and_no_matmul_mutation():
    t = SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace(allow_tf32=True),
                        cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False))))
    with pytest.raises(RuntimeError):
        with m.map_tf32(t, False):
            assert t.backends.cudnn.allow_tf32 is False
            raise RuntimeError('failure')
    assert t.backends.cudnn.allow_tf32 is True and t.backends.cuda.matmul.allow_tf32 is False


def test_map_only_comparison_reports_cross_precision_but_keeps_official_replay_gate():
    t = SimpleNamespace(backends=SimpleNamespace(cudnn=SimpleNamespace(allow_tf32=True)))
    mapper = object.__new__(m.StableMaps); mapper.torch = t
    mapper.kernel = np.ones((33, 33))/1089
    settings = []
    def gpu_map(x):
        settings.append(t.backends.cudnn.allow_tf32)
        return np.full((256, 256), x[0], np.float32)
    mapper.runtime = SimpleNamespace(maps=gpu_map)
    mapper.stable = lambda x: m.numpy_reference(x, mapper.kernel)
    mapper.check_unchanged = lambda: None
    report, arrays = mapper.gate(np.ones(4096), np.ones(4096)*1.02, np.ones((256, 256)))
    assert settings == [True, False, True, False] and t.backends.cudnn.allow_tf32
    assert report['cross_precision_diagnostic_only']['outside_float64_tolerance'] == 65536
    assert len(arrays) == 6
    with pytest.raises(ValueError, match='Official stored-distance'):
        mapper.gate(np.ones(4096), np.ones(4096), np.ones((256, 256))*2)


def fixture_run(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('fullmaps_fixture', Path(__file__).with_name('test_mpi_patchcore_full_maps.py'))
    helper = importlib.util.module_from_spec(spec); spec.loader.exec_module(helper)
    archive, runtime, state, parent, parent_sig = helper.fixture_run(tmp_path, monkeypatch)
    monkeypatch.setattr(v, 'POLICY', {**v.POLICY, 'tiles': 1, 'training_photos': 2, 'candidates': 24576, 'bank': 2})
    record = v.c.select_tiles()[0]
    old = tmp_path/'artifacts/analyses'/v.d.NAME; old.mkdir(parents=True)
    identity = {'name': v.d.NAME, 'policy': v.d.POLICY, 'tiles': [record], 'parent_execution_sha256': parent_sig,
                'checkpoint_sha256': parent['model-receipt.json']['files']['model.pt'],
                'sources_lf_sha256': {n: hashlib.sha256((tmp_path/n).read_bytes()).hexdigest() for n in v.d.SOURCES}}
    sig = v.fit.digest(identity); v.p.publish(old, 'identity', sig, identity)
    v.d.collect_tile(old, 'tile-00', sig, record, v.c.read_overlap(None), runtime({}))
    for n in v.SOURCES: (tmp_path/n).write_text('fixture')
    reviewed = {'failed-identity.json': v.fit.canonical(identity),
                'plan.json': v.fit.canonical({'failed_identity_sha256': sig, 'sources_lf_sha256': {}}),
                'tile-00/double-distances.npy': v.c.np_bytes(np.ones(4096, np.float64)),
                'tile-00/double-indices.npy': v.c.np_bytes(np.zeros(4096, np.int64)),
                'tile-00/saved-map.npy': v.c.np_bytes(np.ones((256, 256), np.float32)),
                'tile-00/report.json': v.fit.canonical({'key': record['key'], 'region': record['source_row']['region']})}
    monkeypatch.setattr(v, 'checked_audit', lambda path: reviewed)
    class Mapper:
        calls = 0; fail = False
        def __init__(self, runtime):
            self.kernel = np.ones((33, 33))/1089
            self.meta = {'fixture': True}
        def stable(self, x):
            if Mapper.fail: raise ValueError('reference gate failed')
            return np.full((256, 256), x[0], np.float64)
        def gate(self, n, b, old):
            Mapper.calls += 1
            arrays = {f'{a}-{b}.npy': np.ones((256, 256), np.float64) for a in ('native', 'double')
                      for b in ('gpu-tf32-true', 'gpu-tf32-false', 'cpu-float64')}
            return {'passed': True}, arrays
        def check_unchanged(self): pass
    return archive, runtime, state, Mapper, old, reviewed


def snapshot(folder):
    return {p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in folder.rglob('*') if p.is_file()}


def test_end_to_end_old_preserved_interrupt_resume_reexport(tmp_path, monkeypatch):
    archive, runtime, state, mapper, old, reviewed = fixture_run(tmp_path, monkeypatch)
    before = snapshot(old); state.interrupt_at = 2
    with pytest.raises(KeyboardInterrupt): v.run(tmp_path, archive, runtime, mapper)
    root = tmp_path/'artifacts/analyses'/v.NAME
    assert (root/'precision-gate/value.json').exists() and (root/'search-000/cost.json').exists()
    assert not (root/'search-001').exists()
    state.interrupt_at = None
    result = v.run(tmp_path, archive, runtime, mapper)
    handoff = Path(result['output']); data = handoff.read_bytes()
    assert v.run(tmp_path, archive, runtime, mapper) == result and handoff.read_bytes() == data
    assert mapper.calls == 1 and state.traced == 1 and snapshot(old) == before
    with zipfile.ZipFile(handoff) as z:
        report = json.loads(z.read('result.json'))
        assert not report['training'] and not report['threshold_changed'] and not report['photo_predictions_computed']
        assert np.load(io.BytesIO(z.read('bank-maps.npy')), allow_pickle=False).dtype == np.float64
        assert 'map-gate-00/double-gpu-tf32-false.npy' in z.namelist()
    (root/'map-gate-00/report.json').write_text('{}')
    with pytest.raises(ValueError): v.run(tmp_path, archive, runtime, mapper)


def test_failed_reference_gate_prevents_full_search(tmp_path, monkeypatch):
    archive, runtime, state, mapper, old, _ = fixture_run(tmp_path, monkeypatch)
    mapper.fail = True; before = snapshot(old)
    with pytest.raises(ValueError, match='reference gate'): v.run(tmp_path, archive, runtime, mapper)
    assert state.searches == 0 and snapshot(old) == before


@pytest.mark.parametrize('bad_kind', ['source', 'feature', 'audited_index', 'audited_map'])
def test_corrupted_input_rejected_before_search(tmp_path, monkeypatch, bad_kind):
    archive, runtime, state, mapper, old, reviewed = fixture_run(tmp_path, monkeypatch)
    if bad_kind == 'source': (tmp_path/v.d.SOURCES[0]).write_text('changed')
    elif bad_kind == 'feature': (old/'tile-00/features.npy').write_bytes(b'changed')
    elif bad_kind == 'audited_index': reviewed['tile-00/double-indices.npy'] = v.c.np_bytes(np.ones(4096, np.int64))
    else: reviewed['tile-00/saved-map.npy'] = v.c.np_bytes(np.ones((256, 256), np.float32)*2)
    with pytest.raises(ValueError): v.run(tmp_path, archive, runtime, mapper)
    assert state.searches == 0


def test_checked_audit_hash_inventory_and_payload(tmp_path, monkeypatch):
    path = tmp_path/'audit.zip'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a', b'payload'); z.writestr('handoff-manifest.json', json.dumps({'a': hashlib.sha256(b'payload').hexdigest()}))
    with pytest.raises(ValueError, match='reviewed'): v.checked_audit(path)
    monkeypatch.setattr(v, 'AUDIT_SHA', v.fit.sha256_file(path))
    assert v.checked_audit(path)['a'] == b'payload'
    with zipfile.ZipFile(path, 'w') as z:
        z.writestr('a', b'changed'); z.writestr('handoff-manifest.json', json.dumps({'a': hashlib.sha256(b'payload').hexdigest()}))
    monkeypatch.setattr(v, 'AUDIT_SHA', v.fit.sha256_file(path))
    with pytest.raises(ValueError, match='payload'): v.checked_audit(path)


def test_actual_cpu_generator_if_installed():
    t = pytest.importorskip('torch')
    am = pytest.importorskip('anomalib.models.image.patchcore.anomaly_map')
    runtime = SimpleNamespace(torch=t, model=SimpleNamespace(model=SimpleNamespace(anomaly_map_generator=am.AnomalyMapGenerator())))
    mapper = m.StableMaps(runtime)
    x = np.random.default_rng(71).uniform(0, 70, 4096)
    assert mapper.stable(x).shape == (256, 256)
    assert runtime.model.model.anomaly_map_generator.blur.kernel.dtype == t.float32


def test_actual_gpu_map_isolation_if_available():
    t = pytest.importorskip('torch')
    if not t.cuda.is_available(): pytest.skip('Actual CUDA required')
    am = pytest.importorskip('anomalib.models.image.patchcore.anomaly_map')
    runtime = object.__new__(v.d.Runtime); runtime.torch = t
    runtime.model = SimpleNamespace(model=SimpleNamespace(anomaly_map_generator=am.AnomalyMapGenerator().cuda()))
    previous = t.backends.cudnn.allow_tf32
    try:
        t.backends.cudnn.allow_tf32 = True
        mapper = m.StableMaps(runtime); x = np.linspace(0, 60, 4096).astype(np.float32)
        saved = runtime.maps(x); report, arrays = mapper.gate(x, x.astype(np.float64), saved)
        assert report['official_replay_maximum_difference'] <= 1e-4
        assert arrays['double-cpu-float64.npy'].dtype == np.float64
        mapper.check_unchanged()
    finally: t.backends.cudnn.allow_tf32 = previous
