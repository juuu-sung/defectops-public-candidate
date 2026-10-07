from contextlib import contextmanager, nullcontext
import copy
import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import zipfile

import numpy as np
from PIL import Image
import pytest

from scripts import diagnose_mpi_patchcore_candidate_coverage as d


def test_subset_exact_and_corruption():
    raw = np.arange(40, dtype=np.float32).reshape(10, 4)
    selected = d.validate_indices([8, 1, 6], 10, 3); bank = raw[selected].copy()
    assert d.validate_subset(bank, raw[:5], selected, 0) == 1
    assert d.validate_subset(bank, raw[5:], selected, 5) == 2
    bank[0, 0] += 1
    with pytest.raises(ValueError, match='exact recorded subset'): d.validate_subset(bank, raw, selected, 0)


@pytest.mark.parametrize('ids', [[0, 0], [-1, 1], [0, 4], [0., 1.], [0]])
def test_bad_selection_rejected(ids):
    with pytest.raises(ValueError): d.validate_indices(ids, 4, 2)


def test_merge_preserves_global_tie_and_nonfinite_rejected():
    distances, indices = d.merge_best(np.array([2., 3., 4.]), np.array([9, 7, 2]), np.array([2., 2., 4.]), np.array([1, 12, 8]))
    assert indices.tolist() == [1, 12, 2] and distances.tolist() == [2., 2., 4.]
    with pytest.raises(ValueError): d.merge_best(distances, indices, np.array([2., np.nan, 1.]), indices)


def test_origin_stride_and_full_never_worse():
    training = {'photo_ids': ['a', 'b'], 'rows': [{'n': i} for i in range(24)]}
    source = d.origin(12288+1024+33, training)
    assert source['photo_id'] == 'b' and source['row']['n'] == 13
    assert (source['grid_y'], source['grid_x']) == (2, 2)
    with pytest.raises(ValueError, match='exceeds subset'):
        d.comparison([{}], np.array([1.]), np.array([0]), np.array([2.]), np.array([0]), np.array([0]), training)
    out = d.comparison([{}], np.array([2.]), np.array([0]), np.array([1.]), np.array([1]), np.array([0]), training)
    assert out[0]['relative_drop'] == .5 and not out[0]['full_neighbor_is_in_bank']


def test_capture_restores_after_error():
    class Tensor:
        shape = (4096, 768)
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return np.zeros(2)
    class Core:
        def nearest_neighbors(self, embedding, n_neighbors): return Tensor(), Tensor()
    core = Core()
    with pytest.raises(RuntimeError):
        with d.capture(core) as out:
            core.nearest_neighbors(Tensor(), 1)
            assert len(out) == 1
            raise RuntimeError('stop')
    assert 'nearest_neighbors' not in vars(core)


def test_float_replay_only_not_identifiers():
    assert d.same_numbers({'v': 1.0, 'id': 'a'}, {'v': 1.+1e-15, 'id': 'a'})
    assert not d.same_numbers({'v': 1.0}, {'v': 1.01})
    assert not d.same_numbers({'id': 'a'}, {'id': 'b'})


def test_fixed_selection_requires_six_gt_and_fourteen_winners():
    report = {'cases': [{'key': f'validation/{i}', 'label': 0, 'prediction': {'winning_region': 'r'}} for i in range(14)]}
    records = [{'key': f'validation/{i}', 'label': 0, 'prefix': str(i), 'source_row': {'region': 'r'}} for i in range(14)]
    payload = {}
    for i in range(12):
        records.append({'key': 'defect/0024', 'label': 1, 'prefix': f'd{i}', 'source_row': {'region': f'r{i}'}})
        payload[f'd{i}/mask.npy'] = d.np_bytes(np.full((256, 256), i < 6, dtype=bool))
    selected = d.select_tiles(report, records, payload)
    assert len(selected) == 20 and sum(r['label'] for r in selected) == 6
    payload['d0/mask.npy'] = d.np_bytes(np.zeros((256, 256), dtype=bool))
    with pytest.raises(ValueError): d.select_tiles(report, records, payload)


def test_search_algorithm_cpu_double_with_fake_cuda_api():
    """Exercise real chunk/reduction code, explicitly not CUDA numerical validation."""
    class Tensor:
        def __init__(self, x): self.x = np.asarray(x)
        def cpu(self): return self
        def numpy(self): return self.x
        def min(self, dim):
            ids = self.x.argmin(axis=dim)
            return Tensor(self.x[np.arange(len(ids)), ids]), Tensor(ids)
    class Torch:
        inference_mode = staticmethod(nullcontext)
        as_tensor = staticmethod(lambda x, device: Tensor(x))
        @staticmethod
        def cdist(q, c, p, compute_mode):
            assert p == 2 and compute_mode == 'donot_use_mm_for_euclid_dist'
            return Tensor(np.linalg.norm(q.x[:, None]-c.x[None], axis=2))
    runtime = object.__new__(d.Runtime); runtime.torch = Torch()
    assert runtime.search_gate(lambda: None)['passed']


def test_query_replay_gate_and_masks():
    b = io.BytesIO(); Image.new('RGB', (256, 256)).save(b, format='PNG')
    mask = np.zeros((256, 256), dtype=bool); mask[:8, :8] = True
    payload = {'a/source.png': b.getvalue(), 'a/map.npy': d.np_bytes(np.ones((256, 256), dtype=np.float32)), 'a/mask.npy': d.np_bytes(mask)}
    record = {'prefix': 'a', 'key': 'defect/0024', 'label': 1, 'source_row': {'region': 'r'}, 'sample_region': {'top1': 1., 'native_raw': 1.}}
    runtime = SimpleNamespace(trace=lambda image: (1., np.ones((256, 256), dtype=np.float32), np.zeros((4096, 768), dtype=np.float32), np.arange(4096, dtype=np.float32)))
    value, features = d.query_tile(runtime, record, payload)
    assert features.shape[1] == 768 and any('GT_' in role for q in value['queries'] for role in q['roles'])
    record['sample_region']['top1'] = 2.
    with pytest.raises(ValueError, match='replay differs'): d.query_tile(runtime, record, payload)


def fixture_run(tmp_path, monkeypatch):
    """Real atomic receipts/export with tiny two-photo pool and fake inference; no CUDA claim."""
    policy = {**d.POLICY, 'training_photos': 2, 'candidates': 24576, 'bank': 2}
    monkeypatch.setattr(d, 'POLICY', policy)
    for name in d.SOURCES:
        path = tmp_path/name; path.parent.mkdir(parents=True, exist_ok=True); path.write_text('fixture')
    image = Image.new('RGB', (256, 256), (10, 20, 30)); stream = io.BytesIO(); image.save(stream, format='PNG'); png = stream.getvalue()
    row = {'member': 'normal.png', 'member_sha256': hashlib.sha256(png).hexdigest(), 'decoded_rgb_sha256': hashlib.sha256(image.tobytes()).hexdigest(), 'region': d.fit.REGIONS[0]}
    archive = tmp_path/'source.zip'
    with zipfile.ZipFile(archive, 'w') as z: z.writestr('normal.png', png)
    training = {'photo_ids': ['a', 'b'], 'rows': [copy.deepcopy(row) for _ in range(24)]}
    plan = {'training': training, 'archive_sha256': d.fit.sha256_file(archive), 'sources_lf_sha256': {}, 'config': {}, 'environment': {}}
    signature = d.fit.digest(plan); old = tmp_path/'artifacts'/d.p.NAME
    old.mkdir(parents=True)
    with d.fit.phase_stage(old, 'model') as stage:
        (stage/'model.pt').write_bytes(b'fake model'); d.fit.write_json(stage/'context.json', {})
        d.fit.seal_phase(stage, 'model', signature, {}, ['model.pt', 'context.json'])
    receipts = []
    (old/'features').mkdir()
    for i in range(2):
        raw = np.zeros((12288, 768), dtype=np.float32); raw[:, 0] = i+1
        with d.fit.phase_stage(old/'features', f'photo-{i:03d}') as stage:
            np.save(stage/'features.npy', raw, allow_pickle=False)
            d.fit.seal_phase(stage, f'photo-{i:03d}', signature, {}, ['features.npy'])
        receipts.append(d.fit.read_phase(old/'features', f'photo-{i:03d}', signature, ['features.npy']))
    parent = {'plan.json': plan, 'feature-receipts.json': receipts, 'model-receipt.json': d.fit.read_phase(old, 'model', signature, ['model.pt', 'context.json']),
              'selected-indices.json': {'indices': [0, 12288]}, 'fit-report.json': {'context': {}}, 'contract.json': {}}
    record = {'key': 'validation/a', 'label': 0, 'prefix': 'a', 'source_row': row, 'sample_region': {'top1': 1., 'native_raw': 1.}}
    payload = {'scores.json': b'{}', 'review.json': json.dumps({'records': [record]}).encode(), 'a/source.png': png, 'a/map.npy': d.np_bytes(np.ones((256, 256), dtype=np.float32))}
    monkeypatch.setattr(d.overlap, 'load', lambda handoffs: ({}, parent, {}))
    monkeypatch.setattr(d.overlap, 'analyze', lambda *args: {})
    monkeypatch.setattr(d, 'read_overlap', lambda path: payload)
    monkeypatch.setattr(d, 'select_tiles', lambda *args: [record])
    monkeypatch.setattr(d, 'environment', lambda: {})
    monkeypatch.setattr(d.engine, 'normal_gate', lambda *args: {'passed': True})
    class Tensor:
        def __init__(self, values): self.values = values
        def detach(self): return self
        def cpu(self): return self
        def numpy(self): return self.values
    class Runtime:
        searches = 0; traced = 0; interrupt_at = None
        def __init__(self, config):
            bank = np.zeros((2, 768), dtype=np.float32); bank[:, 0] = [1, 2]
            self.model = SimpleNamespace(model=SimpleNamespace(memory_bank=Tensor(bank)))
        def load(self, *args): pass
        def search_gate(self, check): return {'passed': True}
        def trace(self, image):
            Runtime.traced += 1
            return 1., np.ones((256, 256), dtype=np.float32), np.zeros((4096, 768), dtype=np.float32), np.ones(4096, dtype=np.float32)
        def search(self, q, c, check):
            Runtime.searches += 1
            if Runtime.searches == Runtime.interrupt_at: raise KeyboardInterrupt()
            i = np.full(len(q), int(np.argmin(c[:, 0])), dtype=np.int64)
            return c[i, 0].astype(np.float64), i
    @contextmanager
    def attempt(root, signature, policy):
        yield SimpleNamespace(runtime=None, check=lambda: None, snapshot=lambda: {'fake_runtime': True})
    monkeypatch.setattr(d.engine, 'attempt', attempt)
    return archive, Runtime, parent, signature


def test_end_to_end_interrupt_resume_export_replay_and_tamper(tmp_path, monkeypatch):
    archive, runtime, parent, sig = fixture_run(tmp_path, monkeypatch)
    runtime.interrupt_at = 3  # bank, photo0, interrupt photo1
    with pytest.raises(KeyboardInterrupt): d.run(tmp_path, archive, runtime)
    root = tmp_path/'artifacts/analyses'/d.NAME
    assert (root/'search-000/value.json').exists() and not (root/'search-001').exists()
    runtime.interrupt_at = None
    result = d.run(tmp_path, archive, runtime)
    assert result['full_candidates_searched'] == 24576 and result['subset_vectors_verified'] == 2
    assert runtime.traced == 1 and runtime.searches == 5
    handoff = Path(result['output']); before = handoff.read_bytes()
    assert d.run(tmp_path, archive, runtime) == result
    assert handoff.read_bytes() == before and runtime.traced == 1 and runtime.searches == 6
    with zipfile.ZipFile(handoff) as z:
        assert len([n for n in z.namelist() if n.startswith('cards/')]) == 2
        assert np.load(io.BytesIO(z.read('full-neighbors.npy')), allow_pickle=False).shape == (2, 768)
        with Image.open(io.BytesIO(z.read('cards/query-000.png'))) as im: assert im.size == (768, 296)
    (root/'search-000/value.json').write_text('{}')
    with pytest.raises(ValueError): d.run(tmp_path, archive, runtime)


def test_cuda_exhaustive_search_reference_if_available():
    torch = pytest.importorskip('torch')
    if not torch.cuda.is_available(): pytest.skip('CUDA required; not substituted by CPU')
    runtime = object.__new__(d.Runtime); runtime.torch = torch
    assert runtime.search_gate(lambda: None)['passed']
