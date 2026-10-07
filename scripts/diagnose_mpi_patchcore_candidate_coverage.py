"""Fixed-query coreset versus retained full candidate pool; no model fitting."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import review_mpi_patchcore_score_overlap as overlap
from scripts import mpi_patch_trace_protocol as trace
from scripts.mpi_patchcore_scaling_runtime import Runtime as BaseRuntime, environment
from scripts.export_mpi_patchcore_scaling_review import checked_png, write_bundle

fit, p, engine = overlap.fit, overlap.p, overlap.engine
NAME = 'mpi-patchcore-candidate-coverage-v1'
OVERLAP_SHA = '735cb18242c6f01d6f0d61707acb23d9c924850e89730ea67c86741ea2388d11'
POLICY = {
    'normal_tiles': 'all14 frozen overlap normal winning tiles',
    'defect_tiles': 'all6 nonempty GT tiles of defect/0024',
    'queries': trace.POLICY['query_cells'],
    'cell_gt': trace.POLICY['cell_gt'],
    'query_ties': trace.POLICY['query_ties'],
    'training_photos': 256, 'candidates': 3145728, 'bank': 78643, 'dimensions': 768,
    'candidate_scope': 'original stride2 retained pool; not every native training-grid position',
    'distance': 'exhaustive unnormalised Euclidean; float64 direct cdist for BOTH pools',
    'chunk_rows': 4096, 'ties': 'first index in frozen pool order for exactly equal computed distances',
    'map_replay_absolute_tolerance': 1e-4,
    'native_distance_check': {'atol': .001, 'rtol': .0001},
    'resource_plan': p.POLICY['resource_plan'], 'time_limit_seconds': None,
    'resume': 'Ctrl+C only; atomic per-tile queries and per-training-photo search; failure requires review',
    'training': False, 'threshold_changed': False, 'service_model_changed': False,
    'automatic_pass_allowed': False,
    'interpretation': 'post hoc local feature diagnostic, not new photo scores or accuracy',
}
SOURCES = ['scripts/diagnose_mpi_patchcore_candidate_coverage.py',
           'scripts/run_mpi_patchcore_candidate_coverage.ps1',
           'scripts/review_mpi_patchcore_score_overlap.py',
           'scripts/export_mpi_patchcore_scaling_review.py',
           'scripts/mpi_patch_trace_protocol.py']


def read_overlap(path):
    if fit.sha256_file(path) != OVERLAP_SHA:
        raise ValueError('Expected the reviewed overlap ZIP')
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {'handoff-manifest.json'}:
            raise ValueError('Overlap inventory differs')
        payload = {n: z.read(n) for n in manifest}
        if any(hashlib.sha256(payload[n]).hexdigest() != sha for n, sha in manifest.items()):
            raise ValueError('Overlap payload differs')
    return payload


def np_bytes(values):
    stream = io.BytesIO(); np.save(stream, values, allow_pickle=False)
    return stream.getvalue()


def same_numbers(a, b):
    """Cross-platform JSON arithmetic only; identifiers and inventory remain exact."""
    if isinstance(a, dict):
        return isinstance(b, dict) and set(a) == set(b) and all(same_numbers(a[k], b[k]) for k in a)
    if isinstance(a, list):
        return isinstance(b, list) and len(a) == len(b) and all(same_numbers(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, (int, float)) and np.isfinite(a) and np.isfinite(b) and abs(a-b) <= 1e-10
    return a == b


def select_tiles(report, records, payload):
    winners = {c['key']: c['prediction']['winning_region'] for c in report['cases'] if not c['label']}
    chosen = []
    for record in records:
        key, region, prefix = record['key'], record['source_row']['region'], record['prefix']
        mask = np.load(io.BytesIO(payload[prefix+'/mask.npy']), allow_pickle=False) if record['label'] else None
        if mask is not None and (mask.dtype != np.bool_ or mask.shape != (256, 256)):
            raise ValueError('Invalid frozen GT')
        if ((record['label'] and key == 'defect/0024' and mask.any()) or
                (not record['label'] and winners.get(key) == region)):
            chosen.append(record)
    if (len(chosen) != 20 or sum(r['label'] for r in chosen) != 6 or
            len({(r['key'], r['source_row']['region']) for r in chosen}) != 20):
        raise ValueError('Expected14 normal winning tiles and6 GT defect tiles')
    return chosen


def validate_indices(indices, n, k):
    ids = np.asarray(indices)
    if (ids.shape != (k,) or not np.issubdtype(ids.dtype, np.integer) or
            np.any(ids < 0) or np.any(ids >= n) or len(np.unique(ids)) != k):
        raise ValueError('Invalid frozen selected indices')
    return ids


def validate_subset(bank, raw, selected, start):
    """Every coreset row must equal the original raw candidate, byte for byte."""
    positions = np.flatnonzero((selected >= start) & (selected < start+len(raw)))
    expected = np.ascontiguousarray(raw[selected[positions]-start])
    if bank[positions].tobytes() != expected.tobytes():
        raise ValueError('Bank is not the exact recorded subset of cached candidates')
    return len(positions)


def merge_best(best_d, best_i, distances, indices):
    if (distances.shape != best_d.shape or indices.shape != best_i.shape or
            not np.isfinite(distances).all() or np.any(distances < 0) or np.any(indices < 0)):
        raise ValueError('Invalid nearest-neighbor result')
    take = (distances < best_d) | ((distances == best_d) & (indices < best_i))
    return np.where(take, distances, best_d), np.where(take, indices, best_i)


def origin(index, training):
    photo, within = divmod(int(index), 12288)
    region, slot = divmod(within, 1024)
    row = training['rows'][photo*12+region]
    grid_index = fit.spatial_patch_indices(64, 2)[slot]
    return {'candidate_index': int(index), 'photo_id': training['photo_ids'][photo],
            'row': row, 'grid_y': grid_index//64, 'grid_x': grid_index % 64}


def comparison(queries, bank_d, bank_i, full_d, full_i, selected, training):
    if not np.isfinite(bank_d).all() or not np.isfinite(full_d).all() or np.any(full_d > bank_d+1e-8):
        raise ValueError('Full-pool distance exceeds subset distance')
    result = []
    for q, bd, bi, fd, fi in zip(queries, bank_d, bank_i, full_d, full_i, strict=True):
        result.append({**q, 'bank_distance': float(bd), 'full_distance': float(fd),
            'distance_drop': float(bd-fd), 'relative_drop': float((bd-fd)/bd) if bd else None,
            'bank_neighbor': origin(selected[int(bi)], training),
            'full_neighbor': origin(fi, training), 'full_neighbor_is_in_bank': bool(np.any(selected == fi)),
            'ties': 'one deterministic nearest source; equal vectors may have other origins'})
    return result


@contextmanager
def capture(core):
    previous, had = core.nearest_neighbors, 'nearest_neighbors' in vars(core)
    override, records = vars(core).get('nearest_neighbors'), []
    def record(embedding, n_neighbors):
        out = previous(embedding, n_neighbors)
        if n_neighbors == 1 and tuple(embedding.shape) == (4096, 768):
            records.append(tuple(v.detach().cpu().numpy().copy() for v in (embedding, *out)))
        return out
    core.nearest_neighbors = record
    try:
        yield records
    finally:
        if had: core.nearest_neighbors = override
        else: del core.nearest_neighbors


class Runtime(BaseRuntime):
    def trace(self, image):
        with capture(self.model.model) as records:
            native, values = self.predict(image)
        if len(records) != 1:
            raise ValueError('Expected one public-forward4096-patch feature capture')
        features, distances, indices = records[0]
        if (features.dtype != np.float32 or not np.isfinite(features).all() or
                distances.shape != (4096,) or indices.shape != (4096,)):
            raise ValueError('Invalid public-forward features')
        return native, values, features, distances

    def search(self, queries, candidates, check):
        """Small query set, streaming candidates. No projection or feature normalization."""
        torch = self.torch
        if (queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]
                or not len(queries) or not len(candidates) or
                not np.isfinite(queries).all() or not np.isfinite(candidates).all()):
            raise ValueError('Invalid search inputs')
        q = torch.as_tensor(np.array(queries, dtype=np.float64), device='cuda')
        best_d = np.full(len(queries), np.inf); best_i = np.full(len(queries), np.iinfo(np.int64).max, dtype=np.int64)
        with torch.inference_mode():
            for start in range(0, len(candidates), POLICY['chunk_rows']):
                check()
                c = torch.as_tensor(np.array(candidates[start:start+POLICY['chunk_rows']], dtype=np.float64), device='cuda')
                d = torch.cdist(q, c, p=2, compute_mode='donot_use_mm_for_euclid_dist')
                minimum, index = d.min(dim=1)
                best_d, best_i = merge_best(best_d, best_i, minimum.cpu().numpy(), index.cpu().numpy()+start)
                del c, d, minimum, index
        direct = np.linalg.norm(queries.astype(np.float64)-candidates[best_i].astype(np.float64), axis=1)
        if not np.allclose(direct, best_d, atol=1e-8, rtol=1e-9):
            raise ValueError('GPU distance differs from direct CPU float64 winner check')
        return best_d, best_i

    def search_gate(self, check):
        rng = np.random.default_rng(8241)
        bank = rng.normal(size=(4103, 768)).astype(np.float32)
        bank[4100] = bank[0]
        queries = np.stack([bank[0], bank[4102], rng.normal(size=768).astype(np.float32)])
        distances, indices = self.search(queries, bank, check)
        cpu = np.linalg.norm(queries[:, None].astype(np.float64)-bank[None].astype(np.float64), axis=2)
        expected = cpu.argmin(axis=1)
        if not np.array_equal(indices, expected) or not np.allclose(distances, cpu[np.arange(3), expected], atol=1e-8, rtol=1e-9):
            raise ValueError('Chunked GPU search failed full CPU reference/tie check')
        return {'passed': True, 'queries': 3, 'candidates': 4103, 'includes_cross_chunk_tie': True}


def query_tile(runtime, record, payload):
    prefix = record['prefix']
    with Image.open(io.BytesIO(payload[prefix+'/source.png'])) as image:
        image.load()
        native, values, features, distances = runtime.trace(image)
    saved = np.load(io.BytesIO(payload[prefix+'/map.npy']), allow_pickle=False)
    difference = float(np.max(np.abs(np.asarray(values)-saved)))
    expected = record['sample_region']
    top1 = fit.candidate_image_score(native, values, fit.CANDIDATE)
    if (not np.isfinite(difference) or max(difference, abs(native-expected['native_raw']),
            abs(top1-expected['top1'])) > POLICY['map_replay_absolute_tolerance']):
        raise ValueError('Frozen map/score replay differs; no recalibration allowed')
    mask = np.load(io.BytesIO(payload[prefix+'/mask.npy']), allow_pickle=False) if record['label'] else None
    queries = trace.select_query_cells(distances, mask)
    for q in queries:
        q.update(key=record['key'], label=record['label'], region=record['source_row']['region'], prefix=prefix)
    return {'queries': queries, 'map_maximum_absolute_difference': difference}, features[[q['flat_index'] for q in queries]]


def cards(results, payload, source):
    """Display cells mark query coordinates, not the CNN receptive field or defect box."""
    exported = {}
    for i, result in enumerate(results):
        canvas = Image.new('RGB', (768, 296), 'white'); draw = ImageDraw.Draw(canvas)
        entries = [(payload[result['prefix']+'/source.png'], result['grid_x'], result['grid_y'], 'query '+result['key']),
            (checked_png(source, result['bank_neighbor']['row']), result['bank_neighbor']['grid_x'], result['bank_neighbor']['grid_y'],
             'bank d=%.6f'%result['bank_distance']),
            (checked_png(source, result['full_neighbor']['row']), result['full_neighbor']['grid_x'], result['full_neighbor']['grid_y'],
             'full d=%.6f'%result['full_distance'])]
        for j, (raw, x, y, title) in enumerate(entries):
            with Image.open(io.BytesIO(raw)) as image: canvas.paste(image.convert('RGB'), (j*256, 40))
            draw.text((j*256+4, 3), title, fill='black')
            draw.text((j*256+4, 19), '4x4 display cell; NOT receptive field', fill='black')
            draw.rectangle((j*256+x*4, 40+y*4, j*256+x*4+3, 40+y*4+3), outline='red', width=1)
        stream = io.BytesIO(); canvas.save(stream, format='PNG'); exported[f'cards/query-{i:03d}.png'] = stream.getvalue()
    return exported


def run(project, archive_path, runtime_factory=Runtime):
    handoffs = project/'artifacts/handoff'
    current, parent, raw = overlap.load(handoffs)
    payload = read_overlap(handoffs/(overlap.NAME+'.zip'))
    report, reviewed = json.loads(payload['scores.json']), json.loads(payload['review.json'])
    if not same_numbers(report, overlap.analyze(current, parent, raw)):
        raise ValueError('Frozen case selection differs')
    records = select_tiles(report, reviewed['records'], payload)
    plan = parent['plan.json']; parent_sig = fit.digest(plan)
    if (len(parent['feature-receipts.json']) != POLICY['training_photos'] or
            len(plan['training']['photo_ids']) != POLICY['training_photos'] or
            len(plan['training']['rows']) != POLICY['training_photos']*12):
        raise ValueError('Frozen training pool size differs')
    if fit.sha256_file(archive_path) != plan['archive_sha256']:
        raise ValueError('Original archive differs')
    for name, sha in plan['sources_lf_sha256'].items():
        if hashlib.sha256((project/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != sha:
            raise ValueError(f'Frozen fit dependency differs: {name}')
    old = project/'artifacts'/p.NAME
    model_receipt = fit.read_phase(old, 'model', parent_sig, ['model.pt', 'context.json'])
    if model_receipt != parent['model-receipt.json']:
        raise ValueError('Model receipt differs')
    selected = validate_indices(parent['selected-indices.json']['indices'], POLICY['candidates'], POLICY['bank'])
    identity = {'name': NAME, 'policy': POLICY, 'parent_sha256': overlap.d.evidence.INPUT_SHA,
        'development_sha256': overlap.INPUT_SHA, 'overlap_sha256': OVERLAP_SHA,
        'checkpoint_sha256': model_receipt['files']['model.pt'], 'parent_execution_sha256': parent_sig,
        'archive_sha256': plan['archive_sha256'], 'selected_tiles': records,
        'sources_lf_sha256': {n: hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in SOURCES}}
    root = project/'artifacts/analyses'/NAME
    with fit.locked_run(root):
        signature = engine.initialize(root, identity)
        with engine.attempt(root, signature, POLICY) as monitor, zipfile.ZipFile(archive_path) as source:
            if len(source.namelist()) != len(set(source.namelist())):
                raise ValueError('Duplicate source members')
            runtime = runtime_factory(plan['config']); monitor.runtime = runtime
            if environment() != plan['environment']:
                raise ValueError('GPU/runtime environment differs from frozen fit')
            runtime.load(old/'model/model.pt', parent['fit-report.json']['context'])
            print('1/3 Replaying8 normal photos and20 fixed query tiles; no fitting.', flush=True)
            gate = engine.normal_gate(old, parent_sig, source, plan, runtime, identity['checkpoint_sha256'], parent['contract.json'])
            search_gate = runtime.search_gate(monitor.check)
            bank = runtime.model.model.memory_bank.detach().cpu().numpy().copy()
            if bank.shape != (POLICY['bank'], POLICY['dimensions']) or bank.dtype != np.float32 or not np.isfinite(bank).all():
                raise ValueError('Invalid bank')
            queries, feature_rows = [], []
            for i, record in enumerate(records):
                name = f'query-{i:02d}'
                if not (root/name).exists():
                    value, values = query_tile(runtime, record, payload)
                    with fit.phase_stage(root, name) as stage:
                        fit.write_json(stage/'value.json', value); np.save(stage/'features.npy', values, allow_pickle=False)
                        fit.seal_phase(stage, name, signature, {'record_sha256': fit.digest(record)}, ['value.json', 'features.npy'])
                receipt = fit.read_phase(root, name, signature, ['value.json', 'features.npy'])
                if receipt['record_sha256'] != fit.digest(record): raise ValueError('Query input differs')
                value = json.loads((root/name/'value.json').read_bytes())
                values = p.array(root/name/'features.npy', (len(value['queries']), 768))
                queries.extend(value['queries']); feature_rows.append(np.array(values))
                print(f'query tile {i+1}/20', flush=True)
            features = np.concatenate(feature_rows)
            bank_d, bank_i = runtime.search(features, bank, monitor.check)
            if not np.allclose(bank_d, [q['native_patch_distance'] for q in queries], **POLICY['native_distance_check']):
                raise ValueError('Float64 coreset search disagrees with native float32 distances')
            print(f'2/3 {len(queries)} fixed queries against all3145728 retained candidates;256 resumable blocks.', flush=True)
            best_d = np.full(len(queries), np.inf); best_i = np.full(len(queries), np.iinfo(np.int64).max, dtype=np.int64)
            covered = 0; search_signature = fit.digest({'features': hashlib.sha256(features.tobytes()).hexdigest(), 'queries': queries})
            nearest_features = np.zeros(features.shape, dtype=np.float32)
            for i in range(POLICY['training_photos']):
                monitor.check(); name = f'photo-{i:03d}'
                receipt = fit.read_phase(old/'features', name, parent_sig, ['features.npy'])
                if receipt != parent['feature-receipts.json'][i]: raise ValueError('Original feature receipt differs')
                candidates = p.array(old/'features'/name/'features.npy', (12288, 768))
                covered += validate_subset(bank, candidates, selected, i*12288)
                stage_name = f'search-{i:03d}'
                if not (root/stage_name).exists():
                    distances, indices = runtime.search(features, candidates, monitor.check)
                    p.publish(root, stage_name, signature, {'queries_sha256': search_signature,
                        'feature_receipt_sha256': receipt['receipt_sha256'],
                        'distances': distances.tolist(), 'indices': (indices+i*12288).tolist()})
                result = p.value(root, stage_name, signature)
                if result['queries_sha256'] != search_signature or result['feature_receipt_sha256'] != receipt['receipt_sha256']:
                    raise ValueError('Cached search input differs')
                distances, indices = np.asarray(result['distances']), np.asarray(result['indices'], dtype=np.int64)
                if np.any(indices < i*12288) or np.any(indices >= (i+1)*12288): raise ValueError('Search index outside source photo')
                direct = np.linalg.norm(features.astype(np.float64)-candidates[indices-i*12288].astype(np.float64), axis=1)
                if not np.allclose(direct, distances, atol=1e-8, rtol=1e-9): raise ValueError('Cached neighbor distance differs')
                next_d, next_i = merge_best(best_d, best_i, distances, indices)
                replaced = next_i != best_i
                nearest_features[replaced] = candidates[indices[replaced]-i*12288]
                best_d, best_i = next_d, next_i
                if (i+1) % 8 == 0: print(f'candidate coverage {i+1}/256 photos', flush=True)
            if covered != POLICY['bank']: raise ValueError('Not all bank vectors verified against source pool')
            if not np.array_equal(bank, runtime.model.model.memory_bank.detach().cpu().numpy()):
                raise ValueError('Memory bank changed during diagnosis')
            results = comparison(queries, bank_d, bank_i, best_d, best_i, selected, plan['training'])
            print('3/3 Exporting local-distance changes and matched training-image cards.', flush=True)
            summary = {'status': 'diagnosed_fixed_query_candidate_coverage', 'query_count': len(queries),
                'normal_tiles': 14, 'defect_GT_tiles': 6, 'subset_vectors_verified': covered,
                'full_candidates_searched': POLICY['candidates'], 'normal_gate': gate, 'search_gate': search_gate,
                'native_bank_maximum_distance_difference': float(np.max(np.abs(bank_d-[q['native_patch_distance'] for q in queries]))),
                'training': False, 'threshold_changed': False, 'automatic_pass_allowed': False,
                'interpretation': POLICY['interpretation'],
                'warning': 'Nearest-feature distance is NOT the smoothed map or photo score. No detection/FP improvement inferred.'}
            p.publish(root, 'result', signature, {'summary': summary, 'queries': results})
            output = handoffs/(NAME+'.zip')
            export = {'plan.json': fit.canonical(identity), 'result.json': fit.canonical({'summary': summary, 'queries': results}),
                      'queries.npy': np_bytes(features), 'bank-neighbors.npy': np_bytes(bank[bank_i]),
                      'full-neighbors.npy': np_bytes(nearest_features),
                      'search-receipts.json': fit.canonical([fit.read_phase(root, f'search-{i:03d}', signature, ['value.json'])
                                                          for i in range(POLICY['training_photos'])]),
                      'resource-snapshot.json': fit.canonical(monitor.snapshot())}
            # Preserve the first successful package on rerun; verify its core content before returning.
            if output.exists():
                with zipfile.ZipFile(output) as z:
                    manifest = json.loads(z.read('handoff-manifest.json'))
                    if (len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {'handoff-manifest.json'} or
                            any(hashlib.sha256(z.read(n)).hexdigest() != sha for n, sha in manifest.items()) or
                            any(z.read(n) != raw for n, raw in export.items() if n != 'resource-snapshot.json')):
                        raise ValueError('Existing handoff differs; preserved')
            else:
                export.update(cards(results, payload, source)); write_bundle(output, export)
            monitor.check()
            return {**summary, 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(json.dumps(run(args.project, args.archive), indent=2)); return 0
    except KeyboardInterrupt:
        print('Interrupted. Preserve files; rerun the same command to resume completed blocks.'); return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve files and send the error; no retraining.'); return 1


if __name__ == '__main__':
    raise SystemExit(main())
