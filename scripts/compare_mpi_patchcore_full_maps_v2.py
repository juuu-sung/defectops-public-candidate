"""Map precision gate followed by the fixed20 exhaustive comparison, in a NEW run."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import compare_mpi_patchcore_full_maps as d
from scripts import diagnose_mpi_patchcore_map_parity as audit
from scripts.mpi_patchcore_stable_maps import StableMaps, ATOL, RTOL

fit, p, engine, c = d.fit, d.p, d.engine, d.c
NAME = 'mpi-patchcore-full-map-comparison-v2'
AUDIT_SHA = 'b5e521f5fb3f511829697dad299b90d3660f44b75c1b45af60edefcb4098e08a'
POLICY = {k: v for k, v in d.POLICY.items() if not k.startswith('float64_vs_native')}
POLICY.update(map='deepcopy installed map generator; CPU float64; exact installed Gaussian buffer',
              map_reference='independent NumPy float64 FFT nearest4x reflect16 cross-correlation',
              stable_map_atol=ATOL, stable_map_rtol=RTOL,
              cross_precision='diagnostic only; not equivalence gate or photo threshold',
              audited_bank_distances='reuse pinned audit, verify all winning vectors against unchanged bank',
              failed_v1='read-only receipt-verified cached features; never mutate or resume failed attempt')
SOURCES = ['scripts/compare_mpi_patchcore_full_maps_v2.py', 'scripts/mpi_patchcore_stable_maps.py',
           'scripts/run_mpi_patchcore_full_maps_v2.ps1']


def checked_audit(path):
    if fit.sha256_file(path) != AUDIT_SHA:
        raise ValueError('Expected reviewed map-parity audit ZIP')
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest) | {'handoff-manifest.json'}:
            raise ValueError('Audit inventory differs')
        data = {n: z.read(n) for n in manifest}
        if any(hashlib.sha256(data[n]).hexdigest() != h for n, h in manifest.items()):
            raise ValueError('Audit payload differs')
    return data


def array(data, name, shape, dtype):
    a = np.load(io.BytesIO(data[name]), allow_pickle=False)
    if a.shape != shape or a.dtype != dtype or not np.isfinite(a).all():
        raise ValueError(f'Invalid audited array: {name}')
    return a


def source_check(project, sources):
    for n, h in sources.items():
        if hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != h:
            raise ValueError(f'Frozen source changed: {n}')


def load_inputs(project, archive_path):
    handoffs = project/'artifacts/handoff'
    current, parent, raw = c.overlap.load(handoffs)
    prior_bytes = d.checked_input(handoffs/(c.NAME+'.zip'))
    prior_plan = json.loads(prior_bytes['plan.json'])
    payload = c.read_overlap(handoffs/(c.overlap.NAME+'.zip'))
    expected = c.select_tiles(c.overlap.analyze(current, parent, raw), json.loads(payload['review.json'])['records'], payload)
    plan = parent['plan.json']; parent_sig = fit.digest(plan)
    if (prior_plan['selected_tiles'] != expected or prior_plan['parent_execution_sha256'] != parent_sig
            or prior_plan['policy'] != c.POLICY or len(expected) != POLICY['tiles']
            or len(parent['feature-receipts.json']) != POLICY['training_photos']):
        raise ValueError('Frozen tile/pool selection differs')
    if fit.sha256_file(archive_path) != plan['archive_sha256']:
        raise ValueError('Original archive differs')
    reviewed = checked_audit(handoffs/(audit.NAME+'.zip'))
    failed_identity = json.loads(reviewed['failed-identity.json']); old_sig = fit.digest(failed_identity)
    audit_plan = json.loads(reviewed['plan.json'])
    failed = project/'artifacts/analyses'/d.NAME
    if (p.value(failed, 'identity', old_sig) != failed_identity
            or failed_identity['name'] != d.NAME or failed_identity['policy'] != d.POLICY
            or failed_identity['tiles'] != expected or failed_identity['parent_execution_sha256'] != parent_sig
            or audit_plan['failed_identity_sha256'] != old_sig):
        raise ValueError('Reviewed failed run or source cohort differs')
    for sources in [plan['sources_lf_sha256'], prior_plan['sources_lf_sha256'],
                    failed_identity['sources_lf_sha256'], audit_plan['sources_lf_sha256']]:
        source_check(project, sources)
    model_root = project/'artifacts'/p.NAME
    model = fit.read_phase(model_root, 'model', parent_sig, ['model.pt', 'context.json'])
    if model != parent['model-receipt.json'] or model['files']['model.pt'] != failed_identity['checkpoint_sha256']:
        raise ValueError('Original checkpoint differs')
    return parent, payload, prior_bytes, reviewed, failed_identity, expected


def cached_tiles(project, failed_identity, expected, payload, reviewed, bank):
    """Read-only reuse of the exact features already audited; no new feature extraction."""
    root = project/'artifacts/analyses'/d.NAME; sig = fit.digest(failed_identity)
    features, natives, doubles, indices, maps, receipts = [], [], [], [], [], []
    for i, record in enumerate(expected):
        name = f'tile-{i:02d}'
        receipt = fit.read_phase(root, name, sig, ['features.npy', 'native-distances.npy', 'replay.json'])
        if receipt['record_sha256'] != fit.digest(record):
            raise ValueError('Cached tile identity differs')
        q = np.array(p.array(root/name/'features.npy', (4096, 768)))
        native = np.array(p.array(root/name/'native-distances.npy', (4096,)))
        double = array(reviewed, name+'/double-distances.npy', (4096,), np.float64)
        ids = array(reviewed, name+'/double-indices.npy', (4096,), np.int64)
        saved = np.load(io.BytesIO(payload[record['prefix']+'/map.npy']), allow_pickle=False)
        if not np.array_equal(saved, array(reviewed, name+'/saved-map.npy', (256, 256), np.float32)):
            raise ValueError('Original/audited saved map differs')
        tile_report = json.loads(reviewed[name+'/report.json'])
        if (tile_report['key'], tile_report['region']) != (record['key'], record['source_row']['region']):
            raise ValueError('Audited tile identity differs')
        d.check_distances(double, audit.direct_distances(q, bank, ids), 'audited bank winning vectors')
        features.append(q); natives.append(native); doubles.append(double); indices.append(ids)
        maps.append(saved); receipts.append(receipt)
    return np.concatenate(features), natives, np.stack(doubles), np.stack(indices), maps, receipts


def map_gate(root, sig, expected, natives, doubles, old_maps, mapper, monitor):
    """All20 technical map checks finish before any expensive full-pool search."""
    exports, reports, bank_maps = {}, [], []
    synthetic = [np.zeros(4096), np.ones(4096)*50, np.linspace(0, 100, 4096)]
    corner = np.zeros(4096); corner[0] = 100; corner[-1] = 50; synthetic.append(corner)
    for values in synthetic:
        monitor.check(); mapper.stable(values)
    for i, record in enumerate(expected):
        monitor.check(); name = f'map-gate-{i:02d}'
        names = [f'{a}-{b}.npy' for a in ('native', 'double')
                 for b in ('gpu-tf32-true', 'gpu-tf32-false', 'cpu-float64')]
        files = ['report.json', *names]
        if not (root/name).exists():
            try:
                report, arrays = mapper.gate(natives[i], doubles[i], old_maps[i])
            except ValueError as exc:
                raise ValueError(f"Map gate tile={i} {record['key']} {record['source_row']['region']}: {exc}") from exc
            report.update(key=record['key'], region=record['source_row']['region'])
            with fit.phase_stage(root, name) as stage:
                fit.write_json(stage/'report.json', report)
                for n, a in arrays.items(): np.save(stage/n, a, allow_pickle=False)
                fit.seal_phase(stage, name, sig, {'map_contract': mapper.meta}, files)
        receipt = fit.read_phase(root, name, sig, files)
        if receipt['map_contract'] != mapper.meta:
            raise ValueError('Stable map kernel/settings differ on resume')
        for n in files: exports[name+'/'+n] = (root/name/n).read_bytes()
        reports.append(json.loads(exports[name+'/report.json']))
        bank_maps.append(np.array(p.array(root/name/'double-cpu-float64.npy', (256, 256), np.float64)))
        print(f'Map precision gate {i+1}/{len(expected)} passed (cross-precision deltas reported, not suppressed).', flush=True)
    return bank_maps, reports, exports


def run(project, archive_path, runtime_factory=d.Runtime, mapper_factory=StableMaps):
    parent, payload, prior_bytes, reviewed, failed_identity, expected = load_inputs(project, archive_path)
    plan = parent['plan.json']; parent_sig = fit.digest(plan)
    model_root = project/'artifacts'/p.NAME
    identity = {'name': NAME, 'policy': POLICY, 'audit_sha256': AUDIT_SHA, 'coverage_sha256': d.INPUT_SHA,
                'parent_execution_sha256': parent_sig, 'failed_identity_sha256': fit.digest(failed_identity),
                'checkpoint_sha256': failed_identity['checkpoint_sha256'], 'tiles': expected,
                'sources_lf_sha256': {n: hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in SOURCES}}
    root = project/'artifacts/analyses'/NAME
    with fit.locked_run(root):
        sig = engine.initialize(root, identity)
        with engine.attempt(root, sig, POLICY) as monitor, zipfile.ZipFile(archive_path) as archive:
            if c.environment() != plan['environment']: raise ValueError('GPU/runtime differs')
            runtime = runtime_factory(plan['config']); monitor.runtime = runtime
            runtime.load(model_root/'model/model.pt', parent['fit-report.json']['context'])
            print('1/4 Original normal replay and exact double search gate. No backbone changes.', flush=True)
            normal_gate = engine.normal_gate(model_root, parent_sig, archive, plan, runtime,
                                            identity['checkpoint_sha256'], parent['contract.json'])
            fast_gate = runtime.fast_gate(monitor.check)
            # Revalidate on EVERY process, retain the first passing evidence for stable re-export.
            if not (root/'initial-validation').exists():
                p.publish(root, 'initial-validation', sig, {'normal_gate': normal_gate, 'fast_gate': fast_gate})
            initial_validation = p.value(root, 'initial-validation', sig)
            bank = runtime.model.model.memory_bank.detach().cpu().numpy().copy()
            if bank.shape != (POLICY['bank'], 768) or bank.dtype != np.float32 or not np.isfinite(bank).all():
                raise ValueError('Bank differs')
            features, natives, bank_d, bank_i, old_maps, tile_receipts = cached_tiles(
                project, failed_identity, expected, payload, reviewed, bank)
            print('2/4 Map-only TF32 on/off and CPU float64 independent-reference validation.', flush=True)
            mapper = mapper_factory(runtime)
            bank_maps, gate, export = map_gate(root, sig, expected, natives, bank_d, old_maps, mapper, monitor)
            p.publish(root, 'precision-gate', sig, {'passed': True, 'contract': mapper.meta, 'tiles': gate})
            # Persist the gate before the long search, even if a later stage stops.
            print('3/4 Same81920 queries against full3145728 retained candidates. No time cap.', flush=True)
            full_d, full_i, scan, receipts = d.scan_pool(root, sig, model_root, parent, runtime, features, bank, monitor)
            if np.any(full_d > bank_d.ravel()+POLICY['distance_atol']):
                raise ValueError('Full distance exceeds subset')
            prior = json.loads(prior_bytes['result.json'])
            points = d.point_replay(expected, features, bank_d.ravel(), full_d, prior,
                                    np.load(io.BytesIO(prior_bytes['queries.npy']), allow_pickle=False))
            print('4/4 Same CPU float64 maps for both pools, top656 and GT response. No photo decisions.', flush=True)
            full_maps = []
            for distances in full_d.reshape(-1, 4096):
                monitor.check(); full_maps.append(mapper.stable(distances))
            limit = float(np.max(bank_maps)); rows = []
            for i, (record, old, subset, full) in enumerate(zip(expected, old_maps, bank_maps, full_maps, strict=True)):
                # Propagate the EXISTING distance numerical bound through the positive kernel.
                map_subset_bound = POLICY['distance_atol']*mapper.kernel.sum()+ATOL+RTOL*np.abs(subset)
                if np.any(full > subset+map_subset_bound):
                    raise ValueError('Stable full map exceeds stable subset map')
                prefix = record['prefix']
                mask = np.load(io.BytesIO(payload[prefix+'/mask.npy']), allow_pickle=False) if record['label'] else None
                row = d.describe(record, old, subset, full, mask)
                row['full_display_above_range_fraction'] = float(np.mean(full > limit)); rows.append(row)
                caption = record['key'].split('/')[-1]+' / '+record['source_row']['region']
                export[f'cards/tile-{i:02d}.png'] = d.card(payload[prefix+'/source.png'], subset, full, mask, limit, caption)
                export[f'sources/tile-{i:02d}.png'] = payload[prefix+'/source.png']
                if mask is not None: export[f'masks/tile-{i:02d}.npy'] = c.np_bytes(mask)
            mapper.check_unchanged()
            if not np.array_equal(bank, runtime.model.model.memory_bank.detach().cpu().numpy()):
                raise ValueError('Bank mutated')
            report = {'status': 'completed_fixed20_stable_full_map_diagnostic',
                      'normal_gate': initial_validation['normal_gate'],
                      'fast_search_gate': initial_validation['fast_gate'], 'map_contract': mapper.meta, 'map_precision_gate': gate,
                      'prior53_replay': points, 'scan': scan, 'tiles': rows, 'display_maximum': limit,
                      'training': False, 'threshold_changed': False, 'photo_predictions_computed': False,
                      'automatic_pass_allowed': False, 'service_model_changed': False}
            p.publish(root, 'result', sig, report)
            if not (root/'resource').exists(): p.publish(root, 'resource', sig, monitor.snapshot())
            export.update({'plan.json': fit.canonical(identity), 'result.json': fit.canonical(report),
                           'resource.json': fit.canonical(p.value(root, 'resource', sig)),
                           'tile-receipts.json': fit.canonical(tile_receipts), 'search-receipts.json': fit.canonical(receipts),
                           'kernel.npy': c.np_bytes(mapper.kernel), 'bank-distances.npy': c.np_bytes(bank_d),
                           'full-distances.npy': c.np_bytes(full_d.reshape(-1, 4096)), 'bank-indices.npy': c.np_bytes(bank_i),
                           'full-indices.npy': c.np_bytes(full_i.reshape(-1, 4096)),
                           'official-maps.npy': c.np_bytes(np.stack(old_maps)),
                           'bank-maps.npy': c.np_bytes(np.stack(bank_maps)), 'full-maps.npy': c.np_bytes(np.stack(full_maps))})
            monitor.check(); output = project/'artifacts/handoff'/(NAME+'.zip'); c.write_bundle(output, export)
            return {'status': report['status'], 'tiles': len(rows), 'locations': len(features),
                    'search_seconds_sum': scan['search_seconds_sum'], 'output': str(output), 'automatic_pass_allowed': False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT); parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args(argv)
    try: print(json.dumps(run(args.project, args.archive), indent=2)); return 0
    except KeyboardInterrupt:
        print('Interrupted. Preserve all files; same v2 command resumes completed phases.'); return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve files; no retraining or threshold changes.'); return 1


if __name__ == '__main__': raise SystemExit(main())
