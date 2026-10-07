"""Read-only diagnosis of the stopped full-map comparison; never authorizes resume."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import compare_mpi_patchcore_full_maps as d

fit, p, engine = d.fit, d.p, d.engine
NAME = 'mpi-patchcore-map-parity-audit-v1'
SOURCES = ['scripts/diagnose_mpi_patchcore_map_parity.py']


def difference(a, b, atol, rtol=0.):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or not a.size or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid diagnostic arrays')
    delta = np.abs(a-b); pos = int(delta.argmax())
    return {'max_absolute_difference': float(delta.flat[pos]), 'mean_absolute_difference': float(delta.mean()),
            'outside_tolerance': int((delta > atol+rtol*np.abs(b)).sum()),
            'maximum_difference_index': list(map(int, np.unravel_index(pos, a.shape))),
            'a_at_maximum': float(a.flat[pos]), 'b_at_maximum': float(b.flat[pos])}


def direct_distances(features, bank, indices):
    indices = np.asarray(indices)
    if indices.shape != (len(features),) or indices.dtype.kind not in 'iu' or np.any(indices < 0) or np.any(indices >= len(bank)):
        raise ValueError('Invalid diagnostic neighbor indices')
    return np.concatenate([np.linalg.norm(features[i:i+512].astype(np.float64)-bank[indices[i:i+512]].astype(np.float64),axis=1)
                           for i in range(0,len(features),512)])


class Runtime(d.Runtime):
    def native_neighbors(self, features):
        with self.torch.inference_mode():
            q = self.torch.as_tensor(np.array(features, dtype=np.float32), device='cuda')
            distances, indices = self.model.model.nearest_neighbors(q, 1)
        return distances.cpu().numpy().copy(), indices.cpu().numpy().copy()


def tile_audit(runtime, features, saved_native, saved_map, bank, check):
    native, native_i = runtime.native_neighbors(features)
    double, double_i = runtime.fast_search(features, bank, check)
    native_exact = direct_distances(features, bank, native_i)
    double_exact = direct_distances(features, bank, double_i)
    d.check_distances(double, double_exact, 'diagnostic double winners')
    # Error-selected locations are for numerical diagnosis, not performance selection.
    probe_ids = sorted({int(np.argmax(np.abs(native-double))), int(np.argmin(double))})
    ref, ref_i = runtime.search(features[probe_ids], bank, check)
    d.check_distances(double[probe_ids], ref, 'independent exhaustive direct-cdist queries')
    reconstructed = runtime.maps(saved_native)
    matched = runtime.maps(double)
    report = {
        'saved_native_vs_public_repeat': difference(saved_native, native, 1e-4),
        'saved_map_vs_native_reconstruction': difference(saved_map, reconstructed, 1e-4),
        'native_vs_double_distances': difference(native, double, d.POLICY['distance_atol'], d.POLICY['distance_rtol']),
        'native_vs_exact_same_native_neighbor': difference(native, native_exact, 1e-4),
        'native_neighbor_exact_vs_double_minimum': difference(native_exact, double, 1e-4),
        'saved_map_vs_matched_double': difference(saved_map, matched, d.POLICY['float64_vs_native_map_atol'], d.POLICY['float64_vs_native_map_rtol']),
        'neighbor_indices_different': int(np.sum(native_i != double_i)),
        'native_zero_distances': int(np.sum(native == 0)), 'double_minimum': float(double.min()),
        'saved_top656': float(np.partition(saved_map.ravel().astype(np.float64), -656)[-656:].mean()),
        'double_top656': float(np.partition(matched.ravel().astype(np.float64), -656)[-656:].mean()),
        'independent_reference': [{'flat_index':i, 'native':float(native[i]), 'double':float(double[i]),
             'direct_cdist':float(ref[j]), 'direct_neighbor_index':int(ref_i[j]),
             'same_native_neighbor_exact':float(native_exact[i]), 'query_norm':float(np.linalg.norm(features[i].astype(np.float64)))}
             for j,i in enumerate(probe_ids)],
    }
    arrays = {'native-distances.npy':native, 'double-distances.npy':double,
              'native-indices.npy':native_i, 'double-indices.npy':double_i,
              'native-neighbor-exact.npy':native_exact, 'saved-map.npy':saved_map,
              'native-reconstruction.npy':reconstructed, 'double-map.npy':matched}
    return report, arrays


def run(project, runtime_factory=Runtime):
    old = project/'artifacts/analyses'/d.NAME
    identity = json.loads((old/'identity/value.json').read_bytes()); old_sig = fit.digest(identity)
    p.value(old, 'identity', old_sig)
    if identity['name'] != d.NAME or identity['policy'] != d.POLICY:
        raise ValueError('Expected unchanged full-map v1 identity')
    for name, sha in identity['sources_lf_sha256'].items():
        if hashlib.sha256((project/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != sha:
            raise ValueError(f'Failed-run source changed: {name}')
    failures = []
    for folder in sorted(old.glob('attempt-*')):
        record = json.loads((folder/'finished.json').read_bytes())
        if record['identity'] != old_sig: raise ValueError('Attempt identity differs')
        failures.append(record)
    if not any(r['status']=='failed' and r['error']=='ValueError: Matched double-precision map too far from native map' for r in failures):
        raise ValueError('Expected recorded map-parity failure; no bypass of other failures')
    handoffs = project/'artifacts/handoff'
    _,parent,_ = d.c.overlap.load(handoffs)
    plan = parent['plan.json']; parent_sig = fit.digest(plan)
    prior_bytes = d.checked_input(handoffs/(d.c.NAME+'.zip'))
    prior_plan = json.loads(prior_bytes['plan.json'])
    if identity['parent_execution_sha256'] != parent_sig or identity['tiles'] != prior_plan['selected_tiles']:
        raise ValueError('Parent or fixed tile selection differs')
    for name, sha in plan['sources_lf_sha256'].items():
        if hashlib.sha256((project/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest() != sha:
            raise ValueError(f'Parent source changed: {name}')
    model_root = project/'artifacts'/p.NAME
    model_receipt = fit.read_phase(model_root,'model',parent_sig,['model.pt','context.json'])
    if model_receipt != parent['model-receipt.json'] or model_receipt['files']['model.pt'] != identity['checkpoint_sha256']:
        raise ValueError('Frozen model differs')
    payload = d.c.read_overlap(handoffs/(d.c.overlap.NAME+'.zip'))
    evidence = {'name':NAME, 'failed_identity_sha256':old_sig, 'failed_attempts':failures,
                'policy':{'diagnostic_only':True, 'threshold_changed':False, 'training':False,
                          'full_pool_search':False, 'permission_to_resume':False},
                'sources_lf_sha256':{n:hashlib.sha256((project/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in SOURCES}}
    root = project/'artifacts/analyses'/NAME
    with fit.locked_run(root):
        sig = engine.initialize(root,evidence)
        with engine.attempt(root,sig,d.POLICY) as monitor:
            if d.c.environment() != plan['environment']: raise ValueError('Environment differs')
            runtime = runtime_factory(plan['config']); monitor.runtime = runtime
            runtime.load(model_root/'model/model.pt',parent['fit-report.json']['context'])
            bank = runtime.model.model.memory_bank.detach().cpu().numpy().copy()
            if bank.shape != (d.POLICY['bank'],768): raise ValueError('Bank shape differs')
            exported = {}; rows = []
            for i, record in enumerate(identity['tiles']):
                monitor.check(); name=f'tile-{i:02d}'
                receipt = fit.read_phase(old,name,old_sig,['features.npy','native-distances.npy','replay.json'])
                if receipt['record_sha256'] != fit.digest(record): raise ValueError('Tile identity differs')
                files=['report.json','native-distances.npy','double-distances.npy','native-indices.npy','double-indices.npy',
                       'native-neighbor-exact.npy','saved-map.npy','native-reconstruction.npy','double-map.npy']
                if not (root/name).exists():
                    features=p.array(old/name/'features.npy',(4096,768))
                    native=p.array(old/name/'native-distances.npy',(4096,))
                    saved=np.load(io.BytesIO(payload[record['prefix']+'/map.npy']),allow_pickle=False)
                    result, arrays = tile_audit(runtime,features,native,saved,bank,monitor.check)
                    result.update(key=record['key'],region=record['source_row']['region'])
                    with fit.phase_stage(root,name) as stage:
                        fit.write_json(stage/'report.json',result)
                        for n, array in arrays.items(): np.save(stage/n,array,allow_pickle=False)
                        fit.seal_phase(stage,name,sig,{'source_receipt_sha256':receipt['receipt_sha256']},files)
                audited=fit.read_phase(root,name,sig,files)
                if audited['source_receipt_sha256'] != receipt['receipt_sha256']: raise ValueError('Audit source differs')
                result=json.loads((root/name/'report.json').read_bytes());rows.append(result)
                for n in files: exported[name+'/'+n]=(root/name/n).read_bytes()
                print(f"audit {i+1}/{len(identity['tiles'])}: {result['key']} {result['region']} map delta={result['saved_map_vs_matched_double']['max_absolute_difference']:.8g}, pixels outside={result['saved_map_vs_matched_double']['outside_tolerance']}",flush=True)
            if not np.array_equal(bank,runtime.model.model.memory_bank.detach().cpu().numpy()): raise ValueError('Bank mutated')
            report={'status':'diagnostic_completed_not_resume_authorization','tiles':rows,
                    'tiles_outside_existing_tolerance':sum(r['saved_map_vs_matched_double']['outside_tolerance']>0 for r in rows),
                    'training':False,'threshold_changed':False,'full_pool_search':False,'permission_to_resume':False}
            p.publish(root,'report',sig,report)
            if not (root/'resource').exists(): p.publish(root,'resource',sig,monitor.snapshot())
            exported.update({'report.json':fit.canonical(report),'plan.json':fit.canonical(evidence),
                             'failed-identity.json':fit.canonical(identity),'resource.json':fit.canonical(p.value(root,'resource',sig))})
            monitor.check(); output=handoffs/(NAME+'.zip'); d.c.write_bundle(output,exported)
            return {'status':report['status'],'tiles_outside_existing_tolerance':report['tiles_outside_existing_tolerance'],'output':str(output)}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--project',type=Path,default=ROOT)
    args=parser.parse_args(argv)
    try: print(json.dumps(run(args.project),indent=2));return 0
    except KeyboardInterrupt: print('Interrupted; preserve files. Same diagnostic command resumes completed tiles.');return 130
    except Exception as exc: print(f'ERROR: {type(exc).__name__}: {exc}. Preserve original and diagnostic outputs.');return 1


if __name__=='__main__':raise SystemExit(main())
