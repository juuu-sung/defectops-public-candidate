"""Read-only fixed 14 normal / weakest defect score and saved-map review."""
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
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import mpi_patchcore_development_scaling as d
from scripts import run_mpi_patchcore_scaling as engine
from scripts.export_mpi_patchcore_scaling_review import checked_png, write_bundle
from scripts import mpi_supersimplenet_map_diagnostic as maps

NAME = 'mpi-patchcore-score-overlap-v1'
INPUT_SHA = '992b0c47ef2b2ee4641077ad2ca75b2d07533bfedf776c903473cdb352e609b8'
fit, p = d.fit, d.p


def load(handoffs):
    current = d.evidence.checked_bundle(handoffs / 'mpi-patchcore-development-scaling-v1.zip', INPUT_SHA)
    parent = d.evidence.checked_bundle(handoffs / 'mpi-patchcore-scaling-v1.zip', d.evidence.INPUT_SHA)
    raw = {**parent['normal-scores.json'], **parent['evaluation/scores.json']}
    plan = current['plan.json']
    for name, expected in plan['sources_lf_sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != expected:
            raise ValueError(f'Frozen scoring dependency changed: {name}')
    if (plan['policy'] != d.POLICY or plan['source_zip_sha256'] != d.evidence.INPUT_SHA or
            plan['parent_execution_sha256'] != fit.digest(parent['plan.json']) or
            plan['original_scores_sha256'] != fit.digest(raw) or
            plan['checkpoint_sha256'] != parent['fit-report.json']['checkpoint_sha256']):
        raise ValueError('Frozen model or score provenance differs')
    expanded = d.validate_expanded(raw, current['additional-scores.json'], plan)
    arr = p.comparison.matrix(expanded)
    method = p.comparison.fit_method(arr['development'], arr['calibration'])
    if method != current['contract.json']:
        raise ValueError('Contract replay differs')
    replay = d.compare(raw, expanded, parent['contract.json'], method)
    # Scalars replay on CPU; floating-point tolerance only for numerical values.
    def same(a, b):
        if isinstance(a, dict): return set(a) == set(b) and all(same(a[k], b[k]) for k in a)
        if isinstance(a, list): return len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
        if isinstance(a, float): return np.isfinite(a) and np.isfinite(b) and abs(a-b) <= 1e-10
        return a == b
    if not same(replay, current['comparison.json']):
        raise ValueError('Saved predictions do not replay')
    return current, parent, raw


def decomposition(normal, defect):
    """Symmetric 3-factor arithmetic Shapley, not a causal attribution."""
    def score(x): return (x['raw'] - x['center']) / x['scale']
    names = ('raw', 'center', 'scale')
    values = [score({k: normal[k] if bits & (1 << i) else defect[k]
                     for i, k in enumerate(names)}) for bits in range(8)]
    old = d.evidence.allocation(values)
    contributions = dict(zip(names, old.values(), strict=True))
    return {'normal_minus_defect':values[7]-values[0], 'contributions':contributions,
            'scope':'Arithmetic decomposition of two winning tiles; includes interaction allocation, not physical cause or an adopted alternative scoring rule'}


def analyze(current, parent, raw):
    method = current['contract.json']
    predictions = current['comparison.json']['predictions']['candidate']
    weakest = min((s for s in predictions['test'] if s['label']), key=lambda s:(s['score'], s['key']))
    normals = sorted((s for s in predictions['validation'] if s['score'] >= weakest['score']), key=lambda s:s['key'])
    if weakest['key'] != 'defect/0024' or len(normals) != 14:
        raise ValueError('Frozen 14-normal / defect0024 selection differs')
    by_key = {s['key']:s for group in raw.values() for s in group}
    def tile(key, region):
        j = fit.REGIONS.index(region)
        return {'region':region, 'raw':by_key[key]['regions'][j]['top1'],
                'center':method['regions'][j]['center'], 'scale':method['regions'][j]['scale']}
    ref = tile(weakest['key'], weakest['winning_region'])
    cases = []
    for prediction in [weakest] + normals:
        key = prediction['key']
        tiles = []
        for region in fit.REGIONS:
            x = tile(key, region); reference_same = tile(weakest['key'], region)
            x['normalized_score'] = (x['raw']-x['center'])/x['scale']
            x['raw_boundary'] = x['center']+x['scale']*method['threshold']
            x['raw_delta_vs_defect_same_region'] = x['raw']-reference_same['raw']
            tiles.append(x)
        winner = tile(key, prediction['winning_region'])
        cases.append({'key':key, 'label':prediction['label'], 'prediction':prediction,
            'score_margin_to_threshold':prediction['score']-method['threshold'], 'tiles':tiles,
            'winning_raw_delta_vs_defect_winner':winner['raw']-ref['raw'],
            'winner_comparison':decomposition(winner, ref)})
    return {'status':'fixed_score_overlap_diagnostic', 'selection':[s['key'] for s in [weakest]+normals],
        'cases':cases, 'defect_reference':weakest, 'normal_count':14,
        'raw_winner_at_least_defect_winner':sum(c['winning_raw_delta_vs_defect_winner']>=0 for c in cases[1:]),
        'normal_winner_same_region_as_defect':sum(c['prediction']['winning_region']==weakest['winning_region'] for c in cases[1:]),
        'theoretical_fp_floor_at_all55':14, 'threshold':method['threshold'],
        'contract_changed':False, 'automatic_pass_allowed':False,
        'interpretation':'Observed diagnostic only; no threshold fitted to labels, no training, no causal conclusion from cross-region raw scores'}


def export(project, archive_path, current, parent, report):
    plan = parent['plan.json']
    if fit.sha256_file(archive_path) != plan['archive_sha256']:
        raise ValueError('Archive fingerprint differs')
    signature = fit.digest(plan)
    checkpoint = parent['fit-report.json']['checkpoint_sha256']
    cache_root = project / 'artifacts/analyses' / p.NAME
    index = {photo['key']:(g,i,photo) for g,photos in plan['cohorts'].items() for i,photo in enumerate(photos)}
    payload = {'scores.json':fit.canonical(report)}
    records = []
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive member names')
        for n,key in enumerate(report['selection']):
            group,i,photo = index[key]
            if group not in ('test','validation'):
                raise ValueError('Unexpected review cohort')
            name = f'photo-{i:03d}'
            expected = parent['evaluation/prediction-receipts.json'][f'{group}\\{name}\\receipt.json']
            actual = fit.read_phase(cache_root/group, name, signature, ['sample.json','maps.npy'])
            if actual != expected:
                raise ValueError('Map receipt differs from frozen evaluation')
            sample = engine.read_sample(cache_root/group, name, signature, photo, checkpoint)
            original_sample = parent['evaluation/scores.json'][group][i]
            if sample != original_sample:
                raise ValueError('Cached sample differs from frozen evaluation')
            values = p.array(cache_root/group/name/'maps.npy', (12,256,256))
            masks = engine.ev.read_masks(archive, photo) if photo['label'] else [None]*12
            for j,row in enumerate(photo['regions']):
                prefix = f'photos/{n:02d}/{j:02d}'
                payload[prefix+'/source.png'] = checked_png(archive,row)
                buffer = io.BytesIO(); np.save(buffer,values[j],allow_pickle=False)
                payload[prefix+'/map.npy'] = buffer.getvalue()
                if masks[j] is not None:
                    mask_raw = archive.read(row['mask_member'])
                    if hashlib.sha256(mask_raw).hexdigest() != row['mask_sha256']:
                        raise ValueError('Mask bytes differ')
                    payload[prefix+'/mask-source.png'] = mask_raw
                    buffer = io.BytesIO(); np.save(buffer,masks[j],allow_pickle=False)
                    payload[prefix+'/mask.npy'] = buffer.getvalue()
                records.append({'prefix':prefix,'key':key,'label':photo['label'],'source_row':row,
                    'sample_region':sample['regions'][j], 'saved_receipt':expected,
                    'photo_sha256':fit.digest(photo),
                    'map_stats':{k:v for k,v in maps.map_stats(values[j].astype(np.float64), masks[j]).items() if not k.startswith('display_')}})
            print(f'Read-only export {n+1}/15 photos; all12 regions, no inference.',flush=True)
    payload['review.json'] = fit.canonical({'status':'saved_map_review_exported', 'records':records,
        'input_sha256':INPUT_SHA,'parent_sha256':d.evidence.INPUT_SHA,'archive_sha256':plan['archive_sha256'],
        'checkpoint_sha256':checkpoint,'contract':current['contract.json'],
        'mask_rule':'Same frozen read_masks nonzero parser as prior evaluation; all12 tiles retained, including empty masks',
        'photos':15,'normal_photos':14,'defect_photos':1,'tiles':180,
        'new_inference':False,'training':False,'threshold_changed':False,'automatic_pass_allowed':False,
        'rights':'Private review only; do not redistribute source images or masks',
        'sources_lf_sha256':{f:hashlib.sha256((project/f).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
            for f in ('scripts/review_mpi_patchcore_score_overlap.py','scripts/run_mpi_patchcore_score_overlap.ps1')}})
    output = project/'artifacts/handoff'/(NAME+'.zip')
    with fit.locked_run(project/'artifacts/analyses'/(NAME+'-export-lock')):
        write_bundle(output,payload)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT)
    parser.add_argument('--archive',type=Path)
    parser.add_argument('--handoffs',type=Path)
    parser.add_argument('--scores-only',action='store_true')
    args = parser.parse_args(argv)
    current,parent,raw = load(args.handoffs or args.project/'artifacts/handoff')
    report = analyze(current,parent,raw)
    if args.scores_only:
        print(json.dumps(report,indent=2)); return 0
    if args.archive is None: parser.error('--archive is required for map export')
    output = export(args.project,args.archive,current,parent,report)
    print(json.dumps({'status':'exported_no_inference','photos':15,'tiles':180,'output':str(output)},indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
