"""One locked normal-only scale-shrinkage comparison; CPU saved scores only."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import mpi_patchcore_raw_bank as raw_bank
from scripts.export_mpi_patchcore_scaling_review import write_bundle

p, fit = raw_bank.p, raw_bank.fit
comparison = p.comparison
NAME = 'mpi-patchcore-scale-shrinkage-v1'
INPUT_SHA = 'be102c68fe55468f472966701663d7787b3b8001c3d463cf33c722bbfee67b17'
EXECUTION_SHA = '53f546546dae7ae2713381b9f84d38f2b4e455abdc010db6e7bb79dd115b7b43'
POLICY = {
    'single_change': 'region Q90-minus-median width blended with median of12 widths',
    'weight_local': 0.5, 'weight_pooled': 0.5, 'pooled_estimator': 'median',
    'center': 'development region median unchanged',
    'threshold': 'calibration photo Q95 higher; >=; numerical value refitted',
    'candidate_search': False, 'training': False, 'new_inference': False,
    'accuracy_targets': {'tp': 55, 'fp54_max': 2, 'fp256_max': 12},
    'stability': p.evaluation.POLICY['acceptance']['stability'],
    'bootstrap': 'exact600 parent photo draws; recompute pooled width in each draw',
    'model_and_maps_changed': False, 'automatic_pass_allowed': False,
    'service_model_changed': False, 'time_limit_seconds': None,
    'interpretation': 'observed exploratory regression; no new generalization claim',
}


def fit_method(development, calibration):
    """No labels, validation scores, trial outcomes or tunable alpha accepted."""
    for x in (development, calibration):
        if (x.ndim != 2 or x.shape[1] != 12 or len(x) < 2
                or not np.isfinite(x).all()):
            raise ValueError('Expected finite normal score matrix with12 regions')
    original = comparison.fit_method(development, calibration)
    widths = np.array([r['scale'] for r in original['regions']])
    pooled = float(np.median(widths))
    scale = .5 * widths + .5 * pooled
    center = np.array([r['center'] for r in original['regions']])
    threshold = float(np.quantile(np.max((calibration-center)/scale, axis=1), .95, method='higher'))
    if not np.isfinite(threshold):
        raise ValueError('Nonfinite candidate threshold')
    return {'field': 'top1', 'scale_kind': 'q90_half_pooled_median',
            'threshold': threshold, 'pooled_width': pooled,
            'unblended_widths': widths.tolist(), 'weight_local': .5,
            'regions': [{'region': r, 'center': float(c), 'scale': float(s)}
                        for r, c, s in zip(fit.REGIONS, center, scale, strict=True)]}


def checked_input(path, project):
    data = raw_bank.development.evidence.checked_bundle(path, INPUT_SHA)
    plan = data['plan.json']
    if fit.digest(plan) != EXECUTION_SHA:
        raise ValueError('Reviewed execution changed')
    for name, sha in plan['sources_lf_sha256'].items():
        if hashlib.sha256((project/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != sha:
            raise ValueError('Frozen source changed: '+name)
    raw_bank.prior.validate_roles(plan['training'], plan['cohorts'])
    raw = {**data['normal-scores.json'], **data['evaluation/scores.json']}
    raw_bank.validate_raw(raw, plan['cohorts'])
    raw_bank.contract_replay(raw, data['contract.json'])
    if p.reload_check(data['normal-scores.json'], data['reload-scores.json'], data['contract.json']) != data['reload-check.json']:
        raise ValueError('Saved reload record differs')
    return data, raw


def make_plan(project, data):
    paths = [Path(__file__).relative_to(ROOT).as_posix(),
             'docs/mpi-patchcore-scale-shrinkage-v1.md',
             'tests/test_mpi_patchcore_scale_shrinkage.py',
             'scripts/export_mpi_patchcore_scaling_review.py']
    return {'name': NAME, 'policy': POLICY, 'input_sha256': INPUT_SHA,
            'parent_execution_sha256': EXECUTION_SHA,
            'checkpoint_sha256': data['fit-report.json']['checkpoint_sha256'],
            'baseline_contract': data['contract.json'],
            'normal_scores_sha256': fit.digest(data['normal-scores.json']),
            'cohorts_sha256': fit.digest(data['plan.json']['cohorts']),
            'numpy': np.__version__, 'python': platform.python_version(),
            'sources_lf_sha256': {n: hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in paths}}


def compare(raw, original, candidate):
    source = {'reference': raw, 'contract': {'candidate_top1_q90': original}}
    result = raw_bank.evaluation.compare(source, raw, candidate)
    count = result['counts']['candidate']
    result['gates'] = {'all55_preserved': count['tp'] == 55 and not result['lost_baseline_detections'],
                       'fp54_at_most2': count['fp54'] <= 2, 'fp256_at_most12': count['fp256'] <= 12}
    result['passed_accuracy_targets'] = all(result['gates'].values())
    result['comparison'] = 'Same raw768 bank and maps; original scale versus half-pooled scale'
    return result


def summarize_trials(trials):
    summaries = {scope: p.evaluation.summarize(rows) for scope, rows in trials.items()}
    descriptive = {scope: {m: {
        'valid_draws': sum(r[m]['valid'] for r in rows),
        'all55_draws': sum(r[m]['valid'] and r[m]['counts']['tp'] == 55 for r in rows),
        'minimum_tp': min((r[m]['counts']['tp'] for r in rows if r[m]['valid']), default=None),
    } for m in ('baseline', 'candidate')} for scope, rows in trials.items()}
    return {'trials': trials, 'summaries': summaries, 'descriptive': descriptive,
            'passed': all(s['passed'] for s in summaries.values()),
            'interpretation': 'Paired normal resampling, not population confidence intervals; invalid draws retained'}


def stability(raw, old_trials):
    values = comparison.matrix(raw)
    labels = [s['label'] for s in raw['test']]
    trials = {}
    for scope, i, _, di, ci in raw_bank.development.draws():
        old = old_trials[scope][i]
        if (old['iteration'] != i or old['development_indices'] != di.tolist()
                or old['calibration_indices'] != ci.tolist()):
            raise ValueError('Parent resampling indices changed')
        row = {'scope': scope, 'iteration': i, 'development_indices': di.tolist(),
               'calibration_indices': ci.tolist()}
        for name, factory in [('baseline', comparison.fit_method), ('candidate', fit_method)]:
            try:
                method = factory(values['development'][di], values['calibration'][ci])
                row[name] = {'valid': True, 'counts': comparison.counts(values, method, labels),
                             'threshold': method['threshold'], 'contract': method}
            except ValueError as exc:
                row[name] = {'valid': False, 'error': str(exc)}
        if {k: v for k, v in row['baseline'].items() if k != 'contract'} != old['candidate']:
            raise ValueError('Raw-bank baseline sensitivity does not reproduce')
        row['valid'] = all(row[n]['valid'] for n in ('baseline', 'candidate'))
        trials.setdefault(scope, []).append(row)
    if set(old_trials) != set(trials) or any(len(old_trials[s]) != 200 for s in trials):
        raise ValueError('Expected exactly600 unchanged parent draws')
    return summarize_trials(trials)


def boundaries(before, after):
    return [{'region': a['region'], 'center': a['center'], 'scale_before': a['scale'],
             'scale_after': b['scale'],
             'raw_boundary_before': a['center']+a['scale']*before['threshold'],
             'raw_boundary_after': b['center']+b['scale']*after['threshold']}
            for a, b in zip(before['regions'], after['regions'], strict=True)]


def run(project, path, root, phase):
    data, raw = checked_input(path, project)
    identity = make_plan(project, data)
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if phase == 'plan':
            p.publish(root, 'plan', signature, identity)
            return {'status': 'normal_scale_plan_frozen', 'plan_sha256': signature,
                    'evaluation_performed': False, 'output': str(root)}
        if not (root/'plan').exists() or p.value(root, 'plan', signature) != identity:
            raise ValueError('Matching frozen plan required before evaluation')
        if (root/'result').exists():
            result = p.value(root, 'result', signature)
        else:
            started = time.perf_counter()
            values = comparison.matrix(raw)
            # Candidate fit receives only normal development and calibration.
            contract = fit_method(values['development'], values['calibration'])
            p.publish(root, 'contract', signature, contract)
            result_compare = compare(raw, data['contract.json'], contract)
            if result_compare['counts']['baseline'] != data['evaluation/comparison.json']['counts']['candidate']:
                raise ValueError('Fixed baseline counts changed')
            stable = stability(raw, data['evaluation/stability.json']['trials'])
            for name, payload in [('comparison', result_compare), ('stability', stable),
                                  ('boundaries', boundaries(data['contract.json'], contract))]:
                p.publish(root, name, signature, payload)
            result = {'status': 'observed_cpu_calibration_comparison_completed',
                'counts': result_compare['counts'], 'accuracy_targets_passed': result_compare['passed_accuracy_targets'],
                'stability_targets_passed': stable['passed'], 'model_changed': False,
                'service_model_changed': False, 'automatic_pass_allowed': False,
                'new_image_reads': 0, 'new_inference': 0, 'training_updates': 0,
                'maps_and_within_tile_location_metrics_changed': False,
                'warm_inference_rebenchmarked': False,
                'scope': 'Raw scores/maps frozen; only post-map calibration changes; inherited location/warm records are not fresh measurements',
                'cpu_analysis_seconds': time.perf_counter()-started, 'plan_sha256': signature}
            p.publish(root, 'result', signature, result)
        names = ('plan', 'contract', 'comparison', 'stability', 'boundaries', 'result')
        payload = {n+'.json': fit.canonical(p.value(root, n, signature)) for n in names}
        write_bundle(project/'artifacts/handoff'/(NAME+'.zip'), payload)
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--phase', choices=('plan', 'evaluate'), required=True)
    args = parser.parse_args()
    try:
        result = run(ROOT, args.input, ROOT/'artifacts/analyses'/NAME, args.phase)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (ValueError, OSError, KeyError) as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no model or threshold replacement.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
