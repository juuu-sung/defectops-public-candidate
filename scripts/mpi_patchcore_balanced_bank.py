"""One fixed region-balanced selection; inference still searches the whole bank."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from scripts import mpi_patchcore_development_scaling as development
from scripts import mpi_patchcore_scaling_evaluation as evaluation

p, fit = development.p, development.fit
NAME = 'mpi-patchcore-balanced-bank-v1'
DEVELOPMENT_SHA = '992b0c47ef2b2ee4641077ad2ca75b2d07533bfedf776c903473cdb352e609b8'
POLICY = {
    **p.POLICY, 'normal_development': 256, 'initial_free_disk_GiB': 10,
    'selection': '12 independent region k-center greedy selections in the SAME cached 369D projection; concatenate in fixed REGIONS order',
    'quotas': [6554]*7+[6553]*5,
    'initial_center': 'parent projection start_index modulo 262144, shared local photo/patch index in every region',
    'query_search': 'all 78643 representatives, unrestricted by query region',
    'accuracy_targets': {'preserve_all_55': True, 'fp54_max': 2, 'fp256_max': 12},
    'resume': 'atomic per-region selection checkpoints every1000 plus final; photos; Ctrl+C only; failed/unaccounted attempts require review',
    'projection_caveat': 'Reuse exact parent projected cache; no new projection or feature extraction',
    'selection_inputs': 'only frozen 256 training photos and original projection; no score/map/GT/hard-case input',
    'baseline': 'same256 training bank with development256 contract, TP55 FP1/54 FP15/256',
}


def quotas(total=78643, regions=12):
    if regions < 1 or total < regions:
        raise ValueError('Positive quota for every region required')
    q, r = divmod(total, regions)
    return [q+int(i < r) for i in range(regions)]


def global_indices(region, photos=256, regions=12, patches=1024):
    if not 0 <= region < regions:
        raise ValueError('Region out of bounds')
    return (np.arange(photos, dtype=np.int64)[:, None]*regions*patches
            + region*patches + np.arange(patches, dtype=np.int64)).ravel()


def distribution(indices, photos=256, regions=12, patches=1024):
    ids = np.asarray(indices)
    if (ids.ndim != 1 or not np.issubdtype(ids.dtype, np.integer)
            or len(np.unique(ids)) != len(ids) or np.any(ids < 0)
            or np.any(ids >= photos*regions*patches)):
        raise ValueError('Invalid or repeated representative index')
    return {'regions': np.bincount(ids//patches % regions, minlength=regions).tolist(),
            'photos': np.bincount(ids//(regions*patches), minlength=photos).tolist()}


def validate_roles(training, cohorts):
    development.pool_protocol.validate_group(training, 256)
    seen_ids = set(training['photo_ids'])
    seen_pixels = {r['decoded_rgb_sha256'] for r in training['rows']}
    for group in ('development', 'calibration', 'validation', 'test'):
        photos = cohorts[group]
        expected = {'development':256, 'calibration':169, 'validation':256, 'test':109}[group]
        if len(photos) != expected or len({x['key'] for x in photos}) != expected:
            raise ValueError('Cohort size/keys changed')
        ids = {x['photo_id'] for x in photos}
        pixels = {r['decoded_rgb_sha256'] for x in photos for r in x['regions']}
        if group != 'test' and (len(ids) != expected or ids & seen_ids):
            raise ValueError('Normal photo roles overlap')
        if pixels & seen_pixels:
            raise ValueError('Exact RGB duplicate crosses roles')
        if group != 'test': seen_ids.update(ids)
        seen_pixels.update(pixels)
        labels = [x['label'] for x in photos]
        if (group == 'test' and (labels.count(1), labels.count(0)) != (55,54)) or (group != 'test' and any(labels)):
            raise ValueError('Cohort labels changed')


def validate_raw(raw, cohorts):
    if set(raw) != set(cohorts): raise ValueError('Score cohorts incomplete')
    for group, photos in cohorts.items():
        if len(raw[group]) != len(photos): raise ValueError('Missing scores')
        for sample, photo in zip(raw[group], photos, strict=True):
            p.comparison.c.scoring.old.validate_sample(sample, photo)


def contract_replay(raw, method):
    values = p.comparison.matrix(raw)
    if p.comparison.fit_method(values['development'], values['calibration']) != method:
        raise ValueError('Frozen development256/calibration169 contract differs')


def prepare(project: Path, handoffs=None):
    handoffs = handoffs or project/'artifacts/handoff'
    parent = development.evidence.checked_bundle(handoffs/'mpi-patchcore-scaling-v1.zip', development.evidence.INPUT_SHA)
    dev = development.evidence.checked_bundle(handoffs/(development.NAME+'.zip'), DEVELOPMENT_SHA)
    plan = parent['plan.json']
    if (plan['policy'] != p.POLICY or dev['plan.json']['policy'] != development.POLICY
            or dev['plan.json']['parent_execution_sha256'] != fit.digest(plan)
            or dev['plan.json']['checkpoint_sha256'] != parent['fit-report.json']['checkpoint_sha256']):
        raise ValueError('Reviewed parent recipe/identity differs')
    for record in (plan, dev['plan.json']):
        for name, sha in record['sources_lf_sha256'].items():
            if hashlib.sha256((project/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != sha:
                raise ValueError('Frozen dependency differs: '+name)
    cohorts = {**plan['cohorts'], 'development': dev['plan.json']['expanded_development']}
    validate_roles(plan['training'], cohorts)
    raw = {**parent['normal-scores.json'], **parent['evaluation/scores.json']}
    raw = development.validate_expanded(raw, dev['additional-scores.json'], dev['plan.json'])
    validate_raw(raw, cohorts)
    contract_replay(raw, dev['contract.json'])
    counts = p.comparison.counts(p.comparison.matrix(raw), dev['contract.json'], [s['label'] for s in raw['test']])
    if counts != {'tp':55, 'fn':0, 'fp54':1, 'tn54':53, 'fp256':15}:
        raise ValueError('Reviewed comparison baseline changed')
    selected = parent['selected-indices.json']['indices']
    if len(selected) != POLICY['representatives']: raise ValueError('Bank size differs')
    paths = sorted([*project.glob('scripts/*.py'), *project.glob('src/defectops/**/*.py'),
                    project/'scripts/run_mpi_patchcore_balanced_bank.ps1',
                    project/'docs/mpi-patchcore-balanced-bank-v1.md'])
    identity = {'name':NAME, 'policy':POLICY, 'training':plan['training'], 'cohorts':cohorts,
        'archive_sha256':plan['archive_sha256'], 'config':plan['config'], 'environment':plan['environment'],
        'parent_execution_sha256':fit.digest(plan), 'development_execution_sha256':fit.digest(dev['plan.json']),
        'parent_zip_sha256':development.evidence.INPUT_SHA, 'development_zip_sha256':DEVELOPMENT_SHA,
        'parent_checkpoint_sha256':parent['fit-report.json']['checkpoint_sha256'],
        'baseline_contract':dev['contract.json'], 'baseline_counts':counts,
        'baseline_scores_sha256':fit.digest(raw), 'parent_distribution':distribution(selected),
        'projection_receipts_sha256':fit.digest(parent['projection-receipts.json']),
        'feature_receipts_sha256':fit.digest(parent['feature-receipts.json']),
        'sources_lf_sha256':{f.relative_to(project).as_posix():hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in paths}}
    return {'parent':parent, 'development':dev, 'reference':raw, 'cohorts':cohorts,
            'config':plan['config'], 'contract':{'candidate_top1_q90':dev['contract.json']}}, identity


def compare(source, candidate, method):
    validate_raw(candidate, source['cohorts'])
    result = evaluation.compare(source, candidate, method)
    result['gates'] = {'all55_preserved':result['counts']['candidate']['tp']==55 and not result['lost_baseline_detections'],
        'fp54_at_most2':result['counts']['candidate']['fp54']<=2,
        'fp256_at_most12':result['counts']['candidate']['fp256']<=12}
    result['passed_accuracy_targets'] = all(result['gates'].values())
    result['comparison'] = 'Global greedy versus region-balanced greedy; both256 development and same169 calibration'
    return result


def stability(baseline, candidate, check=lambda:None):
    arrays = {k:p.comparison.matrix(v) for k,v in [('baseline',baseline), ('candidate',candidate)]}
    labels = [x['label'] for x in baseline['test']]
    trials = {}
    # Reuse EXACT expanded256 draws from the preceding development experiment for BOTH banks.
    for scope,i,_,di,ci in development.draws():
        check()
        row = {'scope':scope, 'iteration':i, 'development_indices':di.tolist(), 'calibration_indices':ci.tolist()}
        for model,values in arrays.items():
            try:
                method = p.comparison.fit_method(values['development'][di], values['calibration'][ci])
                row[model] = {'valid':True, 'counts':p.comparison.counts(values,method,labels), 'threshold':method['threshold']}
            except ValueError as exc:
                row[model] = {'valid':False, 'error':str(exc)}
        row['valid'] = all(row[m]['valid'] for m in arrays)
        trials.setdefault(scope, []).append(row)
    summaries = {s:p.evaluation.summarize(rows) for s,rows in trials.items()}
    return {'trials':trials, 'summaries':summaries, 'passed':all(s['passed'] for s in summaries.values()),
            'interpretation':'600 paired normal-sample sensitivity draws; not population confidence intervals; no rerolls'}
