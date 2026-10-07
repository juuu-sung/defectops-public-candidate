"""Frozen bank; nested development128->256, never fit on test or pick thresholds."""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from scripts import analyze_mpi_patchcore_scaling_sensitivity as evidence
from scripts import mpi_supersimplenet_scaling_protocol as pool_protocol
from scripts import mpi_patchcore_scaling_evaluation as evaluation

p, fit = evidence.p, evidence.p.fit
NAME = "mpi-patchcore-development-scaling-v1"
POOL_ZIP = "mpi-normal-diversity-preparation-v1.zip"
POLICY = {
    "model_training": False, "existing_development": 128, "additional_development": 128,
    "expanded_development": 256, "calibration": 169, "test_defects": 55,
    "test_normals": 54, "validation_normals": 256,
    "selection": "SHA256(name,NUL,additional-development,NUL,photo_id),ascending; no scores",
    "score": p.POLICY["score"], "threshold": p.POLICY["threshold"],
    "preserve_all_55": True, "fp54_max": 2, "fp256_max": 12,
    "stability": p.POLICY["stability"],
    "expanded_draw_seeds": {"development_only": 81001, "both": 81003},
    "draw_pairing": "Original baseline draws unchanged; shared calibration169 indices; independent full256 development draws. Not identical development indices or stratified128+128 draws.",
    "reload_photos": 8, "reload_absolute_tolerance": 1e-4,
    "resource_plan": p.POLICY["resource_plan"], "time_limit_seconds": None,
    "free_disk_GiB": 5, "free_gpu_GiB": 6,
    "new_test_image_reads": 0, "new_test_mask_reads": 0,
    "automatic_pass_allowed": False, "service_model_changed": False,
    "interpretation": "observed exploratory regression, not independent generalization",
}


def rank(pid):
    return hashlib.sha256(f"{NAME}\0additional-development\0{pid}".encode()).hexdigest(), pid


def select_additional(pool, plan, prior_training):
    """Input pool already excludes damaged/incomplete/exact duplicate/test tiles."""
    pool_protocol.validate_group({"photo_ids": pool["ids"], "rows": pool["rows"]}, len(pool["ids"]))
    cohorts = plan["cohorts"]
    training = plan["training"]
    pool_protocol.validate_group(training, 256)
    by_id = {pid: pool["rows"][i*12:(i+1)*12] for i, pid in enumerate(pool["ids"])}
    if any(by_id.get(pid) != training["rows"][i*12:(i+1)*12] for i,pid in enumerate(training["photo_ids"])):
        raise ValueError("Current model training not contained unchanged in reviewed pool")
    prior_ids = prior_training["training_photo_ids"]
    if len(prior_ids) != 64 or len(set(prior_ids)) != 64 or not set(prior_ids) <= set(by_id):
        raise ValueError("Historical diversity training IDs differ")
    prior_rows = { (r['photo_id'], r['region']): r for r in prior_training['training_rows'] }
    if len(prior_rows) != 64*12 or any(prior_rows.get((pid,r['region'])) != r for pid in prior_ids for r in by_id[pid]):
        raise ValueError("Historical training fingerprints differ")
    blocked_ids = set(training["photo_ids"]) | set(prior_ids)
    # Test IDs live in another namespace; block their RGB fingerprints, not numbers.
    blocked_pixels = {r['decoded_rgb_sha256'] for r in training['rows']}
    for group, photos in cohorts.items():
        if group != 'test': blocked_ids.update(photo['photo_id'] for photo in photos)
        blocked_pixels.update(r['decoded_rgb_sha256'] for photo in photos for r in photo['regions'])
    blocked_pixels.update(r['decoded_rgb_sha256'] for r in prior_rows.values())
    eligible = [pid for pid in pool['ids'] if pid not in blocked_ids and
                not any(r['decoded_rgb_sha256'] in blocked_pixels for r in by_id[pid])]
    if len(eligible) < 128:
        raise ValueError("Insufficient eligible normals; no role changes or duplicate replacement")
    ids = sorted(eligible, key=rank)[:128]
    photos = [{"key": "development-added/"+pid, "photo_id": pid, "label": 0, "regions": by_id[pid]} for pid in ids]
    return {"pool_count":len(pool['ids']), "eligible_count":len(eligible),
            "excluded_current_training_count":256, "historical_training_count":64,
            "additional_photos":photos, "additional_ids_sha256":fit.digest(ids),
            "additional_rows_sha256":fit.digest([r for photo in photos for r in photo['regions']]),
            "physical_board_independence_proven":False,
            "scope":"Unused for retained model training/development/calibration/evaluation; historical pool descriptors were observed. Exact RGB separation is not session/physical-board independence."}


def prepare(project, handoffs):
    data = evidence.checked_bundle(handoffs/'mpi-patchcore-scaling-v1.zip', evidence.INPUT_SHA)
    old_plan = data['plan.json']
    if old_plan['policy'] != p.POLICY:
        raise ValueError("Reviewed model recipe changed")
    for name, sha in old_plan['sources_lf_sha256'].items():
        if hashlib.sha256((project/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=sha:
            raise ValueError(f"Frozen dependency differs: {name}")
    normal_pool = pool_protocol.read_source(handoffs/POOL_ZIP, ('pool.json','selection.json'))
    pool = normal_pool['pool.json']
    if (len(pool['ids'])!=1086 or fit.digest(pool['ids'])!=pool_protocol.POOL_IDS_SHA
            or fit.digest(pool['rows'])!=pool_protocol.POOL_ROWS_SHA):
        raise ValueError("Reviewed eligible normal pool changed")
    raw = {**data['normal-scores.json'], **data['evaluation/scores.json']}
    p.comparison.validate_raw(raw, old_plan['cohorts'])
    p.comparison.contract_replay(raw, data['contract.json'])
    if p.comparison.counts(p.comparison.matrix(raw), data['contract.json'], [s['label'] for s in raw['test']]) != {
            'tp':55,'fn':0,'fp54':1,'tn54':53,'fp256':16}:
        raise ValueError("Reviewed fixed256-bank baseline changed")
    selection=select_additional(pool,old_plan,normal_pool['selection.json'])
    files=sorted([*project.glob('scripts/*.py'),*project.glob('src/defectops/**/*.py'),
                  project/'scripts/run_mpi_patchcore_development_scaling.ps1',
                  project/'docs/mpi-patchcore-development-scaling-v1.md'])
    identity={"name":NAME,"policy":POLICY,"selection":selection,
        "source_zip_sha256":evidence.INPUT_SHA,"pool_zip_sha256":pool_protocol.SOURCES[POOL_ZIP],
        "parent_execution_sha256":fit.digest(old_plan),"archive_sha256":old_plan['archive_sha256'],
        "checkpoint_sha256":data['fit-report.json']['checkpoint_sha256'],
        "original_contract":data['contract.json'],"original_scores_sha256":fit.digest(raw),
        "expanded_development":old_plan['cohorts']['development']+selection['additional_photos'],
        "environment":old_plan['environment'],
        "sources_lf_sha256":{f.relative_to(project).as_posix():hashlib.sha256(f.read_bytes().replace(b'\r\n',b'\n')).hexdigest() for f in files}}
    return data,raw,identity


def validate_expanded(raw, additional, identity):
    if len(raw['development'])!=128 or len(additional)!=128:
        raise ValueError("Expected128 retained and128 added normals")
    for sample,photo in zip(additional,identity['selection']['additional_photos'],strict=True):
        p.comparison.c.scoring.old.validate_sample(sample,photo)
        if sample['label']!=0: raise ValueError("Only normals allowed in development")
    expanded={**raw,'development':raw['development']+additional}
    if len({s['key'] for s in expanded['development']})!=256:
        raise ValueError("Duplicate development sample")
    return expanded


def compare(raw, expanded, old_method, method):
    result=evaluation.compare({'reference':raw,'contract':{'candidate_top1_q90':old_method}},expanded,method)
    result['gates']={'tp55_preserved':result['counts']['candidate']['tp']==55 and not result['lost_baseline_detections'],
                     'fp54_at_most2':result['counts']['candidate']['fp54']<=2,
                     'fp256_at_most12':result['counts']['candidate']['fp256']<=12}
    result['passed_accuracy_targets']=all(result['gates'].values())
    result['comparison']='Same frozen256-photo bank; development128 versus development256'
    result['map_values_changed']=False
    return result


def draws():
    rngs={s:np.random.Generator(np.random.PCG64(seed)) for s,seed in POLICY['expanded_draw_seeds'].items()}
    for scope,i,di,ci in p.evaluation.draws():
        expanded=rngs[scope].integers(0,256,256) if scope in rngs else np.arange(256)
        yield scope,i,di,expanded,ci


def stability(raw, expanded, check=lambda:None):
    arrays={'baseline':p.comparison.matrix(raw),'candidate':p.comparison.matrix(expanded)}
    labels=[s['label'] for s in raw['test']]
    trials={}
    for scope,i,di,edi,ci in draws():
        check()
        row={'scope':scope,'iteration':i,'baseline_development_indices':di.tolist(),
             'candidate_development_indices':edi.tolist(),'calibration_indices':ci.tolist()}
        for name,idx in [('baseline',di),('candidate',edi)]:
            values=arrays[name]
            try:
                method=p.comparison.fit_method(values['development'][idx],values['calibration'][ci])
                row[name]={'valid':True,'counts':p.comparison.counts(values,method,labels),'threshold':method['threshold']}
            except ValueError as exc:
                row[name]={'valid':False,'error':str(exc)}
        row['valid']=all(row[m]['valid'] for m in arrays)
        trials.setdefault(scope,[]).append(row)
    summaries={s:p.evaluation.summarize(rows) for s,rows in trials.items()}
    return {'trials':trials,'summaries':summaries,'passed':all(s['passed'] for s in summaries.values()),
            'pairing':POLICY['draw_pairing'],'interpretation':'Normal sample sensitivity, not population confidence intervals; invalid draws retained'}


def analyze(raw, additional, data, identity, check=lambda:None):
    expanded=validate_expanded(raw,additional,identity)
    values=p.comparison.matrix(expanded)
    method=p.comparison.fit_method(values['development'],values['calibration'])
    result=compare(raw,expanded,data['contract.json'],method)
    stable=stability(raw,expanded,check)
    for scope,rows in stable['trials'].items():
        for row,old in zip(rows,data['evaluation/stability.json']['trials'][scope],strict=True):
            if (row['baseline']['valid']!=old['candidate']['valid'] or
                    row['baseline'].get('counts')!=old['candidate'].get('counts') or
                    abs(row['baseline'].get('threshold',0)-old['candidate'].get('threshold',0))>1e-10):
                raise ValueError("Baseline stability replay differs")
    changes=[]
    for before,after in zip(data['contract.json']['regions'],method['regions'],strict=True):
        changes.append({'region':before['region'],'center_before':before['center'],'center_after':after['center'],
            'scale_before':before['scale'],'scale_after':after['scale'],
            'raw_boundary_before':before['center']+before['scale']*data['contract.json']['threshold'],
            'raw_boundary_after':after['center']+after['scale']*method['threshold']})
    return {'contract':method,'comparison':result,'stability':stable,'region_changes':changes}
