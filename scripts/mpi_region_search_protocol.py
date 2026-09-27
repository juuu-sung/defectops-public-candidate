"""Frozen, post-test region-routing ablation; never select a service model."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from scripts import mpi_patch_trace_protocol as trace

fit, ev, scoring = trace.fit, trace.ev, trace.score
NAME = "mpi-pcb-region-search-v1"
TRACE_SHA256 = "7fe091c22ec0c4f1891fd8e940ec387e6a6d8d7f1aa42560869fe1afc3e9fd1e"
ARMS = ("shared_A", "same_region_A")
POLICY = {
    "schema_version": 1,
    "source_trace_zip_sha256": TRACE_SHA256,
    "source_score_zip_sha256": trace.SCORE_ZIP_SHA256,
    "bank_shape": [78643, 768],
    "routing": "allow every bank index with at least one exact original stride2 source in the input region",
    "multi_region_aliases": "allow in ALL exact source regions; keep all aliases; no arbitrary first-source routing",
    "unresolved_sources": "stop before any candidate scoring; no approximation or global fallback",
    "minimum_bank_vectors_per_region": 9,
    "maximum_origin_records": 786432,
    "spatial_coordinate_constraint": False,
    "new_coreset_selection": False,
    "normal_development_photos": 128,
    "normal_calibration_photos": 169,
    "normalization": scoring.POLICY["normalization"],
    "minimum_mad_exclusive": scoring.POLICY["minimum_mad_exclusive"],
    "photo_score": "max_of_all_12_regions",
    "calibration_quantile": .95,
    "quantile_method": "higher",
    "keep_calibration_0411": True,
    "scoring": "unchanged smoothed-map top1% mean then normal-only region median/MAD",
    "pair_order": "alternate by tile ordinal; same image decoded once; six first tiles per arm per photo",
    "replay_tolerance": 1e-4,
    "map_monotonicity_tolerance": .002,
    "test_previously_observed": True,
    "excluded_test_originals": ["good/0013"],
    "location_excluded_originals": ["defect/0006"],
    "sensitivity_excluded_originals": ["defect/0054"],
    "training_performed": False,
    "existing_threshold_changed": False,
    "candidate_threshold_recalculated_normals_only": True,
    "service_model_changed": False,
    "automatic_pass_allowed": False,
    "confirmatory_claim_allowed": False,
}


def make_routing(bank_hashes, chunks, preparation, minimum=9):
    """Route by every exact alias, preserving original bank indices and order."""
    if not bank_hashes or any(len(h) != 64 or any(c not in '0123456789abcdef' for c in h) for h in bank_hashes):
        raise ValueError("Invalid bank hash inventory")
    train = {r["member"]: r for r in preparation["training_rows"]}
    wanted, aliases, seen = set(bank_hashes), defaultdict(list), set()
    if [c["photo_id"] for c in chunks] != preparation["training_photo_ids"]:
        raise ValueError("Source scan training photo inventory differs")
    for chunk in chunks:
        if chunk["candidate_embeddings_checked"] != 12*1024:
            raise ValueError("Incomplete original candidate scan")
        for o in chunk["origins"]:
            key = (o["member"], o["candidate_flat_index"])
            row = train.get(o["member"])
            if (row is None or o["photo_id"] != chunk["photo_id"] or key in seen
                    or o["bank_vector_sha256"] not in wanted
                    or any(o[k] != row[k] for k in ("photo_id", "region", "member_sha256", "decoded_rgb_sha256"))
                    or not (0 <= o["grid_y"] < 64 and 0 <= o["grid_x"] < 64)
                    or o["grid_y"] % 2 or o["grid_x"] % 2
                    or o["candidate_flat_index"] != o["grid_y"]*64+o["grid_x"]):
                raise ValueError("Invalid/duplicate/non-training exact origin")
            seen.add(key)
            aliases[o["bank_vector_sha256"]].append(o)
            if len(seen) > POLICY["maximum_origin_records"]:
                raise ValueError("Too many source aliases; no truncation")
    if wanted - aliases.keys():
        raise ValueError(f"Unresolved bank vectors: {sum(h not in aliases for h in bank_hashes)}; no global fallback")
    ordered = {h: sorted(aliases[h], key=lambda o: (o['photo_id'], o['region'], o['candidate_flat_index'])) for h in sorted(wanted)}
    sources = [{o['region'] for o in ordered[h]} for h in bank_hashes]
    indices = {r: [i for i,s in enumerate(sources) if r in s] for r in fit.REGIONS}
    if any(len(ids) < minimum for ids in indices.values()):
        raise ValueError("Insufficient same-region bank; no fallback or refit")
    return {"bank_hashes": bank_hashes, "region_indices": indices,
            "per_region_bank_vectors": {r:len(v) for r,v in indices.items()},
            "bank_vectors":len(bank_hashes), "unique_feature_hashes":len(wanted),
            "multi_origin_vectors":sum(len(ordered[h]) > 1 for h in bank_hashes),
            "multi_region_vectors":sum(len(s) > 1 for s in sources),
            "origin_records":len(seen), "origins_sha256":fit.digest(ordered)}, ordered


def validate_top_sample(sample, photo):
    if (any(sample[k] != photo[k] for k in ("key", "photo_id", "label"))
            or [(r['region'],r['member']) for r in sample['regions']] != [(r['region'],r['member']) for r in photo['regions']]
            or len(sample['regions']) != 12):
        raise ValueError("Sample photo/region identity differs")
    if any(not isinstance(r['top1'], (int,float)) or not math.isfinite(r['top1']) or r['top1'] < 0 for r in sample['regions']):
        raise ValueError("Invalid top1 score")


def a_score(sample, normalization):
    return scoring.score(sample, trace.A, normalization)


def check_shared(sample, expected, frozen):
    validate_top_sample(sample, expected)
    diff=max(abs(a['top1']-b['top1']) for a,b in zip(sample['regions'],expected['regions'],strict=True))
    a,b=a_score(sample,frozen['normalization']),a_score(expected,frozen['normalization'])
    if max(diff,abs(a-b)) > POLICY['replay_tolerance']:
        raise ValueError("Published shared-bank score drift")
    if (a >= frozen['thresholds'][trace.A]) != (b >= frozen['thresholds'][trace.A]):
        raise ValueError("Published shared-bank decision flip")
    return max(diff,abs(a-b))


def calibrate(development, calibration):
    """Separate candidate contract; no published-threshold assertion or test access."""
    dev_ids={s['photo_id'] for s in development}
    cal_ids={s['photo_id'] for s in calibration}
    if (not calibration or len(dev_ids) != len(development) or len(cal_ids) != len(calibration)
            or dev_ids & cal_ids or any(s['label'] != 0 or not s['key'].startswith('calibration/') for s in calibration)):
        raise ValueError("Candidate calibration must use separate unique normals")
    for s in development+calibration:
        validate_top_sample(s,s)
    normalization=scoring.fit_normalization(development)
    threshold=float(np.quantile([a_score(s,normalization) for s in calibration],.95,method='higher'))
    return {'normalization':normalization,'threshold':threshold,'test_scores_used':0,
            'development_scores_sha256':fit.digest(development),'calibration_scores_sha256':fit.digest(calibration),
            'automatic_pass_allowed':False,'rule':'normal 95% quantile, higher; no test optimization'}


def classify(samples, normalization, threshold):
    return [{k:s[k] for k in ('key','photo_id','label')} | {
        'score':a_score(s,normalization),'threshold':threshold,'automatic_pass_allowed':False,
        'decision':'REVIEW_REQUIRED' if a_score(s,normalization) >= threshold else 'BELOW_THRESHOLD_NOT_AUTO_PASS'} for s in samples]


def compare(samples, contract, published):
    new=classify(samples,contract['normalization'],contract['threshold'])
    pairs=ev.paired_changes(published,new)
    arms={ARMS[0]:published,ARMS[1]:new}
    return {'models':{k:ev.model_metrics(v,v[0]['threshold']) for k,v in arms.items()},
            'paired_changes_from_shared_A':pairs,
            'sensitivity_without_reconstructed_photo':{k:ev.model_metrics([s for s in v if s['key']!='defect/0054'],v[0]['threshold']) for k,v in arms.items()},
            'original_photos':len(new),'normal_photos':sum(s['label']==0 for s in new),
            'defect_photos':sum(s['label']==1 for s in new)},arms
