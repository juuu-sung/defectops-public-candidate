"""One area-matched spatial aggregation ablation; not a new independent test."""
import math

import numpy as np

from scripts import mpi_score_validation_protocol as p

FIELD = "window26"


def spatial_score(values):
    values = np.asarray(values)
    if values.shape != (256, 256) or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("Expected finite nonnegative 256x256 map")
    # 26**2 = 676, close to the existing 656-pixel pool. Only geometry changes.
    integral = np.pad(values.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    means = (integral[26:, 26:] - integral[:-26, 26:] - integral[26:, :-26] + integral[:-26, :-26]) / 676
    y, x = np.unravel_index(int(np.argmax(means)), means.shape)
    return {FIELD: float(means[y, x]), "window_xyxy": [int(x), int(y), int(x)+26, int(y)+26]}


def validate(sample, photo):
    p.old.validate_sample(sample, photo)
    for row in sample['regions']:
        v, box = row[FIELD], row['window_xyxy']
        if not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0:
            raise ValueError("Invalid spatial score")
        if (len(box) != 4 or any(type(n) is not int for n in box)
                or not 0 <= box[0] <= 230 or not 0 <= box[1] <= 230
                or box[2] != box[0]+26 or box[3] != box[1]+26):
            raise ValueError("Invalid spatial window")


def fit_contract(development, calibration):
    # Reuse exactly the same Q90 recipe, changing only its input field.
    for group, samples in [('development', development), ('calibration', calibration)]:
        p.validate_samples(samples, group)
        for sample in samples:
            validate(sample, sample)
    # Do not compute unused MAD candidates: Q90 alone must be nondegenerate.
    if {s['photo_id'] for s in development} & {s['photo_id'] for s in calibration}:
        raise ValueError("Development/calibration overlap")
    x = np.array([[row[FIELD] for row in s['regions']] for s in development], dtype=np.float64)
    center = np.median(x, axis=0)
    scale = np.quantile(x, .9, axis=0, method='higher') - center
    if not np.isfinite(scale).all() or np.any(scale <= 1e-6):
        raise ValueError("Degenerate spatial Q90 scale; no fallback")
    method = {'field': FIELD, 'scale_kind': 'q90', 'threshold': 0., 'regions':[
        {'region':r,'center':float(c),'scale':float(w)} for r,c,w in zip(p.fit.REGIONS,center,scale,strict=True)]}
    method['threshold'] = float(np.quantile([p.score(s, method)['score'] for s in calibration], .95, method='higher'))
    return method


def compare(samples, control, candidate, max_cost_fraction=.1):
    methods = {'control_q90': control, 'window26_q90': candidate}
    test = p.observed_metrics(samples['test'], methods)
    normal = {}
    for name, method in methods.items():
        alerts = [s['key'] for s in samples['validation'] if p.score(s,method)['alert']]
        normal[name] = {'photos':len(samples['validation']),'false_positives':len(alerts),
                        'rate':len(alerts)/len(samples['validation']),'alert_keys':alerts,
                        'within_descriptive_5_percent_budget':len(alerts)/len(samples['validation']) <= .05}
    added, lost = [], []
    for s in samples['test']:
        if s['label']:
            a,b = p.score(s,control)['alert'], p.score(s,candidate)['alert']
            if b and not a:
                added.append(s['key'])
            if a and not b:
                lost.append(s['key'])
    photos = [s for group in samples.values() for s in group]
    total = sum(s['timing']['photo_scoring_ms'] for s in photos)
    extra = sum(sum(s['timing']['spatial_ms']) for s in photos)
    fraction = extra/total if total > 0 else None
    cost_ok = fraction is not None and fraction <= max_cost_fraction
    passed = (bool(added) and not lost and test['window26_q90']['counts']['fp'] <= test['control_q90']['counts']['fp']
              and normal['window26_q90']['false_positives'] <= normal['control_q90']['false_positives'] and cost_ok)
    return {'observed_test':test,'observed_normals':normal,'added_true_positives':added,'lost_true_positives':lost,
            'screening_passed':passed,'automatic_promotion':False,
            'timing':{'photos':len(photos),'scope':'checked image decode, shared inference, top1 and spatial scoring; excludes loading, probe and persistence',
                      'photo_mean_ms':total/len(photos),'photo_p95_ms':float(np.quantile([s['timing']['photo_scoring_ms'] for s in photos],.95,method='higher')),
                      'spatial_total_ms':extra,'spatial_fraction_of_measured_total':fraction,'within_cost_limit':cost_ok,
                      'spatial_tile_p95_ms':float(np.quantile([v for s in photos for v in s['timing']['spatial_ms']],.95,method='higher'))}}
