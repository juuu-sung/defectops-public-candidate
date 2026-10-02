"""Label-aware OR-threshold feasibility diagnostic. NEVER a deployable model."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_mpi_qn_scale as q


def mask(values):
    return sum(1 << i for i in np.flatnonzero(values).tolist())


def feasible(targets, normals_a, normals_b, cap_a=2, cap_b=12, *, seconds=30., max_nodes=100000,
             max_rss=2*1024**3):
    """Complete finite existence search unless resource-limited; not optimization."""
    start = time.monotonic()
    targets, normals_a, normals_b = [np.asarray(v, dtype=float) for v in (targets, normals_a, normals_b)]
    if (targets.ndim != 2 or not targets.shape[1] or cap_a < 0 or cap_b < 0 or
        any(v.ndim != 2 or v.shape[1] != targets.shape[1] or not np.isfinite(v).all()
            for v in (targets, normals_a, normals_b))):
        raise ValueError("Invalid finite matrices or budgets")
    options, seen, nodes = [], set(), 0
    complete = (1 << len(targets))-1

    class Limit(Exception):
        pass

    def check():
        if (time.monotonic()-start > seconds or nodes >= max_nodes or q.common.peak_rss_bytes() > max_rss):
            raise Limit("Search resource limit; feasibility unknown")

    def search(covered, na, nb, selected):
        nonlocal nodes
        check()
        if covered == complete:
            return selected
        state = (covered, na, nb)
        if state in seen:
            return None
        seen.add(state); nodes += 1
        valid = [(idx,o) for idx,o in enumerate(options)
                 if (na|o[3]).bit_count() <= cap_a and (nb|o[4]).bit_count() <= cap_b]
        smallest = None
        for i in range(len(targets)):
            if covered & (1 << i):
                continue
            choices = [(idx,o) for idx,o in valid if o[2] & (1 << i)]
            if not choices:
                return None
            if smallest is None or len(choices) < len(smallest):
                smallest = choices
        smallest.sort(key=lambda v: (((v[1][3]&~na).bit_count()+(v[1][4]&~nb).bit_count()),
                                    -(v[1][2]&~covered).bit_count(), v[1][0], v[1][1]))
        for idx,o in smallest:
            result = search(covered|o[2],na|o[3],nb|o[4],selected+[idx])
            if result is not None:
                return result
        return None

    result, status, error = None, None, None
    try:
        for region in range(targets.shape[1]):
            for threshold in np.unique(targets[:,region]):
                check()
                a,b = mask(normals_a[:,region]>=threshold),mask(normals_b[:,region]>=threshold)
                if a.bit_count()<=cap_a and b.bit_count()<=cap_b:
                    options.append((region,float(threshold),mask(targets[:,region]>=threshold),a,b))
        result=search(0,0,0,[])
        status="feasible_label_aware_witness" if result is not None else "infeasible_exhaustive"
    except Limit as exc:
        status,error="unknown_resource_limit",str(exc)
    witness=None
    if result is not None:
        bounds=[None]*targets.shape[1]
        for index in result:
            r,t,*_=options[index]
            bounds[r]=t if bounds[r] is None else min(bounds[r],t)
        threshold=np.array([np.inf if v is None else v for v in bounds])
        hit=lambda x:np.any(x>=threshold,axis=1)
        if not hit(targets).all() or sum(hit(normals_a))>cap_a or sum(hit(normals_b))>cap_b:
            raise ValueError("Independent witness verification failed")
        witness={"DO_NOT_DEPLOY_label_fitted_boundaries":bounds,"target_hits":int(sum(hit(targets))),
                 "fp_a":int(sum(hit(normals_a))),"fp_b":int(sum(hit(normals_b)))}
    return {"status":status,"error":error,"witness_not_model":witness,"visited_states":nodes,
            "candidate_boundaries_after_individual_budget_pruning":len(options),
            "elapsed_seconds":time.monotonic()-start,"peak_rss_bytes":q.common.peak_rss_bytes(),
            "labels_used_for_fitting":True,"performance_evaluation":False,"deployable":False}


def run(source, output):
    cfg=q.load_config()
    raw,a,stored,reference=q.load_source(source,cfg)
    control,replay=q.common.replay(a,raw,stored,reference,cfg)
    pred=q.common.predictions(raw,a,control)
    keep=np.array([r['label']==1 and r['alert'] for r in pred['test']])
    normal=np.array([r['label']==0 for r in raw['test']])
    if int(sum(keep))!=53 or int(sum(normal))!=54:
        raise ValueError('Frozen baseline inventory differs')
    record={"name":"mpi-region-threshold-feasibility-v1","source_zip_sha256":cfg['source_zip_sha256'],
            "source_sha256":q.common.sha(Path(__file__).read_bytes()),
            "protocol_sha256":q.common.sha((ROOT/'docs/mpi-region-threshold-feasibility-v1.md').read_bytes()),
            "reference_code_sha256":{name:q.common.sha((ROOT/name).read_bytes()) for name in
                ['scripts/run_mpi_qn_scale.py','scripts/run_mpi_scale_shrinkage.py']},
            "budgets":{"seconds":30,"states":100000,"rss_bytes":2*1024**3,"fp54":2,"fp256":12},
            "targets":[r['key'] for r,k in zip(raw['test'],keep) if k],"replay":replay}
    identity=q.common.sha(q.common.canonical(record))
    with q.common.locked(output):
        q.common.publish(output/'identity.json',identity,record)
        if (output/'report.json').exists():
            return q.common.read_sealed(output/'report.json',identity)
        # An interrupted diagnostic is not automatically retried with a fresh clock.
        q.common.publish(output/'started.json',identity,{'unix_seconds':time.time()})
        result=feasible(a['test'][keep],a['test'][normal],a['validation'])
        q.common.publish(output/'report.json',identity,result)
        return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,default=ROOT/'artifacts/analyses/mpi-region-threshold-feasibility-v1')
    args=parser.parse_args()
    print(q.common.canonical(run(args.source,args.output)).decode())
