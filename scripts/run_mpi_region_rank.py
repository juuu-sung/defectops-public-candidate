"""One normal-only, equal-risk region-rank candidate; no use of oracle bounds."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts import run_mpi_qn_scale as q


def fit_rank(dev,cal):
    dev,cal=np.asarray(dev,dtype=float),np.asarray(cal,dtype=float)
    if dev.shape!=(128,12) or cal.shape!=(169,12) or not np.isfinite(dev).all() or not np.isfinite(cal).all():
        raise ValueError('Requires finite frozen 128/169 normal rows')
    return {'strict_upper_bounds':np.concatenate([dev,cal]).max(axis=0).tolist(),
            'normal_count':297,'tail_denominator':298,'per_region_alpha':'1/240',
            'decision':'any raw > region maximum; ties do not alert'}


def predictions(raw,arrays,method):
    bounds=np.array(method['strict_upper_bounds'])
    return {g:[{'key':row['key'],'label':row['label'],'alert':bool(np.any(x>bounds)),
                 'exceeding_regions':[q.common.REGIONS[i] for i in np.flatnonzero(x>bounds)],
                 'raw_margin_max':float(np.max(x-bounds))} for row,x in zip(raw[g],a,strict=True)]
            for g,a in arrays.items()}


def counts(arrays,labels,method):
    bounds=np.array(method['strict_upper_bounds']);p=(arrays['test']>bounds).any(axis=1)
    return {'tp':int(sum(p&(labels==1))),'fn':int(sum(~p&(labels==1))),
            'fp':int(sum(p&(labels==0))),'tn':int(sum(~p&(labels==0))),
            'fp_validation':int(sum((arrays['validation']>bounds).any(axis=1)))}


def run(source,output):
    cfg=q.load_config();start=time.monotonic()
    raw,a,stored,ref=q.load_source(source,cfg)
    control,replay=q.common.replay(a,raw,stored,ref,cfg)
    identity_record={'name':'mpi-region-rank-v1','source_zip_sha256':cfg['source_zip_sha256'],
        'source_sha256':q.common.sha(Path(__file__).read_bytes()),
        'protocol_sha256':q.common.sha((ROOT/'docs/mpi-region-rank-v1.md').read_bytes()),
        'helpers':{n:q.common.sha((ROOT/n).read_bytes()) for n in ['scripts/run_mpi_qn_scale.py','scripts/run_mpi_scale_shrinkage.py']},
        'config':cfg,'roles':'development128+calibration169 pooled for candidate only',
        'early_stop':'fixed performance gate failure skips bootstrap; no stability claim'}
    identity=q.common.sha(q.common.canonical(identity_record))
    with q.common.locked(output):
        q.common.publish(output/'identity.json',identity,identity_record)
        if (output/'report.json').exists():return q.common.read_sealed(output/'report.json',identity)
        q.common.publish(output/'started.json',identity,{'unix_seconds':time.time()})
        candidate=fit_rank(a['development'],a['calibration'])
        q.common.publish(output/'parameters.json',identity,{'control':control,'candidate':candidate})
        pred={'control':q.common.predictions(raw,a,control),'candidate':predictions(raw,a,candidate)}
        q.common.publish(output/'predictions.json',identity,pred)
        labels=np.array([r['label'] for r in raw['test']])
        metrics={'control':q.common.counts(a,labels,control),'candidate':counts(a,labels,candidate)}
        lost=[x['key'] for x,y in zip(pred['control']['test'],pred['candidate']['test']) if x['label'] and x['alert'] and not y['alert']]
        gained=[x['key'] for x,y in zip(pred['control']['test'],pred['candidate']['test']) if x['label'] and not x['alert'] and y['alert']]
        gate={'tp':metrics['candidate']['tp']>=53,'no_lost_tp':not lost,
              'fp54':metrics['candidate']['fp']<=2,'fp256':metrics['candidate']['fp_validation']<=12}
        stable={}
        if all(gate.values()):
            for scope in cfg['bootstrap']['scopes']:
                rows=[]
                for i,di,ci in q.common.trial_indices(scope,cfg['bootstrap'],128,169):
                    q.budget(start,0,cfg['cpu_budget'])
                    d,c=a['development'][di],a['calibration'][ci]
                    row={'development_indices':di.tolist(),'calibration_indices':ci.tolist(),'methods':{}}
                    for name,fit,measure in [('control',lambda d,c:q.common.fit_method(d,c,0),q.common.counts),('candidate',fit_rank,counts)]:
                        try:
                            m=fit(d,c);row['methods'][name]={'valid':True,'parameters':m,'counts':measure(a,labels,m)}
                        except ValueError as exc:row['methods'][name]={'valid':False,'error':str(exc)}
                    q.common.publish(output/'trials'/scope/f'{i:04d}.json',identity,row);rows.append(row)
                stable[scope]=q.common.summarize_trials(rows,cfg['development_screening'])
        elapsed,rss=q.budget(start,0,cfg['cpu_budget'])
        report={'name':'mpi-region-rank-v1','replay':replay,'counts':metrics,'conditions':gate,
                'lost_baseline_detections':lost,'gained_detections':gained,'candidate_parameters':candidate,
                'point_passed':all(gate.values()),'stability':stable or None,
                'stability_status':'completed' if stable else 'not_run_fixed_gate_failed',
                'decision':'research_candidate_only' if all(gate.values()) and all(v['passed'] for v in stable.values()) else 'not_adopted',
                'cost':{'elapsed_seconds':elapsed,'peak_rss_bytes':rss},
                'training_performed':False,'automatic_pass_allowed':False,'independent_performance_claim_allowed':False}
        q.common.publish(output/'report.json',identity,report)
        return report


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,default=ROOT/'artifacts/analyses/mpi-region-rank-v1')
    args=p.parse_args();print(q.common.canonical(run(args.source,args.output)).decode())
