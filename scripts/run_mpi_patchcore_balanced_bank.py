"""Fixed-size region-balanced bank -> normal calibration -> complete observed comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import traceback
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT),str(ROOT/'src')]
from scripts import mpi_patchcore_balanced_bank as b
from scripts import run_mpi_patchcore_scaling as engine
from scripts.mpi_patchcore_balanced_runtime import Runtime
from scripts.export_mpi_patchcore_scaling_review import write_bundle

p,fit,ev = b.p,b.fit,engine.ev


def checked_model(project,source):
    root = project/'artifacts'/p.NAME
    parent = source['parent']
    signature = fit.digest(parent['plan.json'])
    receipt = fit.read_phase(root,'model',signature,['model.pt','context.json'])
    if receipt != parent['model-receipt.json']:
        raise ValueError('Original bank/model changed')
    if json.loads((root/'model/context.json').read_bytes()) != parent['fit-report.json']['context']:
        raise ValueError('Original model context changed')
    return root


def required_parent_maps(project, source):
    """Fail early on missing old maps; full hash/pixel checks occur when compared."""
    parent=source['parent']
    for key,base in [('prediction-receipts.json',project/'artifacts'/p.NAME),
                     ('evaluation/prediction-receipts.json',project/'artifacts/analyses'/p.NAME)]:
        for name in parent[key]:
            relative=Path(name.replace('\\','/'))
            if relative.is_absolute() or '..' in relative.parts:
                raise ValueError('Unsafe parent receipt path')
            directory=base/relative.parent
            for leaf in ('receipt.json','sample.json','maps.npy'):
                if not (directory/leaf).is_file():
                    raise ValueError('Required original comparison cache missing; preserve/recover it: '+str(directory/leaf))
    base=project/'artifacts/analyses'/b.development.NAME/'additional'
    for name in source['development']['prediction-receipts.json']:
        if Path(name).name!=name or not name.startswith('photo-'):
            raise ValueError('Unsafe added-normal receipt path')
        for leaf in ('receipt.json','sample.json','maps.npy'):
            if not (base/name/leaf).is_file():raise ValueError('Required additional-development cache missing: '+str(base/name/leaf))
    repair=project/'artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png'
    if not repair.is_file():raise ValueError('Approved0054 recovery file missing; no automatic replacement')


def checked_cache(project,source,check):
    root = checked_model(project,source)
    parent = source['parent']; signature = fit.digest(parent['plan.json'])
    raw_paths, projected_paths = [], []
    if len(parent['feature-receipts.json'])!=256 or len(parent['projection-receipts.json'])!=49:
        raise ValueError('Missing source cache receipts')
    for i,expected in enumerate(parent['feature-receipts.json']):
        check(); name=f'photo-{i:03d}'
        actual = fit.read_phase(root/'features',name,signature,['features.npy'])
        rows = parent['plan.json']['training']['rows'][i*12:(i+1)*12]
        if (actual!=expected or actual['photo_id']!=parent['plan.json']['training']['photo_ids'][i]
                or actual['rows_sha256']!=fit.digest(rows)):
            raise ValueError('Training feature provenance differs')
        path=root/'features'/name/'features.npy'
        p.array(path,(12288,768)); raw_paths.append(path)
    for i,expected in enumerate(parent['projection-receipts.json']):
        check(); name=expected['phase']
        actual=fit.read_phase(root/'projection',name,signature,list(expected['files']))
        if actual!=expected: raise ValueError('Frozen projected cache differs')
        if i:
            a,z = p.probe.slices(3145728,65536)[i-1]
            if name!=f'chunk-{i-1:03d}' or actual['start']!=a or actual['end']!=z:
                raise ValueError('Projected candidate order differs')
            path=root/'projection'/name/'features.npy'
            p.array(path,(z-a,369)); projected_paths.append(path)
        else:
            if name!='initial': raise ValueError('Projection initialization missing')
            p.array(root/'projection/initial/matrix.npy',(369,768))
    start = parent['projection-receipts.json'][0]['start_index']
    return raw_paths,projected_paths,start


def read_normal(root,signature,identity,checkpoint,prefix='first'):
    return engine.read_normals(root,signature,identity['cohorts'],checkpoint,prefix)


def check_fit(root,signature,identity):
    report=p.value(root,'result',signature)
    receipt=fit.read_phase(root,'model',signature,['model.pt','context.json'])
    if receipt['files']['model.pt']!=report['checkpoint_sha256']:
        raise ValueError('Candidate model changed')
    if json.loads((root/'model/context.json').read_bytes())!=report['context']:
        raise ValueError('Candidate context changed')
    first=read_normal(root,signature,identity,report['checkpoint_sha256'])
    second=read_normal(root,signature,identity,report['checkpoint_sha256'],'reload')
    b.contract_replay(first,report['contract'])
    if p.reload_check(first,second,report['contract'])!=p.value(root,'reload-check',signature):
        raise ValueError('Reload evidence differs')
    if p.value(root,'contract',signature)!=report['contract']:
        raise ValueError('Candidate normal contract changed')
    return report,first


def fit_run(project,root,archive_path,source,identity,runtime_factory=Runtime):
    signature=engine.initialize(root,identity)
    if (root/'result').exists(): return check_fit(root,signature,identity)[0]
    with engine.attempt(root,signature,b.POLICY) as monitor,zipfile.ZipFile(archive_path) as archive:
        print('1/4 Verify frozen training-only caches and original normal replay. No feature retraining.',flush=True)
        parent_root=checked_model(project,source)
        raw_paths,projected,start=checked_cache(project,source,monitor.check)
        p.publish(root,'cache-check',signature,{'raw_photos':256,'projected_chunks':48,
            'raw_receipts_sha256':identity['feature_receipts_sha256'],
            'projection_receipts_sha256':identity['projection_receipts_sha256'],'passed':True})
        runtime=runtime_factory(identity['config']);monitor.runtime=runtime
        runtime.load(parent_root/'model/model.pt',source['parent']['fit-report.json']['context'])
        gate=engine.normal_gate(parent_root,identity['parent_execution_sha256'],archive,source,runtime,
            identity['parent_checkpoint_sha256'],identity['baseline_contract'])
        if not (root/'baseline-gate').exists():p.publish(root,'baseline-gate',signature,gate)
        else:p.value(root,'baseline-gate',signature)
        runtime.release()
        print('2/4 Region-balanced training-only selection: 12 quotas, total78643. Query search remains global.',flush=True)
        selected,receipts=runtime.balanced_select(root/'selection',signature,projected,start,monitor.check)
        p.publish(root,'selected-indices',signature,{'count':len(selected),'indices':selected,'distribution':b.distribution(selected)})
        p.publish(root,'selection-receipts',signature,receipts)
        if not (root/'model').exists():
            runtime.assemble(parent_root/'model/model.pt',raw_paths,selected,monitor.check)
            context={'category':b.NAME,'seed':42,'manifest_sha256':fit.digest(identity['training']),
                     'execution_config_sha256':signature}
            with fit.phase_stage(root,'model') as stage:
                runtime.save(stage/'model.pt',context)
                fit.write_json(stage/'context.json',context)
                fit.seal_phase(stage,'model',signature,{'representatives':78643},['model.pt','context.json'])
        receipt=fit.read_phase(root,'model',signature,['model.pt','context.json'])
        context=json.loads((root/'model/context.json').read_bytes());checkpoint=receipt['files']['model.pt']
        runtime.release()
        monitor.check()
        check=runtime.verify_bank(root/'model/model.pt',parent_root/'model/model.pt',raw_paths,selected)
        p.publish(root,'bank-check',signature,check)
        print('3/4 Score normal development256/calibration169; freeze SAME Q90/Q95 rule, new normal-derived values.',flush=True)
        runtime.load(root/'model/model.pt',context)
        first={g:engine.collect(root,'first-'+g,signature,archive,identity['cohorts'][g],runtime,monitor,checkpoint)
               for g in ('development','calibration')}
        values=p.comparison.matrix(first)
        method=p.comparison.fit_method(values['development'],values['calibration'])
        p.publish(root,'contract',signature,method)
        print('4/4 Reload saved candidate; replay all425 normals. Test remains unopened in fit worker.',flush=True)
        runtime.release();runtime.load(root/'model/model.pt',context)
        second={g:engine.collect(root,'reload-'+g,signature,archive,identity['cohorts'][g],runtime,monitor,checkpoint)
                for g in ('development','calibration')}
        reload=p.reload_check(first,second,method)
        p.publish(root,'reload-check',signature,reload)
        checked_model(project,source)
    report={'status':'balanced_bank_fit_calibrated_reloaded','checkpoint_sha256':checkpoint,'context':context,
        'contract':method,'reload':reload,'training_photos':256,'memory_bank_shape':[78643,768],
        'quotas':b.POLICY['quotas'],'query_search':'global','test_images_decoded':0,
        'automatic_pass_allowed':False,'service_model_changed':False,
        'cumulative_seconds':sum(a['seconds'] for a in p.attempts(root))}
    p.publish(root,'result',signature,report)
    return report


def parent_maps(project,source,group,i):
    parent=source['parent']; dev=source['development']
    if group=='development' and i>=128:
        root=project/'artifacts/analyses'/b.development.NAME/'additional'; index=i-128
        signature=fit.digest(dev['plan.json']);expected=dev['prediction-receipts.json'][f'photo-{index:03d}']
    else:
        phase='first-'+group if group in ('development','calibration') else group
        root=project/('artifacts' if phase.startswith('first-') else 'artifacts/analyses')/p.NAME/phase
        index=i;signature=fit.digest(parent['plan.json'])
        records=parent['prediction-receipts.json' if phase.startswith('first-') else 'evaluation/prediction-receipts.json']
        normalized={k.replace('\\','/'):v for k,v in records.items()}
        expected=normalized[f'{phase}/photo-{i:03d}/receipt.json']
    name=f'photo-{index:03d}'
    actual=fit.read_phase(root,name,signature,['sample.json','maps.npy'])
    if actual!=expected: raise ValueError('Parent map receipt differs')
    sample=engine.read_sample(root,name,signature,source['cohorts'][group][i],source['parent']['fit-report.json']['checkpoint_sha256'])
    if sample!=source['reference'][group][i]: raise ValueError('Parent map/score pairing changed')
    return p.array(root/name/'maps.npy',(12,256,256))


def location(project,fit_root,root,signature,archive,source,candidate,method,checkpoint,check):
    entries=[('test',i,'all_defects') for i,s in enumerate(candidate['test']) if s['label']]
    for group in candidate:
        ordered=sorted((i for i,s in enumerate(candidate[group]) if not s['label']),
            key=lambda i:(-p.comparison.c.scoring.score(candidate[group][i],method)['score'],candidate[group][i]['key']))[:3]
        entries.extend((group,i,'highest_normals') for i in ordered)
    output=[]
    for group,i,role in entries:
        check();photo=source['cohorts'][group][i]
        directory=fit_root/('first-'+group) if group in ('development','calibration') else root/group
        engine.read_sample(directory,f'photo-{i:03d}',signature,photo,checkpoint)
        values=p.array(directory/f'photo-{i:03d}'/'maps.npy',(12,256,256))
        prior=parent_maps(project,source,group,i)
        eligible=bool(photo['label']) and photo['key'] not in b.POLICY['location_excluded']
        masks=ev.read_masks(archive,photo) if eligible else [None]*12
        tiles=[]
        for j,region in enumerate(fit.REGIONS):
            def stats(v):
                return {k:x for k,x in b.evaluation.maps.map_stats(v,masks[j]).items() if not k.startswith('display_')}
            tiles.append({'region':region,'baseline':stats(prior[j]),'candidate':stats(values[j]),
                'baseline_top1':source['reference'][group][i]['regions'][j]['top1'],
                'candidate_top1':candidate[group][i]['regions'][j]['top1']})
        output.append({'group':group,'key':photo['key'],'role':role,'location_eligible':eligible,'tiles':tiles})
    return {'cases':output,'defect_photos':55,'normal_high_photos':12,
        'location_excluded':b.POLICY['location_excluded'],'new_pixel_threshold':False,
        'scope':'Same native float32 inference maps; GT ranking/top656 overlap; no GT enters selection, scoring or calibration'}


def evaluate(project,fit_root,root,archive_path,source,identity,runtime_factory=Runtime):
    signature=engine.initialize(root,identity)
    report,first=check_fit(fit_root,signature,identity)
    checkpoint,method=report['checkpoint_sha256'],report['contract']
    if not (root/'result').exists():
        with engine.attempt(root,signature,b.POLICY) as monitor,zipfile.ZipFile(archive_path) as archive:
            runtime=runtime_factory(identity['config']);monitor.runtime=runtime
            runtime.load(fit_root/'model/model.pt',report['context'])
            gate=engine.normal_gate(fit_root,signature,archive,source,runtime,checkpoint,method)
            if not (root/'initial-normal-gate').exists():p.publish(root,'initial-normal-gate',signature,gate)
            recovery=project/'artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png'
            repaired=[r for photo in identity['cohorts']['test'] for r in photo['regions'] if r['member']==ev.REPAIR_MEMBER]
            if len(repaired)!=1: raise ValueError('Approved repair inventory changed')
            previous={'recovery.json':{'recovered_png_sha256':repaired[0]['member_sha256']}}
            print('Evaluation1/4: all55 defects + normal54/256. Prior12 is descriptive, never an early-stop.',flush=True)
            candidate=dict(first)
            for group in ('test','validation'):
                candidate[group]=engine.collect(root,group,signature,archive,identity['cohorts'][group],runtime,monitor,checkpoint,recovery,previous)
            result=b.compare(source,candidate,method)
            p.publish(root,'comparison',signature,result)
            print('Evaluation2/4: both banks warm20 photos x3 repeats. No elapsed-time limit.',flush=True)
            if not (root/'warm').exists():
                timing={'candidate':b.evaluation.warm(runtime,archive,identity['cohorts']['calibration'],first['calibration'],method,monitor)}
                runtime.release()
                parent_root=checked_model(project,source)
                runtime.load(parent_root/'model/model.pt',source['parent']['fit-report.json']['context'])
                timing['baseline']=b.evaluation.warm(runtime,archive,identity['cohorts']['calibration'],source['reference']['calibration'],identity['baseline_contract'],monitor)
                p.publish(root,'warm',signature,timing)
            else:p.value(root,'warm',signature)
            monitor.check();runtime.release();monitor.runtime=None
            print('Evaluation3/4: saved map/GT comparison, all55 defects and12 high normals.',flush=True)
            if not (root/'location').exists():
                p.publish(root,'location',signature,location(project,fit_root,root,signature,archive,source,candidate,method,checkpoint,monitor.check))
            else:p.value(root,'location',signature)
            print('Evaluation4/4: 600 paired normal resampling draws; no dropping failed draws.',flush=True)
            stable=b.stability(source['reference'],candidate,monitor.check)
            for scope,rows in stable['trials'].items():
                for row,old in zip(rows,source['development']['stability.json']['trials'][scope],strict=True):
                    if row['baseline']!=old['candidate']:raise ValueError('Parent development256 stability replay differs')
            p.publish(root,'stability',signature,stable)
            checked_model(project,source)
            if fit.sha256_file(fit_root/'model/model.pt')!=checkpoint:raise ValueError('Candidate model changed')
        p.publish(root,'result',signature,{'status':'balanced_bank_observed_comparison_completed',
            'counts':result['counts'],'accuracy_targets_passed':result['passed_accuracy_targets'],
            'stability_targets_passed':stable['passed'],'prior12':result['prior12'],
            'all55_defects_evaluated':True,'normal54_and256_evaluated':True,
            'stability_draws':600,'location_evaluated':True,'warm_evaluated':True,
            'independent_performance_claim_allowed':False,'automatic_pass_allowed':False,'service_model_changed':False,
            'checkpoint_sha256':checkpoint,'cumulative_seconds':sum(a['seconds'] for a in p.attempts(root))})
    candidate=dict(first)
    for group in ('test','validation'):
        candidate[group]=[engine.read_sample(root/group,f'photo-{i:03d}',signature,photo,checkpoint) for i,photo in enumerate(identity['cohorts'][group])]
    if b.compare(source,candidate,method)!=p.value(root,'comparison',signature):raise ValueError('Saved decisions differ')
    for phase in ('warm','location','stability'):p.value(root,phase,signature)
    return p.value(root,'result',signature)


def export(fit_root,eval_root,output,identity):
    signature=fit.digest(identity)
    report,first=check_fit(fit_root,signature,identity)
    payload={'plan.json':identity,'fit-report.json':report,'normal-scores.json':first,
        'reload-scores.json':read_normal(fit_root,signature,identity,report['checkpoint_sha256'],'reload'),
        'fit-attempts.json':p.attempts(fit_root),
        'model-receipt.json':fit.read_phase(fit_root,'model',signature,['model.pt','context.json'])}
    for phase in ('cache-check','baseline-gate','selected-indices','selection-receipts','bank-check','contract','reload-check'):
        payload[phase+'.json']=p.value(fit_root,phase,signature)
    if eval_root is not None:
        for phase in ('initial-normal-gate','comparison','warm','location','stability','result'):
            payload['evaluation/'+phase+'.json']=p.value(eval_root,phase,signature)
        payload['evaluation/attempts.json']=p.attempts(eval_root)
        payload['evaluation/scores.json']={g:[engine.read_sample(eval_root/g,f'photo-{i:03d}',signature,photo,report['checkpoint_sha256'])
            for i,photo in enumerate(identity['cohorts'][g])] for g in ('test','validation')}
    for prefix,root in [('fit',fit_root),('evaluation',eval_root)]:
        if root is not None:
            payload[prefix+'/prediction-receipts.json']={x.relative_to(root).as_posix():json.loads(x.read_bytes())
                for x in sorted(root.glob('*/photo-*/receipt.json'))}
    write_bundle(output,{k:fit.canonical(v) for k,v in payload.items()})


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    parser.add_argument('--worker',choices=('fit','evaluation'),help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    project=args.project.resolve();archive=args.archive.resolve()
    source,identity=b.prepare(project)
    if args.plan_only:print(json.dumps(identity,indent=2));return 0
    if engine.environment()!=identity['environment']:raise ValueError('Frozen GPU/software environment differs')
    if fit.sha256_file(archive)!=identity['archive_sha256']:raise ValueError('Original archive changed')
    required_parent_maps(project,source)
    if shutil.disk_usage(project).free<10*1024**3:raise RuntimeError('Need10 GiB free disk; no deletion')
    root=project/'artifacts'/b.NAME;eval_root=project/'artifacts/analyses'/b.NAME
    parent=project/'artifacts'/p.NAME
    if args.worker:
        with fit.locked_run(parent),fit.locked_run(root):
            if args.worker=='fit':
                fit_run(project,root,archive,source,identity)
                export(root,None,project/'artifacts/handoff'/(b.NAME+'-fit.zip'),identity)
            else:
                with fit.locked_run(eval_root):
                    evaluate(project,root,eval_root,archive,source,identity)
                    export(root,eval_root,project/'artifacts/handoff'/(b.NAME+'.zip'),identity)
        return 0
    with fit.locked_run(project/'artifacts'/(b.NAME+'-launcher')):
        for worker in ('fit','evaluation'):
            result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--project',str(project),
                '--archive',str(archive),'--worker',worker],check=False)
            if result.returncode:raise RuntimeError(worker+' stopped; preserve files and send error')
    print(json.dumps({'status':'completed_exploratory','handoff':str(project/'artifacts/handoff'/(b.NAME+'.zip'))},indent=2))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Wait for cleanup; same command resumes sealed checkpoints. Preserve outputs.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        traceback.print_exc()
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no reroll, shrinking or threshold override.',file=sys.stderr)
        raise SystemExit(1)
