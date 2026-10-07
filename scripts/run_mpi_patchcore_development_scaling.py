"""Inference on added normals only; preserve model, original contracts and all data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import sys
import traceback
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import mpi_patchcore_development_scaling as d
from scripts import run_mpi_patchcore_scaling as engine
from scripts.export_mpi_patchcore_scaling_review import write_bundle

p,fit=d.p,d.fit


def verify_model(project,data):
    root=project/'artifacts'/p.NAME
    receipt=fit.read_phase(root,'model',fit.digest(data['plan.json']),['model.pt','context.json'])
    if receipt!=data['model-receipt.json'] or json.loads((root/'model/context.json').read_bytes())!=data['fit-report.json']['context']:
        raise ValueError('Frozen model/context differs')
    return root


def reload_added(root,signature,archive,photos,runtime,checkpoint,method,monitor):
    maximum=0.; first=[]; second=[]
    for i,photo in enumerate(photos[:d.POLICY['reload_photos']]):
        monitor.check()
        expected=engine.read_sample(root/'additional',f'photo-{i:03d}',signature,photo,checkpoint)
        actual,values=engine.score_photo(archive,photo,runtime)
        saved=p.array(root/'additional'/f'photo-{i:03d}'/'maps.npy',(12,256,256))
        maximum=max(maximum,float(np.max(np.abs(values-saved))))
        first.append(expected);second.append(actual)
    result=p.reload_check({'development':first,'calibration':[]},{'development':second,'calibration':[]},method)
    if maximum>d.POLICY['reload_absolute_tolerance']: raise ValueError('Added-normal reload maps differ')
    return {**result,'map_maximum_absolute_difference':maximum,'fresh_model_load':True}


def export(root,output,identity):
    signature=fit.digest(identity)
    names=('plan','preflight','initial-normal-gate','additional-scores','contract','reload-check',
           'comparison','stability','region-changes','result')
    payload={n+'.json':fit.canonical(p.value(root,n,signature)) for n in names}
    payload['attempts.json']=fit.canonical(p.attempts(root))
    payload['prediction-receipts.json']=fit.canonical({x.parent.name:json.loads(x.read_bytes()) for x in sorted((root/'additional').glob('photo-*/receipt.json'))})
    write_bundle(output,payload)


def run(project,archive_path,data,raw,identity,runtime_factory=engine.Runtime):
    root=project/'artifacts/analyses'/d.NAME
    output=project/'artifacts/handoff'/(d.NAME+'.zip')
    if fit.sha256_file(archive_path)!=identity['archive_sha256']: raise ValueError('Original archive changed')
    fit_root=project/'artifacts'/p.NAME
    with fit.locked_run(fit_root),fit.locked_run(root):
        verify_model(project,data)
        signature=engine.initialize(root,identity)
        p.publish(root,'plan',signature,identity)
        photos=identity['selection']['additional_photos']
        checkpoint=identity['checkpoint_sha256']
        if not (root/'result').exists():
            with engine.attempt(root,signature,d.POLICY) as monitor,zipfile.ZipFile(archive_path) as archive:
                print('1/4 Verify128 selected normal photos and replay8 frozen calibration photos. No training.',flush=True)
                if not (root/'preflight').exists():
                    for photo in photos:
                        for row in photo['regions']:
                            monitor.check()
                            with fit.checked_image(archive,row): pass
                    p.publish(root,'preflight',signature,{'normal_photos':128,'normal_tiles':1536,
                        'additional_rows_sha256':identity['selection']['additional_rows_sha256'],'passed':True,
                        'test_images_decoded':0,'test_masks_decoded':0})
                else:p.value(root,'preflight',signature)
                runtime=runtime_factory(data['plan.json']['config']);monitor.runtime=runtime
                runtime.load(fit_root/'model/model.pt',data['fit-report.json']['context'])
                gate=engine.normal_gate(fit_root,identity['parent_execution_sha256'],archive,
                    {'cohorts':data['plan.json']['cohorts']},runtime,checkpoint,data['contract.json'])
                if not (root/'initial-normal-gate').exists():p.publish(root,'initial-normal-gate',signature,gate)
                else:p.value(root,'initial-normal-gate',signature)
                print('2/4 Score only128 added normals; reuse all prior development/calibration/test/validation scores.',flush=True)
                added=engine.collect(root,'additional',signature,archive,photos,runtime,monitor,checkpoint)
                p.publish(root,'additional-scores',signature,added)
                expanded=d.validate_expanded(raw,added,identity)
                values=p.comparison.matrix(expanded)
                method=p.comparison.fit_method(values['development'],values['calibration'])
                p.publish(root,'contract',signature,method)
                print('3/4 Reload same model; verify8 added normals with frozen new contract.',flush=True)
                runtime.release(); runtime.load(fit_root/'model/model.pt',data['fit-report.json']['context'])
                check=reload_added(root,signature,archive,photos,runtime,checkpoint,method,monitor)
                if not (root/'reload-check').exists():p.publish(root,'reload-check',signature,check)
                else:p.value(root,'reload-check',signature)
                runtime.release();monitor.runtime=None
                print('4/4 CPU comparison of all55 defects,54/256 normals and600 sensitivity draws. No test reads.',flush=True)
                result=d.analyze(raw,added,data,identity,monitor.check)
                for key in ('comparison','stability','region_changes'):
                    p.publish(root,key.replace('_','-'),signature,result[key])
                verify_model(project,data)
            summary={'status':'development_scaling_observed_comparison_completed','checkpoint_sha256':checkpoint,
                'counts':result['comparison']['counts'],'accuracy_targets_passed':result['comparison']['passed_accuracy_targets'],
                'stability_targets_passed':result['stability']['passed'],'model_changed':False,'training_performed':False,
                'original_contract_changed':False,'candidate_contract_created':True,'new_test_images_decoded':0,
                'new_test_masks_decoded':0,'automatic_pass_allowed':False,'service_model_changed':False,
                'new_inference_photos':128,'inference_replay_photos_per_full_attempt':16,
                'location_maps':'Unchanged; parent evaluation/location.json remains evidence, not a new localization measurement',
                'warm_timing':'Inherited parent benchmark only; this run does not re-benchmark inference',
                'cumulative_seconds':sum(x['seconds'] for x in p.attempts(root))}
            p.publish(root,'result',signature,summary)
        # Revalidate cached maps and recompute CPU results on completed reruns too.
        added=[engine.read_sample(root/'additional',f'photo-{i:03d}',signature,photo,checkpoint) for i,photo in enumerate(photos)]
        if added!=p.value(root,'additional-scores',signature):raise ValueError('Cached score export differs')
        results=d.analyze(raw,added,data,identity)
        for k in ('contract','comparison','stability','region_changes'):p.publish(root,k.replace('_','-'),signature,results[k])
        export(root,output,identity)
        return p.value(root,'result',signature),output


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT)
    parser.add_argument('--archive',type=Path,required=True)
    parser.add_argument('--plan-only',action='store_true')
    args=parser.parse_args(argv)
    project=args.project.resolve()
    data,raw,identity=d.prepare(project,project/'artifacts/handoff')
    if args.plan_only:
        print(json.dumps(identity,indent=2));return 0
    if engine.environment()!=identity['environment']:raise ValueError('Frozen lab environment differs')
    if shutil.disk_usage(project).free<d.POLICY['free_disk_GiB']*1024**3:raise RuntimeError('Need5 GiB free disk; no deletion')
    summary,output=run(project,args.archive.resolve(),data,raw,identity)
    print(json.dumps({**summary,'handoff':str(output)},indent=2));return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except KeyboardInterrupt:
        print('Interrupted. Wait for cleanup; same command resumes completed photos. Preserve files.',file=sys.stderr)
        raise SystemExit(130)
    except Exception as exc:
        traceback.print_exc()
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve files; no retraining, reroll or threshold override.',file=sys.stderr)
        raise SystemExit(1)
