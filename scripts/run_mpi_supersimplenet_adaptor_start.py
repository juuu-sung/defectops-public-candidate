"""Two matched adaptor-start arms; frozen inputs, no elapsed-time cap or test access."""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import time
import traceback
import zipfile

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
from scripts import mpi_supersimplenet_adaptor_start_protocol as p
from scripts.mpi_supersimplenet_adaptor_start_runtime import Runtime,environment
from scripts.run_mpi_supersimplenet_update_audit import Monitor
from scripts.run_mpi_supersimplenet_parity import check_inputs
from scripts.run_mpi_supersimplenet_varying import evaluate,check_targets,reload_verify
from scripts.run_mpi_efficientad_probe import publish,attempts

fit=p.fit


def export(root,handoff,identity,report):
    raw={'plan.json':fit.canonical(identity),'report.json':fit.canonical(report),'attempts.json':fit.canonical(attempts(root))}
    for arm in p.ARMS:
        for name in ('value.json','receipt.json'):
            key=f'{arm}/initialization/{name}';raw[key]=(root/key).read_bytes()
        for step in p.POLICY['snapshots']:
            for name in p.EVIDENCE_FILES+['receipt.json']:
                key=f'{arm}/step-{step:04d}/{name}';raw[key]=(root/key).read_bytes()
    raw['handoff-manifest.json']=fit.canonical({k:hashlib.sha256(v).hexdigest() for k,v in raw.items()})
    if handoff.exists():
        with zipfile.ZipFile(handoff) as z:
            if len(z.namelist()) != len(raw) or set(z.namelist()) != set(raw) or any(z.read(k)!=v for k,v in raw.items()):
                raise ValueError('Existing handoff differs; preserved')
        return
    handoff.parent.mkdir(parents=True,exist_ok=True)
    staged=Path(tempfile.mkdtemp(prefix='.adaptor-export-',dir=handoff.parent))/handoff.name
    with zipfile.ZipFile(staged,'x',zipfile.ZIP_DEFLATED) as z:
        for k,v in raw.items():z.writestr(k,v)
    staged.rename(handoff)


def read_arm(root,arm,signature,identity):
    folder=root/arm
    if not (folder/'initialization').exists():
        if folder.exists() and any(x.name.startswith('step-') for x in folder.iterdir()):
            raise ValueError('Checkpoint without initialization')
        return None,[]
    initial=p.parity.e.value(folder,'initialization',signature)
    p.validate_initialization(initial,identity,arm)
    return initial,p.read_checkpoints(folder,signature,initial)


def execute(root,archive,fit_root,identity,handoff,runtime_factory=Runtime,monitor_factory=Monitor):
    signature=fit.digest(identity)
    check_inputs(fit_root,identity)
    with fit.locked_run(root):
        if not (root/'identity').exists() and any(x.name!='.run.lock' for x in root.iterdir()):
            raise ValueError('Output without identity; preserve and review')
        publish(root,'identity',signature,identity)
        previous=attempts(root)
        prior_seconds=sum(a['seconds'] for a in previous)
        prior_updates=sum(a['attempted_updates'] for a in previous)
        histories={arm:read_arm(root,arm,signature,identity) for arm in p.ARMS}
        a,b=(histories[arm][1] for arm in p.ARMS)
        if b and len(a)!=len(p.POLICY['snapshots']):raise ValueError('Second arm precedes completed first arm')
        for x,y in zip(a,b):p.validate_matched(x,y)
        if (root/'result').exists():
            report=p.parity.e.value(root,'result',signature)
            if any(len(h[1])!=len(p.POLICY['snapshots']) for h in histories.values()):raise ValueError('Incomplete final result')
            export(root,handoff,identity,report);return report
        attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=root))
        fit.write_json(attempt/'started.json',{'identity':signature,'unix_seconds':time.time()})
        started=time.perf_counter();runtime=None;error=None;interrupted=False;attempted=0
        monitor=monitor_factory(None,prior_seconds,started,p.POLICY)
        reloads={}
        try:
            with monitor,zipfile.ZipFile(archive) as z,ExitStack() as stack:
                names=z.namelist()
                if any(names.count(row['member'])!=1 or '/train/good/' not in row['member'] for row in identity['rows']):
                    raise ValueError('Missing, duplicate or non-normal input')
                images=[stack.enter_context(fit.checked_image(z,row)) for row in identity['rows']]
                for arm in p.ARMS:
                    initial,trajectory=histories[arm]
                    folder=root/arm;folder.mkdir(exist_ok=True)
                    print(f'{arm}: identical saved initial head; 500 updates; no time cap',flush=True)
                    runtime=runtime_factory(fit_root/'step-028800/model.pt',identity['fit_signature'])
                    monitor.runtime=runtime
                    fresh=runtime.initialize(fit_root/'step-000000/model.pt',identity,arm)
                    p.validate_initialization(fresh,identity,arm)
                    if initial is not None and fresh!=initial:raise ValueError('Initialization differs from saved record')
                    publish(folder,'initialization',signature,fresh);initial=fresh
                    if trajectory:reload_verify(runtime,folder,trajectory[-1],signature,images,monitor)
                    losses=trajectory[-1]['losses'].copy() if trajectory else []
                    completed=trajectory[-1]['step'] if trajectory else -1
                    first_history=histories[p.ARMS[0]][1]
                    for stop in p.POLICY['snapshots']:
                        if stop<=completed:continue
                        while len(losses)<stop:
                            monitor.check()
                            if prior_updates+attempted>=p.POLICY['maximum_attempted_updates']:
                                raise RuntimeError('Attempted-update cap exceeded; no automatic extension')
                            attempted+=1
                            row=runtime.step(images,len(losses))
                            if arm==p.ARMS[1]:
                                expected=first_history[-1]['losses'][len(losses)]
                                if any(row[k]!=expected[k] for k in ('step','seed','target_sha256','positive_samples','positive_pixels')):
                                    raise ValueError('Training synthetic draws differ between arms')
                            losses.append(row)
                            if len(losses)%25==0:print(f'{arm} {len(losses)}/500; loss={row["loss"]:.6f}',flush=True)
                        evaluations,flat=evaluate(runtime,images,monitor)
                        if trajectory:check_targets(evaluations,trajectory[0])
                        progress={'step':stop,'losses':losses.copy(),'evaluations':evaluations,'fingerprints':runtime.fingerprints()}
                        if stop in (0,p.POLICY['updates']):progress['feature_statistics']=runtime.feature_statistics(images)
                        p.validate_progress(progress,initial)
                        if arm==p.ARMS[1]:p.validate_matched(first_history[p.POLICY['snapshots'].index(stop)],progress)
                        phase=f'step-{stop:04d}'
                        with fit.phase_stage(folder,phase) as stage:
                            runtime.save(stage/'model.pt',signature,stop)
                            np.savez_compressed(stage/'arrays.npz',**flat)
                            fit.write_json(stage/'progress.json',progress)
                            monitor.check();fit.seal_phase(stage,phase,signature,{},p.FILES)
                        trajectory.append(progress)
                    reloads[arm]=reload_verify(runtime,folder,trajectory[-1],signature,images,monitor)
                    monitor.check();histories[arm]=(initial,trajectory)
                    monitor.runtime=None;runtime.release();runtime=None
                check_inputs(fit_root,identity)
        except BaseException as exc:
            error=f'{type(exc).__name__}: {exc}';interrupted=isinstance(exc,KeyboardInterrupt)
            raise
        finally:
            elapsed=time.perf_counter()-started
            fit.write_json(attempt/'finished.json',{'seconds':elapsed,'attempted_updates':attempted,
                'status':'interrupted' if interrupted else ('failed' if error else 'completed'),
                'error':error,'memory':monitor.snapshot()})
            monitor.runtime=None
            if runtime is not None:runtime.release()
        # Re-read sealed evidence before constructing the comparison or exporting it.
        histories={arm:read_arm(root,arm,signature,identity) for arm in p.ARMS}
        for x,y in zip(*(histories[arm][1] for arm in p.ARMS),strict=True):p.validate_matched(x,y)
        summaries={arm:[{'step':row['step'],'heldout':p.v.aggregate(row['evaluations'])} for row in h[1]]
                   for arm,h in histories.items()}
        final={arm:rows[-1]['heldout'] for arm,rows in summaries.items()}
        difference={k:final[p.ARMS[1]][k]['mean_over_defined_seeds']-final[p.ARMS[0]][k]['mean_over_defined_seeds']
                    if all(final[arm][k]['mean_over_defined_seeds'] is not None for arm in p.ARMS) else None
                    for k in final[p.ARMS[0]]}
        report={'status':'adaptor_start_comparison_completed','completed_updates_per_arm':p.POLICY['updates'],
            'completed_updates_total':len(p.ARMS)*p.POLICY['updates'],
            'attempted_updates_including_replayed_work':prior_updates+attempted,'cumulative_compute_seconds':prior_seconds+elapsed,
            'elapsed_time_cap_seconds':None,'primary_comparison_step':500,'trajectory':summaries,
            'initial_adaptor_minus_trained_adaptor_at_500':difference,'all_synthetic_targets_matched':True,
            'final_reload_max_abs_difference':reloads,'original_checkpoints_unchanged':True,
            'test_images_decoded':0,'test_masks_decoded':0,'threshold_changed':False,'service_model_changed':False,
            'automatic_pass_allowed':False,'automatic_model_selection':False,'later_phases_automatically_started':False,
            'note':'Same one training photo, held-out synthetic seeds only. No independent-image or real-defect claim.'}
        publish(root,'result',signature,report);export(root,handoff,identity,report)
        return report


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args(argv);project=args.project.resolve()
    if fit.sha256_file(args.archive)!=fit.EXPECTED_ARCHIVE_SHA256:raise ValueError('Original archive changed')
    if shutil.disk_usage(project).free<p.POLICY['free_disk_GiB']*1024**3:raise ValueError('Need 8 GiB free disk')
    fit_root=project/'artifacts'/p.parity.e.p.NAME
    if not fit_root.is_dir():raise ValueError('Preserved formal fit folder missing; do not retrain')
    with fit.locked_run(fit_root):
        identity=p.prepare(project,project/'artifacts/handoff',fit_root)
        current=environment()
        for key in ('torch','torchvision','anomalib','timm','supersimplenet_sources'):
            if current[key]!=identity['environment'][key]:raise ValueError(f'Environment changed: {key}')
        identity['runtime_environment']=current
        report=execute(project/'artifacts/analyses'/p.NAME,args.archive,fit_root,identity,
                       project/'artifacts/handoff'/(p.NAME+'.zip'))
    print(json.dumps({k:v for k,v in report.items() if k!='trajectory'},ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    try:raise SystemExit(main())
    except (Exception,KeyboardInterrupt) as exc:
        traceback.print_exc()
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve files; do not reroll or change thresholds.',file=sys.stderr)
        raise SystemExit(1)
