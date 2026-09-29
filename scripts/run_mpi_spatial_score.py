"""Frozen single-window score comparison, normal-only calibration before observed test."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import time
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
from scripts import review_mpi_score_cases as reviewed
from scripts import mpi_spatial_score_protocol as spatial

v, p, fit, ev = reviewed.validation, reviewed.p, reviewed.fit, reviewed.ev
NAME = 'mpi-spatial-score-v1'
GROUPS = ('development', 'calibration', 'test', 'validation')
POLICY_LF_SHA256 = '5a250f2b298be4051cc456d05d585b48a8d44db72ebb54f424197c0229a480c1'


def create_sample(source, photo, runtime, recovered, expected, contract):
    rows, tile_ms, spatial_ms = [], [], []
    started = time.perf_counter()
    for row in photo['regions']:
        runtime.synchronize()
        before = time.perf_counter()
        # No mask reads, ROI based on labels, border deletion or new image resizing.
        image = ev.read_tile(source,row,recovered) if photo['key'].startswith(('good/','defect/')) else fit.checked_image(source,row)
        try:
            raw, values = runtime.predict(image)
            runtime.synchronize()
            values = np.asarray(values,dtype=np.float32)
            top1 = fit.candidate_image_score(raw,values,fit.CANDIDATE)
            began_spatial = time.perf_counter()
            extra = spatial.spatial_score(values)
            spatial_ms.append((time.perf_counter()-began_spatial)*1000)
            rows.append({k:row[k] for k in ('region','member')} | {'top1':top1,'native_raw':float(raw)} | extra)
            tile_ms.append((time.perf_counter()-before)*1000)
        finally:
            image.close()
    sample = {k:photo[k] for k in ('key','photo_id','label')} | {'regions':rows}
    sample['timing'] = {'tile_ms':tile_ms,'spatial_ms':spatial_ms,'photo_scoring_ms':(time.perf_counter()-started)*1000}
    sample['maximum_absolute_replay_difference'] = p.replay(sample,expected,contract)
    spatial.validate(sample,photo)
    return sample


def read_sample(root, group, name, signature, photo, expected, contract, records):
    receipt = fit.read_phase(root/group,name,signature,['sample.json'])
    if receipt['input_photo_sha256'] != fit.digest(photo):
        raise ValueError('Cached spatial sample input differs')
    gate_path = (root/receipt['probe_path']).resolve()
    gate_path.relative_to(root.resolve())
    if fit.sha256_file(gate_path) != receipt['probe_sha256']:
        raise ValueError('Runtime probe evidence differs')
    gate = json.loads(gate_path.read_bytes())
    if not gate['passed'] or gate['runtime'] != records['reload.json']['receipt']['runtime']:
        raise ValueError('Invalid cached runtime gate')
    sample = json.loads((root/group/name/'sample.json').read_bytes())
    spatial.validate(sample,photo)
    if p.replay(sample,expected,contract) != sample['maximum_absolute_replay_difference']:
        raise ValueError('Cached replay differs')
    return sample, receipt


def execute(root, source, checkpoint, recovered_path, config, preparation, records, previous,
            cohorts, reference, contract, policy, identity, handoff, runtime_factory=fit.TorchRuntime):
    if tuple(cohorts) != GROUPS or tuple(reference) != GROUPS:
        raise ValueError('Normal calibration must precede observed evaluation')
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if not (root/'identity').exists() and any((root/n).exists() for n in (*GROUPS,'calibrated','result')):
            raise ValueError('Output without input identity; preserved')
        v.publish(root,'identity',signature,{'plan.json':{'identity':identity,'policy':policy}})
        cached, receipts = {}, {}
        for group in GROUPS:
            (root/group).mkdir(exist_ok=True)
            for i,(photo,expected) in enumerate(zip(cohorts[group],reference[group],strict=True)):
                name = f'photo-{i+1:03d}'
                if (root/group/name).exists():
                    cached[group,i], receipts[group,i] = read_sample(root,group,name,signature,photo,expected,contract,records)
        normal_done = all((g,i) in cached for g in GROUPS[:2] for i in range(len(cohorts[g])))
        if (root/'calibrated').exists() and not normal_done:
            raise ValueError('Calibration receipt missing normal evidence')
        if any(g in GROUPS[2:] for g,i in cached) and not (root/'calibrated').exists():
            raise ValueError('Evaluation sample before calibration freeze')
        count = sum(map(len,cohorts.values()))
        if (root/'result').exists() and len(cached) != count:
            raise ValueError('Published result missing samples')
        runtime = None
        attempt = Path(tempfile.mkdtemp(prefix='attempt-',dir=root)) if len(cached) < count else None
        samples = {g:[] for g in GROUPS}
        try:
            if attempt is not None:
                runtime = runtime_factory(config)
                runtime.load(checkpoint,records['fit.json']['checkpoint_context'])
                if runtime.environment() != records['reload.json']['receipt']['runtime']:
                    raise ValueError('Runtime environment differs')
                gate = ev.normal_probe(source,preparation,records['reload.json']['photos'],runtime)
                gate['runtime'] = runtime.environment()
                fit.write_json(attempt/'normal-probe.json',gate)
                if not gate['passed']:
                    raise ValueError(f"Normal probe blocked: {gate['reasons']}")
                probe_path = (attempt/'normal-probe.json').relative_to(root).as_posix()
            recovered = None
            for group in GROUPS:
                if group == 'test':
                    candidate = spatial.fit_contract(samples['development'],samples['calibration'])
                    calibration = {'method':candidate,'normal_source_sha256':fit.digest({g:samples[g] for g in GROUPS[:2]}),
                                   'evaluation_scores_used':0,'policy_sha256':fit.digest(policy)}
                    v.publish(root,'calibrated',signature,{'calibration.json':calibration})
                    if any((group,i) not in cached for i in range(len(cohorts[group]))):
                        recovered = ev.recovered_bytes(recovered_path,previous)
                print(f'{group}: {len(cohorts[group])} photos; reused {sum((group,i) in cached for i in range(len(cohorts[group])))}',flush=True)
                for i,(photo,expected) in enumerate(zip(cohorts[group],reference[group],strict=True)):
                    name = f'photo-{i+1:03d}'
                    if (group,i) not in cached:
                        with fit.phase_stage(root/group,name) as stage:
                            sample = create_sample(source,photo,runtime,recovered,expected,contract)
                            fit.write_json(stage/'sample.json',sample)
                            fit.seal_phase(stage,name,signature,{'input_photo_sha256':fit.digest(photo),
                                'probe_path':probe_path,'probe_sha256':fit.sha256_file(root/probe_path)},['sample.json'])
                        cached[group,i],receipts[group,i] = read_sample(root,group,name,signature,photo,expected,contract,records)
                    samples[group].append(cached[group,i])
                    if (i+1)%10 == 0 or i+1 == len(cohorts[group]):
                        print(f'{group} {i+1}/{len(cohorts[group])}',flush=True)
            comparison = spatial.compare(samples,contract['candidate_top1_q90'],candidate,policy['maximum_added_scoring_fraction'])
            report = {'status':'spatial_score_comparison_complete_exploratory','policy':policy,'comparison':comparison,
                      'maximum_absolute_replay_difference':max(s['maximum_absolute_replay_difference'] for group in samples.values() for s in group),
                      'photos':count,'training_performed':False,'new_independent_evaluation':False,
                      'control_threshold_changed':False,'candidate_calibrated_on_normals_only':True,'automatic_pass_allowed':False}
            probes = {r['probe_path']:json.loads((root/r['probe_path']).read_bytes()) for r in receipts.values()}
            payloads = {'report.json':report,'raw-scores.json':samples,'calibration.json':calibration,
                        'control.json':contract,'plan.json':json.loads((root/'identity/plan.json').read_bytes()),'runtime-probes.json':probes}
            payloads['handoff-manifest.json'] = {n:hashlib.sha256(fit.canonical(value)).hexdigest() for n,value in payloads.items()}
            v.publish(root,'result',signature,payloads)
            v.probe.export(root/'result',handoff)
            return report
        except Exception as exc:
            if attempt is not None:
                fit.write_json(attempt/'failure.json',{'error':f'{type(exc).__name__}: {exc}','completed_photos':len(cached)})
                print(f"Failure evidence: {attempt/'failure.json'}",file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                runtime.release()


def prepare(project, handoffs):
    data, _, config, preparation, records = reviewed.load_inputs(project,handoffs)
    _, cohorts, _, _, _, previous = v.prepare(project,handoffs)
    policy_path = project/'configs/mpi-spatial-score-v1.json'
    if hashlib.sha256(policy_path.read_bytes().replace(b'\r\n',b'\n')).hexdigest() != POLICY_LF_SHA256:
        raise ValueError('Preregistered spatial policy changed; do not tune this version')
    policy = json.loads(policy_path.read_bytes())
    # The recipe is not an experiment-search CLI. Changing it requires a new protocol version.
    expected = {'window_size':26,'stride':1,'padding':'none','candidate':'window26_q90',
                'normalization':'median_and_q90_higher_minus_median','calibration_quantile':.95,
                'quantile_method':'higher','decision_operator':'>=','maximum_added_scoring_fraction':.1}
    if any(policy.get(k) != val for k,val in expected.items()):
        raise ValueError('Spatial preregistration differs from implemented recipe')
    counts = [policy[k] for k in ('development_count','calibration_count','observed_test_count','observed_normal_count')]
    if [len(cohorts[g]) for g in GROUPS] != counts:
        raise ValueError('Frozen cohort counts differ')
    reference = {g:data['raw-scores.json'][g] for g in GROUPS}
    identity = {'policy':policy,'cohorts_sha256':fit.digest(cohorts),'source_validation_identity':data['plan.json']['identity'],
                'validation_zip_sha256':reviewed.VALIDATION_SHA256,'reference_sha256':fit.digest(reference),
                'source_files_lf_sha256':{f'scripts/{name}':hashlib.sha256((project/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                                        for name in ('run_mpi_spatial_score.py','mpi_spatial_score_protocol.py','review_mpi_score_cases.py')}}
    return dict(config=config,preparation=preparation,records=records,previous=previous,cohorts=cohorts,
                reference=reference,contract=data['freeze.json']['contract'],policy=policy,identity=identity)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        project = args.project.resolve()
        prepared = prepare(project,project/'artifacts/handoff')
        checkpoint = project/'artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt'
        if fit.sha256_file(checkpoint) != ev.CHECKPOINT_SHA256 or fit.sha256_file(args.archive) != fit.EXPECTED_ARCHIVE_SHA256:
            raise ValueError('Checkpoint or original archive identity differs')
        print('One frozen 26x26 window candidate. No training. All evaluation data already observed.',flush=True)
        with zipfile.ZipFile(args.archive) as source:
            report = execute(project/'artifacts/analyses'/NAME,source,checkpoint,
                project/'artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png',
                **prepared,handoff=project/'artifacts/handoff'/f'{NAME}.zip')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Existing data/results preserved.',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
