"""Six fixed misses: map attribution only, no tuning or automatic cause assignment."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import tempfile
import zipfile

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
from scripts import run_mpi_spatial_score as run
from scripts import mpi_spatial_miss_cards as cards

fit, ev, p, spatial = run.fit, run.ev, run.p, run.spatial
NAME = 'mpi-spatial-miss-review-v1'
SOURCE_SHA256 = '8429a970864c1116d5f52b786057fc84c3e48f465b5c502715ab7f61d35e13dc'
SELECTION = [('0019', 'remaining_miss'), ('0024', 'remaining_miss'),
             ('0021', 'lost_detection'), ('0027', 'lost_detection'),
             ('0028', 'lost_detection'), ('0047', 'lost_detection')]
FILES = ['sample.json', 'case.json', 'maps.npz', 'card-1.jpg', 'card-2.jpg', 'focus.jpg']


def prepare(project, handoffs):
    previous = ev.checked_bundle(handoffs / 'mpi-spatial-score-v1.zip', SOURCE_SHA256)
    source = run.prepare(project, handoffs)
    if previous['plan.json']['identity'] != source['identity'] or previous['control.json'] != source['contract']:
        raise ValueError('Spatial source identity differs')
    samples = previous['raw-scores.json']
    for group in run.GROUPS:
        for s, photo, ref in zip(samples[group], source['cohorts'][group], source['reference'][group], strict=True):
            spatial.validate(s, photo)
            if p.replay(s, ref, source['contract']) != s['maximum_absolute_replay_difference']:
                raise ValueError('Spatial source replay differs')
    candidate = spatial.fit_contract(samples['development'], samples['calibration'])
    if previous['calibration.json'] != {
        'method': candidate, 'evaluation_scores_used': 0,
        'normal_source_sha256': fit.digest({g:samples[g] for g in run.GROUPS[:2]}),
        'policy_sha256': fit.digest(source['policy'])}:
        raise ValueError('Spatial source calibration differs')
    control = source['contract']['candidate_top1_q90']
    if spatial.compare(samples, control, candidate) != previous['report.json']['comparison']:
        raise ValueError('Spatial source result differs')
    methods = {'control_q90':control, 'window26_q90':candidate}
    selected = []
    for photo_id, reason in SELECTION:
        key = 'defect/' + photo_id
        sample = next(s for s in samples['test'] if s['key'] == key)
        photo = next(s for s in source['cohorts']['test'] if s['key'] == key)
        if not sample['label'] or p.score(sample, candidate)['alert'] or (
                p.score(sample, control)['alert'] != (reason == 'lost_detection')):
            raise ValueError('Fixed selection no longer matches source decisions')
        selected.append({'photo':photo, 'expected':sample, 'reason':reason})
    identity = {'spatial_zip_sha256':SOURCE_SHA256, 'source_identity':source['identity'],
                'selection':selected, 'methods':methods, 'old_contract':source['contract'],
                'source_files_lf_sha256':{f'scripts/{n}':hashlib.sha256((project/'scripts'/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
                    for n in ('review_mpi_spatial_misses.py', 'mpi_spatial_miss_cards.py',
                              'mpi_score_case_cards.py', 'mpi_score_probe_cards.py')}}
    return {k:source[k] for k in ('config','preparation','records')} | {
        'selected':selected, 'methods':methods, 'old_contract':source['contract'], 'identity':identity}


def replay(sample, expected, old_contract, methods):
    spatial.validate(sample, expected)
    maximum = p.replay(sample, expected, old_contract)
    for row, ref in zip(sample['regions'], expected['regions'], strict=True):
        maximum = max(maximum, abs(row['window26']-ref['window26']))
        if row['window_xyxy'] != ref['window_xyxy']:
            raise ValueError('Spatial winning window replay differs')
    if maximum > 1e-4:
        raise ValueError('Spatial score replay drift')
    for method in methods.values():
        if p.score(sample, method)['alert'] != p.score(expected, method)['alert']:
            raise ValueError('Frozen decision changed')
    return maximum


def describe(sample, arrays, masks, methods, reason):
    # GT is only consulted here, after frozen scores and positions were computed.
    if arrays.shape != (12,256,256) or arrays.dtype != np.float32:
        raise ValueError('Expected twelve float32 maps')
    if masks.shape != arrays.shape or masks.dtype != np.bool_:
        raise ValueError('Expected twelve intact boolean masks')
    rows = []
    for i, row in enumerate(sample['regions']):
        values, mask = arrays[i], masks[i]
        stats = ev.maps.region_statistics(values, mask, row['top1'])
        computed = spatial.spatial_score(values)
        if abs(computed['window26']-row['window26']) > 1e-4 or computed['window_xyxy'] != row['window_xyxy']:
            raise ValueError('Map/window score differs')
        x,y,x2,y2 = row['window_xyxy']
        inside, gt = values[y:y2,x:x2], mask[y:y2,x:x2]
        overlap, total_gt = int(gt.sum()), int(mask.sum())
        total_score = float(inside.sum(dtype=np.float64))
        attribution = {'window_xyxy':row['window_xyxy'], 'pixels':676,
                       'gt_overlap_pixels':overlap, 'gt_recall':overlap/total_gt if total_gt else None,
                       'gt_fraction_of_window':overlap/676,
                       'gt_score_share':float(inside[gt].sum(dtype=np.float64))/total_score if total_score > 0 else None,
                       'gt_mean':float(inside[gt].mean(dtype=np.float64)) if overlap else None,
                       'non_gt_mean':float(inside[~gt].mean(dtype=np.float64)) if overlap < 676 else None}
        boundaries = {}
        for name, method in methods.items():
            param = method['regions'][i]
            if param['region'] != row['region']:
                raise ValueError('Normalization order differs')
            value = (row[method['field']]-param['center'])/param['scale']
            boundaries[name] = {'normalized_score':value, 'margin':value-method['threshold'],
                'raw_boundary':param['center']+param['scale']*method['threshold']}
        rows.append({'region':row['region'], 'map':stats, 'window':attribution, 'methods':boundaries})
    return {'key':sample['key'], 'label':sample['label'], 'reason':reason, 'sample':sample,
            'methods':{name:p.score(sample,m) | {'threshold':m['threshold'], 'margin':p.score(sample,m)['score']-m['threshold']}
                       for name,m in methods.items()}, 'regions':rows,
            'automatic_pass_allowed':False, 'automatic_cause_assigned':False}


def create_case(source, entry, runtime, old_contract, methods, stage):
    photo = entry['photo']
    images, arrays, rows = [], [], []
    try:
        for row in photo['regions']:
            image = ev.read_tile(source, row, None)
            images.append(image)
            raw, values = runtime.predict(image)
            runtime.synchronize()
            values = np.asarray(values,dtype=np.float32)
            arrays.append(values.copy())
            rows.append({k:row[k] for k in ('region','member')} | {
                'top1':fit.candidate_image_score(raw,values,fit.CANDIDATE), 'native_raw':float(raw)} | spatial.spatial_score(values))
        sample = {k:photo[k] for k in ('key','photo_id','label')} | {'regions':rows}
        delta = replay(sample, entry['expected'], old_contract, methods)
        masks = ev.read_masks(source,photo)
        if any(m is None for m in masks):
            raise ValueError('Six-case review requires intact official masks')
        arrays, masks = np.stack(arrays), np.stack(masks)
        case = describe(sample,arrays,masks,methods,entry['reason'])
        case['maximum_absolute_replay_difference'] = delta
        fit.write_json(stage/'sample.json',sample)
        fit.write_json(stage/'case.json',case)
        with (stage/'maps.npz').open('xb') as target:
            np.savez_compressed(target,anomaly_maps=arrays,gt_masks=masks,regions=np.array(fit.REGIONS))
        for page in range(2):
            with cards.render(case,images,arrays,masks,page) as card:
                card.save(stage/f'card-{page+1}.jpg',quality=92)
        with cards.focus(case,images,arrays,masks) as card:
            card.save(stage/'focus.jpg',quality=92)
    finally:
        for image in images:
            image.close()


def read_case(root,name,signature,entry,records,old_contract,methods):
    receipt = fit.read_phase(root/'cases',name,signature,FILES)
    if receipt['entry_sha256'] != fit.digest(entry):
        raise ValueError('Cached input changed')
    gate_path = (root/receipt['probe_path']).resolve()
    gate_path.relative_to(root.resolve())
    if fit.sha256_file(gate_path) != receipt['probe_sha256']:
        raise ValueError('Runtime gate changed')
    gate = json.loads(gate_path.read_bytes())
    if not gate['passed'] or gate['runtime'] != records['reload.json']['receipt']['runtime']:
        raise ValueError('Invalid cached runtime gate')
    case = json.loads((root/'cases'/name/'case.json').read_bytes())
    sample = json.loads((root/'cases'/name/'sample.json').read_bytes())
    with np.load(root/'cases'/name/'maps.npz',allow_pickle=False) as data:
        if list(data['regions']) != list(fit.REGIONS):
            raise ValueError('Saved map order differs')
        expected = describe(sample,data['anomaly_maps'],data['gt_masks'],methods,entry['reason'])
    expected['maximum_absolute_replay_difference'] = replay(sample,entry['expected'],old_contract,methods)
    if case != expected:
        raise ValueError('Cached map attribution differs')
    return case,receipt


def execute(root,source,checkpoint,config,preparation,records,selected,methods,old_contract,identity,
            handoff,runtime_factory=fit.TorchRuntime):
    signature = fit.digest(identity)
    with fit.locked_run(root):
        if not (root/'identity').exists() and any((root/n).exists() for n in ('cases','result')):
            raise ValueError('Output without identity; preserved')
        run.v.publish(root,'identity',signature,{'plan.json':{'identity':identity}})
        (root/'cases').mkdir(exist_ok=True)
        cases,receipts = {},{}
        for entry in selected:
            name = entry['photo']['key'].replace('/','-')
            if (root/'cases'/name).exists():
                cases[name],receipts[name] = read_case(root,name,signature,entry,records,old_contract,methods)
        if (root/'result').exists() and len(cases) != len(selected):
            raise ValueError('Published review missing cases')
        runtime = None
        attempt = Path(tempfile.mkdtemp(prefix='attempt-',dir=root)) if len(cases)<len(selected) else None
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
                    raise ValueError('Normal probe blocked')
                probe_path = (attempt/'normal-probe.json').relative_to(root).as_posix()
            for entry in selected:
                name = entry['photo']['key'].replace('/','-')
                if name not in cases:
                    with fit.phase_stage(root/'cases',name) as stage:
                        create_case(source,entry,runtime,old_contract,methods,stage)
                        fit.seal_phase(stage,name,signature,{'entry_sha256':fit.digest(entry),
                            'probe_path':probe_path,'probe_sha256':fit.sha256_file(root/probe_path)},FILES)
                    cases[name],receipts[name] = read_case(root,name,signature,entry,records,old_contract,methods)
                print(f"Six-case review {len(cases)}/{len(selected)}: {entry['photo']['key']}",flush=True)
            ordered = [cases[e['photo']['key'].replace('/','-')] for e in selected]
            report = {'status':'six_spatial_misses_review_complete_exploratory','case_count':len(ordered),
                'spatial_zip_sha256':SOURCE_SHA256,'execution_sha256':signature,
                'maximum_absolute_replay_difference':max(c['maximum_absolute_replay_difference'] for c in ordered),
                'training_performed':False,'thresholds_changed':False,'new_performance_evaluation':False,
                'automatic_pass_allowed':False,'automatic_cause_assigned':False,'source_attribution':ev.maps.SOURCE_ATTRIBUTION}
            payloads = {'report.json':report,'cases.json':ordered,'freeze.json':methods,'plan.json':{'identity':identity},
                'runtime-probes.json':{r['probe_path']:json.loads((root/r['probe_path']).read_bytes()) for r in receipts.values()}}
            paths = {f"cards/{name}-{suffix}.jpg":root/'cases'/name/file for name in cases
                     for suffix,file in (('1','card-1.jpg'),('2','card-2.jpg'),('focus','focus.jpg'))}
            paths.update({f'maps/{name}.npz':root/'cases'/name/'maps.npz' for name in cases})
            names = list(payloads)+list(paths)+['index.html','handoff-manifest.json']
            if not (root/'result').exists():
                with fit.phase_stage(root,'result') as stage:
                    for name,value in payloads.items(): fit.write_json(stage/name,value)
                    for name,path in paths.items():
                        (stage/name).parent.mkdir(exist_ok=True)
                        shutil.copyfile(path,stage/name)
                    (stage/'index.html').write_text(cards.index(ordered),encoding='utf-8')
                    fit.write_json(stage/'handoff-manifest.json',{n:fit.sha256_file(stage/n) for n in names if n!='handoff-manifest.json'})
                    fit.seal_phase(stage,'result',signature,{},names)
            fit.read_phase(root,'result',signature,names)
            for name,value in payloads.items():
                if json.loads((root/'result'/name).read_bytes()) != value:
                    raise ValueError('Published content differs')
            run.v.probe.export(root/'result',handoff)
            return report
        except Exception as exc:
            if attempt is not None:
                fit.write_json(attempt/'failure.json',{'error':f'{type(exc).__name__}: {exc}','completed_cases':len(cases)})
            raise
        finally:
            if runtime is not None: runtime.release()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path,required=True)
    args = parser.parse_args(argv)
    try:
        project = args.project.resolve()
        prepared = prepare(project,project/'artifacts/handoff')
        checkpoint = project/'artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt'
        if fit.sha256_file(args.archive)!=fit.EXPECTED_ARCHIVE_SHA256 or fit.sha256_file(checkpoint)!=ev.CHECKPOINT_SHA256:
            raise ValueError('Archive or checkpoint changed')
        print('Six fixed defects, 72 tiles. Read-only attribution; no fitting or threshold changes.',flush=True)
        with zipfile.ZipFile(args.archive) as source:
            report = execute(project/'artifacts/analyses'/NAME,source,checkpoint,**prepared,
                             handoff=project/'artifacts/handoff'/f'{NAME}.zip')
        print(json.dumps(report,ensure_ascii=False,indent=2))
        return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Existing data/results preserved.',file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
