"""Fixed20 tile maps: representative bank vs exhaustive retained candidate pool."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'src')]
from scripts import diagnose_mpi_patchcore_candidate_coverage as c
from scripts import mpi_supersimplenet_map_diagnostic as metrics

fit, p, engine = c.fit, c.p, c.engine
NAME = 'mpi-patchcore-full-map-comparison-v1'
INPUT_SHA = '74c90571b5942618f221bf61f93f9faf2e7deb6a513b8daf66499258d6d50228'
POLICY = {
    'tiles': 20, 'feature_rows_per_tile': 4096, 'dimensions': 768,
    'training_photos': 256, 'candidates': 3145728, 'bank': 78643,
    'query_chunk': 512, 'candidate_chunk': 4096,
    'distance': 'installed euclidean_dist float64 GEMM; all retained stride2 candidates; no normalization',
    'distance_atol': 1e-4, 'distance_rtol': 1e-8,
    'native_replay_atol': 1e-4,
    'float64_vs_native_map_atol': .001, 'float64_vs_native_map_rtol': .0001,
    'map': 'installed anomaly_map_generator, distances cast to float32;64x64 to256x256',
    'score': 'top656 map mean only; no weighted native image score or new photo decision',
    'selection': 'exact previous20 tiles; all locations, no outcome-based selection',
    'display': 'common old-bank map maximum across20 tiles; clipping fraction recorded',
    'resource_plan': p.POLICY['resource_plan'], 'time_limit_seconds': None,
    'training': False, 'threshold_changed': False, 'automatic_pass_allowed': False,
    'service_model_changed': False, 'new_photo_predictions': False,
    'resume': 'atomic tile/normal-photo phases; Ctrl+C resumes; failed/unaccounted attempts require review',
}
SOURCES = ['scripts/compare_mpi_patchcore_full_maps.py', 'scripts/run_mpi_patchcore_full_maps.ps1',
           'scripts/mpi_supersimplenet_map_diagnostic.py']


def checked_input(path):
    if fit.sha256_file(path) != INPUT_SHA: raise ValueError('Expected reviewed candidate-coverage ZIP')
    with zipfile.ZipFile(path) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest)|{'handoff-manifest.json'}:
            raise ValueError('Coverage ZIP inventory differs')
        data = {n:z.read(n) for n in manifest}
        if any(hashlib.sha256(data[n]).hexdigest() != h for n,h in manifest.items()):
            raise ValueError('Coverage ZIP payload differs')
    return data


def check_distances(actual, expected, why):
    actual, expected = np.asarray(actual), np.asarray(expected)
    if (actual.shape != expected.shape or not np.isfinite(actual).all() or not np.isfinite(expected).all()
            or np.any(actual < 0) or np.any(expected < 0) or not np.allclose(actual, expected,
                atol=POLICY['distance_atol'], rtol=POLICY['distance_rtol'])):
        raise ValueError(f'Distance parity failed: {why}')
    return float(np.max(np.abs(actual-expected)))


def point_replay(records, features, bank, full, prior, arrays):
    lookup = {(r['key'],r['source_row']['region']):i for i,r in enumerate(records)}
    ids = [lookup[q['key'],q['region']]*4096+q['flat_index'] for q in prior['queries']]
    if not np.array_equal(features[ids], arrays): raise ValueError('Frozen53 query features changed')
    return {'queries':len(ids), 'bank_max_difference':check_distances(bank[ids], [q['bank_distance'] for q in prior['queries']], '53 bank queries'),
            'full_max_difference':check_distances(full[ids], [q['full_distance'] for q in prior['queries']], '53 full queries')}


def describe(record, old, bank, full, mask):
    for values in (old, bank, full):
        if values.shape != (256,256) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError('Invalid tile map')
    if np.any(full > bank+POLICY['float64_vs_native_map_atol']): raise ValueError('Full-pool map exceeds subset map')
    def stat(values):
        result = {k:v for k,v in metrics.map_stats(values.astype(np.float64),mask).items() if not k.startswith('display_')}
        result['map_maximum_yx'] = list(map(int,np.unravel_index(np.argmax(values),values.shape)))
        return result
    old_s, bank_s, full_s = map(stat,(old,bank,full))
    # A paired source query is used for both banks; maxima are allowed to move.
    y,x = bank_s['map_maximum_yx']
    return {'key':record['key'],'region':record['source_row']['region'],'label':record['label'],
        'official_saved':old_s,'matched_bank':bank_s,'full_pool':full_s,
        'top656_relative_drop':float((bank_s['top656_mean']-full_s['top656_mean'])/bank_s['top656_mean']) if bank_s['top656_mean'] else None,
        'full_value_at_bank_peak':float(full[y,x]),
        'maximum_moved':bank_s['map_maximum_yx'] != full_s['map_maximum_yx'],
        'map_reduction_mean':float(np.mean(bank.astype(np.float64)-full)),
        'photo_prediction':None,'threshold_applied':False}


class Runtime(c.Runtime):
    def fast_search(self, queries, candidates, check):
        """Same Euclidean metric, double GEMM workspace bounded on both axes."""
        t = self.torch
        if (queries.ndim != 2 or candidates.ndim != 2 or queries.shape[1] != candidates.shape[1]
                or not len(queries) or not len(candidates) or not np.isfinite(queries).all() or not np.isfinite(candidates).all()):
            raise ValueError('Invalid feature search')
        result = np.full(len(queries), np.inf); indices = np.full(len(queries),np.iinfo(np.int64).max,dtype=np.int64)
        with t.inference_mode():
            all_q = t.as_tensor(np.array(queries,dtype=np.float64),device='cuda')
            for cs in range(0,len(candidates),POLICY['candidate_chunk']):
                cand = t.as_tensor(np.array(candidates[cs:cs+POLICY['candidate_chunk']],dtype=np.float64),device='cuda')
                for qs in range(0,len(queries),POLICY['query_chunk']):
                    check(); end=min(qs+POLICY['query_chunk'],len(queries))
                    dist=self.model.model.euclidean_dist(all_q[qs:end],cand)
                    dd,ii=dist.min(dim=1)
                    result[qs:end],indices[qs:end]=c.merge_best(result[qs:end],indices[qs:end],dd.cpu().numpy(),ii.cpu().numpy()+cs)
                    del dist,dd,ii
                del cand
        return result,indices

    def fast_gate(self, check):
        rng=np.random.default_rng(6421)
        q=rng.normal(size=(517,768)).astype(np.float32)
        bank=rng.normal(size=(4103,768)).astype(np.float32)
        bank[0]=q[0];bank[4100]=bank[0];bank[-1]=q[-1]
        actual,indices=self.fast_search(q,bank,check)
        # Independent direct differences, bounded CPU memory; includes query/candidate chunk edges.
        expected=[]; winners=[]
        for row in q:
            distances=np.linalg.norm(bank.astype(np.float64)-row.astype(np.float64),axis=1)
            winners.append(int(np.argmin(distances)));expected.append(float(distances.min()))
        delta=check_distances(actual,expected,'synthetic direct CPU reference')
        if not np.array_equal(indices,winners):raise ValueError('Synthetic nearest indices/tie parity differs')
        return {'passed':True,'maximum_distance_difference':delta,'queries':517,'candidates':4103}

    def maps(self, distances):
        t=self.torch
        with t.inference_mode():
            tensor=t.as_tensor(np.array(distances,dtype=np.float32).reshape(1,1,64,64),device='cuda')
            result=self.model.model.anomaly_map_generator(tensor,(256,256))
        return result.detach().cpu().numpy().reshape(256,256).copy()


def collect_tile(root,name,sig,record,payload,runtime):
    files=['features.npy','native-distances.npy','replay.json']
    if not (root/name).exists():
        with Image.open(io.BytesIO(payload[record['prefix']+'/source.png'])) as image:
            image.load();native,values,features,distances=runtime.trace(image)
        saved=np.load(io.BytesIO(payload[record['prefix']+'/map.npy']),allow_pickle=False)
        replay_map=runtime.maps(distances)
        delta=max(float(np.max(np.abs(values-saved))),float(np.max(np.abs(replay_map-saved))),
                  abs(native-record['sample_region']['native_raw']),
                  abs(fit.candidate_image_score(native,values,fit.CANDIDATE)-record['sample_region']['top1']))
        if not np.isfinite(delta) or delta>POLICY['native_replay_atol']:
            raise ValueError('Official map/score reconstruction differs; no threshold change')
        with fit.phase_stage(root,name) as stage:
            np.save(stage/'features.npy',features,allow_pickle=False)
            np.save(stage/'native-distances.npy',distances,allow_pickle=False)
            fit.write_json(stage/'replay.json',{'maximum_absolute_difference':delta})
            fit.seal_phase(stage,name,sig,{'record_sha256':fit.digest(record)},files)
    receipt=fit.read_phase(root,name,sig,files)
    if receipt['record_sha256']!=fit.digest(record):raise ValueError('Query tile identity differs')
    return np.array(p.array(root/name/'features.npy',(4096,768))),receipt


def scan_pool(root,sig,old,parent,runtime,features,bank,monitor):
    """Save each complete training-photo scan; unfinished blocks never become committed results."""
    plan=parent['plan.json'];parent_sig=fit.digest(plan)
    selected=c.validate_indices(parent['selected-indices.json']['indices'],POLICY['candidates'],POLICY['bank'])
    best=np.full(len(features),np.inf);best_i=np.full(len(features),np.iinfo(np.int64).max,dtype=np.int64)
    nearest=np.zeros_like(features);covered=0;receipts=[];timings=[]
    for i in range(POLICY['training_photos']):
        monitor.check();name=f'photo-{i:03d}'
        receipt=fit.read_phase(old/'features',name,parent_sig,['features.npy'])
        if receipt!=parent['feature-receipts.json'][i]:raise ValueError('Original feature cache receipt differs')
        candidates=p.array(old/'features'/name/'features.npy',(12288,768))
        covered+=c.validate_subset(bank,candidates,selected,i*12288)
        phase=f'search-{i:03d}';files=['distances.npy','indices.npy','cost.json']
        if not (root/phase).exists():
            runtime.synchronize();started=time.perf_counter()
            distances,indices=runtime.fast_search(features,candidates,monitor.check)
            runtime.synchronize();seconds=time.perf_counter()-started
            with fit.phase_stage(root,phase) as stage:
                np.save(stage/'distances.npy',distances,allow_pickle=False)
                np.save(stage/'indices.npy',indices+i*12288,allow_pickle=False)
                fit.write_json(stage/'cost.json',{'search_seconds':seconds})
                fit.seal_phase(stage,phase,sig,{'feature_receipt_sha256':receipt['receipt_sha256']},files)
        sr=fit.read_phase(root,phase,sig,files)
        if sr['feature_receipt_sha256']!=receipt['receipt_sha256']:raise ValueError('Scan source differs')
        distances=p.array(root/phase/'distances.npy',(len(features),),np.float64)
        indices=p.array(root/phase/'indices.npy',(len(features),),np.int64)
        if np.any(indices<i*12288) or np.any(indices>=(i+1)*12288):raise ValueError('Nearest index outside block')
        next_d,next_i=c.merge_best(best,best_i,distances,indices)
        replaced=next_i!=best_i;nearest[replaced]=candidates[indices[replaced]-i*12288]
        best,best_i=next_d,next_i;receipts.append(sr)
        cost=json.loads((root/phase/'cost.json').read_bytes())['search_seconds'];timings.append(cost)
        if i==0:
            print(f'Technical first block: {cost:.2f}s; linear search-only estimate {cost*POLICY["training_photos"]/3600:.2f}h. Not an ETA or cutoff.',flush=True)
        if (i+1)%8==0:print(f'full map search {i+1}/{POLICY["training_photos"]} training photos',flush=True)
    if covered!=POLICY['bank']:raise ValueError('Bank subset verification incomplete')
    direct=np.concatenate([np.linalg.norm(features[i:i+512].astype(np.float64)-nearest[i:i+512].astype(np.float64),axis=1) for i in range(0,len(features),512)])
    delta=check_distances(best,direct,'all final winning vectors CPU direct norm')
    return best,best_i,{'subset_verified':covered,'winner_distance_maximum_difference':delta,
                       'successful_block_search_seconds':timings,'search_seconds_sum':sum(timings)},receipts


def card(source,bank,full,mask,limit,caption=''):
    if not np.isfinite(limit) or limit < 0: raise ValueError('Invalid display limit')
    canvas=Image.new('RGB',(768,292),'white');draw=ImageDraw.Draw(canvas)
    with Image.open(io.BytesIO(source)) as image:canvas.paste(image.convert('RGB'),(0,36))
    draw.text((4,4),'Source; green on maps = GT',fill='black')
    draw.text((4,19),caption,fill='black')
    for j,(name,values) in enumerate([('Matched bank',bank),('Full candidates',full)],1):
        v=np.clip(values/limit,0,1) if limit else np.zeros_like(values)
        rgb=np.stack([v,np.zeros_like(v),1-v],axis=2)
        pixels=np.uint8(np.rint(rgb*255))
        if mask is not None:
            pad=np.pad(mask,1)
            inner=pad[1:-1,1:-1]&pad[:-2,1:-1]&pad[2:,1:-1]&pad[1:-1,:-2]&pad[1:-1,2:]
            pixels[mask&~inner]=[0,255,0]
        canvas.paste(Image.fromarray(pixels),(j*256,36))
        draw.text((j*256+4,4),name,fill='black')
        draw.text((j*256+4,19),f'common distance scale 0 to {limit:.3f}',fill='black')
    b=io.BytesIO();canvas.save(b,format='PNG');return b.getvalue()


def run(project,archive_path,runtime_factory=Runtime):
    handoffs=project/'artifacts/handoff';current,parent,raw=c.overlap.load(handoffs)
    prior_bytes=checked_input(handoffs/(c.NAME+'.zip'))
    prior_plan=json.loads(prior_bytes['plan.json']);prior=json.loads(prior_bytes['result.json'])
    payload=c.read_overlap(handoffs/(c.overlap.NAME+'.zip'))
    expected=c.select_tiles(c.overlap.analyze(current,parent,raw),json.loads(payload['review.json'])['records'],payload)
    plan=parent['plan.json'];parent_sig=fit.digest(plan)
    if (prior_plan['selected_tiles']!=expected or prior_plan['parent_execution_sha256']!=parent_sig
            or prior_plan['policy']!=c.POLICY or len(expected)!=POLICY['tiles']
            or len(parent['feature-receipts.json'])!=POLICY['training_photos']):raise ValueError('Frozen selection/pool differs')
    if fit.sha256_file(archive_path)!=plan['archive_sha256']:raise ValueError('Original archive differs')
    for source_plan in [plan,prior_plan]:
        for n,h in source_plan['sources_lf_sha256'].items():
            if hashlib.sha256((project/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:raise ValueError(f'Frozen source changed: {n}')
    old=project/'artifacts'/p.NAME
    model=fit.read_phase(old,'model',parent_sig,['model.pt','context.json'])
    if model!=parent['model-receipt.json']:raise ValueError('Model receipt differs')
    identity={'name':NAME,'policy':POLICY,'coverage_sha256':INPUT_SHA,'parent_execution_sha256':parent_sig,
              'checkpoint_sha256':model['files']['model.pt'],'tiles':expected,
              'sources_lf_sha256':{n:hashlib.sha256((project/n).read_bytes().replace(b'\r\n',b'\n')).hexdigest() for n in SOURCES}}
    root=project/'artifacts/analyses'/NAME
    with fit.locked_run(root):
        sig=engine.initialize(root,identity)
        with engine.attempt(root,sig,POLICY) as monitor,zipfile.ZipFile(archive_path) as archive:
            runtime=runtime_factory(plan['config']);monitor.runtime=runtime
            if c.environment()!=plan['environment']:raise ValueError('GPU/runtime differs')
            runtime.load(old/'model/model.pt',parent['fit-report.json']['context'])
            print('1/4 Fixed normal replay and float64 matrix-search correctness check.',flush=True)
            normal_gate=engine.normal_gate(old,parent_sig,archive,plan,runtime,identity['checkpoint_sha256'],parent['contract.json'])
            fast_gate=runtime.fast_gate(monitor.check)
            bank=runtime.model.model.memory_bank.detach().cpu().numpy().copy()
            if bank.shape!=(POLICY['bank'],768) or bank.dtype!=np.float32 or not np.isfinite(bank).all():raise ValueError('Bank differs')
            features=[];tile_receipts=[]
            for i,record in enumerate(expected):
                f,receipt=collect_tile(root,f'tile-{i:02d}',sig,record,payload,runtime)
                features.append(f);tile_receipts.append(receipt)
            features=np.concatenate(features)
            print('2/4 Matched float64 representative maps; original model untouched.',flush=True)
            bank_d,bank_i=runtime.fast_search(features,bank,monitor.check)
            direct=np.concatenate([np.linalg.norm(features[i:i+512].astype(np.float64)-bank[bank_i[i:i+512]].astype(np.float64),axis=1) for i in range(0,len(features),512)])
            check_distances(bank_d,direct,'all representative winners')
            old_maps=[np.load(io.BytesIO(payload[r['prefix']+'/map.npy']),allow_pickle=False) for r in expected]
            bank_maps=[runtime.maps(x) for x in bank_d.reshape(-1,4096)]
            for a,b in zip(old_maps,bank_maps,strict=True):
                if not np.allclose(a,b,atol=POLICY['float64_vs_native_map_atol'],rtol=POLICY['float64_vs_native_map_rtol']):
                    raise ValueError('Matched double-precision map too far from native map')
            print('3/4 All81920 locations against full retained pool; timing first block, no time cap.',flush=True)
            full_d,full_i,scan,receipts=scan_pool(root,sig,old,parent,runtime,features,bank,monitor)
            if np.any(full_d>bank_d+POLICY['distance_atol']):raise ValueError('Full distance exceeds subset')
            points=point_replay(expected,features,bank_d,full_d,prior,np.load(io.BytesIO(prior_bytes['queries.npy']),allow_pickle=False))
            if not np.array_equal(bank,runtime.model.model.memory_bank.detach().cpu().numpy()):raise ValueError('Bank mutated')
            print('4/4 Fixed map generation, top656 aggregation, GT response and private cards.',flush=True)
            full_maps=[runtime.maps(x) for x in full_d.reshape(-1,4096)]
            limit=float(np.max(bank_maps));rows=[];export={}
            for i,(record,a,b,new) in enumerate(zip(expected,old_maps,bank_maps,full_maps,strict=True)):
                prefix=record['prefix'];mask=np.load(io.BytesIO(payload[prefix+'/mask.npy']),allow_pickle=False) if record['label'] else None
                rows.append(describe(record,a,b,new,mask))
                rows[-1]['full_display_above_range_fraction']=float(np.mean(new>limit))
                caption=record['key'].split('/')[-1]+' / '+record['source_row']['region']
                export[f'cards/tile-{i:02d}.png']=card(payload[prefix+'/source.png'],b,new,mask,limit,caption)
                export[f'sources/tile-{i:02d}.png']=payload[prefix+'/source.png']
                if mask is not None:export[f'masks/tile-{i:02d}.npy']=c.np_bytes(mask)
            report={'status':'completed_fixed20_full_map_diagnostic','normal_gate':normal_gate,'fast_search_gate':fast_gate,
                    'prior53_replay':points,'scan':scan,'tiles':rows,'display_maximum':limit,
                    'native_vs_matched_map_max_difference':float(np.max(np.abs(np.stack(old_maps)-np.stack(bank_maps)))),
                    'training':False,'threshold_changed':False,'photo_predictions_computed':False,'automatic_pass_allowed':False}
            p.publish(root,'result',sig,report)
            export.update({'plan.json':fit.canonical(identity),'result.json':fit.canonical(report),
                'bank-distances.npy':c.np_bytes(bank_d.reshape(-1,4096)), 'full-distances.npy':c.np_bytes(full_d.reshape(-1,4096)),
                'bank-indices.npy':c.np_bytes(bank_i.reshape(-1,4096)), 'full-indices.npy':c.np_bytes(full_i.reshape(-1,4096)),
                'official-maps.npy':c.np_bytes(np.stack(old_maps)), 'bank-maps.npy':c.np_bytes(np.stack(bank_maps)),
                'full-maps.npy':c.np_bytes(np.stack(full_maps)), 'tile-receipts.json':fit.canonical(tile_receipts),
                'search-receipts.json':fit.canonical(receipts)})
            # Freeze cost at first export so a verified completed rerun does not rewrite it.
            if not (root/'resource').exists():p.publish(root,'resource',sig,monitor.snapshot())
            export['resource.json']=fit.canonical(p.value(root,'resource',sig))
            monitor.check();output=handoffs/(NAME+'.zip');c.write_bundle(output,export)
            return {'status':report['status'],'tiles':len(rows),'locations':len(features),
                    'search_seconds_sum':scan['search_seconds_sum'],'output':str(output),'automatic_pass_allowed':False}


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=ROOT);parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args(argv)
    try:print(json.dumps(run(args.project,args.archive),indent=2));return 0
    except KeyboardInterrupt:
        print('Interrupted. Keep files; same command resumes completed training-photo blocks.');return 130
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no retraining or threshold change.');return 1


if __name__=='__main__':raise SystemExit(main())
