"""Resumable same-region search ablation on a frozen MPI PatchCore checkpoint."""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import os
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parents[1]/'src')]
from scripts import mpi_region_search_protocol as p
from scripts.mpi_region_runtime import RegionRuntime
from scripts import probe_mpi_scores as old

fit, ev = p.fit, p.ev
RESULT_FILES = ['comparison.json','scores.json','raw-scores.json','candidate-calibration.json',
                'shared-A-calibration.json','routing.json','origins.json','plan.json','index.html','handoff-manifest.json']


def read_phase(root, name, signature, files):
    receipt = fit.read_phase(root,name,signature,files)
    return {n:json.loads((root/name/n).read_bytes()) for n in files if n.endswith('.json')}, receipt


def publish(root, name, signature, values, metadata=None):
    with fit.phase_stage(root,name) as stage:
        for n,v in values.items():
            fit.write_json(stage/n,v)
        return fit.seal_phase(stage,name,signature,metadata or {},list(values))


def sample_files(test):
    return ['sample.json'] + (['maps.npz'] if test else [])


def check_sample(data, photo, expected, frozen, test, path):
    for arm in p.ARMS:
        p.validate_top_sample(data['arms'][arm],photo)
        times=data['timing']['inference_map_transfer_ms'][arm]
        if len(times)!=12:
            raise ValueError('Incomplete paired timing')
        ev.latency_summary(times)
    if len(data['timing']['decode_ms'])!=12:
        raise ValueError('Incomplete decode timing')
    ev.latency_summary(data['timing']['decode_ms'])
    diff=p.check_shared(data['arms'][p.ARMS[0]],expected,frozen)
    if data['shared_maximum_replay_difference']!=diff:
        raise ValueError('Cached replay difference disagrees')
    if (not np.isfinite(data['minimum_candidate_minus_shared_map'])
            or data['minimum_candidate_minus_shared_map'] < -p.POLICY['map_monotonicity_tolerance']):
        raise ValueError('Restricted map unexpectedly below shared map')
    if test:
        with np.load(path/'maps.npz',allow_pickle=False) as z:
            maps=z['anomaly_maps']
            if set(z.files)!={'anomaly_maps','regions'} or maps.shape!=(12,256,256) or maps.dtype!=np.float32 or z['regions'].tolist()!=fit.REGIONS:
                raise ValueError('Cached candidate map layout differs')
            for array,row in zip(maps,data['arms'][p.ARMS[1]]['regions'],strict=True):
                if not np.isfinite(array).all() or np.any(array<0) or fit.candidate_image_score(0.,array,fit.CANDIDATE)!=row['top1']:
                    raise ValueError('Cached candidate map score differs')


def score_photo(source, photo, expected, frozen, routing, runtime, recovered, stage, test):
    rows={a:[] for a in p.ARMS}
    timing={'decode_ms':[], 'inference_map_transfer_ms':{a:[] for a in p.ARMS},
            'scope':'decode once; each arm includes preprocessing, feature extraction, native search/map, GPU transfer; excludes model load, JSON/NPZ, HTTP'}
    candidate_maps=[]
    minimum=0.
    started=time.perf_counter()
    for i,row in enumerate(photo['regions']):
        before=time.perf_counter()
        image=ev.read_tile(source,row,recovered) if test else fit.checked_image(source,row)
        timing['decode_ms'].append((time.perf_counter()-before)*1000)
        maps={}
        try:
            for arm in (p.ARMS if i%2==0 else p.ARMS[::-1]):
                runtime.synchronize(); before=time.perf_counter()
                raw,array=(runtime.predict(image) if arm==p.ARMS[0] else runtime.predict_region(image,routing['region_indices'][row['region']]))
                runtime.synchronize()
                timing['inference_map_transfer_ms'][arm].append((time.perf_counter()-before)*1000)
                array=np.asarray(array,dtype=np.float32)
                if array.shape!=(256,256) or not np.isfinite(array).all() or np.any(array<0):
                    raise ValueError('Invalid anomaly map')
                maps[arm]=array
                rows[arm].append({k:row[k] for k in ('region','member')} | {'top1':fit.candidate_image_score(raw,array,fit.CANDIDATE)})
        finally:
            image.close()
        minimum=min(minimum,float(np.min(maps[p.ARMS[1]]-maps[p.ARMS[0]])))
        if minimum < -p.POLICY['map_monotonicity_tolerance']:
            raise ValueError('Same-region search violated distance/map monotonicity')
        if test:
            candidate_maps.append(maps[p.ARMS[1]])
    arms={a:{k:photo[k] for k in ('key','photo_id','label')} | {'regions':rows[a]} for a in p.ARMS}
    timing['paired_photo_ms']=(time.perf_counter()-started)*1000
    result={'arms':arms,'timing':timing,'shared_maximum_replay_difference':p.check_shared(arms[p.ARMS[0]],expected,frozen),
            'minimum_candidate_minus_shared_map':minimum}
    if test:
        with (stage/'maps.npz').open('xb') as stream:
            np.savez_compressed(stream,anomaly_maps=np.stack(candidate_maps),regions=np.array(fit.REGIONS))
    fit.write_json(stage/'sample.json',result)
    check_sample(result,photo,expected,frozen,test,stage)


def candidate_gate(source, preparation, routing, runtime):
    rows=preparation['calibration_rows'][:12]
    # Each region's subset has a different size. Warm every region before timing.
    for row in rows:
        with fit.checked_image(source,row) as image:
            runtime.predict_region(image,routing['region_indices'][row['region']])
    times=[]
    for row in rows:
        runtime.synchronize(); before=time.perf_counter()
        with fit.checked_image(source,row) as image:
            raw,array=runtime.predict_region(image,routing['region_indices'][row['region']])
        runtime.synchronize(); times.append((time.perf_counter()-before)*1000)
        if np.shape(array)!=(256,256) or not np.isfinite(array).all() or np.any(np.asarray(array)<0):
            raise ValueError('Invalid candidate normal gate map')
    timing=ev.latency_summary(times)
    if timing['mean']>ev.PROBE['maximum_mean_tile_ms'] or timing['p95_higher']>ev.PROBE['maximum_p95_tile_ms']:
        raise ValueError('Severe same-region runtime regression before candidate scoring')
    return {'passed':True,'normal_tiles':len(rows),'warmup_tiles':len(rows),'latency_ms':timing,
            'scope':'normal decode+single candidate; regression gate, not service SLA'}


def verify_gates(root, receipts, expected_bank):
    hashes={r['gate_sha256'] for r in receipts}
    gates=[json.loads(f.read_bytes()) for f in sorted(root.glob('attempt-*/gate.json')) if fit.sha256_file(f) in hashes]
    if len(gates)!=len(hashes) or not all(g['normal']['passed'] and g['feature']['direct_replay_exact'] and g['bank']==expected_bank for g in gates):
        raise ValueError('Missing or invalid contributing normal/feature gate')
    return gates


def verify_candidate_gates(root, receipts, routing):
    hashes={r['candidate_gate_sha256'] for r in receipts}
    gates=[]
    for path in sorted(root.glob('attempt-*/candidate-gate.json')):
        if fit.sha256_file(path) in hashes:
            gate=json.loads(path.read_bytes())
            if (not gate['passed'] or gate['normal_tiles']!=12 or gate['warmup_tiles']!=12
                    or gate['routing_sha256']!=fit.digest(routing)
                    or gate['shared_gate_sha256']!=fit.sha256_file(path.parent/'gate.json')
                    or any(r['gate_sha256']!=gate['shared_gate_sha256'] for r in receipts
                           if r['candidate_gate_sha256']==fit.sha256_file(path))):
                raise ValueError('Invalid contributing candidate normal gate')
            gates.append(gate)
    if len(gates)!=len(hashes):
        raise ValueError('Missing contributing candidate normal gate')
    return gates


def result_html(report, scores):
    rows=[]
    labels={p.ARMS[0]:'기존 A · 전체 영역 검색',p.ARMS[1]:'후보 A · 같은 영역만 검색'}
    for arm in p.ARMS:
        m=report['models'][arm]
        c=m['fixed_threshold']['counts'];rates=m['fixed_threshold']['rates']
        latency=report['latency_ms']['test'][arm]
        rows.append(f'<tr><td>{labels[arm]}</td><td>{c["tp"]} / {c["tp"]+c["fn"]}</td>'
                    f'<td>{c["fn"]}</td><td>{c["fp"]} / {c["fp"]+c["tn"]}</td>'
                    f'<td>{rates["recall"]:.1%}</td><td>{rates["overkill_rate"]:.1%}</td>'
                    f'<td>{latency["mean"]:.1f}</td><td>{latency["p95_higher"]:.1f}</td></tr>')
    changed=[]
    decision=lambda s:'검토 경고' if s=='REVIEW_REQUIRED' else '기준 미만 · 합격 아님'
    for before,after in zip(scores[p.ARMS[0]],scores[p.ARMS[1]],strict=True):
        if before['decision']!=after['decision']:
            changed.append(f"<tr><td>{html.escape(before['key'])}</td><td>{'불량' if before['label'] else '정상'}</td><td>{decision(before['decision'])}</td><td>{decision(after['decision'])}</td></tr>")
    pairs=report['paired_changes_from_shared_A']
    return ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>MPI 동일 영역 비교</title>'
            '<style>body{max-width:1100px;margin:32px auto;font:17px/1.7 sans-serif}td,th{border:1px solid #bbb;padding:10px}table{border-collapse:collapse}td{overflow-wrap:anywhere}</style>'
            '<h1>같은 영역의 정상 패치만 비교하면 좋아질까요?</h1><p>shared_A: 기존 공용 기준표. same_region_A: 같은 출처 영역만 검색. '
            '저장된 정상 벡터와 특징 추출기는 같습니다. 후보 기준은 같은 정상 개발·보정 자료로 다시 산출했습니다.</p>'
            '<p>이미 본 시험 자료의 탐색 비교입니다. 기준 미만도 자동 합격이 아닙니다. 서비스 모델을 바꾸지 않았습니다.</p>'
            '<table><tr><th>방식</th><th>잡은 불량</th><th>놓친 불량</th><th>정상 오경보</th><th>불량 검출률</th><th>정상 오경보율</th><th>평균(ms)</th><th>p95(ms)</th></tr>'+''.join(rows)+'</table>'
            '<p>시간은 원본 한 장의 12개 영역 처리 합계입니다. 공통 사진 읽기·모델 로딩·결과 저장·HTTP를 제외한 비교용 측정으로, 서비스 응답 시간이 아닙니다.</p>'
            f'<p>새로 잡은 불량 {pairs["defect"]["counts"]["new_only_review"]}장 / 새로 놓친 불량 {pairs["defect"]["counts"]["old_only_review"]}장. '
            f'새 정상 오경보 {pairs["normal"]["counts"]["new_only_review"]}장 / 줄어든 정상 오경보 {pairs["normal"]["counts"]["old_only_review"]}장.</p>'
            '<h2>판정이 바뀐 사진</h2>'
            '<table><tr><th>사진</th><th>라벨</th><th>기존</th><th>후보</th></tr>'+''.join(changed)+'</table>'
            '<h2>다음 확인</h2><p>추가 검출과 새 미탐, 정상 오경보, 시간·메모리를 함께 검토합니다. 미탐만 줄었다고 성공으로 선언하지 않습니다. '
            '이번에는 마스크 성능을 다시 계산하지 않았습니다. 후보 시험 지도는 로컬 캐시에 저장돼 있습니다.</p></html>')


def execute(root,source,checkpoint,recovered_path,config,preparation,records,previous,reviewed,development,
            expected_bank,identity,git,handoff,runtime_factory=RegionRuntime):
    signature=fit.digest(identity)
    frozen=reviewed['calibration.json']
    audit={v['key']:v for v in previous['test-audit.json']['photos']}
    cohorts={'development':old.normal_photos(development['photo_ids'],development['rows'],'development'),
             'calibration':old.normal_photos(preparation['calibration_photo_ids'],preparation['calibration_rows'],'calibration'),
             'test':[audit[s['key']] for s in reviewed['scores.json'][p.trace.A]]}
    expected={s['key']:s for g in cohorts for s in reviewed['raw-scores.json'][g]}
    train_ids=preparation['training_photo_ids']
    with fit.locked_run(root):
        if (root/'integrity-block.json').exists():
            raise ValueError('Prior bank integrity failure; preserve and request review')
        if not (root/'identity').exists():
            if any((root/n).exists() for n in ('bank','sources','routing','contract','result',*cohorts)):
                raise ValueError('Outputs without frozen identity; preserved')
            publish(root,'identity',signature,{'plan.json':{'identity':identity,'git':git,'development':development}})
        read_phase(root,'identity',signature,['plan.json'])
        for n in ('sources',*cohorts):
            (root/n).mkdir(exist_ok=True)
        bank=None
        if (root/'bank').exists():
            data,_=read_phase(root,'bank',signature,['bank.json']);bank=data['bank.json']
            if bank['fingerprint']!=expected_bank:
                raise ValueError('Cached bank fingerprint differs')
        sources={}
        for i,pid in enumerate(train_ids):
            name=f'photo-{i+1:03d}'
            if (root/'sources'/name).exists():
                data,receipt=read_phase(root/'sources',name,signature,['origins.json'])
                rows=preparation['training_rows'][i*12:(i+1)*12]
                if bank is None or receipt['rows_sha256']!=fit.digest(rows) or receipt['bank_sha256']!=fit.digest(bank) or data['origins.json']['photo_id']!=pid:
                    raise ValueError('Cached source scan inputs differ')
                sources[i]=(data['origins.json'],receipt)
        routing=origins=None
        if (root/'routing').exists():
            data,_=read_phase(root,'routing',signature,['routing.json','origins.json'])
            if len(sources)!=len(train_ids) or bank is None:
                raise ValueError('Routing without complete original source scan')
            routing,origins=p.make_routing(bank['hashes'],[sources[i][0] for i in range(len(train_ids))],preparation)
            if data!={'routing.json':routing,'origins.json':origins}:
                raise ValueError('Cached routing does not reproduce')
        cached={g:{} for g in cohorts}
        receipts=[v[1] for v in sources.values()]
        for group,photos in cohorts.items():
            for i,photo in enumerate(photos):
                name=f'photo-{i+1:03d}'
                if (root/group/name).exists():
                    data,receipt=read_phase(root/group,name,signature,sample_files(group=='test'))
                    if routing is None or receipt['routing_sha256']!=fit.digest(routing) or receipt['photo_sha256']!=fit.digest(photo):
                        raise ValueError('Cached paired sample inputs differ')
                    check_sample(data['sample.json'],photo,expected[photo['key']],frozen,group=='test',root/group/name)
                    cached[group][i]=(data['sample.json'],receipt);receipts.append(receipt)
        verify_gates(root,receipts,expected_bank)
        sample_receipts=[v[1] for g in cached.values() for v in g.values()]
        verify_candidate_gates(root,sample_receipts,routing)
        contract=None
        if (root/'contract').exists():
            data,_=read_phase(root,'contract',signature,['calibration.json'])
            if any(len(cached[g])!=len(cohorts[g]) for g in ('development','calibration')):
                raise ValueError('Candidate contract lacks complete normals')
            contract=p.calibrate(*[[cached[g][i][0]['arms'][p.ARMS[1]] for i in range(len(cohorts[g]))] for g in ('development','calibration')])
            if data['calibration.json']!=contract:
                raise ValueError('Candidate normal-only contract changed')
        if cached['test'] and contract is None:
            raise ValueError('Test samples without frozen candidate contract')
        if any(v[1]['contract_sha256']!=fit.digest(contract) for v in cached['test'].values()):
            raise ValueError('Test scores bound to different candidate contract')
        if (root/'result').exists():
            data,_=read_phase(root,'result',signature,RESULT_FILES)
            if contract is None or any(len(cached[g])!=len(cohorts[g]) for g in cohorts):
                raise ValueError('Completed report missing verified sample caches')
            recalculated,scores=p.compare([cached['test'][i][0]['arms'][p.ARMS[1]] for i in range(len(cohorts['test']))],
                                         contract,reviewed['scores.json'][p.trace.A])
            if (data['scores.json']!=scores or data['candidate-calibration.json']!=contract
                    or data['shared-A-calibration.json']!=frozen or data['routing.json']!=routing
                    or data['origins.json']!=origins
                    or any(data['comparison.json'][k]!=v for k,v in recalculated.items())):
                raise ValueError('Completed report does not reproduce from verified caches')
            old.export(root/'result',handoff)
            return data['comparison.json']
        attempt=Path(tempfile.mkdtemp(prefix='attempt-',dir=root))
        runtime=None;before_bank=None;gate_sha=None;candidate_gate_sha=None
        try:
            need_runtime=len(sources)<len(train_ids) or any(len(cached[g])<len(cohorts[g]) for g in cohorts)
            if need_runtime:
                print('1/5 Loading frozen MPI bank and verifying shared normal scores (no fitting)...',flush=True)
                runtime=runtime_factory(config);runtime.load(checkpoint,records['fit.json']['checkpoint_context'])
                if runtime.environment()!=records['reload.json']['receipt']['runtime']:
                    raise ValueError('Runtime differs from reviewed checkpoint')
                before_bank=runtime.bank_fingerprint()
                if before_bank!=expected_bank:
                    raise ValueError('Loaded bank differs from reviewed patch trace')
                normal=ev.normal_probe(source,preparation,records['reload.json']['photos'],runtime)
                if not normal['passed']:
                    raise ValueError(f"Shared normal gate failed: {normal['reasons']}")
                with fit.checked_image(source,preparation['training_rows'][0]) as image:
                    feature=runtime.verify_feature_path(image)
                if not feature['direct_replay_exact']:
                    raise ValueError('Feature replay differs')
                actual={'fingerprint':before_bank,'hashes':runtime.all_bank_hashes()}
                if len(actual['hashes'])!=before_bank['shape'][0]:
                    raise ValueError('Incomplete bank vector hashes')
                if bank is None:
                    bank=actual;publish(root,'bank',signature,{'bank.json':bank})
                elif bank!=actual:
                    raise ValueError('Bank row hashes changed')
                fit.write_json(attempt/'gate.json',{'normal':normal,'feature':feature,'bank':before_bank,'process_id':os.getpid(),'runtime':runtime.environment()})
                gate_sha=fit.sha256_file(attempt/'gate.json')
            wanted=set(bank['hashes'])
            print(f'2/5 Exact origin scan: reusing {len(sources)}/{len(train_ids)} original normal photos...',flush=True)
            for i,pid in enumerate(train_ids):
                if i not in sources:
                    rows=preparation['training_rows'][i*12:(i+1)*12];found=[];count=0;started=time.perf_counter()
                    for row in rows:
                        with fit.checked_image(source,row) as image:
                            values,indices=runtime.candidate_features(image)
                        if np.shape(values)!=(1024,config['feature_dimension']) or indices!=fit.spatial_patch_indices(64,2):
                            raise ValueError('Original candidate shape/stride differs')
                        found.extend(p.trace.find_origins(values,indices,row,wanted));count+=len(indices)
                    chunk={'photo_id':pid,'candidate_embeddings_checked':count,'origins':found,'seconds':time.perf_counter()-started}
                    receipt=publish(root/'sources',f'photo-{i+1:03d}',signature,{'origins.json':chunk},
                                    {'rows_sha256':fit.digest(rows),'bank_sha256':fit.digest(bank),'gate_sha256':gate_sha})
                    sources[i]=(chunk,receipt)
                print(f'normal origin scan {i+1}/{len(train_ids)}',flush=True)
            routing,origins=p.make_routing(bank['hashes'],[sources[i][0] for i in range(len(train_ids))],preparation)
            if not (root/'routing').exists():
                publish(root,'routing',signature,{'routing.json':routing,'origins.json':origins})
            if runtime is not None:
                fit.write_json(attempt/'candidate-gate.json',candidate_gate(source,preparation,routing,runtime) |
                               {'routing_sha256':fit.digest(routing),'shared_gate_sha256':gate_sha})
                candidate_gate_sha=fit.sha256_file(attempt/'candidate-gate.json')
                runtime.reset_peak()
            samples={};recovered=None
            for group,photos in cohorts.items():
                if group=='test':
                    contract=p.calibrate(samples['development'],samples['calibration'])
                    if not (root/'contract').exists():
                        publish(root,'contract',signature,{'calibration.json':contract})
                    data,_=read_phase(root,'contract',signature,['calibration.json'])
                    if data['calibration.json']!=contract:
                        raise ValueError('Candidate contract differs before test')
                    if len(cached[group])<len(photos):
                        recovered=ev.recovered_bytes(recovered_path,previous)
                print(f"{'4/5' if group=='test' else '3/5'} {group}: {len(photos)} photos, cached {len(cached[group])}...",flush=True)
                for i,photo in enumerate(photos):
                    if i not in cached[group]:
                        name=f'photo-{i+1:03d}'
                        with fit.phase_stage(root/group,name) as stage:
                            score_photo(source,photo,expected[photo['key']],frozen,routing,runtime,recovered,stage,group=='test')
                            receipt=fit.seal_phase(stage,name,signature,{'photo_sha256':fit.digest(photo),'routing_sha256':fit.digest(routing),
                                'gate_sha256':gate_sha,'candidate_gate_sha256':candidate_gate_sha,
                                'contract_sha256':fit.digest(contract) if group=='test' else None},sample_files(group=='test'))
                        data=json.loads((root/group/name/'sample.json').read_bytes());cached[group][i]=(data,receipt)
                    if group=='test' and cached[group][i][1]['contract_sha256']!=fit.digest(contract):
                        raise ValueError('Test scores bound to different candidate contract')
                    if (i+1)%10==0 or i+1==len(photos):
                        print(f'{group} {i+1}/{len(photos)}',flush=True)
                samples[group]=[cached[group][i][0]['arms'][p.ARMS[1]] for i in range(len(photos))]
                if group=='development':
                    p.scoring.fit_normalization(samples[group])
                if runtime is not None and runtime.bank_fingerprint()!=before_bank:
                    raise ValueError('Bank changed during paired scoring')
            report,scores=p.compare(samples['test'],contract,reviewed['scores.json'][p.trace.A])
            receipts=[v[1] for v in sources.values()]+[v[1] for g in cached.values() for v in g.values()]
            gates=verify_gates(root,receipts,expected_bank)
            if runtime is not None:
                if runtime.bank_fingerprint()!=before_bank:
                    raise ValueError('Bank changed before publication')
                fit.write_json(attempt/'runtime.json',{'process_id':os.getpid(),'bank_unchanged':True,'memory':runtime.memory()})
            all_samples={g:[cached[g][i][0] for i in range(len(cohorts[g]))] for g in cohorts}
            report.update(status='region_search_evaluated_exploratory',policy=p.POLICY,execution_sha256=signature,
                checkpoint_sha256=ev.CHECKPOINT_SHA256,bank_fingerprint=expected_bank,
                source_scan_seconds=sum(sources[i][0]['seconds'] for i in sources),
                per_region_bank_vectors=routing['per_region_bank_vectors'],
                shared_maximum_replay_difference=max(s['shared_maximum_replay_difference'] for g in all_samples.values() for s in g),
                latency_ms={g:{a:ev.latency_summary([sum(s['timing']['inference_map_transfer_ms'][a]) for s in ps]) for a in p.ARMS} for g,ps in all_samples.items()},
                latency_scope='paired alternating arm order; 12-tile inference+preprocess+map transfer per photo, excludes shared decode, JSON/NPZ, load, HTTP; not production SLA',
                mixed_attempts=len(gates)>1,normal_gates=gates,
                candidate_gates=verify_candidate_gates(root,[v[1] for g in cached.values() for v in g.values()],routing),
                runtime_attempts=[json.loads(f.read_bytes()) for f in sorted(root.glob('attempt-*/runtime.json'))],
                interrupted_attempts=[json.loads(f.read_bytes()) for f in sorted(root.glob('attempt-*/failure.json'))],
                gpu_memory_scope='peak across both arms; not exclusive per-arm memory',
                location_note='Classification comparison only; no masks read or new localization claim. Candidate maps kept locally. Future location analysis must exclude all of defect/0006.',
                training_performed=False,existing_threshold_changed=False,service_model_changed=False,
                automatic_pass_allowed=False,confirmatory_claim_allowed=False)
            print('5/5 Saving exploratory comparison; no automatic winner or service update...',flush=True)
            with fit.phase_stage(root,'result') as stage:
                payload={'comparison.json':report,'scores.json':scores,'raw-scores.json':all_samples,'candidate-calibration.json':contract,
                         'shared-A-calibration.json':frozen,'routing.json':routing,'origins.json':origins,
                         'plan.json':json.loads((root/'identity/plan.json').read_bytes())}
                for n,v in payload.items():fit.write_json(stage/n,v)
                (stage/'index.html').write_text(result_html(report,scores),encoding='utf-8')
                fit.write_json(stage/'handoff-manifest.json',{n:fit.sha256_file(stage/n) for n in RESULT_FILES if n!='handoff-manifest.json'})
                fit.seal_phase(stage,'result',signature,{},RESULT_FILES)
            old.export(root/'result',handoff)
            return report
        except BaseException as exc:
            fit.write_json(attempt/'failure.json',{'error':f'{type(exc).__name__}: {exc}',
                'completed_source_photos':len(sources),'completed_photos':{g:len(v) for g,v in cached.items()},
                'process_id':os.getpid(),'automatic_pass_allowed':False,'note':'Partial run, no valid full comparison. Preserve old data/model and completed phases.'})
            print(f'Failure record: {attempt / "failure.json"}',file=sys.stderr)
            raise
        finally:
            if runtime is not None:
                if before_bank is not None:
                    try: unchanged=runtime.bank_fingerprint()==before_bank
                    except Exception: unchanged=False
                    if not unchanged:
                        fit.write_json(root/'integrity-block.json',{'reason':'bank_changed_or_unverifiable','attempt':attempt.name})
                runtime.release()


def run(project,archive):
    handoffs=project/'artifacts/handoff'
    config=fit.load_config(project/'configs/mpi-pcb-normal-bank-pilot-v1.toml')
    preparation=fit.load_preparation(handoffs/'mpi-pcb-normal-bank-preparation-v1.zip',config)
    records=ev.load_fit(handoffs/'mpi-pcb-normal-bank-fit-v1.zip',project,config,preparation)
    previous,_=ev.load_prior(project,handoffs)
    reviewed=ev.checked_bundle(handoffs/'mpi-pcb-score-probe-v1.zip',p.trace.SCORE_ZIP_SHA256)
    trace=ev.checked_bundle(handoffs/'mpi-pcb-patch-trace-v1.zip',p.TRACE_SHA256)
    for name,h in trace['plan.json']['identity']['implementation']['source_files_lf_sha256'].items():
        if hashlib.sha256((project/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()!=h:
            raise ValueError(f'Prior implementation changed: {name}')
    if trace['A-calibration.json']!=reviewed['calibration.json'] or p.scoring.calibration_contract(reviewed['raw-scores.json']['development'],reviewed['raw-scores.json']['calibration'])!=reviewed['calibration.json']:
        raise ValueError('Reviewed A contract differs')
    if [s['key'] for s in reviewed['scores.json'][p.trace.A]]!=previous['cohort-amendment.json']['photo_keys']:
        raise ValueError('Fixed test cohort changed')
    fingerprints=ev.checked_bundle(handoffs/'mpi-pcb-normal-bank-preparation-v1.zip',fit.PREPARATION_SHA256)['normal-fingerprints.json']
    development=p.scoring.select_development(fingerprints,preparation,previous['test-audit.json']['photos'])
    if development!=reviewed['plan.json']['development']:
        raise ValueError('Normal development cohort differs')
    if (len(development['photo_ids'])!=128 or len(preparation['calibration_photo_ids'])!=169
            or len(preparation['training_photo_ids'])!=64 or '0411' not in preparation['calibration_photo_ids']
            or len(reviewed['scores.json'][p.trace.A])!=109
            or sum(s['label'] for s in reviewed['scores.json'][p.trace.A])!=55):
        raise ValueError('Fixed cohort size or retained calibration photo differs')
    expected_bank=trace['diagnosis.json']['normal_probes'][0]['bank']
    if expected_bank['shape']!=p.POLICY['bank_shape']:
        raise ValueError('Reviewed bank shape differs')
    checkpoint=project/'artifacts/mpi-pcb-normal-bank-pilot-v1/fit/model.pt'
    if fit.sha256_file(checkpoint)!=ev.CHECKPOINT_SHA256 or fit.sha256_file(archive)!=fit.EXPECTED_ARCHIVE_SHA256:
        raise ValueError('Source archive/checkpoint differs; no replacement or repair')
    implementation=ev.run_identity(project)
    for name in ('compare_mpi_regions.py','mpi_region_search_protocol.py','mpi_region_runtime.py',
                 'trace_mpi_patches.py','mpi_patch_trace_protocol.py','mpi_patch_trace_cards.py',
                 'probe_mpi_scores.py','mpi_score_probe_protocol.py','mpi_score_probe_cards.py'):
        implementation['source_files_lf_sha256'][f'scripts/{name}']=hashlib.sha256((project/'scripts'/name).read_bytes().replace(b'\r\n',b'\n')).hexdigest()
    identity={'policy':p.POLICY,'implementation':implementation,'expected_bank':expected_bank,
              'development_sha256':fit.digest(development),'shared_contract_sha256':fit.digest(reviewed['calibration.json'])}
    with zipfile.ZipFile(archive) as source:
        return execute(project/'artifacts/analyses'/p.NAME,source,checkpoint,
            project/'artifacts/evaluations/mpi-pcb-external-evaluation-v2/recovery/grid1-3-defect-0054.png',
            config,preparation,records,previous,reviewed,development,expected_bank,identity,fit.git_state(project),handoffs/f'{p.NAME}.zip')


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument('--archive',type=Path,required=True)
    args=parser.parse_args(argv)
    try:
        report=run(args.project.resolve(),args.archive.resolve())
        print(json.dumps({k:report[k] for k in ('status','models','paired_changes_from_shared_A','per_region_bank_vectors','latency_ms')},indent=2,ensure_ascii=False,allow_nan=False))
        return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}',file=sys.stderr)
        return 1


if __name__=='__main__':
    raise SystemExit(main())
