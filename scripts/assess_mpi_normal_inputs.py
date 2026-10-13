"""Offline input observations alongside frozen model results; never a PASS gate."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import io
import json
from pathlib import Path
import sys
from uuid import uuid4
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import review_mpi_normal_appearance as census

fit = census.fit
NAME = 'mpi-normal-input-assessment-v1'
CENSUS_SHA = 'bbfb5c0fdd897a4cb7ca92183bf73b4cb13d00c29966171af4ec8bff949a4e59'
SOURCES = ['scripts/assess_mpi_normal_inputs.py', 'scripts/run_mpi_normal_input_assessment.ps1',
           'tests/test_mpi_normal_input_assessment.py', 'docs/inspection-input-assessability-v1.md']
STATES = ('INSUFFICIENT_INFORMATION', 'NEEDS_INPUT_REVIEW', 'NO_LISTED_ISSUE_DETECTED')
LABELS = dict(zip(STATES, ('필수 타일의 정보 없음', '입력 검토 필요', '제한 점검 이상 없음')))
POLICY = {'profile': 'mpi-fixed12-rgb256-v1', 'schema_version': 1,
          'whole_zero': 'all RGB values exactly zero in required tile',
          'partial_zero': 'any frame-connected exact zero; review only, not rejection',
          'automatic_pass_allowed': False, 'service_model_changed': False,
          'training_updates': 0, 'new_inference': 0, 'threshold_changed': False,
          'test_images_decoded': 0, 'test_masks_decoded': 0,
          'limits': ['Observed research normals; not independent validation.',
                     'No listed issue does not establish input suitability.',
                     'Near-black, blur, glare, wrong product and natural occlusion not assessed.',
                     'One empty tile does not prove all overlapping views are empty.',
                     'No original FP is removed; holds are not detections.']}


def tile_state(statistics):
    area = statistics['connected_black_pixels']
    if type(area) is not int or not 0 <= area <= 65536:
        raise ValueError('Invalid connected black area')
    if area == 65536:
        return STATES[0], 'ALL_ZERO_REQUIRED_TILE'
    if area:
        return STATES[1], 'CONNECTED_ZERO_PRESENT'
    return STATES[2], None


def photo_state(regions):
    if (len(regions) != 12 or sorted(r['region'] for r in regions) != sorted(fit.REGIONS)
            or any(r['state'] not in STATES for r in regions)):
        raise ValueError('Missing, duplicate or unassessed required region')
    return min((r['state'] for r in regions), key=STATES.index)


def model_predictions(raw, contract):
    if [r['region'] for r in contract['regions']] != list(fit.REGIONS):
        raise ValueError('Contract region order differs')
    result = {}
    center = np.array([r['center'] for r in contract['regions']])
    scale = np.array([r['scale'] for r in contract['regions']])
    if not np.isfinite(center).all() or not np.isfinite(scale).all() or np.any(scale <= 0):
        raise ValueError('Invalid fixed contract')
    if not np.isfinite(contract['threshold']):
        raise ValueError('Invalid threshold')
    for group in ('test', 'validation'):
        result[group] = []
        for row in raw[group]:
            if [r['region'] for r in row['regions']] != list(fit.REGIONS):
                raise ValueError('Score region order differs')
            values = np.array([r['top1'] for r in row['regions']])
            if not np.isfinite(values).all():
                raise ValueError('Nonfinite saved model score')
            normalized = (values-center)/scale
            score = float(normalized.max())
            result[group].append({'key': row['key'], 'photo_id': row['photo_id'], 'label': row['label'],
                                  'score': score, 'alert': score >= contract['threshold'],
                                  'winning_region': fit.REGIONS[int(normalized.argmax())]})
    return result


def load_evidence(project, census_path, baseline_path):
    if fit.sha256_file(census_path) != CENSUS_SHA:
        raise ValueError('Frozen census ZIP differs')
    data, raw = census.parent.checked_input(baseline_path, project)
    groups = census.normal_groups(data['plan.json'])
    with zipfile.ZipFile(census_path) as z:
        manifest = json.loads(z.read('handoff-manifest.json'))
        if len(z.namelist()) != len(set(z.namelist())) or set(z.namelist()) != set(manifest)|{'handoff-manifest.json'}:
            raise ValueError('Census manifest membership differs')
        payload = {n: z.read(n) for n in manifest}
    if any(hashlib.sha256(payload[n]).hexdigest() != h for n, h in manifest.items()):
        raise ValueError('Census content checksum differs')
    plan, review = (json.loads(payload[n]) for n in ('plan.json', 'review.json'))
    if (groups != plan['groups'] or fit.digest(plan) != review['execution_sha256']
            or plan['policy'] != census.POLICY or review['policy'] != census.POLICY
            or plan['input_sha256'] != census.parent.INPUT_SHA
            or plan['archive_sha256'] != data['plan.json']['archive_sha256']
            or plan['checkpoint_sha256'] != data['fit-report.json']['checkpoint_sha256']):
        raise ValueError('Census identity differs')
    for n, h in plan['sources_lf_sha256'].items():
        if hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != h:
            raise ValueError('Census source differs: '+n)
    lookup = {(r['group'], r['photo_id'], r['region']): r for r in review['rows']}
    if len(lookup) != 6144 or len(review['rows']) != 6144:
        raise ValueError('Expected 6144 unique frozen rows')
    for g, photos in groups.items():
        for photo in photos:
            for source in photo['regions']:
                if lookup[g, photo['photo_id'], source['region']]['source'] != source:
                    raise ValueError('Frozen image source differs')
    if census.summarize(review['rows']) != review['summary']:
        raise ValueError('Census summary differs')
    images, verified = {}, set()
    for n, content in payload.items():
        if '/focus/' not in n and '/black-examples/' not in n:
            continue
        g = n.split('/')[0]
        region, pid = ((census.FOCUS, Path(n).stem) if '/focus/' in n
                       else Path(n).stem.rsplit('-', 1))
        key = (g, pid, region); row = lookup[key]
        check_pixels(content, row)
        images[key] = content; verified.add(key)
    predicted = model_predictions(raw, data['contract.json'])
    saved = data['evaluation/comparison.json']['predictions']['candidate']
    for group in predicted:
        if predicted[group] != saved[group]:
            raise ValueError('Frozen model predictions do not replay')
    counts = {'tp': sum(p['alert'] for p in predicted['test'] if p['label'] == 1),
              'fn': sum(not p['alert'] for p in predicted['test'] if p['label'] == 1),
              'fp54': sum(p['alert'] for p in predicted['test'] if p['label'] == 0),
              'tn54': sum(not p['alert'] for p in predicted['test'] if p['label'] == 0),
              'fp256': sum(p['alert'] for p in predicted['validation'])}
    if counts != data['evaluation/comparison.json']['counts']['candidate']:
        raise ValueError('Baseline counts differ')
    return groups, plan, lookup, images, verified, predicted['validation'], counts, data['contract.json']


def check_pixels(content, row):
    if hashlib.sha256(content).hexdigest() != row['source']['member_sha256']:
        raise ValueError('Image bytes differ')
    with Image.open(io.BytesIO(content)) as im:
        if im.format != 'PNG' or im.mode != 'RGB' or im.size != (256, 256):
            raise ValueError('Expected RGB256 PNG')
        im.verify()
    with Image.open(io.BytesIO(content)) as im:
        im.load(); rgb = np.asarray(im)
        if (hashlib.sha256(rgb.tobytes()).hexdigest() != row['source']['decoded_rgb_sha256']
                or census.black_statistics(rgb) != row['statistics']):
            raise ValueError('Image pixels or saved statistics differ')


def verify_archive(path, expected_sha, lookup, images, verified):
    if fit.sha256_file(path) != expected_sha:
        raise ValueError('Original archive differs')
    with zipfile.ZipFile(path) as z:
        if len(z.namelist()) != len(set(z.namelist())):
            raise ValueError('Duplicate original archive member')
        for i, (key, row) in enumerate(sorted(lookup.items())):
            content = census.checked_png(z, row['source'])
            check_pixels(content, row); verified.add(key)
            if tile_state(row['statistics'])[0] != STATES[2]:
                images[key] = content
            if (i+1) % 384 == 0:
                print(f'Original normal verification {i+1}/{len(lookup)}; CPU only', flush=True)


def assess(groups, lookup, verified, predictions):
    models = {p['photo_id']: p for p in predictions}
    if len(models) != len(predictions) or set(models) != {p['photo_id'] for p in groups['validation']}:
        raise ValueError('Validation model IDs differ')
    photos = []
    for group, inputs in groups.items():
        for photo in inputs:
            pid = photo['photo_id']; regions = []
            for source in photo['regions']:
                key = (group, pid, source['region']); row = lookup[key]
                state, reason = tile_state(row['statistics'])
                regions.append({'region': source['region'], 'state': state, 'reason': reason,
                                'statistics': row['statistics'], 'source': source,
                                'pixel_reverified_here': key in verified})
            photos.append({'group': group, 'photo_id': pid, 'state': photo_state(regions),
                           'regions': regions, 'model_result': models[pid] if group == 'validation' else None,
                           'human_assessment': 'NOT_RECORDED', 'automatic_pass_allowed': False})
    summaries = {}
    for group in groups:
        selected = [p for p in photos if p['group'] == group]
        cross = {s: dict.fromkeys(('alert', 'below_threshold', 'not_scored'), 0) for s in STATES}
        union = intersection = input_count = 0
        for p in selected:
            model = p['model_result']
            status = 'not_scored' if model is None else ('alert' if model['alert'] else 'below_threshold')
            cross[p['state']][status] += 1
            needs = p['state'] != STATES[2]; alert = model is not None and model['alert']
            input_count += needs; intersection += needs and alert; union += needs or alert
        summaries[group] = {'photos': len(selected), 'input_states': dict(Counter(p['state'] for p in selected)),
                            'cross_table': cross, 'input_followup_photos': input_count,
                            'input_and_model_alert': intersection, 'input_or_model_alert': union,
                            'interpretation': 'Triage count only; all photos still require human review; no PASS.'}
    return photos, summaries


def page(title, body):
    return ('<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
            f'<title>{html.escape(title)}</title><style>body{{font:16px system-ui;max-width:1100px;margin:32px auto;padding:0 20px;line-height:1.6}}'
            'table{border-collapse:collapse;width:100%}td,th{border:1px solid #bbb;padding:8px;text-align:left}'
            'img{width:256px;height:256px} .notice{background:#fff2cb;padding:12px}details{margin:12px 0}</style>'
            f'<h1>{html.escape(title)}</h1>{body}</html>').encode('utf-8')


def render(report, images):
    payload = {}; rows = []; model_rows = []
    for p in report['photos']:
        group, pid = p['group'], p['photo_id']; key = group+'/'+pid
        items = []
        for r in p['regions']:
            image_key = (group, pid, r['region']); img = images.get(image_key)
            name = f'images/{group}-{pid}-{r["region"]}.png'
            picture = '<p>이 타일 원본은 첨부되지 않았습니다. 아래는 저장 통계입니다.</p>'
            if img is not None:
                payload[name] = img; picture = f'<img src="../{name}" alt="{pid} {r["region"]} 원본">'
            evidence = '이 실행에서 픽셀 검산' if r['pixel_reverified_here'] else '고정된 연구실 통계 기록'
            items.append(f'<details><summary>{r["region"]} · {LABELS[r["state"]]} · 검은 픽셀 '
                         f'{r["statistics"]["connected_black_pixels"]:,} / 65,536</summary><p>{evidence}</p>{picture}</details>')
        case_name = f'cases/{group}-{pid}.html'
        payload[case_name] = page(key+' 입력 검토', '<p><a href="../index.html">목록으로</a></p>'
            '<p class="notice">제품 판정 아님. 필요한 검사 부위가 실제로 보이는지 확인하세요. '
            '흐림·반사·근접 검은색 등은 자동 점검 범위 밖입니다. 사람 의견 저장 기능은 없습니다.</p>'
            f'<h2>{LABELS[p["state"]]}</h2>'+''.join(items))
        rows.append(f'<tr><td>{group}</td><td><a href="{case_name}">{pid}</a></td><td>{LABELS[p["state"]]}</td></tr>')
        model = p['model_result']
        label = '결과 없음 (학습자료)' if model is None else ('기존 모델 경고' if model['alert'] else '모델 점수 기준 미만 · 합격 아님')
        score = '—' if model is None else f'{model["score"]:.6f}'
        model_rows.append(f'<tr><td>{key}</td><td>{LABELS[p["state"]]}</td><td>{label}</td><td>{score}</td></tr>')
    mode = ('전체 원본 픽셀 재검산' if report['mode'] == 'archive' else
            '저장 기록 기반 분류 · 일부 첨부 원본만 픽셀 재검산')
    counts = ''.join(f'<li>{g}: {s["photos"]}장, 입력 확인 필요 {s["input_followup_photos"]}장</li>' for g, s in report['summary'].items())
    payload['index.html'] = page('MPI 정상 입력 검토 목록', f'<p class="notice">{mode}. 자동 합격·검사 제외 기능 없음.</p>'
        f'<p>원본 픽셀 재검산: {report["pixel_verification"]["unique_tiles"]:,} / 6,144 타일. '
        '목록 전체는 512사진입니다. 입력 검토를 먼저 진행한 뒤 별도 모델 대조표를 보세요.</p>'
        '<p><a href="model-comparison.html">기존 모델 결과 대조</a> · <a href="input-assessment.json">상세 기록</a></p>'
        f'<ul>{counts}</ul><table><tr><th>자료 역할</th><th>사진 ID</th><th>입력 상태</th></tr>'+''.join(rows)+'</table>')
    payload['model-comparison.html'] = page('입력 상태와 기존 모델 결과',
        '<p><a href="index.html">입력 검토 목록</a></p><p class="notice">기존 오경보 15/256은 그대로입니다. '
        '입력 보류를 검출 성공이나 오경보 감소로 계산하지 않습니다. 모델은 새로 실행하지 않았습니다.</p>'
        '<table><tr><th>사진</th><th>입력 상태</th><th>기존 모델</th><th>기존 점수</th></tr>'+''.join(model_rows)+'</table>')
    return payload


def run(project, census_path, baseline_path, archive=None):
    mode = 'archive' if archive is not None else 'records'
    root = project/'artifacts/analyses'/(NAME+'-'+mode)
    started = datetime.now(timezone.utc).isoformat()
    with fit.locked_run(root):
        try:
            groups, plan, lookup, images, verified, models, counts, contract = load_evidence(project, census_path, baseline_path)
            if archive is not None:
                verify_archive(archive, plan['archive_sha256'], lookup, images, verified)
            photos, summary = assess(groups, lookup, verified, models)
            identity = {'name': NAME, 'mode': mode, 'policy': POLICY, 'census_sha256': CENSUS_SHA,
                        'baseline_sha256': census.parent.INPUT_SHA, 'census_execution_sha256': fit.digest(plan),
                        'contract_sha256': fit.digest(contract),
                        'sources_lf_sha256': {n: hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for n in SOURCES}}
            report = {'schema_version': 1, 'status': 'offline_input_assessment_completed', 'mode': mode,
                      'execution_sha256': fit.digest(identity), 'policy': POLICY, 'photos': photos, 'summary': summary,
                      'pixel_verification': {'unique_tiles': len(verified), 'total_tiles': len(lookup)},
                      'baseline_counts_unchanged': counts, 'fixed_model_threshold': contract['threshold'],
                      'model_record_source': 'raw-bank evaluation/comparison.json predictions.candidate',
                      'human_assessment': 'NOT_RECORDED', 'production_approved': False}
            payload = render(report, images)
            payload.update({'plan.json': fit.canonical(identity), 'input-assessment.json': fit.canonical(report)})
            sig = fit.digest(identity)
            if not (root/'review').exists():
                with fit.phase_stage(root, 'review') as stage:
                    for n, raw in payload.items():
                        target = stage/n; target.parent.mkdir(parents=True, exist_ok=True)
                        with target.open('xb') as f: f.write(raw)
                    fit.seal_phase(stage, 'review', sig,
                                   {'started_at': started, 'completed_at': datetime.now(timezone.utc).isoformat()}, list(payload))
            receipt = fit.read_phase(root, 'review', sig, list(payload))
            if receipt['files'] != {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}:
                raise ValueError('Existing assessment differs; preserved')
            output = project/'artifacts/handoff'/(NAME+'-'+mode+'.zip')
            census.write_bundle(output, payload)
            return {'status': report['status'], 'mode': mode, 'summary': summary,
                    'pixel_verification': report['pixel_verification'], 'output': str(output),
                    'index': str(root/'review/index.html')}
        except Exception as exc:
            failure = {'status': 'TECHNICAL_ERROR', 'assessment_completed': False,
                       'started_at': started, 'failed_at': datetime.now(timezone.utc).isoformat(),
                       'error': f'{type(exc).__name__}: {exc}', 'automatic_pass_allowed': False}
            fit.write_json(root/('failure-'+uuid4().hex+'.json'), failure)
            raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--census', type=Path)
    parser.add_argument('--baseline', type=Path)
    parser.add_argument('--archive', type=Path, help='Optional full original verification; no test member reads')
    args = parser.parse_args(argv); project = args.project.resolve(); handoff = project/'artifacts/handoff'
    try:
        result = run(project, args.census or handoff/'mpi-normal-appearance-census-v1.zip',
                     args.baseline or handoff/'mpi-patchcore-raw-bank-v1.zip', args.archive)
        print(json.dumps(result, ensure_ascii=False, indent=2)); return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve files; no model changes.', file=sys.stderr); return 1


if __name__ == '__main__':
    raise SystemExit(main())
