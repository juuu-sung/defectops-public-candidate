"""Frozen paired saved-map review. CPU only; no training, inference or threshold fitting."""
from __future__ import annotations

import argparse
import hashlib
import html
import io
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import mpi_patchcore_balanced_bank as b
from scripts import run_mpi_patchcore_balanced_bank as runner
from scripts.export_mpi_patchcore_scaling_review import checked_png, write_bundle

fit, p, engine = b.fit, b.p, runner.engine
NAME = 'mpi-patchcore-balanced-map-review-v1'
INPUT_SHA = '823ff7b3dcde78fa6d5d2d739edea31c042c9592c5fd60e1b5db7f21c07bc09b'
LOWER_REGIONS = ('grid4-1_s1024', 'grid4-3_s1024')
EXPECTED_NEW = {'test': ['good/0019', 'good/0021', 'good/0028'],
                'validation': ['validation/0288', 'validation/0975', 'validation/1382',
                               'validation/1415', 'validation/1549']}
EXPECTED_DEFECTS = ['defect/0008', 'defect/0024', 'defect/0025', 'defect/0028', 'defect/0033']


def load(project, handoffs):
    data = b.development.evidence.checked_bundle(handoffs / (b.NAME + '.zip'), INPUT_SHA)
    plan = data['plan.json']
    # New diagnostic files are not part of the old execution snapshot. Every old
    # dependency must remain exact; never rebuild a completed run with a new identity.
    for name, sha in plan['sources_lf_sha256'].items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe frozen source path')
        if hashlib.sha256((project / relative).read_bytes().replace(b'\r\n', b'\n')).hexdigest() != sha:
            raise ValueError('Frozen execution source changed: ' + name)
    source, reconstructed = b.prepare(project, handoffs)
    reconstructed['sources_lf_sha256'] = plan['sources_lf_sha256']
    if reconstructed != plan:
        raise ValueError('Frozen source/cohort/contract identity differs')
    candidate = {**data['normal-scores.json'], **data['evaluation/scores.json']}
    b.validate_raw(candidate, source['cohorts'])
    b.contract_replay(candidate, data['contract.json'])
    if (b.compare(source, candidate, data['contract.json']) != data['evaluation/comparison.json']
            or data['evaluation/result.json']['status'] != 'balanced_bank_observed_comparison_completed'):
        raise ValueError('Completed comparison does not replay')
    return data, source, candidate


def selection(data):
    comparison = data['evaluation/comparison.json']
    if {g: sorted(v) for g, v in comparison['new_false_positives'].items()} != EXPECTED_NEW:
        raise ValueError('Reviewed eight new false positives changed')
    if comparison['resolved_false_positives'] != {'test': ['good/0018'], 'validation': []}:
        raise ValueError('Resolved normal control changed')
    entries = {}

    def add(group, key, reason, focus):
        entry = entries.setdefault((group, key), {'group': group, 'key': key, 'reasons': [], 'focus_regions': []})
        entry['reasons'].append(reason)
        entry['focus_regions'] = sorted(set(entry['focus_regions']) | set(focus), key=fit.REGIONS.index)

    for group, keys in EXPECTED_NEW.items():
        for key in keys:
            focus = [next(x for x in comparison['predictions'][m][group] if x['key'] == key)['winning_region']
                     for m in ('baseline', 'candidate')]
            add(group, key, 'new_false_positive', focus)
    add('test', 'good/0018', 'resolved_false_positive_control', ['grid4-1_s1024'])
    location = data['evaluation/location.json']['cases']
    for region in LOWER_REGIONS:
        ranked = []
        for case in location:
            if not case['location_eligible'] or not case['key'].startswith('defect/'):
                continue
            tile = next(t for t in case['tiles'] if t['region'] == region)
            old, new = [tile[m]['location']['within_tile_auroc'] for m in ('baseline', 'candidate')]
            if old is not None and new is not None:
                ranked.append((new - old, case['key']))
        for delta, key in sorted(ranked)[:2]:
            if delta >= 0:
                raise ValueError('Expected two declining lower-region cases')
            add('test', key, 'top2_location_decline:' + region, [region, 'grid3-2_s1024'])
    add('test', 'defect/0024', 'prior_weak_defect_control', [*LOWER_REGIONS, 'grid3-2_s1024'])
    if sorted(e['key'] for e in entries.values() if e['key'].startswith('defect/')) != EXPECTED_DEFECTS:
        raise ValueError('Frozen lower-region review selection changed; no replacement')
    result = sorted(entries.values(), key=lambda e: (e['group'], e['key']))
    if len(result) != 14:
        raise ValueError('Expected fourteen unique photos')
    return result


def arithmetic(sample, method):
    tiles = []
    for row, calibration in zip(sample['regions'], method['regions'], strict=True):
        if row['region'] != calibration['region']:
            raise ValueError('Region order changed')
        raw, center, scale = row['top1'], calibration['center'], calibration['scale']
        if not all(np.isfinite(x) for x in (raw, center, scale, method['threshold'])) or scale <= 0:
            raise ValueError('Invalid score or calibration')
        score = (raw - center) / scale
        boundary = center + scale * method['threshold']
        tiles.append({'region': row['region'], 'raw': raw, 'center': center, 'scale': scale,
                      'score': score, 'threshold': method['threshold'], 'raw_boundary': boundary,
                      'raw_margin': raw - boundary, 'normalized_margin': score - method['threshold']})
    winner = max(tiles, key=lambda t: t['score'])
    return {'score': winner['score'], 'winning_region': winner['region'],
            'alert': winner['score'] >= method['threshold'], 'tiles': tiles}


def analyze(data, source, candidate):
    methods = {'baseline': data['plan.json']['baseline_contract'], 'candidate': data['contract.json']}
    raw = {'baseline': source['reference'], 'candidate': candidate}
    cases = []
    for entry in selection(data):
        group, key = entry['group'], entry['key']
        samples = {m: next(x for x in raw[m][group] if x['key'] == key) for m in raw}
        same = {m: arithmetic(samples[m], methods[m]) for m in methods}
        cross = {scores: {contract: arithmetic(samples[scores], methods[contract])
                         for contract in methods} for scores in raw}
        for m in methods:
            original = next(x for x in data['evaluation/comparison.json']['predictions'][m][group] if x['key'] == key)
            if any(same[m][k] != original[k] for k in ('score', 'winning_region', 'alert')):
                raise ValueError('Selected photo decision does not replay')
        cases.append({**entry, 'label': samples['candidate']['label'], 'models': same,
                      'crossed_scores_and_contracts': cross})
    return {'status': 'fixed_balanced_bank_review_plan', 'cases': cases,
            'photos': 14, 'tiles': 168, 'new_false_positives': 8, 'resolved_controls': 1, 'defect_photos': 5,
            'crossed_contract_interpretation': 'Arithmetic counterfactual only; not causal attribution, a refit, or an adopted scoring rule',
            'selection_bias': 'Observed error/decline-selected cases, not a representative performance sample',
            'training': False, 'new_inference': False, 'threshold_changed': False,
            'automatic_pass_allowed': False, 'service_model_changed': False}


def read_pair(project, data, source, candidate, group, index):
    photo = source['cohorts'][group][index]
    baseline = runner.parent_maps(project, source, group, index)
    root = project / 'artifacts/analyses' / b.NAME / group
    name = f'photo-{index:03d}'
    signature = fit.digest(data['plan.json'])
    expected = data['evaluation/prediction-receipts.json'][f'{group}/{name}/receipt.json']
    if fit.read_phase(root, name, signature, ['sample.json', 'maps.npy']) != expected:
        raise ValueError('Balanced map receipt differs from reviewed ZIP')
    sample = engine.read_sample(root, name, signature, photo, data['fit-report.json']['checkpoint_sha256'])
    if sample != candidate[group][index]:
        raise ValueError('Balanced map/sample pairing changed')
    values = p.array(root / name / 'maps.npy', (12, 256, 256))
    for array in (baseline, values):
        if array.dtype != np.float32 or np.any(array < 0) or not np.isfinite(array).all():
            raise ValueError('Expected finite nonnegative native float32 maps')
    prior_receipts = {k.replace('\\', '/'): v for k, v in source['parent']['evaluation/prediction-receipts.json'].items()}
    return (baseline, values), {'baseline': prior_receipts[f'{group}/{name}/receipt.json'], 'candidate': expected}


def edge(mask):
    a = np.pad(mask, 1, constant_values=False)
    interior = a[1:-1, 1:-1] & a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:]
    return mask & ~interior


def heat(values, low, high):
    # Identical palette and numeric range for both models of a tile; never two auto-scales.
    ratio = np.zeros_like(values, dtype=np.float64) if high == low else (values - low) / (high - low)
    t = np.clip(ratio, 0, 1)
    return np.stack([255*t, 200*(1-np.abs(2*t-1)), 255*(1-t)], axis=-1).astype(np.uint8)


def card(source_png, pair, mask, key, region, numbers):
    low = min(float(v.min()) for v in pair); high = max(float(v.max()) for v in pair)
    image = Image.open(io.BytesIO(source_png)).convert('RGB')
    original = np.array(image)
    gt = np.zeros((256, 256), dtype=bool) if mask is None else edge(mask)
    original[gt] = [0, 255, 0]
    panels = [Image.fromarray(original)]
    for v in pair:
        pixels = (.55*np.array(image) + .45*heat(v, low, high)).astype(np.uint8)
        support = v >= np.partition(v.ravel(), -656)[-656]
        pixels[edge(support)] = [255, 0, 255]
        pixels[gt] = [0, 255, 0]
        panels.append(Image.fromarray(pixels))
    canvas = Image.new('RGB', (1152, 552), 'white'); draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=17); small = ImageFont.load_default(size=14)
    draw.text((10, 8), f'{key}  |  {region}', fill='black', font=font)
    for i, (title, panel) in enumerate(zip(('Original / GT', 'Global greedy baseline', 'Region-balanced candidate'), panels, strict=True)):
        x = i*384
        draw.text((x+8, 40), title, fill='black', font=font)
        canvas.paste(panel.resize((384,384), Image.Resampling.NEAREST), (x, 66))
        if i:
            v = numbers[i-1]
            draw.text((x+8, 457), f"raw {v['raw']:.4f} | boundary {v['raw_boundary']:.4f}", fill='black', font=small)
            draw.text((x+8, 478), f"z {v['score']:.4f} | threshold {v['threshold']:.4f}", fill='black', font=small)
    draw.text((8, 507), f'Shared RAW color range: {low:.4f} to {high:.4f} (blue to red); pair-specific, not across cards.', fill='black', font=small)
    draw.text((8, 530), 'Green: GT. Magenta: top656 support including ties, NOT a pixel decision threshold. Read-only review.', fill='black', font=small)
    output = io.BytesIO(); canvas.save(output, format='PNG')
    return output.getvalue(), [low, high]


def npy(value):
    out = io.BytesIO(); np.save(out, value, allow_pickle=False); return out.getvalue()


def index_html(report):
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>정상 대표 균형 비교</title>',
             '<style>body{max-width:1200px;margin:32px auto;padding:0 20px;font:17px/1.6 system-ui;color:#17212b}img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{padding:7px 14px;border:1px solid #ccc}section{margin:40px 0}summary{cursor:pointer}a{color:#005faf}</style>',
             '<h1>정상 대표 균형 비교</h1><p>저장 결과만 읽은 진단입니다. 새 학습·추론·경고선 변경은 없습니다.</p>',
             '<p>신규 오경보 8장, 해소 정상 1장, 불량 5장입니다. 오류를 보고 고른 사례이므로 전체 성능 표본이 아닙니다.</p>',
             '<p>왼쪽 원본, 가운데 기존 모델, 오른쪽 균형 모델입니다. 녹색은 실제 결함 위치, 자홍색은 상위656픽셀(동점 포함)이며 픽셀 판정선이 아닙니다. 두 지도는 같은 원점수 색 범위를 쓰지만 서로 다른 카드끼리 색을 직접 비교하지 않습니다.</p>',
             '<p>숫자: raw=영역 원점수, boundary=원점수 단위 경고 경계, z=보정 점수. 경고는 사진의12영역 최고 z로 결정됩니다. 아래 교차표는 원점수와 보정계약의 산술 영향을 구분할 뿐, 새 판정을 채택하거나 원인을 확정하지 않습니다.</p><ul>']
    for n, case in enumerate(report['cases']):
        parts.append(f'<li><a href="#p{n}">{html.escape(case["key"])}</a> — {html.escape(", ".join(case["reasons"]))}</li>')
    parts.append('</ul>')
    for n, case in enumerate(report['cases']):
        parts.append(f'<section id="p{n}"><h2>{html.escape(case["key"])}</h2><table><tr><th>점수 / 보정계약</th><th>기존 보정</th><th>균형 모델 보정</th></tr>')
        for scores, label in [('baseline', '기존 원점수'), ('candidate', '균형 모델 원점수')]:
            parts.append('<tr><td>'+label+'</td>')
            for contract in ('baseline', 'candidate'):
                r = case['crossed_scores_and_contracts'][scores][contract]
                parts.append(f'<td>{r["score"]:.4f} / {"경고" if r["alert"] else "기준 미만"}<br>{html.escape(r["winning_region"])}</td>')
            parts.append('</tr>')
        parts.append('</table><p>기준 미만은 자동 합격을 뜻하지 않습니다.</p>')
        for j, region in enumerate(fit.REGIONS):
            focus = region in case['focus_regions']
            parts.append(f'<details{" open" if focus else ""}><summary>{html.escape(region)}{" — 주요 비교 영역" if focus else ""}</summary><img loading="lazy" src="photos/{n:02d}/{j:02d}/card.png" alt="{html.escape(case["key"] + " " + region)}"></details>')
        parts.append('</section>')
    parts.append('<p>이미지·마스크는 개인 프로젝트 검토용입니다. 공개 재배포하지 마세요.</p></html>')
    return '\n'.join(parts).encode('utf-8')


def close_stats(actual, expected):
    """Float32 reduction metadata may vary by CPU; cached bytes/top1 use exact checks."""
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(close_stats(actual[k], v) for k, v in expected.items())
    if isinstance(expected, list):
        return len(actual) == len(expected) and all(close_stats(a, e) for a, e in zip(actual, expected))
    if isinstance(expected, float):
        return bool(np.isclose(actual, expected, rtol=1e-7, atol=1e-6))
    return actual == expected


def export(project, archive_path, data, source, candidate, report):
    if fit.sha256_file(archive_path) != data['plan.json']['archive_sha256']:
        raise ValueError('Original archive fingerprint changed')
    payload = {'scores.json': fit.canonical(report), 'index.html': index_html(report)}
    records = []
    location = {(c['group'], c['key']): c for c in data['evaluation/location.json']['cases']}
    with zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive member names')
        for n, case in enumerate(report['cases']):
            group, key = case['group'], case['key']
            if group not in ('test', 'validation') or key in b.POLICY['location_excluded']:
                raise ValueError('Unexpected/excluded review photo')
            index = next(i for i, photo in enumerate(source['cohorts'][group]) if photo['key'] == key)
            photo = source['cohorts'][group][index]
            pair, receipts = read_pair(project, data, source, candidate, group, index)
            masks = engine.ev.read_masks(archive, photo) if photo['label'] else [None]*12
            for j, row in enumerate(photo['regions']):
                prefix = f'photos/{n:02d}/{j:02d}'
                image = checked_png(archive, row)
                payload[prefix+'/source.png'] = image
                maps = [v[j] for v in pair]
                stats = {}
                for model, value in zip(('baseline', 'candidate'), maps, strict=True):
                    payload[prefix+'/'+model+'-map.npy'] = npy(value)
                    stats[model] = {k: v for k, v in b.evaluation.maps.map_stats(value, masks[j]).items() if not k.startswith('display_')}
                    if (group, key) in location:
                        expected = location[group, key]['tiles'][j]
                        if expected['region'] != row['region'] or not close_stats(stats[model], expected[model]):
                            raise ValueError('Paired map statistics differ from evaluation')
                if masks[j] is not None:
                    if row['mask_member'] is not None:
                        payload[prefix+'/mask-source.png'] = archive.read(row['mask_member'])
                    payload[prefix+'/mask.npy'] = npy(masks[j])
                numbers = [case['models'][m]['tiles'][j] for m in ('baseline', 'candidate')]
                png, limits = card(image, maps, masks[j], key, row['region'], numbers)
                payload[prefix+'/card.png'] = png
                records.append({'prefix': prefix, 'key': key, 'group': group, 'source_row': row,
                    'photo_sha256': fit.digest(photo), 'receipts': receipts, 'stats': stats,
                    'scores': dict(zip(('baseline','candidate'), numbers, strict=True)), 'shared_raw_display_range': limits})
            print(f'Read-only paired cards {n+1}/{len(report["cases"])} photos; all12 regions. No inference.', flush=True)
    files = ['scripts/review_mpi_patchcore_balanced_maps.py', 'scripts/run_mpi_patchcore_balanced_map_review.ps1',
             'docs/mpi-patchcore-balanced-map-review-v1.md']
    payload['review.json'] = fit.canonical({'status': 'balanced_bank_saved_maps_exported', 'records': records,
        'input_zip_sha256': INPUT_SHA, 'execution_sha256': fit.digest(data['plan.json']),
        'checkpoints': {'baseline': data['plan.json']['parent_checkpoint_sha256'], 'candidate': data['fit-report.json']['checkpoint_sha256']},
        'archive_sha256': data['plan.json']['archive_sha256'], 'contracts': {'baseline': data['plan.json']['baseline_contract'], 'candidate': data['contract.json']},
        'photos': len(report['cases']), 'tiles': len(records), 'new_inference': False, 'training': False,
        'threshold_changed': False, 'automatic_pass_allowed': False, 'service_model_changed': False,
        'sources_lf_sha256': {name: hashlib.sha256((project/name).read_bytes().replace(b'\r\n', b'\n')).hexdigest() for name in files},
        'display': 'Shared paired raw min/max, no per-model auto-scaling. Green GT; magenta top656 including ties, not pixel threshold.',
        'rights': 'Private project review only. Do not redistribute source images or masks.'})
    output = project / 'artifacts/handoff' / (NAME+'.zip')
    with fit.locked_run(project/'artifacts/analyses'/(NAME+'-export-lock')):
        write_bundle(output, payload)
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--handoffs', type=Path)
    parser.add_argument('--archive', type=Path)
    parser.add_argument('--scores-only', action='store_true')
    args = parser.parse_args(argv)
    if not args.scores_only and args.archive is None:
        parser.error('--archive is required unless --scores-only')
    data, source, candidate = load(args.project, args.handoffs or args.project/'artifacts/handoff')
    report = analyze(data, source, candidate)
    if args.scores_only:
        print(json.dumps(report, ensure_ascii=False, indent=2)); return 0
    output = export(args.project, args.archive, data, source, candidate, report)
    print(json.dumps({'status': 'exported_no_inference', 'photos': 14, 'tiles': 168, 'output': str(output)}, indent=2))
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve caches and ZIPs; do not retrain or change thresholds.', file=sys.stderr)
        raise SystemExit(1)
