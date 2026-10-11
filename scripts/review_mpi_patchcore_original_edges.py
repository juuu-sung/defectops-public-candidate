"""Read-only boundary audit of original saved raw-bank maps. No model inference."""
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
from scripts import compare_mpi_patchcore_scale_shrinkage as parent
from scripts import run_mpi_patchcore_scaling as engine
from scripts.export_mpi_patchcore_scaling_review import checked_png, write_bundle

fit, p, ev = parent.fit, parent.p, engine.ev
NAME = 'mpi-patchcore-original-edge-review-v1'
EXPECTED_FP = ['0147', '0274', '0275', '0276', '0288', '0335', '0337', '0361',
               '0656', '0706', '0961', '0984', '0998', '1390', '1436']
SOURCES = ['scripts/review_mpi_patchcore_original_edges.py',
           'scripts/compare_mpi_patchcore_scale_shrinkage.py',
           'scripts/run_mpi_patchcore_original_edge_review.ps1',
           'scripts/install_mpi_patchcore_original_edge_review.ps1',
           'tests/test_mpi_patchcore_original_edges.py',
           'docs/mpi-patchcore-original-edge-review-v1.md']
BORDER = 32
Y, X = np.indices((256, 256))
OUTER = (X < BORDER) | (X >= 256-BORDER) | (Y < BORDER) | (Y >= 256-BORDER)


def decision(sample, contract):
    return p.comparison.c.scoring.score(sample, contract)


def prepare(project, path):
    data, raw = parent.checked_input(path, project)
    if data['evaluation/result.json']['status'] != 'raw_bank_observed_comparison_completed':
        raise ValueError('Expected completed raw-bank evaluation')
    predictions = {g: [{**{k: s[k] for k in ('key', 'label', 'photo_id')},
                        **decision(s, data['contract.json'])} for s in raw[g]] for g in raw}
    if predictions != data['evaluation/comparison.json']['predictions']['candidate']:
        raise ValueError('Frozen predictions do not replay')
    fps = sorted(s['photo_id'] for s in predictions['validation'] if s['alert'])
    if fps != EXPECTED_FP or len(raw['validation']) != 256:
        raise ValueError('Expected exact fifteen alerts in256 validation normals')
    if sum(s['label'] for s in raw['test']) != 55:
        raise ValueError('Expected all55 observed defect photos')
    regions = sorted({s['winning_region'] for s in predictions['validation'] if s['alert']},
                     key=fit.REGIONS.index)
    if data['evaluation/location.json']['location_excluded'] != ['defect/0006']:
        raise ValueError('Known incomplete-GT exclusion changed')
    identity = {'name': NAME, 'input_sha256': parent.INPUT_SHA,
        'parent_execution_sha256': parent.EXECUTION_SHA,
        'checkpoint_sha256': data['fit-report.json']['checkpoint_sha256'],
        'archive_sha256': data['plan.json']['archive_sha256'],
        'contract': data['contract.json'], 'false_positive_ids': fps, 'focus_regions': regions,
        'normal_controls': 'all241 non-alert validation normals, no subsampling',
        'defects': 'all55 maps; defect/0006 excluded from GT location analysis as before',
        'boundary_width': BORDER, 'top_pixels': 656,
        'tie_rule': 'stable ascending flat-index argsort; last656; ties recorded',
        'selection': '15 normal winners and all positive-GT tiles in their regions; no top-N picking',
        'inference_calls': 0, 'training_updates': 0, 'threshold_changed': False,
        'automatic_pass_allowed': False, 'service_model_changed': False,
        'sources_lf_sha256': {n: hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                              for n in SOURCES},
        'limits': 'post-hoc descriptive audit, not causal proof or crop/threshold selection'}
    return data, raw, identity


def map_stats(values, mask=None):
    if (values.shape != (256, 256) or values.dtype != np.float32
            or not np.isfinite(values).all() or np.any(values < 0)):
        raise ValueError('Expected finite nonnegative native float32 map')
    flat = values.ravel()
    ids = np.argsort(flat, kind='stable')[-656:]
    y, x = np.divmod(ids, 256)
    cutoff = float(flat[ids[0]])
    peak = int(np.argmax(flat)); py, px = divmod(peak, 256)
    equal = flat == cutoff
    above = flat > cutoff
    needed = 656-int(above.sum())
    edge_above = int(np.count_nonzero(above & OUTER.ravel()))
    edge_equal = int(np.count_nonzero(equal & OUTER.ravel()))
    inner_equal = int(np.count_nonzero(equal & ~OUTER.ravel()))
    out = {'top1': float(np.mean(np.partition(flat.astype(np.float64), -656)[-656:])),
        'top656_outer32_fraction': float(np.mean(OUTER.ravel()[ids])),
        'top656_outer32_tie_bounds': [(edge_above+max(0, needed-inner_equal))/656,
                                    (edge_above+min(needed, edge_equal))/656],
        'cutoff_tie_pixels': int(equal.sum()), 'top656_cutoff': cutoff,
        'top656_sides': {'top': float(np.mean(y < 32)), 'bottom': float(np.mean(y >= 224)),
                         'left': float(np.mean(x < 32)), 'right': float(np.mean(x >= 224))},
        'top656_bbox_xyxy': [int(x.min()), int(y.min()), int(x.max()), int(y.max())],
        'peak_xy': [px, py], 'peak_in_outer32': bool(OUTER[py, px]),
        'outer_max': float(values[OUTER].max()), 'inner_max': float(values[~OUTER].max()),
        'gt': None}
    if mask is not None:
        if mask.shape != values.shape or mask.dtype != np.bool_:
            raise ValueError('Expected native boolean GT mask')
        total = int(mask.sum()); edge = int(np.count_nonzero(mask & OUTER))
        out['gt'] = {'pixels': total, 'outer32_pixels': edge,
                     'outer32_fraction': edge/total if total else None,
                     'top656_gt_pixels': int(np.count_nonzero(mask.ravel()[ids])),
                     'top656_outer32_gt_pixels': int(np.count_nonzero((mask & OUTER).ravel()[ids]))}
    return out


def read_saved(project, data, raw, group, i):
    photo = data['plan.json']['cohorts'][group][i]
    root = project/'artifacts/analyses'/parent.raw_bank.NAME/group
    name = f'photo-{i:03d}'
    receipts = {k.replace('\\', '/'): v for k, v in data['evaluation/prediction-receipts.json'].items()}
    receipt = fit.read_phase(root, name, parent.EXECUTION_SHA, ['sample.json', 'maps.npy'])
    if receipt != receipts[f'{group}/{name}/receipt.json']:
        raise ValueError('Saved map receipt differs from reviewed handoff')
    sample = engine.read_sample(root, name, parent.EXECUTION_SHA, photo,
                                data['fit-report.json']['checkpoint_sha256'])
    if sample != raw[group][i]:
        raise ValueError('Saved sample differs from reviewed scores')
    return photo, sample, p.array(root/name/'maps.npy', (12, 256, 256)), receipt


def png_bytes(image):
    out = io.BytesIO(); image.save(out, format='PNG'); return out.getvalue()


def array_bytes(values):
    out = io.BytesIO(); np.save(out, values, allow_pickle=False); return out.getvalue()


def heat(values, maximum):
    ratio = np.clip(values/max(maximum, 1e-12), 0, 1)
    colors = np.array([[20, 30, 100], [25, 150, 205], [255, 225, 50], [220, 35, 20]])
    return np.rint(np.stack([np.interp(ratio, np.linspace(0, 1, 4), colors[:, i])
                             for i in range(3)], -1)).astype(np.uint8)


def outline(mask):
    a = np.pad(mask, 1)
    return mask & ~(a[:-2, 1:-1] & a[2:, 1:-1] & a[1:-1, :-2] & a[1:-1, 2:])


def card(raw, values, mask, key, region, stats, score, threshold, maximum):
    with Image.open(io.BytesIO(raw)) as image:
        rgb = np.asarray(image).copy()
    blend = np.rint(.55*rgb+.45*heat(values, maximum)).astype(np.uint8)
    support = np.zeros((256, 256, 3), np.uint8)
    support.reshape(-1, 3)[np.argsort(values.ravel(), kind='stable')[-656:]] = [255, 100, 30]
    if mask is not None:
        for panel in (rgb, blend, support): panel[outline(mask)] = [0, 255, 0]
    canvas = Image.new('RGB', (1088, 380), 'white'); draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    draw.text((12, 8), f'{key} | {region} | photo {score:.4f}, threshold {threshold:.4f}', fill='black', font=font)
    for i, (pixels, title) in enumerate(zip([rgb, heat(values, maximum), blend, support],
                                ['Original / GT', 'Raw distance map', 'Overlay / GT', 'Top656 / GT'])):
        im = Image.fromarray(pixels); ImageDraw.Draw(im).rectangle((32, 32, 223, 223), outline='yellow', width=1)
        x = 12+i*270; draw.text((x, 36), title, fill='black', font=font); canvas.paste(im, (x, 58))
    draw.text((12, 326), f'Outer32 top656: {stats["top656_outer32_fraction"]:.3f}. Yellow: boundary; green: GT; orange: ranked pixels, NOT GT.', fill='black', font=font)
    draw.text((12, 350), f'Raw scale 0..{maximum:.3f}, shared within photo. No cropped scoring, pixel threshold, or model changes.', fill='black', font=font)
    return png_bytes(canvas)


def summary(cases, focus_regions):
    normals = [c for c in cases if c['group'] == 'validation']
    result = {'outer32_area_fraction': float(OUTER.mean()), 'normal_groups': {}, 'by_region': {}}
    def describe(values):
        return {'count': len(values), 'mean': float(np.mean(values)) if values else None,
                'median': float(np.median(values)) if values else None,
                'minimum': min(values) if values else None, 'maximum': max(values) if values else None}
    for label, alert in [('false_alert', True), ('non_alert_control', False)]:
        subset = [c for c in normals if c['decision']['alert'] == alert]
        result['normal_groups'][label] = describe([c['tiles'][fit.REGIONS.index(c['decision']['winning_region'])]['top656_outer32_fraction'] for c in subset])
    for j, region in enumerate(fit.REGIONS):
        result['by_region'][region] = {label: describe([c['tiles'][j]['top656_outer32_fraction'] for c in normals
                                          if c['decision']['alert'] == alert])
                                     for label, alert in [('false_alert', True), ('non_alert_control', False)]}
    gt_tiles = [(c, t) for c in cases if c['location_eligible'] for t in c['tiles']
                if t['gt'] is not None and t['gt']['pixels'] > 0]
    for name, tiles in [('all_gt_positive', gt_tiles), ('fp_winning_regions_gt_positive',
                        [(c, t) for c, t in gt_tiles if t['region'] in focus_regions])]:
        result[name] = {'tiles': len(tiles), 'photos': len({c['key'] for c, t in tiles}),
            'tiles_with_outer32_gt': sum(t['gt']['outer32_pixels'] > 0 for c, t in tiles),
            'photos_with_outer32_gt': len({c['key'] for c, t in tiles if t['gt']['outer32_pixels'] > 0}),
            'gt_pixels': sum(t['gt']['pixels'] for c, t in tiles),
            'outer32_gt_pixels': sum(t['gt']['outer32_pixels'] for c, t in tiles)}
    result['interpretation'] = 'Descriptive, observed cases. Region tiles are not independent photos; no significance or automatic candidate gate.'
    return result


def index_html(exports):
    sections = []
    for prefix, title in [('validation/', '정상 오경보 15장'), ('defect/', '대응 영역의 실제 결함')]:
        links = ''.join(f'<li><a href="{html.escape(e["prefix"], quote=True)}/card.png">'
                        f'{html.escape(e["key"])} · {html.escape(e["region"])}</a></li>'
                        for e in exports if e['key'].startswith(prefix))
        sections.append(f'<h2>{title}</h2><ul>{links}</ul>')
    return ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>원본 경계 반응 검토</title>'
            '<body><h1>저장된 원본 지도 검토</h1><p>노란선: 바깥32픽셀 경계 · 초록선: 실제 결함 GT · '
            '주황색: 점수 계산 상위656픽셀. 사진마다 색상 최댓값은 다릅니다. '
            '테두리 제외나 새 판정 기준을 적용한 결과가 아닙니다.</p>'
            + ''.join(sections) + '</body></html>').encode('utf-8')


def run(project, path, archive_path):
    data, raw, identity = prepare(project, path)
    if fit.sha256_file(archive_path) != identity['archive_sha256']:
        raise ValueError('Original archive changed')
    root = project/'artifacts/analyses'/NAME
    sig = fit.digest(identity); contract = data['contract.json']
    with fit.locked_run(root), zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive entries')
        payload = {'plan.json': fit.canonical(identity)}; cases = []; exports = []
        for group in ('validation', 'test'):
            for i, sample in enumerate(raw[group]):
                if group == 'test' and not sample['label']: continue
                photo, sample, maps, receipt = read_saved(project, data, raw, group, i)
                pred = decision(sample, contract)
                eligible = bool(photo['label']) and photo['key'] != 'defect/0006'
                masks = ev.read_masks(archive, photo) if eligible else [None]*12
                tiles = []
                for j, (values, mask) in enumerate(zip(maps, masks, strict=True)):
                    stats = map_stats(values, mask)
                    if stats['top1'] != sample['regions'][j]['top1']:
                        raise ValueError('Independent top656 arithmetic differs')
                    tiles.append({'region': fit.REGIONS[j], **stats})
                case = {'group': group, 'key': photo['key'], 'photo_id': photo['photo_id'],
                        'label': photo['label'], 'decision': pred, 'location_eligible': eligible,
                        'tiles': tiles, 'source_receipt': receipt, 'source_photo': photo}
                cases.append(case)
                normal_alert = group == 'validation' and pred['alert']
                selected = ([fit.REGIONS.index(pred['winning_region'])] if normal_alert else
                            [j for j, t in enumerate(tiles) if eligible and t['region'] in identity['focus_regions']
                             and t['gt'] is not None and t['gt']['pixels'] > 0])
                # Carry all12 original normal tiles/maps for checking the photo winner.
                if normal_alert:
                    prefix = photo['key']; payload[f'{prefix}/maps.npy'] = array_bytes(maps)
                    for j, row in enumerate(photo['regions']):
                        payload[f'{prefix}/original-{j:02d}.png'] = checked_png(archive, row)
                maximum = float(maps.max())
                for j in selected:
                    row = photo['regions'][j]; prefix = f'{photo["key"]}/{j:02d}'
                    source = checked_png(archive, row)
                    payload[prefix+'/original.png'] = source
                    payload[prefix+'/map.npy'] = array_bytes(maps[j])
                    if eligible:
                        payload[prefix+'/mask-original.png'] = archive.read(row['mask_member'])
                        payload[prefix+'/mask.npy'] = array_bytes(masks[j])
                    payload[prefix+'/card.png'] = card(source, maps[j], masks[j], photo['key'], row['region'],
                                                      tiles[j], pred['score'], contract['threshold'], maximum)
                    exports.append({'prefix': prefix, 'key': photo['key'], 'region': row['region'],
                                    'raw_map_range': [0., maximum]})
                if len(cases) % 25 == 0: print(f'Saved-map audit {len(cases)}/311; CPU only', flush=True)
        payload['index.html'] = index_html(exports)
        payload['review.json'] = fit.canonical({'status': 'original_edge_review_completed_no_inference',
            'execution_sha256': sig, 'identity': identity, 'cases': cases, 'exported_tiles': exports,
            'summary': summary(cases, identity['focus_regions']),
            'rights': 'Private project review only. Do not publish original images, masks or cards.'})
        if not (root/'review').exists():
            with fit.phase_stage(root, 'review') as stage:
                for name, value in payload.items():
                    target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as f: f.write(value)
                fit.seal_phase(stage, 'review', sig, {}, list(payload))
        receipt = fit.read_phase(root, 'review', sig, list(payload))
        if receipt['files'] != {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}:
            raise ValueError('Existing review differs; preserved')
        output = project/'artifacts/handoff'/(NAME+'.zip'); write_bundle(output, payload)
        return {'status': 'original_edge_review_completed_no_inference',
                'normal_photos': sum(c['group'] == 'validation' for c in cases),
                'defect_photos': sum(c['label'] == 1 for c in cases),
                'gt_excluded': ['defect/0006'], 'cards': len(exports),
                'model_inference_calls': 0, 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args(argv); project = args.project.resolve()
    try:
        print(json.dumps(run(project, args.input or project/'artifacts/handoff'/(parent.raw_bank.NAME+'.zip'),
                             args.archive), indent=2)); return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no model or threshold changes.', file=sys.stderr)
        return 1


if __name__ == '__main__': raise SystemExit(main())
