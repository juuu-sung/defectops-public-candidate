"""CPU-only census of frozen normal PNGs. No model, masks or test images."""
from __future__ import annotations

import argparse
from collections import deque
import hashlib
import html
import io
import json
import platform
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter
import PIL

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'src')]
from scripts import compare_mpi_patchcore_scale_shrinkage as parent
from scripts.export_mpi_patchcore_scaling_review import checked_png, write_bundle

fit = parent.fit
NAME = 'mpi-normal-appearance-census-v1'
FOCUS = 'grid1-2_s1024'
SOURCES = ['scripts/review_mpi_normal_appearance.py',
           'scripts/run_mpi_normal_appearance.ps1',
           'docs/mpi-normal-appearance-census-v1.md',
           'tests/test_mpi_normal_appearance.py',
           'scripts/compare_mpi_patchcore_scale_shrinkage.py',
           'scripts/export_mpi_patchcore_scaling_review.py']
POLICY = {
    'groups': 'all256 training and all256 validation normal photos; twelve regions each',
    'black_rule': 'R=G=B=0, 4-connected to image frame; NOT a validity mask',
    'neighbor_band': 'existing square dilation radius2, nonblack pixels only; not tuned',
    'boundary_length': 'count horizontal/vertical black-to-nonblack adjacent pairs; exclude image frame',
    'spatial_grid': '4x4 fixed64px cells; counts only, not cropping',
    'sheets': 'all grid1-2 PNGs by ascending photo ID,16 per page,256px unchanged',
    'examples': 'largest connected-black area per group/region, positive only; ties photo ID',
    'inference_calls': 0, 'training_updates': 0, 'test_images_decoded': 0,
    'test_masks_decoded': 0, 'threshold_changed': False,
    'automatic_pass_allowed': False, 'service_model_changed': False,
    'interpretation': 'post-hoc descriptive coverage census; no significance or automatic exclusion',
}


def black_statistics(rgb):
    if rgb.shape != (256, 256, 3) or rgb.dtype != np.uint8:
        raise ValueError('Expected uint8 RGB256 image')
    zero = np.all(rgb == 0, axis=2)
    connected = np.zeros((256, 256), bool)
    queue = deque()
    frame = [(0, x) for x in range(256)] + [(255, x) for x in range(256)]
    frame += [(y, 0) for y in range(256)] + [(y, 255) for y in range(256)]
    for y, x in frame:
        if zero[y, x] and not connected[y, x]:
            connected[y, x] = True; queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for yy, xx in ((y-1, x), (y+1, x), (y, x-1), (y, x+1)):
            if 0 <= yy < 256 and 0 <= xx < 256 and zero[yy, xx] and not connected[yy, xx]:
                connected[yy, xx] = True; queue.append((yy, xx))
    area = int(connected.sum())
    ys, xs = np.nonzero(connected)
    dilated = np.asarray(Image.fromarray(connected.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(5))) > 0
    return {
        'exact_black_pixels': int(zero.sum()), 'connected_black_pixels': area,
        'connected_black_fraction': area/65536,
        'disconnected_black_pixels': int((zero & ~connected).sum()),
        'bbox_xyxy': [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())] if area else None,
        'centroid_xy': [float(xs.mean()), float(ys.mean())] if area else None,
        'interface_edges': int(np.count_nonzero(connected[1:] != connected[:-1]) +
                               np.count_nonzero(connected[:, 1:] != connected[:, :-1])),
        'adjacent_nonblack_2px': int(np.count_nonzero(dilated & ~zero)),
        'interior192_black_pixels': int(connected[32:224, 32:224].sum()),
        'grid4x4_pixels': connected.reshape(4, 64, 4, 64).sum(axis=(1, 3)).tolist(),
        'frame_black_pixels': {'top': int(connected[0].sum()), 'bottom': int(connected[-1].sum()),
                               'left': int(connected[:, 0].sum()), 'right': int(connected[:, -1].sum())},
    }


def normal_groups(plan):
    train = {}
    for row in plan['training']['rows']:
        train.setdefault(row['photo_id'], []).append(row)
    groups = {'training': [{'photo_id': k, 'label': 0, 'regions': v} for k, v in train.items()],
              'validation': plan['cohorts']['validation']}
    id_sets = []
    for group, photos in groups.items():
        ids = [x['photo_id'] for x in photos]
        if len(ids) != 256 or len(set(ids)) != 256:
            raise ValueError('Expected256 distinct normal photos in '+group)
        id_sets.append(set(ids))
        for photo in photos:
            if photo['label'] != 0 or sorted(r['region'] for r in photo['regions']) != sorted(fit.REGIONS):
                raise ValueError('Expected normal with exact12 regions')
            for row in photo['regions']:
                expected = f"extracted/{row['region']}/train/good/{photo['photo_id']}.png"
                if row['photo_id'] != photo['photo_id'] or row['member'] != expected:
                    raise ValueError('Refusing non-frozen training-pool member')
        groups[group] = sorted(photos, key=lambda p: p['photo_id'])
    if id_sets[0] & id_sets[1] or id_sets[0] != set(plan['training']['photo_ids']):
        raise ValueError('Normal roles overlap or training ID mismatch')
    return groups


def prepare(project, path):
    data, _ = parent.checked_input(path, project)
    groups = normal_groups(data['plan.json'])
    identity = {'name': NAME, 'policy': POLICY, 'input_sha256': parent.INPUT_SHA,
                'parent_execution_sha256': parent.EXECUTION_SHA,
                'archive_sha256': data['plan.json']['archive_sha256'],
                'checkpoint_sha256': data['fit-report.json']['checkpoint_sha256'],
                'environment': {'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__},
                'groups': groups,
                'sources_lf_sha256': {n: hashlib.sha256((project/n).read_bytes().replace(b'\r\n', b'\n')).hexdigest()
                                      for n in SOURCES}}
    return groups, identity


def describe(values):
    return {'count': len(values), 'minimum': min(values), 'maximum': max(values),
            'mean': float(np.mean(values)), 'median': float(np.median(values))}


def summarize(rows):
    result = {}
    for group in sorted({r['group'] for r in rows}):
        subset = [r for r in rows if r['group'] == group]
        ids = sorted({r['photo_id'] for r in subset})
        photo_totals = [sum(r['statistics']['connected_black_pixels'] for r in subset if r['photo_id'] == pid) for pid in ids]
        def stats(items):
            return {'tiles': len(items), 'tiles_with_connected_black': sum(r['statistics']['connected_black_pixels'] > 0 for r in items),
                    'area_pixels_all_tiles': describe([r['statistics']['connected_black_pixels'] for r in items]),
                    'interface_edges_all_tiles': describe([r['statistics']['interface_edges'] for r in items]),
                    'mean_grid4x4_black_pixels': np.mean([r['statistics']['grid4x4_pixels'] for r in items], axis=0).tolist(),
                    'tiles_touching_frame': {side: sum(r['statistics']['frame_black_pixels'][side] > 0 for r in items)
                                             for side in ('top', 'bottom', 'left', 'right')}}
        result[group] = {**stats(subset), 'photos': len(ids),
                         'photos_with_connected_black': sum(v > 0 for v in photo_totals),
                         'summed_tile_area_per_photo': describe(photo_totals),
                         'by_region': {region: stats([r for r in subset if r['region'] == region])
                                       for region in fit.REGIONS}}
    return result


def png_bytes(image):
    out = io.BytesIO(); image.save(out, format='PNG'); return out.getvalue()


def contact_sheet(group, entries, page):
    canvas = Image.new('RGB', (1072, 1184), 'white')
    draw = ImageDraw.Draw(canvas); font = ImageFont.load_default(size=14)
    draw.text((12, 8), f'{group} | {FOCUS} | page{page:02d} | fixed ID order; no ranking', fill='black', font=font)
    draw.text((12, 30), 'Original RGB256 tiles. No maps, normality judgement, masking or model changes.', fill='black', font=font)
    for i, (pid, raw) in enumerate(entries):
        x, y = 12+(i % 4)*264, 60+(i//4)*280
        draw.text((x, y), pid, fill='black', font=font)
        with Image.open(io.BytesIO(raw)) as im: canvas.paste(im, (x, y+20))
    return png_bytes(canvas)


def run(project, path, archive_path):
    groups, identity = prepare(project, path)
    if fit.sha256_file(archive_path) != identity['archive_sha256']:
        raise ValueError('Original archive differs')
    root = project/'artifacts/analyses'/NAME
    sig = fit.digest(identity)
    with fit.locked_run(root), zipfile.ZipFile(archive_path) as archive:
        if len(archive.namelist()) != len(set(archive.namelist())):
            raise ValueError('Duplicate archive member')
        payload = {'plan.json': fit.canonical(identity)}; rows = []; sheets = []; examples = []
        for group, photos in groups.items():
            focus = []; largest = {}
            for i, photo in enumerate(photos):
                for row in photo['regions']:
                    raw = checked_png(archive, row)
                    with Image.open(io.BytesIO(raw)) as im: stats = black_statistics(np.asarray(im))
                    rows.append({'group': group, 'photo_id': photo['photo_id'], 'region': row['region'],
                                 'source': row, 'statistics': stats})
                    region = row['region']; area = stats['connected_black_pixels']
                    if area > 0 and (region not in largest or area > largest[region][0]):
                        largest[region] = (area, photo['photo_id'], raw)
                    if region == FOCUS:
                        focus.append((photo['photo_id'], raw))
                        payload[f'{group}/focus/{photo["photo_id"]}.png'] = raw
                if (i+1) % 16 == 0: print(f'{group} normal census {i+1}/{len(photos)}; CPU only', flush=True)
            for start in range(0, len(focus), 16):
                page = start//16+1; name = f'{group}/sheets/page-{page:02d}.png'
                payload[name] = contact_sheet(group, focus[start:start+16], page); sheets.append(name)
            for region, (area, pid, raw) in largest.items():
                name = f'{group}/black-examples/{region}-{pid}.png'
                payload[name] = raw; examples.append({'group': group, 'region': region, 'photo_id': pid,
                                                      'connected_black_pixels': area, 'path': name})
        report = {'status': 'normal_appearance_census_completed_no_inference', 'execution_sha256': sig,
                  'rows': rows, 'summary': summarize(rows), 'sheets': sheets, 'black_examples': examples,
                  'policy': POLICY, 'limits': ['Exact zeros miss near-black fill and do not establish origin.',
                    'Twelve tiles may overlap; summed area is not unique original-photo area.',
                    'Presence, area and visual sheets do not prove feature-space coverage.',
                    'Known validation is descriptive, not new independent generalization.'],
                  'rights': 'Private review; do not publish images or sheets.'}
        payload['review.json'] = fit.canonical(report)
        links = ''.join(f'<li><a href="{html.escape(n, quote=True)}">{html.escape(n)}</a></li>' for n in sheets)
        exlinks = ''.join(f'<li><a href="{e["path"]}">{e["group"]} {e["region"]} {e["photo_id"]}</a></li>' for e in examples)
        overview = ''.join(f'<tr><td>{g}</td><td>{s["photos"]}</td>'
                          f'<td>{s["photos_with_connected_black"]}</td>'
                          f'<td>{s["tiles_with_connected_black"]} / {s["tiles"]}</td>'
                          f'<td>{s["area_pixels_all_tiles"]["mean"]:.2f}</td></tr>'
                          for g, s in report['summary'].items())
        payload['index.html'] = ('<!doctype html><html lang="ko"><meta charset="utf-8"><title>정상 외관 목록</title>'
            '<h1>학습과 검증 정상 외관 목록</h1><p>고정 ID순서의 원본입니다. 정상 판정이나 데이터 제외 기준이 아닙니다.</p>'
            '<table border="1" cellpadding="8"><tr><th>집단</th><th>사진 수</th><th>검은 영역 포함 사진</th>'
            '<th>검은 영역 포함 타일</th><th>전체 타일의 평균 검은 픽셀 수</th></tr>'+overview+'</table>'
            '<p>타일 간 중복 원본 영역이 있어 독립 표본이나 원본 고유 면적으로 해석하지 않습니다. '
            '<a href="review.json">전체 수치와 원본 지문</a></p>'
            '<h2>원통형 부품 영역 전체 목록</h2><ul>'+links+'</ul><h2>영역별 검은 면적 최대 예시</h2>'
            '<p>대표적인 빈도나 원인을 보장하지 않습니다. 빈도는 review.json을 확인하세요.</p><ul>'+exlinks+'</ul></html>').encode('utf-8')
        if not (root/'review').exists():
            with fit.phase_stage(root, 'review') as stage:
                for name, raw in payload.items():
                    target = stage/name; target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open('xb') as stream: stream.write(raw)
                fit.seal_phase(stage, 'review', sig, {}, list(payload))
        receipt = fit.read_phase(root, 'review', sig, list(payload))
        if receipt['files'] != {n: hashlib.sha256(v).hexdigest() for n, v in payload.items()}:
            raise ValueError('Existing result differs; preserved')
        output = project/'artifacts/handoff'/(NAME+'.zip'); write_bundle(output, payload)
        return {'status': report['status'], 'normal_photos': sum(len(p) for p in groups.values()),
                'normal_tiles': len(rows), 'sheets': len(sheets), 'inference_calls': 0, 'output': str(output)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=ROOT)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--archive', type=Path, required=True)
    args = parser.parse_args(argv); project = args.project.resolve()
    try:
        result = run(project, args.input or project/'artifacts/handoff/mpi-patchcore-raw-bank-v1.zip', args.archive)
        print(json.dumps(result, indent=2)); return 0
    except Exception as exc:
        print(f'ERROR: {type(exc).__name__}: {exc}. Preserve outputs; no model/data changes.', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
