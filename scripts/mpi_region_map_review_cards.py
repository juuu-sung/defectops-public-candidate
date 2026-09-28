"""Paired raw-map displays on one shared fixed scale, plus normalized score margins."""
import html

import numpy as np
from PIL import Image, ImageDraw

from scripts import mpi_region_map_review_protocol as q

LABELS = {"new_fn": "새로 놓친 불량", "new_fp": "새 정상 오경보", "new_tp": "새로 잡은 불량",
          "cleared_fp": "줄어든 정상 오경보", "persistent_fn": "계속 놓친 불량",
          "retained_tp": "계속 검출한 불량", "retained_fp": "기존 정상 오경보", "retained_tn": "정상 대조"}


def pixels(original, values, mask):
    source = np.asarray(original).copy()
    edge = q.ev.maps.outline(mask)
    heat = q.ev.maps.colorize_anomaly_values(np.clip(values/q.POLICY["display_raw_map_cap"], 0, 1))
    overlay = np.rint(source*.6 + heat*.4).astype(np.uint8)
    support = source.copy()
    cutoff = np.partition(values.ravel(), -656)[-656]
    chosen = values >= cutoff
    support[chosen] = np.rint(support[chosen]*.5 + np.array([255, 0, 0])*.5).astype(np.uint8)
    for view in (source, overlay, support):
        view[edge] = (0, 255, 0)
    return source, overlay, support


def render(case, images, arrays, masks, page):
    card = Image.new("RGB", (1280, 130+6*314+24), "white")
    draw = ImageDraw.Draw(card)
    draw.text((8, 4), f"MPI {case['key']} | {case['group']} | page {page+1}/2", fill="black")
    draw.text((8, 22), "Green = official GT. Red = top 1% support including ALL cutoff ties (can exceed 656 pixels).", fill="black")
    draw.text((8, 40), f"Both raw-map colors use fixed cap {q.POLICY['display_raw_map_cap']:.4f}; NOT a pixel threshold or A score. Saturation possible.", fill="black")
    for i, arm in enumerate(q.p.ARMS):
        a = case['arms'][arm]
        draw.text((8, 58+i*18), f"{arm}: {a['decision']} | score {a['score']:.4f} / threshold {a['threshold']:.4f} | margin {a['margin']:+.4f}", fill="black")
    if not case['localization_eligible']:
        draw.text((8, 96), "Incomplete GT: WHOLE photo excluded from location statistics. No guessed or repaired mask.", fill="red")
    for i, title in enumerate(("Source + GT", "Shared A map", "Same-region A map", "Shared top 1% + GT", "Same-region top 1% + GT")):
        draw.text((i*256+5, 112), title, fill="black")
    for slot, i in enumerate(range(page*6, (page+1)*6)):
        y = 130 + slot*314
        before = pixels(images[i], arrays[q.p.ARMS[0]][i], masks[i])
        after = pixels(images[i], arrays[q.p.ARMS[1]][i], masks[i])
        for col, view in enumerate((before[0], before[1], after[1], before[2], after[2])):
            card.paste(Image.fromarray(view), (col*256, y))
        draw.text((5, y+258), q.fit.REGIONS[i], fill="black")
        draw.text((5, y+276), f"GT: {int(masks[i].sum()) if masks[i] is not None else 'unavailable'} pixels", fill="black")
        for col, arm in ((1, q.p.ARMS[0]), (2, q.p.ARMS[1])):
            row = case['arms'][arm]['regions'][i]
            draw.text((col*256+5, y+258), f"raw top1 {row['raw_top1']:.3f}", fill="black")
            draw.text((col*256+5, y+276), f"raw boundary {row['raw_equivalent_image_boundary']:.3f}", fill="black")
            draw.text((col*256+5, y+294), f"normalized {row['normalized_score']:.3f}", fill="black")
        for col, arm in ((3, q.p.ARMS[0]), (4, q.p.ARMS[1])):
            values = arrays[arm][i]
            cutoff = case['arms'][arm]['regions'][i]['map']['top1pct_cutoff']
            draw.text((col*256+5, y+258), f"support incl ties: {np.count_nonzero(values>=cutoff)}", fill="black")
    draw.text((5, card.height-20), "MPI-PCB v1.0, Oliveira / Nassu / Wehrmeister, CC BY 4.0, DOI 10.5281/zenodo.8213098 | modified", fill="black")
    return card


def index(cases, report, reference_paths):
    out = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>MPI 저장 지도 비교</title>',
           '<style>body{max-width:1180px;margin:30px auto;font:17px/1.7 sans-serif}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}img{max-width:100%;height:auto}summary{cursor:pointer}</style>',
           '<h1>기존 A와 같은 영역 후보: 어디에 반응했을까요?</h1>',
           '<p>새 추론·학습·경고 기준 변경 없이 저장된 지도를 비교했습니다. 이미 본 시험의 탐색 진단이며 자동 합격이나 원인 확정이 아닙니다.</p>',
           '<p>판정 변화 23장 전부와 고정 대조 7장입니다. 선택 사례의 통계로 전체 성능을 다시 주장하지 않습니다.</p>',
           '<h2>보는 순서</h2><ol><li>초록 정답 위치와 원본을 봅니다.</li><li>두 지도에서 결함과 정상 부분 중 어디가 강한지 봅니다.</li>',
           '<li>상위 1% 반응이 결함에 겹치는지 확인합니다. 동점은 전부 표시하므로 빨간 부분은 정확히 1%보다 클 수 있습니다.</li>',
           '<li>원래 점수가 올라도 경고 경계가 더 크게 올랐는지 숫자를 봅니다.</li></ol>',
           '<p>두 지도는 같은 고정 원점수 색상 범위입니다. 이미지마다 색을 최대치에 맞추지 않았습니다. 색과 빨간 영역은 픽셀 불량 판정이 아닙니다.</p>',
           '<h2>전체 109장 성적 — 앞서 확정된 값, 이번에 변경 없음</h2><table><tr><th>방식</th><th>검출 / 미탐</th><th>정상 오경보</th></tr>']
    for arm in q.p.ARMS:
        c = report['frozen_classification']['models'][arm]['fixed_threshold']['counts']
        out.append(f"<tr><td>{arm}</td><td>{c['tp']} / {c['fn']}</td><td>{c['fp']}</td></tr>")
    out += ['</table><h2>사례</h2>']
    for case in cases:
        old, new = (case['arms'][a] for a in q.p.ARMS)
        out.append(f"<details><summary>{html.escape(case['key'])} · {LABELS[case['group']]} · 경계와의 차이 {old['margin']:+.3f} → {new['margin']:+.3f}</summary>")
        if case['reconstructed_image_used']:
            out.append('<p>0054는 앞서 기록·검증한 복구 이미지 1개를 그대로 사용했습니다. 원본 ZIP을 바꾸지 않았습니다.</p>')
        for path in case['cards']:
            out.append(f'<a href="{html.escape(path)}"><img loading="lazy" src="{html.escape(path)}" alt="{html.escape(case["key"])} paired maps"></a>')
        out.append('</details>')
    out += ['<h2>경고 기준을 정한 정상 0411 — 숫자와 과거 참고 그림만</h2>',
            '<p>후보 정상 지도는 저장되지 않아 비교할 수 없습니다. 아래 그림은 과거 공유 bank 방식만 보여주며 후보 지도가 아닙니다. 다시 추론하거나 정상 라벨을 바꾸지 않았습니다.</p>',
            '<table><tr><th>방식</th><th>0411 점수</th><th>경고 기준</th><th>최대 영역</th><th>더 높은 정상 수</th></tr>']
    for arm, v in report['calibration_reference']['arms'].items():
        out.append(f"<tr><td>{arm}</td><td>{v['score']:.6f}</td><td>{v['threshold']:.6f}</td><td>{', '.join(v['winning_regions'])}</td><td>{v['strictly_higher_calibration_photos']}</td></tr>")
    out.append('</table>')
    for path in reference_paths:
        out.append(f'<p><a href="{html.escape(path)}">0411 과거 공유 방식 참고 그림</a></p>')
    out.append('<h2>수치의 한계와 다음 단계</h2><p>지도 안의 결함·배경 순위 AUC는 그 영역 안의 비교일 뿐 전체 Pixel AUROC가 아닙니다. 작은 결함과 동점은 case JSON의 범위값으로 봅니다. '
               '결함 위치에 반응이 약한지, 정상 부분 반응이 점수를 주도하는지, 정상 보정 경계 상승이 영향을 주는지 확인한 뒤 한 가지 변경만 연구합니다. 현재 모델·기준은 그대로 유지합니다.</p></html>')
    return ''.join(out)
