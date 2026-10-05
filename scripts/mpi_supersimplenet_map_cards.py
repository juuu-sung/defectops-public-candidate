"""Saved maps only; fixed SSN colors and separately labelled Q90 ranks."""
import html

import numpy as np
from PIL import Image, ImageDraw

from scripts import mpi_supersimplenet_map_diagnostic as q


def colors(values):
    lo, hi = q.POLICY["display_range"]
    amount = np.clip(np.where(values < 0, values/lo, values/hi), 0, 1)[..., None]
    target = np.where((values < 0)[..., None], [30., 80., 255.], [255., 25., 20.])
    return np.rint(255*(1-amount) + target*amount).astype(np.uint8)


def ranks(values):
    # Midranks: a flat map stays uniformly 0.5, not a spurious raster gradient.
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    right = np.cumsum(counts)
    mid = (right-counts + (counts-1)/2) / (values.size-1)
    return mid[inverse].reshape(values.shape)


def support(source, values, cutoff):
    result = source.copy()
    chosen = values >= cutoff
    result[chosen] = np.rint(.5*result[chosen]+.5*np.array([255, 0, 0])).astype(np.uint8)
    return result


def render(case, images, arrays, masks, page):
    result = Image.new("RGB", (1280, 148+6*332+28), "white")
    draw = ImageDraw.Draw(result)
    maps, baseline = arrays
    s = case["sample"]
    lines = [f"SuperSimpleNet {case['key']} | {case['role']} | page {page+1}/2",
        f"Classifier photo score {s['score']:.6f} / threshold {case['threshold']:.6f} | alert {s['alert']} | winner {s['winning_region']}",
        "SSN raw segmentation map: shared blue -1, white 0, red +1. NOT a probability or classification score.",
        "Q90 display: rank WITHIN EACH tile (0-1), not raw magnitude. Do not compare heatmap colors across models.",
        "Green = official GT; red support = highest 656 pixels INCLUDING cutoff ties. Not a pixel decision.",
        "Incomplete GT: entire photo excluded from location metrics." if not case["location_eligible"] and case["group"] == "test" else
        "Saved maps only. No inference, training, new thresholds or automatic cause assignment."]
    for i, line in enumerate(lines): draw.text((8, 5+i*20), line, fill="black")
    for i, title in enumerate(("Original + GT", "SSN raw map fixed -1 to +1", "SSN top656 support", "Q90 within-tile rank", "Q90 top656 support")):
        draw.text((i*256+5, 128), title, fill="black")
    for slot, i in enumerate(range(page*6, (page+1)*6)):
        y = 148+slot*332
        source = np.asarray(images[i]).copy()
        edge = q.ev.maps.outline(masks[i])
        row = case["regions"][i]
        views = [source.copy(), np.rint(.55*source+.45*colors(maps[i])).astype(np.uint8),
                 support(source, maps[i], row["segmentation"]["top656_cutoff"])]
        if baseline is not None:
            rank = ranks(baseline[i])
            color = np.stack([255*rank, 80*(1-rank), 255*(1-rank)], axis=-1)
            views += [np.rint(.55*source+.45*color).astype(np.uint8),
                      support(source, baseline[i], row["q90_map"]["top656_cutoff"])]
        for col, view in enumerate(views):
            view[edge] = (0, 255, 0)
            with Image.fromarray(view) as panel: result.paste(panel, (col*256, y))
        if baseline is None:
            draw.text((3*256+8, y+100), "No verified saved Q90 map", fill="black")
            draw.text((4*256+8, y+100), "No new inference requested", fill="black")
        draw.text((5, y+259), row["region"], fill="black")
        draw.text((5, y+277), f"GT pixels: {int(masks[i].sum()) if masks[i] is not None else 'unknown'}", fill="black")
        draw.text((5, y+295), f"classifier logit {row['tile_score']:+.5f}", fill="black")
        draw.text((5, y+313), f"standardized {row['standardized_score']:.4f}", fill="black")
        st = row["segmentation"]
        draw.text((261, y+259), f"map max {st['maximum']:+.5f}", fill="black")
        draw.text((261, y+277), f"map top mean {st['top656_mean']:+.5f}", fill="black")
        draw.text((261, y+295), f"display clipped {st['display_below_range_fraction']+st['display_above_range_fraction']:.1%}", fill="black")
        draw.text((517, y+259), f"support incl ties {st['support_including_ties_pixels']}", fill="black")
        draw.text((517, y+277), f"classifier margin {row['tile_margin']:+.5f}", fill="black")
        for col, stats in ((2, st), (4, row["q90_map"])):
            if stats and stats["location"]:
                loc = stats["location"]
                draw.text((col*256+5, y+295), f"GT overlap bounds {loc['top656_gt_overlap_bounds']}", fill="black")
                auc = loc["within_tile_auroc"]
                draw.text((col*256+5, y+313), f"within-tile AUROC {auc:.4f}" if auc is not None else "within-tile AUROC undefined", fill="black")
        if row["q90_map"]:
            st = row["q90_map"]
            draw.text((773, y+259), f"Q90 raw max {st['maximum']:.4f}", fill="black")
            draw.text((773, y+277), f"Q90 raw top mean {st['top656_mean']:.4f}", fill="black")
    draw.text((5, result.height-23), "MPI-PCB / Oliveira, Nassu, Wehrmeister / DOI 10.5281/zenodo.8213098 / CC BY 4.0 / modified", fill="black")
    return result


def index(cases):
    esc = html.escape
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>SuperSimpleNet 미탐 지도 진단</title>',
        '<style>body{max-width:1280px;margin:30px auto;padding:16px;font:17px/1.7 sans-serif}img{max-width:100%;height:auto}summary{cursor:pointer}</style>',
        '<h1>실제 결함에 반응하지 않은 것인지 사진 점수가 낮은 것인지 확인합니다</h1>',
        '<p>고정 미탐12장과 정상 고점수6장, 경고 기준 정상1장, 중앙 점수 정상1장입니다. 저장된 지도만 읽었으며 학습·추론·임곗값 변경은 없습니다.</p>',
        '<ol><li>원본과 초록 정답 위치를 봅니다.</li><li>SSN 지도와 상위656픽셀 위치가 결함을 포함하는지 확인합니다.</li><li>분류 점수가 높아진 영역과 결함 영역이 같은지 비교합니다.</li><li>검증된 Q90 지도가 있으면 같은 위치와 비교합니다.</li></ol>',
        '<p>SSN 원시 지도 색상은 전체 사례에서 -1~1로 고정했습니다. 지도는 사진 판정에 쓰인 분류 점수와 다릅니다. 빨간색은 불량 확률이나 판정선이 아닙니다.</p>',
        '<p>Q90 색상은 각 타일 안에서의 반응 순위입니다. SSN과 색의 강도를 직접 비교하지 마세요. 평평한 지도는 같은 색으로 표시하고 경계 동점 픽셀을 모두 표시합니다.</p>',
        '<p>0006은 기존 손상 마스크 예외로 사진 전체 위치 지표를 제외합니다. 0054는 승인된 복구 이미지를 그대로 사용합니다. 정상 사진에는 결함 위치 통계를 만들지 않습니다.</p>']
    roles = {"fixed_miss":"놓친 불량", "normal_high":"정상 고점수", "normal_threshold_anchor":"경고 기준 정상", "normal_median":"중앙 점수 정상"}
    for i, case in enumerate(cases):
        parts.append(f'<details><summary>{esc(case["key"])} · {roles[case["role"]]} · 분류 점수 {case["sample"]["score"]:.4f} / 기준 {case["threshold"]:.4f}</summary>')
        if not case["baseline_available"]: parts.append('<p>이번 사례에 검증된 저장 Q90 지도가 없습니다. 새 추론으로 채우지 않았습니다.</p>')
        for page in (1, 2):
            src = f'cases/case-{i:02d}/card-{page}.png'
            parts.append(f'<a href="{src}"><img loading="lazy" src="{src}" alt="{esc(case["key"])} page {page}"></a>')
        parts.append('</details>')
    parts.append('<p>위치 AUROC와 픽셀 겹침은 선택 사례의 원인 진단용입니다. 전체 성능이나 새 점수의 성능을 주장하지 않습니다. 지도에 반응이 있다고 분류 head만 문제라고 확정할 수 없습니다.</p>')
    parts.append(f'<p>{esc(q.ev.maps.SOURCE_ATTRIBUTION)}</p></html>')
    return ''.join(parts)
