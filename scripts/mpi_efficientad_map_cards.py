"""Fixed signed-scale diagnostic cards; visualization never changes model scores."""
import html

import numpy as np
from PIL import Image, ImageDraw

from scripts import mpi_efficientad_map_diagnostic as q


def colors(values):
    lo, hi = q.POLICY["display_range"]
    # White at zero, blue negative, red positive. No per-image scaling.
    amount = np.clip(np.where(values < 0, values/lo, values/hi), 0, 1)[..., None]
    target = np.where((values < 0)[..., None], [30., 80., 255.], [255., 25., 20.])
    return np.rint(255*(1-amount) + target*amount).astype(np.uint8)


def render(case, images, arrays, masks, page):
    result = Image.new("RGB", (1280, 148+6*328+28), "white")
    draw = ImageDraw.Draw(result)
    s = case["sample"]
    lines = [f"EfficientAD {case['key']} | {case['role']} | page {page+1}/2",
             f"Frozen score {s['score']:.6f} / threshold {case['threshold']:.6f} | alert {s['alert']} | winner {s['winning_region']}",
             "ST = teacher/student difference; AE = autoencoder/student difference. Shown AFTER fixed map normalization.",
             "Shared colors: blue -0.05, white 0, red +0.50. Values outside range saturate; numbers below stay unclipped.",
             "Green = official GT; red support = combined top656 INCLUDING ALL cutoff ties. Neither is a new pixel decision.",
             "Incomplete GT: whole photo excluded from location metrics." if not case["location_eligible"] else
             "Contributions use the SAME combined top656 positions, with stable raster-order ties; sum equals tile score."]
    for i, line in enumerate(lines):
        draw.text((8, 5+i*20), line, fill="black")
    for i, title in enumerate(("Original + GT", "ST normalized", "AE normalized", "Combined (ST+AE)/2", "Combined top656 support")):
        draw.text((i*256+5, 128), title, fill="black")
    for slot, i in enumerate(range(page*6, (page+1)*6)):
        y = 148+slot*328
        source = np.asarray(images[i]).copy()
        edge = q.ev.maps.outline(masks[i])
        views = [source.copy()]
        for a in arrays:
            views.append(np.rint(.55*source + .45*colors(a[i])).astype(np.uint8))
        cutoff = case["regions"][i]["combined"]["top656_cutoff"]
        support = source.copy()
        chosen = arrays[2][i] >= cutoff
        support[chosen] = np.rint(.5*support[chosen] + .5*np.array([255, 0, 0])).astype(np.uint8)
        views.append(support)
        for col, view in enumerate(views):
            view[edge] = (0, 255, 0)
            with Image.fromarray(view) as panel:
                result.paste(panel, (col*256, y))
        row = case["regions"][i]
        draw.text((5, y+259), row["region"], fill="black")
        draw.text((5, y+277), f"GT pixels: {int(masks[i].sum()) if masks[i] is not None else 'unknown'}", fill="black")
        draw.text((5, y+295), f"region score {row['standardized_score']:.4f}", fill="black")
        for col, name in ((1, "st"), (2, "ae"), (3, "combined")):
            stats = row[name]
            draw.text((col*256+5, y+259), f"max {stats['maximum']:.4f} | top mean {stats['top656_mean']:.4f}", fill="black")
            draw.text((col*256+5, y+277), f"display high clipped {stats['display_above_range_fraction']:.1%}", fill="black")
            if col < 3:
                c = row[name+"_half_contribution_on_combined_top656"]
                draw.text((col*256+5, y+295), f"half-contribution {c:+.6f}", fill="black")
        draw.text((3*256+5, y+295), f"tile {row['tile_score']:.6f}", fill="black")
        draw.text((4*256+5, y+259), f"tile boundary {row['raw_equivalent_photo_boundary']:.6f}", fill="black")
        draw.text((4*256+5, y+277), f"tile margin {row['tile_margin']:+.6f}", fill="black")
        draw.text((4*256+5, y+295), f"support incl ties {int(chosen.sum())}", fill="black")
    draw.text((5, result.height-23), "MPI-PCB / Oliveira, Nassu, Wehrmeister / DOI 10.5281/zenodo.8213098 / CC BY 4.0 / modified", fill="black")
    return result


def index(cases):
    esc = html.escape
    parts = ['<!doctype html><html lang="ko"><meta charset="utf-8"><title>EfficientAD 미탐 원인 진단</title>',
             '<style>body{max-width:1280px;margin:30px auto;padding:16px;font:17px/1.7 sans-serif}img{max-width:100%;height:auto}summary{cursor:pointer}table{border-collapse:collapse}td,th{border:1px solid #aaa;padding:8px}</style>',
             '<h1>왜 기존 불량 12장을 놓쳤을까요?</h1>',
             '<p>새 추론·학습·임곗값 변경 없이 저장된 지도를 읽었습니다. 기존 불량12장, 정상 고반응6장, 경고 기준 사진1장, 정상 중앙 점수 사진1장입니다. 선택 사례이므로 전체 성능 통계가 아닙니다.</p>',
             '<h2>보는 순서</h2><ol><li>원본과 초록색 결함 위치를 먼저 봅니다.</li><li>ST와 AE 중 어느 지도가 결함에 반응하는지 봅니다.</li><li>합성 지도와 빨간 상위픽셀 위치가 결함을 포함하는지 봅니다.</li><li>사진 점수를 주도한 영역과 해당 영역의 점수·경계 차이를 확인합니다.</li></ol>',
             '<p>ST는 teacher/student 특징 차이, AE는 autoencoder/student 특징 차이입니다. 표시 지도는 정상 자료로 이미 보정한 값입니다. ST·AE의 half-contribution은 같은 합성지도 상위656픽셀에서 계산한 기여도이며 둘을 더하면 영역 점수가 됩니다. 서로 다른 위치에서 구한 ST·AE top mean을 더해서는 안 됩니다.</p>',
             '<p>모든 지도에 동일한 색상 범위(-0.05~0.50)를 사용합니다. 음수는 파랑, 0은 흰색, 양수는 빨강입니다. 범위 밖은 색만 포화되며 수치는 잘리지 않습니다. 색상은 고정 경고선이나 불량 확정이 아닙니다. 빨간 support에는 경계 동점이 모두 포함돼656픽셀보다 클 수 있습니다.</p>',
             '<p>0006은 기존 손상 마스크 예외를 유지합니다. 유효한 다른 영역의 정답은 그림에 표시할 수 있지만 사진 전체의 위치 수치는 제외합니다. 정상 사진의 초록 표시가 없는 것은 정상이라는 기존 라벨 때문입니다.</p>',
             '<h2>검토 사례</h2>']
    roles = {"fixed_miss": "놓친 불량", "normal_high": "정상 고반응", "normal_threshold_anchor": "경고 기준을 결정한 정상", "normal_median": "중앙 점수 정상"}
    for i, case in enumerate(cases):
        s = case["sample"]
        parts.append(f'<details><summary>{esc(case["key"])} · {roles[case["role"]]} · 점수 {s["score"]:.4f} / 경고선 {case["threshold"]:.4f}</summary>')
        if case["recovered_image_used"]:
            parts.append('<p>0054는 앞서 승인한 복구 타일을 그대로 사용했습니다. 새 복구가 아닙니다.</p>')
        for page in (1, 2):
            src = f"cases/case-{i:02d}/card-{page}.png"
            parts.append(f'<a href="{src}"><img loading="lazy" src="{src}" alt="{esc(case["key"])} page {page}"></a>')
        parts.append('</details>')
    parts.append('<h2>해석의 한계</h2><p>JSON의 within_tile_auroc와 상위656픽셀 겹침은 결함 위치 진단용이며 전체 Pixel AUROC나 새 모델 성능이 아닙니다. 미탐 원인을 자동 확정하거나 정답 위치로 점수를 다시 계산해 합격시키지 않았습니다. 원인을 확인한 뒤 다음 변경 하나를 논의합니다.</p>')
    parts.append(f'<p>{esc(q.ev.maps.SOURCE_ATTRIBUTION)}</p></html>')
    return ''.join(parts)
