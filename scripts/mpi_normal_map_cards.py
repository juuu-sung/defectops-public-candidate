"""Fixed-scale normal-map diagnostics; no prediction or quality decisions."""
from collections import deque
import html
import math

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageFilter

DISPLAY_MAX = 80.0  # Visualization only. Never a calibrated pixel threshold.
BAND_RADIUS = 2


def spatial_masks(image, values):
    zero = np.all(np.asarray(image) == 0, axis=2)
    connected = np.zeros_like(zero)
    queue = deque()
    height, width = zero.shape
    for y, x in [(0, x) for x in range(width)] + [(height-1, x) for x in range(width)] + [(y, 0) for y in range(height)] + [(y, width-1) for y in range(height)]:
        if zero[y, x] and not connected[y, x]:
            connected[y, x] = True
            queue.append((y, x))
    while queue:
        y, x = queue.popleft()
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            yy, xx = y+dy, x+dx
            if 0 <= yy < height and 0 <= xx < width and zero[yy, xx] and not connected[yy, xx]:
                connected[yy, xx] = True
                queue.append((yy, xx))
    dilated = np.asarray(Image.fromarray(connected.astype(np.uint8)*255).filter(ImageFilter.MaxFilter(2*BAND_RADIUS+1))) > 0
    band = dilated & ~zero
    count = math.ceil(values.size * .01)
    # Deterministic location for boundary ties; the score is unaffected by tie order.
    indices = np.argsort(-values.ravel().astype(np.float64), kind="stable")[:count]
    top = np.zeros(values.size, dtype=bool)
    top[indices] = True
    top = top.reshape(values.shape)
    return top, {"edge_connected_black": connected, "adjacent_nonblack_2px": band,
                 "other_pixels": ~(connected | band)}


def statistics(image, values):
    top, masks = spatial_masks(image, values)
    selected = values[top].astype(np.float64)
    cutoff = float(selected.min())
    total = float(selected.sum())
    return {"top_count": int(top.sum()), "top_mean": float(selected.mean()),
            "top_cutoff": cutoff, "pixels_equal_to_cutoff": int(np.sum(values == cutoff)),
            "pixels_strictly_above_cutoff": int(np.sum(values > cutoff)),
            "tie_policy": "value descending then row-major index; ties do not change top mean",
            "display_clipped_high_pixels": int(np.sum(values > DISPLAY_MAX)),
            "map_minimum": float(values.min()), "map_maximum": float(values.max()),
            "partitions": {name: {"area_pixels": int(mask.sum()),
                "top_pixels": int(np.sum(top & mask)),
                "top_fraction": float(np.sum(top & mask)/top.sum()),
                "top_value_sum_fraction": float(values[top & mask].astype(np.float64).sum()/total) if total else None}
                for name, mask in masks.items()}}


def panels(image, values):
    v = np.clip(values.astype(np.float64)/DISPLAY_MAX, 0, 1)
    # A single common blue->red scale, not per-image min/max stretching.
    heat = np.stack((255*v, 70*(1-np.abs(2*v-1)), 255*(1-v)), axis=-1).astype(np.uint8)
    top, _ = spatial_masks(image, values)
    marks = np.asarray(image).copy()
    marks[top] = [255, 0, 255]
    return image, Image.fromarray(heat), Image.fromarray(marks)


def render(case, images, arrays, indices):
    card = Image.new("RGB", (816, 66+300*len(indices)), "#f5f6f8")
    draw = ImageDraw.Draw(card)
    font = ImageFont.load_default(size=14)
    draw.text((12, 8), f"NORMAL {case['photo_id']} | Q90 {case['decision']['score']:.6f} | not a defect diagnosis", fill="black", font=font)
    draw.text((12, 32), f"Original | raw map fixed 0..{DISPLAY_MAX:g} | magenta: top 1% (656 pixels)", fill="black", font=font)
    for k, i in enumerate(indices):
        row = case["regions"][i]
        y = 64+k*300
        draw.text((12, y), f"{row['region']} | top1 {row['top1']:.4f} | z {row['normalized_score']:.4f}", fill="black", font=font)
        for j, panel in enumerate(panels(images[i], arrays[i])):
            card.paste(panel, (12+j*268, y+26))
    return card


def index(cases):
    sections = []
    for case in cases:
        pid = case["photo_id"]
        sections.append(f'<section><h2>{pid} · {html.escape(case["decision"]["winning_region"])}</h2>'
            f'<p>기존 Q90 점수 {case["decision"]["score"]:.6f}; 경고 기준 {case["threshold"]:.6f}. 정상으로 등록된 진단 대상입니다.</p>'
            f'<img src="cases/{pid}/focus.png" alt="정상 {pid} 최고 점수 영역">'
            f'<p><a href="cases/{pid}/card-1.png">영역 1–6</a> · <a href="cases/{pid}/card-2.png">영역 7–12</a> · '
            f'<a href="cases/{pid}/case.json">수치와 검증 기록</a></p></section>')
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MPI 정상 고득점 위치 검토</title><style>body{font:17px/1.6 system-ui;max-width:1000px;margin:30px auto;padding:0 20px;color:#202632}img{max-width:100%}section{border-top:1px solid #ccc;margin-top:30px}a{color:#1649a0}</style>
<h1>정상 사진 5장의 고득점 위치</h1>
<p>왼쪽은 원본, 가운데는 모든 사진에 같은 0–80 원시 점수 색 범위를 사용한 지도, 오른쪽 자홍색은 이미지 점수 계산에 포함된 상위1% 픽셀입니다. 파랑은0, 빨강은80 이상이며80을 넘는 값의 수는 JSON에 기록합니다. 별도의 색상 자동 확장은 없습니다.</p>
<p>지도 색상 범위는 시각화용이지 픽셀 불량 판정 기준이 아닙니다. 자홍색도 결함 마스크가 아닙니다. 동점 픽셀은 위에서 아래, 왼쪽에서 오른쪽 순서로 골라 위치를 고정하며 점수는 바뀌지 않습니다.</p>
<p>검은 영역은 RGB가 모두0이고 영상 가장자리와4방향으로 연결된 픽셀입니다. 그 바깥2픽셀 띠와 나머지 부분을 별도로 셉니다. 겹침은 기술 통계일 뿐 인과관계나 촬영 품질 탈락 판정이 아닙니다. 반사·흐림은 자동 분류하지 않습니다.</p>
<p>원본과 기존 점수를 재현한 뒤 지도를 보존하는 진단입니다. 재학습·사진 제외·테두리 삭제·새 보정·자동 합격은 하지 않습니다. 이미지와 시각화를 공개하거나 Git·Notion에 올리지 마십시오.</p>
''' + "\n".join(sections) + "</html>"
