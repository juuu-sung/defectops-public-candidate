"""Bounded review cards. Colors are explanatory, never pixel PASS/FAIL."""

import html

import numpy as np
from PIL import Image, ImageDraw

from scripts import evaluate_mpi_pcb_normal_bank as ev


def render(sample, images, arrays, masks, page, reason):
    card = Image.new("RGB", (768, 108 + 6 * 298 + 22), "white")
    draw = ImageDraw.Draw(card)
    draw.text((5, 3), f"MPI {sample['key']} | {reason} | {page+1}/2", fill="black")
    draw.text((5, 18), "Left: source + GT (green). Middle: fixed map display. Right: top 1% support (red).", fill="black")
    draw.text((5, 33), "Color = map / (1.5 x BASELINE image threshold); NOT pixel decision. Saturation possible.", fill="black")
    draw.text((5, 48), "Red support includes all cutoff ties; may exceed 656 pixels. Native raw is NOT map maximum.", fill="black")
    if sample["key"] == "defect/0006":
        draw.text((5, 63), "Incomplete GT: exclude WHOLE photo from localization. No inferred/repaired mask.", fill="red")
    elif sample["label"] == 0:
        draw.text((5, 63), "Labeled normal: inspect lighting/edges/alignment; do not infer real-world defects from color.", fill="black")
    for column, title in enumerate(("Source + official GT", "Baseline anomaly map", "Top 1% support + GT")):
        draw.text((column * 256 + 5, 84), title, fill="black")
    for slot, i in enumerate(range(page * 6, (page+1) * 6)):
        y = 108 + slot * 298
        original, values = np.asarray(images[i]), arrays[i]
        edge = ev.maps.outline(masks[i])
        source = original.copy()
        source[edge] = (0, 255, 0)
        heat = ev.maps.colorize_anomaly_values(np.clip(values / (1.5 * ev.THRESHOLD), 0, 1))
        overlay = np.rint(original * .6 + heat * .4).astype(np.uint8)
        overlay[edge] = (0, 255, 0)
        support = original.copy()
        cutoff = np.partition(values.ravel(), -656)[-656]
        chosen = values >= cutoff
        support[chosen] = np.rint(support[chosen] * .5 + np.array([255, 0, 0]) * .5).astype(np.uint8)
        support[edge] = (0, 255, 0)
        for column, pixels in enumerate((source, overlay, support)):
            card.paste(Image.fromarray(pixels), (column * 256, y))
        row = sample["regions"][i]
        draw.text((5, y+258), row["region"], fill="black")
        draw.text((261, y+258), f"top1 mean: {row['top1']:.4f}", fill="black")
        draw.text((517, y+258), f"native raw: {row['native_raw']:.4f}", fill="black")
        gt = str(int(masks[i].sum())) if masks[i] is not None else "unavailable"
        draw.text((5, y+274), f"GT pixels: {gt} | map max: {float(values.max()):.4f} | support pixels incl. ties: {int(chosen.sum())}", fill="black")
    draw.text((5, card.height-18), "MPI-PCB v1.0 (Oliveira et al.), CC BY 4.0 | DOI: 10.5281/zenodo.8213098 | modified", fill="black")
    return card


def index(entries, report=None):
    items = []
    for key, reason, paths in entries:
        links = " ".join(f'<a href="{html.escape(p, quote=True)}">{i+1}쪽</a>' for i, p in enumerate(paths))
        items.append(f"<li>{html.escape(key)} — {html.escape(reason)}: {links}</li>")
    results = ""
    if report is not None:
        labels = {"baseline_top1": "기존 점수", "a_region_mad_top1": "A: 영역별 정상 차이 보정", "b_native_raw": "B: PatchCore 자체 점수"}
        rows = []
        for key, value in report["models"].items():
            fixed = value["fixed_threshold"]
            c, rates = fixed["counts"], fixed["rates"]
            rows.append(f"<tr><td>{labels[key]}</td><td>{c['tp']}/{c['tp']+c['fn']}</td>"
                        f"<td>{c['fn']}</td><td>{c['fp']}/{c['fp']+c['tn']}</td>"
                        f"<td>{rates['recall']:.1%}</td><td>{value['image_auroc']:.4f}</td></tr>")
        results = ("<h2>동일 시험의 점수 비교</h2><table><tr><th>방식</th><th>검출/불량</th><th>놓친 불량</th>"
                   "<th>정상 오경보/정상</th><th>불량 검출률</th><th>AUROC</th></tr>" + "".join(rows) + "</table>"
                   "<p>놓친 불량과 정상 오경보를 함께 보세요. AUROC는 점수 순위 지표이며 불량 검출률이 아닙니다. "
                   "아래는 이미 본 시험의 탐색 결과입니다. 좋은 숫자가 나와도 자동 합격·서비스 모델 교체를 하지 않습니다. "
                   "복구 사진 제외 결과와 판정 변화 명단은 comparison.json에 함께 있습니다.</p>")
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>MPI 점수 진단</title>"
            "<style>body{max-width:1000px;margin:32px auto;font:17px/1.7 sans-serif}li{margin:8px}"
            "table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:8px;text-align:left}</style>"
            + results +
            "<h1>MPI 정상 고득점·대표 오류 검토</h1><p>왼쪽 원본, 가운데 고정 색상 이상 지도, 오른쪽 상위 1% 반응입니다. "
            "초록 선은 공식 결함 정답입니다. 빨간색만으로 실제 결함이나 원인을 확정하지 않습니다.</p>"
            "<p>정상 사례: 조명·반사, 경계·정렬, 부품의 외관 차이가 강한 반응과 겹치는지 확인합니다. "
            "불량 사례: 결함 부위 반응과 정상 부분 반응을 비교합니다. 이는 사람이 확인할 가설이며 자동 원인 판정이 아닙니다.</p>"
            "<p>새 비교 결과를 보고 사례를 고른 것이 아닙니다. 기존 점수로 정상 상위 10장·중간 3장, "
            "미탐 하위·중간·상위 각 3장, 기존 TP/FP 전체·TN 중간 3장을 고정했습니다. 전수 성능은 별도 JSON을 봅니다.</p><ul>"
            + "".join(items) + "</ul></html>")
