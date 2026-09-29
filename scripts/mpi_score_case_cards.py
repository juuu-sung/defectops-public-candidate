"""Human-readable fixed-score review; one shared anomaly map for all scorers."""
import html

from PIL import Image, ImageDraw

from scripts import mpi_score_probe_cards as previous

LABELS = {"control_top1_mad": "기존 A", "candidate_top1_q90": "Q90",
          "candidate_native_mad": "원시 점수+MAD"}
REASONS = {"added_normal_alert": "Q90에서 새로 경고한 정상",
           "removed_normal_alert": "Q90에서 경고가 해소된 정상",
           "remaining_miss": "Q90에서도 놓친 기존 불량"}


def render(case, images, arrays, masks, page):
    """Reuse the fixed map scale; add both boundaries for every displayed region."""
    base = previous.render(case["sample"], images, arrays, masks, page, case["reason"])
    draw = ImageDraw.Draw(base)
    for slot, i in enumerate(range(page * 6, (page + 1) * 6)):
        y = 108 + slot * 298 + 274
        draw.rectangle((0, y, 767, y + 18), fill="white")
        row = case["regions"][i]
        a, q = row["methods"]["control_top1_mad"], row["methods"]["candidate_top1_q90"]
        draw.text((5, y), f"top1 boundary: A={a['raw_boundary']:.4f} Q90={q['raw_boundary']:.4f}"
                  f" | GT pixels={row['map']['gt']['pixels']}", fill="black")
    return base


def focus(case, images, arrays, masks):
    region = case["methods"]["candidate_top1_q90"]["winning_region"]
    i = next(i for i, row in enumerate(case["regions"]) if row["region"] == region)
    card = render(case, images, arrays, masks, i // 6)
    y = 108 + (i % 6) * 298
    result = Image.new("RGB", (768, 354), "white")
    result.paste(card.crop((0, y, 768, y + 298)), (0, 56))
    card.close()
    draw = ImageDraw.Draw(result)
    draw.text((5, 4), f"{case['key']} | Q90 winning region: {region}", fill="black")
    draw.text((5, 21), "Source + green GT | shared anomaly map | red top-1% support + GT", fill="black")
    draw.text((5, 38), "Image-score boundaries are NOT pixel thresholds. Full 12 regions: cards 1 and 2.", fill="black")
    return result


def index(cases):
    sections = []
    for case in cases:
        name = case["key"].replace("/", "-")
        rows = "".join(f"<tr><td>{LABELS[k]}</td><td>{v['score']:.6f}</td>"
                       f"<td>{v['threshold']:.6f}</td><td>{v['margin']:+.6f}</td>"
                       f"<td>{'경고' if v['alert'] else '기준 미만'}</td>"
                       f"<td>{html.escape(v['winning_region'])}</td></tr>"
                       for k, v in case["methods"].items())
        detail = ""
        for r in case["regions"]:
            gt = r["map"]["gt"]
            if gt["pixels"]:
                detail += (f"<li>{r['region']}: 정답 {gt['pixels']}픽셀; 상위1% 반응의 정답 포함률 "
                           f"{gt['top1pct_gt_recall_min']:.1%}–{gt['top1pct_gt_recall_max']:.1%}; "
                           f"상위1% 점수 중 정답 기여 {gt['top1pct_score_share_min']:.1%}–"
                           f"{gt['top1pct_score_share_max']:.1%}</li>"
                           if gt['top1pct_score_share_min'] is not None else
                           f"<li>{r['region']}: 정답 {gt['pixels']}픽셀; 지도 점수 합0, 기여율 정의 불가</li>")
        question = ("원본에서 반사·밝기·위치·부품 외관 차이가 강한 반응과 겹치는지 확인해 주세요. "
                    "정상1415는 기준 바로 아래인 사례입니다." if case["label"] == 0 else
                    "초록 정답 부위에 반응이 있는지, 다른 정상 부위가 더 강한지 확인해 주세요. "
                    "색상과 사진 단위 경고 여부는 별개입니다.")
        sections.append(f"<section><h2>{html.escape(case['key'])} — {REASONS[case['reason']]}</h2>"
                        f"<p>{question}</p><table><tr><th>방법</th><th>점수</th><th>기준</th>"
                        f"<th>기준과 차이</th><th>판정</th><th>최고 영역</th></tr>{rows}</table>"
                        f"<img src='cards/{name}-focus.jpg' alt='{name} 주요 영역'><ul>{detail}</ul>"
                        f"<p>전체12영역: <a href='cards/{name}-1.jpg'>1쪽</a> · "
                        f"<a href='cards/{name}-2.jpg'>2쪽</a></p></section>")
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>MPI Q90 다섯 사례 검토</title>"
            "<style>body{max-width:1100px;margin:30px auto;padding:16px;font:17px/1.65 sans-serif}"
            "table{border-collapse:collapse}td,th{padding:7px;border:1px solid #bbb}"
            "img{max-width:100%;height:auto}section{margin:40px 0}</style>"
            "<h1>Q90 다섯 사례 검토</h1><p>새 정상 오경보0970·0998, 해소1415, 기존 미탐0019·0024입니다. "
            "결과를 보고 선택한 진단 사례이며 새로운 성능 시험이 아닙니다.</p>"
            "<p>세 방법은 같은 모델·같은 이상 지도를 사용하고 영역 점수 계산과 경고 기준이 다릅니다. "
            "지도 색은 고정된 원점수 척도이며 빨갛다고 사진 단위 경고가 확정되는 것은 아닙니다. "
            "상위1% 표시는 동점 픽셀을 모두 포함할 수 있고 픽셀 합격·불량 판정이 아닙니다.</p>"
            + "".join(sections) + "<p>MPI-PCB v1.0, Oliveira et al., CC BY 4.0; "
            "DOI 10.5281/zenodo.8213098. 원본에 지도·정답 표시를 추가한 파생 시각화.</p></html>")
