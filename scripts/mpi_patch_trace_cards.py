"""Source-aware review cards; visual crops are context, not receptive fields."""

import html

from PIL import Image, ImageDraw

from scripts import mpi_patch_trace_protocol as p


def display_queries(case, sample, contract):
    """At most six examples; complete selected query list remains in JSON."""
    normalized = [(r["top1"]-s["median"])/s["mad"] for r, s in zip(sample["regions"], contract["normalization"], strict=True)]
    winner = max(range(12), key=lambda i: normalized[i])
    pairs = [(i, q) for i, tile in enumerate(case["tiles"]) for q in tile["queries"]]
    chosen = [(i, q) for i, q in pairs if i == winner and "tile_distance_maximum" in q["roles"]]
    for role in ("GT_distance_minimum", "GT_distance_middle", "GT_distance_maximum", "tile_distance_maximum", "tile_distance_middle_control"):
        ranked = sorted([(i, q) for i, q in pairs if role in q["roles"]], key=lambda v: (-normalized[v[0]], v[0], v[1]["flat_index"]))
        # One per role first, so a single role does not fill the whole sheet.
        for item in ranked:
            if item not in chosen:
                chosen.append(item)
                break
    return chosen[:6]


def marked(image, y, x):
    result = image.copy()
    ImageDraw.Draw(result).rectangle((4*x, 4*y, 4*x+3, 4*y+3), outline="yellow", width=2)
    left, top = max(0, min(192, 4*x+2-32)), max(0, min(192, 4*y+2-32))
    crop = result.crop((left, top, left+64, top+64)).resize((256, 256), Image.Resampling.NEAREST)
    return result, crop


def render_neighbors(source, photo, case, provenance, preparation, recovered, contract, sample):
    selected = display_queries(case, sample, contract)
    result = Image.new("RGB", (1024, 120+len(selected)*306+26), "white")
    draw = ImageDraw.Draw(result)
    r = case["replay"]
    draw.text((6, 4), f"{photo['key']} | A={r['A_score']:.4f}, cutoff={r['A_threshold']:.4f} | {r['A_decision']}", fill="black")
    draw.text((6, 21), "Yellow = 4x4 display cell. Zoom = 64x64 context, NOT CNN receptive field. Distances are before map smoothing.", fill="black")
    draw.text((6, 38), "Exact normal source aliases are ALL retained in JSON; FIRST sorted example shown here. Unmatched stays unresolved.", fill="black")
    draw.text((6, 55), "Cross-region match alone does NOT prove a wrong match or cause. This is a post-test diagnostic, not new accuracy.", fill="black")
    draw.text((6, 72), "Subset shown: A winning tile max, then GT low/middle/high and controls. Complete selected queries: queries.json.", fill="black")
    for i, title in enumerate(("Query tile", "Query context zoom", "Exact normal source (one alias)", "Normal context zoom")):
        draw.text((i*256+5, 98), title, fill="black")
    train_rows = {r["member"]: r for r in preparation["training_rows"]}
    for slot, (i, q) in enumerate(selected):
        y = 120+slot*306
        row = photo["regions"][i]
        image = p.fit.checked_image(source, row) if photo["key"].startswith("calibration/") else p.ev.read_tile(source, row, recovered)
        try:
            full, crop = marked(image, q["grid_y"], q["grid_x"])
            result.paste(full, (0, y)); result.paste(crop, (256, y))
            full.close(); crop.close()
        finally:
            image.close()
        aliases = provenance["origins_by_bank_index"][str(q["bank_index"])]
        source_label = "UNRESOLVED - no guessed source"
        if aliases:
            alias = aliases[0]
            with p.fit.checked_image(source, train_rows[alias["member"]]) as normal:
                full, crop = marked(normal, alias["grid_y"], alias["grid_x"])
                result.paste(full, (512, y)); result.paste(crop, (768, y))
                full.close(); crop.close()
            source_label = f"normal/{alias['photo_id']} {alias['region']} ({alias['grid_y']},{alias['grid_x']}) | aliases={len(aliases)}"
        else:
            draw.text((520, y+120), source_label, fill="red")
        draw.text((6, y+258), f"{row['region']} cell=({q['grid_y']},{q['grid_x']}) dist={q['native_patch_distance']:.4f} bank={q['bank_index']}", fill="black")
        draw.text((518, y+258), source_label, fill="black")
        draw.text((6, y+275), " | ".join(q["roles"]), fill="black")
        draw.text((518, y+275), f"Source status: {q['source_status']}", fill="black")
    draw.text((6, result.height-19), "MPI-PCB v1.0 (Oliveira et al.), CC BY 4.0 | DOI: 10.5281/zenodo.8213098 | modified", fill="black")
    return result


def index(report, cases, provenance):
    rows = []
    labels = {"same_region_only": "같은 영역만", "cross_region_only": "다른 영역만", "mixed_source_regions": "출처 여러 영역", "unresolved": "출처 미확인"}
    reasons = {
        "A_false_positive": "정상을 불량 의심으로 경고한 사례",
        "A_normal_calibration_cutoff": "A 경고 기준을 결정한 정상 사진",
        "A_middle_normal_calibration_control": "정상 보정 중간 점수 대조군",
        "A_middle_test_normal_control": "시험 정상 비경고 중간 점수 대조군",
        "lowest_A_miss": "가장 낮은 점수의 미탐",
        "middle_A_miss": "중간 점수의 미탐",
        "nearest_cutoff_A_miss": "경고 기준 바로 아래의 미탐",
        "A_miss_winning_region_without_GT": "최고 점수 영역이 정답 영역 밖인 미탐",
        "retained_old_TP_control": "기존에도 잡았던 불량 대조군",
        "lowest_A_new_TP_control": "A로 새로 잡은 불량 중 최저 점수 대조군",
    }
    summaries = {c["key"]: c for c in provenance["cases"]}
    for c in cases:
        prefix, replay = c["key"].replace("/", "-"), c["replay"]
        counts = " · ".join(f"{labels[k]} {v}" for k, v in summaries[c["key"]]["query_counts"].items())
        decision = "검토 경고" if replay["A_decision"] == "REVIEW_REQUIRED" else "경고 기준 미만 — 합격 아님"
        reason = "; ".join(reasons.get(r, r) for r in c["reasons"])
        rows.append(f"<tr><td>{html.escape(c['key'])}<br>{html.escape(reason)}</td>"
                    f"<td>{replay['A_score']:.4f}<br>{decision}</td><td>{counts}</td>"
                    f"<td><a href='cards/{prefix}-1.jpg'>전체 1</a> · <a href='cards/{prefix}-2.jpg'>전체 2</a><br>"
                    f"<a href='cards/{prefix}-neighbors.jpg'>정상 패치와 나란히 보기</a></td></tr>")
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>MPI 남은 오류와 정상 패치 출처</title>"
            "<style>body{max-width:1150px;margin:32px auto;font:17px/1.7 sans-serif}table{border-collapse:collapse;width:100%}"
            "td,th{border:1px solid #bbb;padding:10px;text-align:left}a{color:#075fa6}</style>"
            "<h1>A 방식의 남은 오류: 무엇과 비교했는가?</h1>"
            "<p>이 단계는 성능을 바꾸는 실험이 아니라 원인을 좁히는 진단입니다. 기존 A는 불량 33/55장 검출, "
            "22장 미탐, 정상 3/54장 오경보입니다. 모델·경고 기준·서비스는 그대로입니다.</p>"
            f"<p>선택한 정상 기준 패치 {report['selected_bank_vectors']}개 중 출처 확인 "
            f"{report['resolved_bank_vectors']}개, 여러 출처 {report['ambiguous_bank_vectors']}개, 미확인 {report['unresolved_bank_vectors']}개.</p>"
            "<p>기존 정상 학습 사진 64장의 특징을 다시 계산해 저장된 패치와 <strong>float32 값이 정확히 같은 경우만</strong> 연결합니다. "
            "출처 미확인이나 여러 출처를 하나로 단정하지 않습니다. 노란 사각형은 특징 격자의 표시 위치이며, 확대 사진은 실제 신경망 수용영역이 아닙니다.</p>"
            "<h2>어떻게 읽나요?</h2><ol><li>전체 카드에서 원본·정답(초록색)·반응 위치를 비교합니다.</li>"
            "<li>정상 패치 카드에서 문제 부위와 모델이 선택한 정상 예시를 나란히 봅니다. 한 사진의 전체 특징이 아니라 선택한 격자 위치의 비교입니다.</li>"
            "<li>다른 영역과의 매칭이 정상 대조군에도 흔한지 확인합니다. 다른 영역이라는 이유만으로 오매칭이라고 단정하지 않습니다.</li></ol>"
            "<p>표의 수는 서로 연관된 선택 지점의 출처 수이지 정확도·미탐률·전체 영역의 매칭 비율이 아닙니다. "
            "현재 미탐을 본 뒤 선정한 탐색 사례이며, 새 독립 성능 검증이 아닙니다.</p>"
            "<table><tr><th>사례·선정 이유</th><th>기존 A 점수·판정</th><th>선택 지점 출처</th><th>검토 카드</th></tr>"
            + "".join(rows) + "</table><h2>다음 결정</h2><p>다른 영역 매칭이 오류와 관련된다는 근거가 있으면 영역별 정상 기준표를, "
            "정상 외관 다양성 부족이 더 뚜렷하면 정상 학습 자료 구성을 바꾸는 한 가지 대조를 설계합니다. "
            "이 화면이 원인을 자동 확정하거나 다음 학습을 실행하지는 않습니다.</p></html>")
