"""Common raw-map display and complete training-normal contact sheets."""
import html
import numpy as np
from PIL import Image, ImageDraw
from scripts import mpi_efficientad_trajectory as t


def render(entry, images, masks, maps, summary, quantiles, page):
    result = Image.new("RGB", (1792, 100+6*300), "white")
    draw = ImageDraw.Draw(result)
    draw.text((8, 8), f"{entry['group']}/{entry['photo']['photo_id']} | RAW response trajectory | page {page+1}/2", fill="black")
    draw.text((8, 28), "No checkpoint decisions, no fitted thresholds. Green = existing GT. Color saturation is display only.", fill="black")
    draw.text((8, 48), f"Shared scales: ST 0..{quantiles[0]['q995']:.6g}; AE 0..{quantiles[1]['q995']:.6g} (final normal q995).", fill="black")
    headings = ["Original + GT", "ST 10000", "ST 30000", "ST 70000", "AE 10000", "AE 30000", "AE 70000"]
    for col, heading in enumerate(headings):
        draw.text((col*256+6, 76), heading, fill="black")
    for slot, i in enumerate(range(page*6, (page+1)*6)):
        y = 100+slot*300
        rgb = np.asarray(images[i])
        edge = t.q.ev.maps.outline(masks[i])
        views = [rgb.copy()]
        for channel in (0, 1):
            upper = quantiles[channel]["q995"]
            if not np.isfinite(upper) or upper <= 0:
                raise ValueError("Invalid common raw display scale")
            for step in (10000, 30000, 70000):
                amount = np.clip(maps[str(step)][i, channel]/upper, 0, 1)[..., None]
                color = 255*(1-amount)+np.array([255, 25, 20])*amount
                views.append(np.rint(.55*rgb+.45*color).astype(np.uint8))
        for col, view in enumerate(views):
            view[edge] = (0, 255, 0)
            with Image.fromarray(view) as tile:
                result.paste(tile, (col*256, y))
        draw.text((5, y+259), t.fit.REGIONS[i], fill="black")
        draw.text((5, y+277), "GT excluded" if entry['photo']['key'] == 'defect/0006' else f"GT pixels {int(masks[i].sum())}", fill="black")
        for channel, name in enumerate(("st", "ae")):
            for offset, step in enumerate((10000, 30000, 70000)):
                stat = summary[str(step)]["regions"][i][name]
                x = (1+channel*3+offset)*256+5
                draw.text((x, y+259), f"top656 mean {stat['top656_mean']:.6g}", fill="black")
                draw.text((x, y+277), f"max {stat['maximum']:.6g}", fill="black")
    return result


def atlas(images, ids, region, page):
    result = Image.new("RGB", (768, 48+4*220), "white")
    draw = ImageDraw.Draw(result)
    draw.text((6, 6), f"TRAIN normal64 | {region} | page {page+1}/4 | display resized 256 -> 192", fill="black")
    draw.text((6, 26), "All training originals, no selection by model response; not used to add/remove training data.", fill="black")
    for slot, (image, pid) in enumerate(zip(images, ids, strict=True)):
        x, y = (slot % 4)*192, 48+(slot//4)*220
        with image.resize((192, 192), Image.Resampling.LANCZOS) as thumb:
            result.paste(thumb, (x, y))
        draw.text((x+5, y+195), pid, fill="black")
    return result


def index():
    return '''<!doctype html><html lang="ko"><meta charset="utf-8"><title>EfficientAD 학습 시점 비교</title>
<style>body{max-width:1792px;margin:24px;padding:16px;font:17px/1.7 sans-serif}img{max-width:100%}summary{cursor:pointer}</style>
<h1>학습이 진행되면서 반응이 어떻게 달라졌나요?</h1>
<p>같은 불량12장·정상8장을 저장된 1만·3만·7만 회 모델에 입력합니다. 새 학습이나 경고선 변경은 하지 않습니다.
아래 수치는 ST·AE 원시 지도입니다. 이전 모델에 7만 회 모델의 경고선을 적용하지 않습니다. 성능 순위나 최적 모델을 자동 선택하지 않습니다.</p>
<p>색은 전체 사례·시점에서 같은 범위입니다. ST와 AE는 각각 7만 회 정상 보정의 q995를 빨간색 상한으로 사용하며,
이는 색 표시용일 뿐 경고 기준이 아닙니다. 포화된 붉은색은 JSON의 실제 수치로 비교하십시오.
녹색은 기존 정답 위치입니다. 0006은 손상 마스크 예외를 유지해 사진 전체 위치 통계에서 제외합니다.</p>
<p>먼저 같은 사진의 ST 세 열을 비교하고 AE 세 열을 보십시오. 정상 고반응이 언제 커지는지,
결함 위치 반응이 감소하는지 살펴보되 원시값 변화만으로 검출 성능·과적합을 확정하지 않습니다.</p>'''


def section(title, paths):
    return '<details><summary>'+html.escape(title)+'</summary>'+''.join(
        f'<a href="{html.escape(p)}"><img loading="lazy" src="{html.escape(p)}" alt="{html.escape(title)}"></a>' for p in paths)+'</details>'
