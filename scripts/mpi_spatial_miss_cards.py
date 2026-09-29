"""Fixed-scale attribution cards, not a new model or pixel decision rule."""
import html

from PIL import Image, ImageDraw

from scripts import mpi_score_probe_cards as previous

LABELS = {'control_q90':'기존 top1% + Q90', 'window26_q90':'공간26×26 + Q90'}


def render(case,images,arrays,masks,page):
    # Reuse the exact existing map scale and GT outlines. Fourth pane adds the
    # saved label-independent winning window, not a GT-selected crop.
    base = previous.render(case['sample'],images,arrays,masks,page,case['reason'])
    card = Image.new('RGB',(1024,base.height),'white')
    card.paste(base,(0,0))
    draw = ImageDraw.Draw(card)
    draw.text((773,84),'Blue: max 26x26; green: GT',fill='black')
    for slot,i in enumerate(range(page*6,(page+1)*6)):
        y = 108+slot*298
        card.paste(base.crop((256,y,512,y+256)),(768,y))
        row = case['regions'][i]
        x1,y1,x2,y2 = row['window']['window_xyxy']
        draw.rectangle((768+x1,y+y1,768+x2-1,y+y2-1),outline=(0,100,255),width=2)
        draw.text((773,y+258),f"window mean: {case['sample']['regions'][i]['window26']:.4f}",fill='black')
        draw.rectangle((0,y+274,1023,y+294),fill='white')
        a,b = [row['methods'][k] for k in LABELS]
        draw.text((5,y+274),f"Margins: top1={a['margin']:+.4f}; window={b['margin']:+.4f} | GT in window: "
                  f"{row['window']['gt_overlap_pixels']}/{row['map']['gt']['pixels']} | NOT pixel PASS/FAIL",fill='black')
    base.close()
    return card


def focus(case,images,arrays,masks):
    # Fixed control winner, not selected using the ground-truth mask.
    region = case['methods']['control_q90']['winning_region']
    i = next(i for i,r in enumerate(case['regions']) if r['region']==region)
    with render(case,images,arrays,masks,i//6) as full:
        result = Image.new('RGB',(1024,354),'white')
        y = 108+(i%6)*298
        result.paste(full.crop((0,y,1024,y+298)),(0,56))
    draw = ImageDraw.Draw(result)
    draw.text((5,4),f"{case['key']} | Fixed control winner: {region}",fill='black')
    draw.text((5,21),'Source + GT | fixed map | top 1% support | blue 26x26 window + GT',fill='black')
    draw.text((5,38),'Full 12 regions in cards 1 and 2. Attribution is NOT causal proof or pixel performance.',fill='black')
    return result


def percent(value):
    return '정의 불가' if value is None else f'{value:.2%}'


def index(cases):
    sections = []
    for case in cases:
        name = case['key'].replace('/','-')
        reason = '기존에도 놓친 불량' if case['reason']=='remaining_miss' else '공간 평균에서 추가로 놓친 불량'
        scores = ''.join(f"<tr><td>{LABELS[k]}</td><td>{m['score']:.6f}</td><td>{m['threshold']:.6f}</td>"
                         f"<td>{m['margin']:+.6f}</td><td>{'경고' if m['alert'] else '기준 미만'}</td>"
                         f"<td>{m['winning_region']}</td></tr>" for k,m in case['methods'].items())
        rows = []
        for row in case['regions']:
            gt,w = row['map']['gt'],row['window']
            rows.append(f"<tr><td>{row['region']}</td><td>{gt['pixels']}</td>"
                        f"<td>{percent(gt['top1pct_gt_recall_min'])}–{percent(gt['top1pct_gt_recall_max'])}</td>"
                        f"<td>{percent(w['gt_recall'])}</td><td>{percent(w['gt_score_share'])}</td>"
                        f"<td>{percent(gt['within_region_rank_auc'])}</td></tr>")
        sections.append(f"<section><h2>{html.escape(case['key'])} — {reason}</h2>"
            f"<table><tr><th>방법</th><th>점수</th><th>기준</th><th>차이</th><th>판정</th><th>최고 영역</th></tr>{scores}</table>"
            f"<img src='cards/{name}-focus.jpg' alt='{name} 기존 방식 최고 영역'>"
            f"<p>전체12영역: <a href='cards/{name}-1.jpg'>1쪽</a> · <a href='cards/{name}-2.jpg'>2쪽</a></p>"
            f"<table><tr><th>영역</th><th>정답 픽셀</th><th>상위1% 정답 포함률(동점 범위)</th>"
            f"<th>창의 정답 포함률</th><th>창 점수 중 정답 기여</th><th>영역 내 정답/배경 순위 AUC</th></tr>{''.join(rows)}</table></section>")
    return ("<!doctype html><html lang='ko'><meta charset='utf-8'><title>MPI 미탐6장 원인 진단</title>"
        "<style>body{max-width:1200px;margin:30px auto;padding:16px;font:17px/1.65 sans-serif}"
        "img{max-width:100%;height:auto}table{border-collapse:collapse}td,th{border:1px solid #bbb;padding:7px}"
        "section{margin:40px 0}a{color:#164a9b}</style><h1>미탐6장: 모델 반응과 점수 집계 구분</h1>"
        "<p>기존 미탐0019·0024와 새 미탐0021·0027·0028·0047만 확인합니다. 재학습·임곗값 변경·새 성능 시험이 아닙니다.</p>"
        "<p>왼쪽부터 원본+공식 정답(초록), 동일 척도 이상 지도, 상위1% 점수 픽셀(빨강), 공간 평균 창(파랑)입니다. "
        "색은 기존 고정 원점수 표시이며 포화될 수 있습니다. 창은 결함 정답을 보지 않고 고릅니다. "
        "정답은 계산 후 설명에만 사용합니다. 서로 다른 점수 체계의 숫자는 각자의 기준 대비로 비교하세요.</p>"
        "<p>확인할 질문: 결함의 높은 반응이 창 밖으로 빠지거나 낮은 배경과 평균되는가? "
        "결함보다 다른 정상 부위가 더 강하게 반응하는가? 두 문제가 함께 있는가?</p>"
        "<p>표는 선택된 사례의 설명 통계입니다. 정답 포함률은 고정 픽셀 임곗값의 검출률이 아니고, "
        "순위 AUC도 새 전체 성능이 아닙니다. 정답 또는 배경이 없으면 해당 통계는 정의 불가입니다. "
        "이 지도만으로 학습 패치 오매칭이나 촬영 원인을 확정할 수 없으며 자동 원인 분류를 하지 않습니다.</p>"
        + ''.join(sections) + "<p>MPI-PCB v1.0 (Oliveira et al.), CC BY 4.0. DOI 10.5281/zenodo.8213098. "
        "원본·공식 정답에 설명 표시를 추가한 파생 시각화. 자동 PASS와 생산 조치 없음.</p></html>")
