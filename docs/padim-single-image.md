# PaDiM 단일 이미지 검사

고정 test 평가가 완료된 fit checkpoint와 calibration 임계값을 그대로 사용해
transistor PNG 한 장을 검사한다. 이 단계는 학습이나 임계값 조정이 아니다.

## 입력과 판정 규칙

- 입력: 1024×1024 RGB PNG 한 장, 최대 50 MiB
- 전처리: 256×256 양선형 resize, ImageNet 평균·표준편차 정규화
- 이미지 판정: 원시 PaDiM score가 `56.027042388916016` 이상이면 `REVIEW_REQUIRED`
- 픽셀·영역: 평가 bundle에 고정된 normal calibration 픽셀 임계값, morphology 없음,
  4연결, 최소 1 map pixel
- 좌표: `xyxy_exclusive`; 256×256 map 좌표와 1024×1024 원본 좌표를 둘 다 저장
- 오류: `INSPECTION_FAILED`로 기록하고 자동 PASS 금지

`PASS`는 안전 인증이 아니라 현재 모델의 결과다. 파일럿 test에서 실제 이상 40장 중
18장을 놓쳤으므로, 이 모델로 실제 자동 PASS를 시행하면 안 된다.

## Windows 검증 실행

기존 fit·evaluation 폴더는 그대로 두고 코드만 받는다. 아래 입력은 이미 평가된
`bent_lead/000.png`이며, 첫 실행은 새 성능 평가가 아니라 산출물·시각화 연결 점검이다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git status --short

$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"

& $defectPython scripts\inspect_padim.py `
    --input ".\data\transistor\test\bent_lead\000.png" `
    --fit-run ".\artifacts\runs\padim-transistor-seed42-retry1" `
    --evaluation ".\artifacts\evaluations\padim-transistor-seed42-eval-v1" `
    --output ".\artifacts\inspections\transistor-bent-lead-000"
```

`git status --short`는 비어 있어야 한다. 성공하면 마지막 JSON의 `status`가 `completed`로 나온다.

```powershell
$inspectionPath = ".\artifacts\inspections\transistor-bent-lead-000"
Get-Content "$inspectionPath\inspection.json" -Raw -Encoding UTF8
Start-Process "$inspectionPath\overlay.png"
```

같은 `--output`은 덮어쓰지 않는다. 실패했다면 먼저
`artifacts\inspections\transistor-bent-lead-000\inspection.json`의 오류를 확인하고,
재시도에는 `transistor-bent-lead-000-retry1` 같은 새 이름을 쓴다.

## 산출물

| 파일 | 내용 |
|---|---|
| `inspection.json` | 점수·임계값·판정·영역·지연시간·commit·checksum·환경 |
| `input.png` | 입력 복사본 |
| `heatmap.png` | 고정 픽셀 임계값을 1로 스케일한 heatmap |
| `overlay.png` | 원본+고정 스케일 heatmap+영역 상자 |
| `anomaly-map.npz` | float32 원시 map과 이미지·픽셀 임계값 |

파일별 최소·최대에 맞춘 시각화는 불량 신호를 과장할 수 있어 사용하지 않는다.
heatmap과 overlay는 평가 때 고정한 픽셀 임계값을 동일하게 사용한다.

## 연구실 확인 결과

2026-09-14 `bent_lead/000.png` 검사에서 `completed`, `REVIEW_REQUIRED`가 반환됐다.
원시 점수는 `56.80778884887695`, 이미지 임곗값은 `56.027042388916016`이었다.
7개 영역과 모든 예정 산출물이 생성됐으며 입력·manifest checksum도 일치했다.
상세 내용은 [단일 이미지 검사 검증 보고서](../reports/padim-single-image-smoke.md)에 기록했다.
