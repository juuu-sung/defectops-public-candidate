# PatchCore 단일 이미지 검사

동결한 PatchCore fit checkpoint와 평가 bundle의 이미지·픽셀 임곗값을 그대로 사용해 transistor
PNG 한 장을 검사한다. 이 명령은 모델을 다시 학습하거나 현재 test 결과에 맞춰 임곗값을 조정하지
않는다.

## 검사 계약

- 입력: 1024×1024 RGB PNG 한 장, 최대 50 MiB
- 전처리: 256×256 bilinear resize와 ImageNet 평균·표준편차 정규화
- 이미지 판정: 원시 PatchCore score가 `40.7912712097168` 이상이면 `REVIEW_REQUIRED`
- 픽셀·영역: 평가 bundle에 고정된 normal calibration pixel threshold, morphology 없음,
  4연결, 최소 1 map pixel
- 출력: 원시 점수, 고정 임곗값, 판정, heatmap, overlay, 원본 좌표 영역, 지연시간,
  model/commit/manifest/checkpoint checksum
- 오류: `INSPECTION_FAILED` 기록을 남기고 자동 PASS를 허용하지 않음

현재 PatchCore는 frozen test 40개 불량 중 37개를 검토 대상으로 보냈지만 3개를 놓쳤다. 또한
PaDiM test 결과를 본 뒤 구성을 고정한 탐색적 baseline이다. 따라서 JSON의 `PASS`는 작업 흐름을
만들기 위한 모델 출력이며 실제 자동 합격이나 안전 인증이 아니다.

## 연구실 Windows 실행

아래는 이미 평가된 `bent_lead/000.png`를 사용해 모델 로드부터 산출물 생성까지 연결되는지
확인하는 smoke test다. 새 성능 평가나 임곗값 튜닝이 아니다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git status --short

$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"

& $defectPython scripts\inspect_patchcore.py `
    --input ".\data\transistor\test\bent_lead\000.png" `
    --fit-run ".\artifacts\runs\patchcore-transistor-seed42" `
    --evaluation ".\artifacts\evaluations\patchcore-transistor-seed42-eval-v1" `
    --output ".\artifacts\inspections\patchcore-transistor-bent-lead-000"
```

`git status --short`는 비어 있어야 한다. 성공하면 출력 JSON에서 `status`는 `completed`, `model`은
`patchcore`, `comparison_status`는 `exploratory_after_padim_test_observed`여야 한다.

```powershell
$inspectionPath = ".\artifacts\inspections\patchcore-transistor-bent-lead-000"
$inspection = Get-Content "$inspectionPath\inspection.json" -Raw -Encoding UTF8 |
    ConvertFrom-Json

[ordered]@{
    status = $inspection.status
    model = $inspection.model
    comparison_status = $inspection.comparison_status
    decision = $inspection.decision
    raw_score = $inspection.raw_score
    image_threshold = $inspection.image_threshold
    anomalous_map_fraction = $inspection.pixel.anomalous_map_fraction
    region_count = $inspection.pixel.region_count
    total_latency_ms = $inspection.latency_ms.total
} | ConvertTo-Json -Depth 5

Start-Process "$inspectionPath\overlay.png"
```

같은 `--output`은 덮어쓰지 않는다. 이미 폴더가 있으면 재시도 이름에 `-retry1`처럼 새 접미사를
붙인다. 실패한 경우 해당 폴더의 `inspection.json`에 오류 유형과 메시지가 남는다.

## 산출물

| 파일 | 내용 |
|---|---|
| `inspection.json` | 모델·탐색 상태·점수·임곗값·판정·영역·지연시간·추적 정보 |
| `input.png` | 입력 PNG 복사본 |
| `heatmap.png` | 고정 pixel threshold 기준 heatmap |
| `overlay.png` | 원본+고정 스케일 heatmap+영역 상자 |
| `anomaly-map.npz` | float32 원시 map과 이미지·픽셀 임곗값 |

시각화는 이미지마다 최소·최대를 다시 맞추지 않는다. 평가 때 고정한 pixel threshold를 1로 두는
동일한 선형 스케일을 사용해 약한 신호가 과장되는 것을 막는다.

## 다음 확인

Windows 출력과 overlay를 받아 다음을 확인한다.

1. 모델과 탐색 상태가 PatchCore bundle과 일치하는가.
2. 입력·checkpoint·manifest checksum과 fit/evaluation ID가 보존되는가.
3. `bent_lead/000.png`의 실제 리드 결함이 overlay 영역에 포함되는가.
4. 모델 최초 로드 시간과 한 장 GPU 추론 시간이 분리 기록되는가.
5. 모든 산출물이 존재하고 `automatic_pass_allowed`가 `false`인가.
