# 연구실 PaDiM 고정 test 평가

이 단계는 seed 42 첫 PaDiM fit이 성공한 뒤 실행한다. `configs/evaluation-v1.toml`에 기록된 규칙을
test 이미지 점수를 보기 전에 고정했으며, manifest의 공식 test 100장(정상 60, 이상 40)만 한 번 평가한다.
이 결과를 보고 모델·전처리·임곗값을 바꾼 후 같은 test 성능을 확증 결과로 다시 제시하지 않는다.

## 동결된 평가 규칙

- 이미지 점수: 원시 PaDiM score
- 이미지 판정: 기존 정상 calibration 43장의 95% `higher` 임곗값, `score >= threshold`
- 픽셀 임곗값: test를 열기 전 정상 calibration map 전체 픽셀의 99.5% `higher` 분위수
- ground-truth mask: 모델 map 크기 256×256으로 nearest-neighbor 변환
- 영역: 픽셀 임곗값 이상, morphology 없음, 4연결, 최소 1픽셀, XYXY 끝점 제외 좌표
- 이미지 지표: AUROC, AP, 고정 임곗값 TP/FP/TN/FN·Precision·Recall·F1·escape·overkill·review rate
- 픽셀 지표: 원시 map AUROC, AUPRO(FPR 0.30까지 정규화), 고정 픽셀 임곗값 혼동행렬
- 이미지 AUROC/AP 불확실성: class-stratified image bootstrap 2,000회
- 비율 불확실성: Wilson 95% 구간. 픽셀을 독립 표본처럼 취급한 신뢰구간은 만들지 않는다.

AUPRO는 0에서 0.30까지 301개 background-FPR 점을 요청하는 결정론적 곡선 적분이다.
세부 규칙과 seed는 `configs/evaluation-v1.toml` 및 평가 산출물에 함께 저장한다.

## Windows 실행

성공한 fit 폴더 `artifacts\runs\padim-transistor-seed42-retry1`가 같은 컴퓨터에 있어야 한다.
이 폴더는 Git에 올라가지 않으며 평가 과정에서도 변경하지 않는다.

```powershell
$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"
$datasetRoot = "$projectPath\data"
Set-Location $projectPath

git pull --ff-only
git status --short

& $defectPython scripts\evaluate_padim.py `
    --root $datasetRoot `
    --fit-run ".\artifacts\runs\padim-transistor-seed42-retry1" `
    --manifest ".\data\manifests\transistor-v1.json" `
    --evaluation-config ".\configs\evaluation-v1.toml" `
    --output ".\artifacts\evaluations\padim-transistor-seed42-eval-v1"
```

`git status --short`는 비어 있어야 한다. 스크립트는 fit checksum·manifest·train/calibration sample ID와
현재 환경을 검증하고, calibration map 임곗값을 파일로 확정한 뒤에만 test 이미지를 읽는다.
checkpoint는 `pre_trained=False`로 재구성한 후 저장 state를 불러오므로 네트워크 다운로드가 없다.

## 성공 조건과 확인

콘솔 마지막 JSON의 `status`가 `evaluated`여야 한다. 평가 폴더에는 다음이 생긴다.

| 파일 | 내용 |
|---|---|
| `evaluation.json` | fit/evaluator commit, 환경, 지표, 지연시간, 한계, artifact checksum |
| `metrics.json` | 이미지·결함 유형별·픽셀 지표와 이미지 단위 불확실성 |
| `predictions.jsonl` | 100장별 원시 점수·판정·지연시간·원본 좌표 영역 |
| `test-maps.npz` | 이미지 경로, float32 원시 map, 평가용 resized mask |
| `calibration-contract.json` | test 로드 전에 확정한 이미지/픽셀 임곗값과 모델 재현 오차 |
| `evaluation-v1.toml`, `manifest.json`, `fit-run.json` | 실행 입력 스냅샷 |

요약 확인:

```powershell
$evaluationPath = ".\artifacts\evaluations\padim-transistor-seed42-eval-v1\evaluation.json"
$evaluation = Get-Content $evaluationPath -Raw | ConvertFrom-Json

$evaluation | Select-Object status, test_evaluated, manifest_sha256, test_elapsed_seconds
$evaluation.counts
$evaluation.metrics.image
$evaluation.metrics.image_by_defect_type
$evaluation.metrics.pixel
$evaluation.latency_ms
```

평가 폴더도 덮어쓰지 않는다. 실패 후 재시도는 원인을 확인한 뒤 `-retry1` 같은 새 출력 이름을 사용한다.
이번 값은 MVTec transistor seed-42 파이프라인 파일럿이며, 실제 공장 성능이나 최종 모델 비교 결과가 아니다.

## 연구실 확인 결과

2026-09-14 `padim-transistor-seed42-eval-v1` 평가가 완료됐다.

- Image AUROC: `0.90125`
- Image AP: `0.8810328434918546`
- 고정 임곗값: TP 22 / FP 3 / TN 57 / FN 18
- Precision / Recall / F1: `0.88 / 0.55 / 0.676923076923077`
- Escape / Overkill / Review rate: `0.45 / 0.05 / 0.25`
- Pixel AUROC: `0.9688963900174326`
- Pixel AUPRO @ FPR 0.30: `0.9005312410862503`

원시 점수와 위치 map의 순위 성능은 유의미하지만, 현재 고정 임곗값은 이상 40장 중 18장을
놓쳤다. 이 결과로 같은 test에 임곗값을 다시 맞추지 않으며 실제 자동 PASS에 사용하지 않는다.
상세 해석은 [파일럿 평가 보고서](../reports/padim-transistor-seed42-pilot.md)에 기록했다.
