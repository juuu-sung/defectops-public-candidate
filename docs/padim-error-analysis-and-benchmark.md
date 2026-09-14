# PaDiM 오류 분석과 warm 추론 측정

고정 test 파일럿에서 나온 FN 18장과 FP 3장을 분석하고, 모델을 한 번만 로드한 상태의 처리 속도를
측정한다. 두 작업 모두 기존 모델·전처리·이미지/픽셀 임곗값을 변경하지 않는다.

## 1. 탐색적 오류 분석

오류 분석기는 평가 bundle의 checksum과 100장 순서를 먼저 검증한다. 저장된 원시 map과 정답 mask로
고정 픽셀 임곗값의 이미지별 IoU·Dice·pixel precision·pixel recall을 계산한다. 이 수치는 실패 원인을
살펴보기 위한 탐색적 결과이며 새 확증 성능이 아니다.

```powershell
$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"
$datasetRoot = "$projectPath\data"

Set-Location $projectPath
git pull --ff-only
git status --short

& $defectPython scripts\analyze_padim_errors.py `
    --root $datasetRoot `
    --fit-run ".\artifacts\runs\padim-transistor-seed42-retry1" `
    --evaluation ".\artifacts\evaluations\padim-transistor-seed42-eval-v1" `
    --output ".\artifacts\analyses\padim-transistor-seed42-errors-v1"
```

생성 파일:

| 파일 | 내용 |
|---|---|
| `analysis.json` | 혼동행렬, 점수 분포, 유형별 recall·픽셀 겹침, 추적 정보 |
| `analysis.md` | 사람이 읽기 쉬운 요약 |
| `cases.jsonl` | 100장별 판정·임곗값 margin·픽셀 겹침 |
| `false-negatives.png` | FN 18장 입력/고정-scale map contact sheet |
| `false-positives.png` | FP 3장 입력/고정-scale map contact sheet |

```powershell
$analysisPath = ".\artifacts\analyses\padim-transistor-seed42-errors-v1"
Get-Content "$analysisPath\analysis.json" -Raw -Encoding UTF8
Start-Process "$analysisPath\false-negatives.png"
Start-Process "$analysisPath\false-positives.png"
```

Contact sheet의 왼쪽은 입력, 오른쪽은 고정 픽셀 임곗값 scale의 map overlay다. 빨간 상자는 예측
픽셀 전체 bbox, 초록 상자는 정답 mask bbox다. 이미지별 min-max 시각화는 사용하지 않는다.

## 2. Warm 일괄 추론 측정

같은 프로세스에서 모델을 한 번 로드하고 첫 이미지로 5회 워밍업한 뒤, frozen test 입력 100장을
batch size 1로 한 번씩 처리한다. 라벨은 속도 측정이나 판정에 사용하지 않는다. 각 점수가 기존 평가
점수와 허용 오차 안에서 같은지도 확인한다.

```powershell
& $defectPython scripts\benchmark_padim_warm.py `
    --root $datasetRoot `
    --fit-run ".\artifacts\runs\padim-transistor-seed42-retry1" `
    --evaluation ".\artifacts\evaluations\padim-transistor-seed42-eval-v1" `
    --warmup 5 `
    --repeats 1 `
    --count 100 `
    --output ".\artifacts\benchmarks\padim-transistor-seed42-warm-v1"
```

```powershell
$benchmarkPath = ".\artifacts\benchmarks\padim-transistor-seed42-warm-v1"
Get-Content "$benchmarkPath\benchmark.json" -Raw -Encoding UTF8
```

`benchmark.json`은 모델 로드 시간과 image decode/checksum, 전처리, GPU 추론/map 전송, sample 전체
시간을 분리한다. 각 구간의 mean·median·p95·min·max, 전체 처리량, GPU peak memory와 기존 평가
점수 대비 최대 차이를 기록한다. `samples.jsonl`에는 100장별 시간이 저장된다.

같은 출력 폴더는 덮어쓰지 않는다. 실패 후 재시도에는 `-retry1` 같은 새 이름을 사용한다.
`git status --short`가 비어 있지 않으면 실행하지 말고 먼저 원인을 확인한다.

## 해석 순서

1. 결함 유형별 FN 집중 여부를 확인한다.
2. 임곗값 바로 아래의 FN과 매우 낮은 점수의 FN을 구분한다.
3. contact sheet에서 위치 신호 부재, 넓은 경계 잡음, 작은 결함 문제를 구분한다.
4. warm GPU 추론 median/p95와 전체 sample median/p95를 따로 본다.
5. 결과를 보고 같은 test에 임곗값을 맞추지 않는다. 개선안은 별도 development protocol에서 선택한다.
