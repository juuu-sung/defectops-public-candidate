# PatchCore 탐색적 기준선

## 목적과 해석 범위

PaDiM 파일럿과 같은 `transistor-v1.json` 분할·입력 전처리·normal-only calibration·평가
지표를 유지한 채 PatchCore의 결과를 비교한다. PaDiM 공식 test 결과를 이미 확인한 뒤 PatchCore
설정을 고정했으므로 이 실행은 `exploratory_after_padim_test_observed`로 기록한다. PatchCore가 더
좋게 나오더라도 독립적인 새 데이터 없이 확증적 모델 선택 결과로 해석하지 않는다.

고정한 PatchCore 설정은 다음과 같다.

- 정상 train 170장으로 메모리 뱅크 구성
- Wide ResNet-50-2의 `layer2`, `layer3` 특징 사용
- 전체 patch embedding 중 10% coreset 선택
- 최근접 이웃 수 9
- RGB PNG 1024×1024 입력을 256×256으로 bilinear resize하고 ImageNet 정규화
- 정상 calibration 43장의 image score 상위 5% 지점으로 고정 판정 임곗값 결정
- test 100장은 fit 명령에서 이미지 파일을 열지 않음

## 연구실 Windows 실행

PowerShell에서 프로젝트와 가상환경이 모두 `C:\개인 프로젝트\defectops` 아래에 있는지 먼저
확인한다. 데이터는 기존처럼 `C:\개인 프로젝트\defectops\data`를 사용한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"

$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"
$datasetRoot = "$projectPath\data"

git pull --ff-only
git status --short
git rev-parse HEAD
```

`git status --short`가 아무것도 출력하지 않아야 실행할 수 있다. 이어서 fit과 normal-only
calibration을 실행한다.

```powershell
& $defectPython scripts\train_patchcore.py `
    --root $datasetRoot `
    --manifest ".\data\manifests\transistor-v1.json" `
    --config ".\configs\patchcore-pilot.toml" `
    --seed 42 `
    --output ".\artifacts\runs\patchcore-transistor-seed42"
```

처음 실행할 때 Wide ResNet-50-2 사전학습 가중치를 내려받을 수 있다. PaDiM보다 큰 특징
추출기와 coreset 선택을 사용하므로 더 오래 걸릴 수 있다. 출력 폴더는 실패하더라도 덮어쓰지
않는다. 같은 이름이 이미 존재한다면 원인을 확인한 뒤 `patchcore-transistor-seed42-retry1`처럼
새 이름을 쓴다.

fit 결과가 `fit_calibrated_exploratory`로 끝난 것을 확인한 뒤에만 공식 test를 평가한다.

```powershell
& $defectPython scripts\evaluate_patchcore.py `
    --root $datasetRoot `
    --fit-run ".\artifacts\runs\patchcore-transistor-seed42" `
    --manifest ".\data\manifests\transistor-v1.json" `
    --evaluation-config ".\configs\patchcore-evaluation-v1.toml" `
    --output ".\artifacts\evaluations\patchcore-transistor-seed42-eval-v1"
```

fit에서 retry 이름을 썼다면 평가의 `--fit-run`도 같은 이름으로 바꾼다. 평가는 저장된
checkpoint와 calibration 점수를 checksum으로 검증하고, 정상 calibration 43장의 점수를 다시
계산해 모델이 정확히 복원됐는지 확인한 후 test 100장을 연다. 결과 상태는
`evaluated_exploratory`이며 PaDiM과 같은 image AUROC/AP, 고정 임곗값 혼동행렬·recall,
pixel AUROC/AUPRO, 결함 유형별 recall을 기록한다.

## 생성 산출물

- `artifacts/runs/<run_id>/`: 메모리 뱅크 checkpoint, calibration 점수, 설정·manifest snapshot
- `artifacts/evaluations/<evaluation_id>/`: 평가 지표, 예측 JSONL, anomaly map, calibration 계약

두 디렉터리 모두 원본 데이터와 마찬가지로 Git에 올리지 않는다. 비교 보고서는 실제 Windows
결과를 받은 뒤 생성한다.

## seed 42 탐색적 결과

연구실 Windows RTX 5060에서 fit과 공식 test 평가를 완료했다.

| 항목 | 결과 |
|---|---:|
| Fit + calibration | 141.677초 |
| Memory bank | 17,408 × 1,536, 약 102 MiB |
| Image AUROC | 0.9904 |
| Image average precision | 0.9884 |
| TP / FP / TN / FN | 37 / 1 / 59 / 3 |
| Precision / Recall / F1 | 0.9737 / 0.9250 / 0.9487 |
| Escape / Overkill | 0.0750 / 0.0167 |
| Pixel AUROC / AUPRO | 0.9720 / 0.9454 |

checkpoint를 다시 로드한 calibration 43장의 최대 점수 차이는 `0.0`이었다. 결함 유형별
recall은 `bent_lead` 0.9, `cut_lead` 1.0, `damaged_case` 1.0, `misplaced` 0.8이다.
따라서 FN 3장은 유형 기준으로 bent lead 1장과 misplaced 2장이다. 유형마다 표본이 10장뿐이라
유형별 수치는 탐색적으로만 해석한다.

## 고정 결과 오류 분석과 warm 측정

최신 코드를 받은 뒤 다음 두 명령을 순서대로 실행한다. 첫 명령은 저장된 결과만 읽어 FN 3장과
FP 1장의 contact sheet를 만든다. 두 번째 명령은 모델을 한 번만 로드한 뒤 같은 test 100장을
속도 측정 용도로 재사용한다. 모델·임곗값·평가 성능은 변경하지 않는다.

```powershell
& $defectPython scripts\analyze_patchcore_errors.py `
    --root $datasetRoot `
    --fit-run ".\artifacts\runs\patchcore-transistor-seed42" `
    --evaluation ".\artifacts\evaluations\patchcore-transistor-seed42-eval-v1" `
    --output ".\artifacts\analyses\patchcore-transistor-seed42-errors-v1"
```

```powershell
& $defectPython scripts\benchmark_patchcore_warm.py `
    --root $datasetRoot `
    --fit-run ".\artifacts\runs\patchcore-transistor-seed42" `
    --evaluation ".\artifacts\evaluations\patchcore-transistor-seed42-eval-v1" `
    --warmup 5 `
    --repeats 1 `
    --count 100 `
    --output ".\artifacts\benchmarks\patchcore-transistor-seed42-warm-v1"
```

두 실행은 연구실 Windows에서 완료했다. 오류 분석 결과는 TP 37 / FP 1 / TN 59 / FN 3으로
평가 bundle과 일치했다. 결함 유형별 평균 IoU / pixel recall은 각각 `bent_lead`
0.2035 / 0.9550, `cut_lead` 0.2154 / 0.7270, `damaged_case` 0.3301 / 0.9428,
`misplaced` 0.4864 / 0.6075다.

warm 측정은 12.5621 images/s였다. GPU 추론·map 전송 중앙값 / p95는 44.13 / 61.59 ms,
전체 sample 중앙값 / p95는 79.88 / 95.91 ms였다. PaDiM의 19.8264 images/s보다 느리지만
현재 1.5초 목표 안이다. 정확도·오류·속도 비교와 결정은
[PaDiM vs PatchCore 탐색적 비교 보고서](../reports/padim-vs-patchcore-transistor-exploratory-v1.md)에
기록했다.
