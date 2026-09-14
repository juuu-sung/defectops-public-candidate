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
