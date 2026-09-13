# 연구실 PaDiM 첫 실행

이 단계는 고정 manifest의 정상 train 170장으로 PaDiM 통계를 fit하고 정상 calibration 43장의
원시 점수로 95% `higher` 분위수 임곗값을 만든다. 공식 test 100장은 목록 분리만 확인하며
이미지를 모델에 입력하거나 평가하지 않는다.

## 사전 조건

- Windows 프로젝트: `C:\개인 프로젝트\defectops`
- 전용 Python: `C:\개인 프로젝트\defectops\.venv\Scripts\python.exe`
- 데이터 루트: `C:\PBL\datasets\mvtec_ad`
- `scripts/check_ml_environment.py` 결과의 `status`가 `passed`
- `scripts/prepare_data.py verify` 결과 manifest SHA-256이
  `9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585`
- Git 작업 폴더가 clean 상태

처음 실행할 때는 사전학습 ResNet18 가중치를 내려받으므로 인터넷 연결이 필요하다.

## 실행

새 PowerShell에서 프로젝트와 Python 경로를 고정한다.

```powershell
$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"
$datasetRoot = "C:\PBL\datasets\mvtec_ad"
Set-Location $projectPath
```

원격의 최신 `main`을 받고 다시 검증한다.

```powershell
git pull --ff-only
git status --short
& $defectPython -m pip install -r ".\requirements\ml-windows-cu128.txt"
& $defectPython -m pip install -e ".[data]"
& $defectPython -m pip check
& $defectPython scripts\check_ml_environment.py --output artifacts\environment-lab-gpu.json
& $defectPython scripts\prepare_data.py verify `
    --root $datasetRoot `
    --manifest ".\data\manifests\transistor-v1.json"
```

`git status --short`는 아무것도 출력하지 않아야 한다.
환경·데이터 검증이 통과하면 seed 42 첫 fit을 실행한다.

```powershell
& $defectPython scripts\train_padim.py `
    --root $datasetRoot `
    --manifest ".\data\manifests\transistor-v1.json" `
    --config ".\configs\pilot.toml" `
    --seed 42 `
    --output ".\artifacts\runs\padim-transistor-seed42"
```

출력 폴더는 덮어쓰지 않는다. 실패 후 재시도할 때는
`padim-transistor-seed42-retry1`처럼 새 이름을 쓴다.

## 성공 조건과 산출물

콘솔 마지막 JSON의 `status`가 `fit_calibrated`여야 한다. 출력 폴더에는 다음 파일이 생긴다.

| 파일 | 내용 |
|---|---|
| `run.json` | commit, manifest hash, 설정, 환경, GPU peak memory, sample ID, 임곗값 |
| `model.pt` | PaDiM 특징 선택과 Gaussian 통계를 포함한 모델 상태 |
| `calibration-scores.jsonl` | 정상 calibration 43장의 원시 점수와 고정 임곗값 판정 |
| `pilot.toml` | 실행 설정 스냅샷 |
| `manifest.json` | 실행 manifest 스냅샷 |

`run.json`은 `train=170`, `calibration=43`, `test_reserved_not_loaded=100`,
`test_evaluated=false`를 기록해야 한다.
이 실행이 통과하기 전에는 공식 test 평가를 시작하지 않는다.
