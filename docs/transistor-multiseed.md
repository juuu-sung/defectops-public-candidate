# Transistor PaDiM·PatchCore 3-seed 탐색 반복

## 목적과 해석 제한

seed 42로 끝낸 단일 실행이 우연한 초기화 결과인지 확인하기 위해 같은 고정 manifest와 설정으로
seed 43·44를 추가한다. PaDiM은 처음부터 42·43·44가 계획돼 있었다. PatchCore 반복은 seed 42
test 결과를 본 뒤 추가했으므로 별도 설정에
`exploratory_repeat_after_patchcore_seed42_test_observed`를 기록한다.

이 실행은 seed 변동성을 확인하는 탐색 반복이다. 같은 MVTec test 100장을 다시 사용하므로
평균·표준편차를 독립적인 배포 성능이나 확증적 모델 선택 결과로 표현하지 않는다. 결과를 보고
하이퍼파라미터, 임곗값 또는 후처리를 바꾸려면 새로운 untouched 데이터가 필요하다.

## 제가 저장소에서 준비한 것

- PatchCore seed 43·44 전용 설정과 평가 설정
- PaDiM seed 43·44용 평가 설정
- 환경·데이터 검증부터 네 번의 fit/evaluation까지 순차 실행하는 PowerShell 스크립트
- 두 모델 × 세 seed의 결과 6개를 checksum과 표본 목록까지 검증해 평균·표본 표준편차를
  계산하는 `summarize_multiseed.py`
- seed 누락, 중복, metrics 변조, manifest/test 목록 불일치 시 집계를 거부하는 테스트

## 연구실 Windows에서 할 일

프로젝트 경로는 `C:\개인 프로젝트\defectops`이고 가상환경도 그 안의 `.venv`를 그대로 쓴다.
새 PowerShell을 열고 다음 명령을 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"

$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"

git pull --ff-only
git status --short
git rev-parse HEAD
& $defectPython -m pip check
```

`git status --short`가 아무것도 출력하지 않고 `pip check`가 오류 없이 끝나야 한다. 데이터와
기존 seed 42 평가 폴더는 다음 위치에 있어야 한다.

```text
C:\개인 프로젝트\defectops\data
C:\개인 프로젝트\defectops\artifacts\evaluations\padim-transistor-seed42-eval-v1
C:\개인 프로젝트\defectops\artifacts\evaluations\patchcore-transistor-seed42-eval-v1
```

GPU를 사용하는 다른 프로그램을 닫고 한 번에 실행한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File ".\scripts\run_transistor_multiseed.ps1"
```

스크립트는 다음 순서로 수행한다.

1. Python 3.12 / PyTorch 2.8.0+cu128 / CUDA / Anomalib GPU 점검
2. 고정 manifest와 전체 데이터 checksum 검증
3. PaDiM seed 43 fit·calibration·test 평가
4. PaDiM seed 44 fit·calibration·test 평가
5. PatchCore seed 43 메모리 뱅크·calibration·test 평가
6. PatchCore seed 44 메모리 뱅크·calibration·test 평가
7. 기존 seed 42 두 결과까지 포함한 3-seed 요약 생성

기존 관찰 시간을 기준으로 PatchCore coreset 선택이 가장 오래 걸린다. 장비 상태에 따라
전체에 약 15~30분을 잡고, 중간에 PowerShell을 닫거나 절전 모드로 전환하지 않는다.
사전학습 가중치는 이미 캐시에 있으면 다시 내려받지 않는다. 최소 2 GiB 이상의 여유 공간을
권장한다.

## 성공 조건

마지막에 다음 문구가 보여야 한다.

```text
Multi-seed exploratory run completed.
Summary: ...\artifacts\summaries\transistor-padim-patchcore-3seed-v1\summary.md
```

생성되는 핵심 결과는 다음과 같다.

```text
artifacts/runs/padim-transistor-seed43/
artifacts/runs/padim-transistor-seed44/
artifacts/runs/patchcore-transistor-seed43/
artifacts/runs/patchcore-transistor-seed44/
artifacts/evaluations/*seed43-eval-v1/
artifacts/evaluations/*seed44-eval-v1/
artifacts/summaries/transistor-padim-patchcore-3seed-v1/summary.json
artifacts/summaries/transistor-padim-patchcore-3seed-v1/summary.md
```

요약을 화면에 출력해 이 대화에 붙여넣는다.

```powershell
Get-Content `
    ".\artifacts\summaries\transistor-padim-patchcore-3seed-v1\summary.json" `
    -Raw -Encoding UTF8
```

학습 checkpoint, test map, 원본 이미지, 전체 artifacts 폴더는 Git에 올리지 않는다.

## 실패하면

스크립트는 첫 오류에서 즉시 멈추며 기존 출력 폴더를 덮어쓰지 않는다. 실패 후 같은 명령을
다시 실행하면 완전한 fit/evaluation은 검증 후 자동으로 재사용한다. 불완전하거나 metadata가
맞지 않는 기존 폴더는 그대로 보존하고 명확히 차단한다. 이 경우 폴더를 임의로 삭제하지 말고
다음 세 가지를 보내면 실패 지점에 맞는 안전한 재시도 이름과 명령을 정리한다.

1. 콘솔의 `ERROR:` 또는 `throw` 이후 전체 문구
2. 마지막으로 완료된 seed와 모델
3. `git rev-parse HEAD` 결과
