# Real-IAD 후속 development 후보 비교 실행

상태: 연구실 Windows GPU 실행·결과 검증 완료, final holdout 미개봉

기준일: 2026-09-22

## 목적

등록·감사를 통과한 Real-IAD `pcb`와 `terminalblock` development만 사용해 사전등록한 PatchCore
후보 네 개를 비교한다. 이 단계는 후보 선택용이며 최종 성능 평가가 아니다. Final holdout 이미지는
추출·디코딩·미리보기·점수 계산을 하지 않는다.

## 실행 전 고정 계약

- 분할 계획 SHA-256:
  `c68562288b378d44d969c3eda68a69a00c6c31e8dad0f4ff21e5c05073ac8f20`
- Seed: 42
- 입력: 256×256, ImageNet 정규화, augmentation 없음
- Backbone: pretrained Wide ResNet-50-2
- Coreset: 10%, 이웃 수: 9
- Image threshold: 정상 calibration score의 higher 95% 분위수
- Pixel threshold: 정상 calibration map pixel의 higher 99.5% 분위수
- Ground truth mask threshold: 128
- Warm 측정: 5회 예열, development 100장, batch 1, 1회 반복
- 실행 계약: `configs/realiad-followup-development-execution-v1.toml`

## 중복 학습을 줄인 구성

후보는 네 개지만 서로 다른 특징 조합은 두 개다. 동일한 특징 모델을 쓰는 후보는 하나의 memory
bank와 anomaly map을 공유하고 image score만 다르게 계산한다.

| 특징 모델 | 후보 | Image score |
|---|---|---|
| layer2 + layer3 | `baseline_raw` | Anomalib PatchCore raw score |
| layer2 + layer3 | `map_top_1pct` | anomaly map 상위 1% 평균 |
| layer2 + layer3 | `map_top_5pct` | anomaly map 상위 5% 평균 |
| layer1 + layer2 | `fine_features_top_1pct` | anomaly map 상위 1% 평균 |

따라서 품목별로 두 모델만 fit하고 네 후보를 비교한다. 후보별 calibration score와 development
prediction은 별도 JSONL로 남긴다.

## 자동 선택 규칙

1. 어느 품목에서도 정상 overkill rate가 10%를 넘지 않는 후보만 통과시킨다.
2. 두 품목 동일 가중 macro escape rate가 가장 낮은 후보를 선택한다.
3. 동률이면 macro Image AUROC가 높은 후보를 선택한다.
4. 다시 동률이면 macro overkill rate, macro warm p95가 낮은 순서로 선택한다.
5. 모든 후보가 overkill 제약을 위반하면 아무 후보도 선택하지 않고 final을 계속 봉인한다.

선택 보고서에는 선택 후보 ID, 보고서 SHA-256, 실행 Git commit, 두 final manifest SHA-256을 기록한다.
이 네 근거가 모두 생겨도 이 단계에서는 final을 열지 않는다.

## 연구실 Windows 실행

저장소를 최신 commit으로 받은 뒤 프로젝트 PowerShell에서 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git rev-parse --short HEAD

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_realiad_followup_development.ps1"
```

정상 완료 시 아래 파일을 Codex 작업에 첨부한다.

`C:\개인 프로젝트\defectops\artifacts\handoff\realiad-followup-development-v1.zip`

Handoff에는 후보 결과·calibration/prediction·선택 보고서·계약·manifest만 포함한다. 대용량 모델
checkpoint와 원본 이미지는 포함하지 않는다.

실행 결과와 선택 근거는
[`reports/realiad-followup-development-selection-v1.md`](../reports/realiad-followup-development-selection-v1.md)에
기록했다. 선택 후보는 `fine_features_top_1pct`다. 이 후보의 warm p95는 약 10초/장으로 매우 느리므로,
final 실행 전에 소요 시간과 재개 방법을 확인한다. 잠금값은
[`configs/realiad-followup-final-lock-v1.toml`](../configs/realiad-followup-final-lock-v1.toml)에 고정했다.

## 안전장치

- Git 작업 트리가 깨끗하지 않으면 실행하지 않는다.
- Development와 final manifest가 등록 당시 체크섬과 다르면 실행하지 않는다.
- Final 이미지나 마스크가 추출된 흔적이 하나라도 있으면 실행하지 않는다.
- 기존 결과가 완전하면 재사용하고, 불완전하거나 체크섬이 다르면 덮어쓰지 않는다.
- 실행 도중 Python 오류가 나면 그 실행이 새로 만든 출력 폴더만 정리한다.
- 자동 PASS는 계속 금지한다.
