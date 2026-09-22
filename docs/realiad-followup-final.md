# Real-IAD 후속 final 평가 — 실행 전 안내

상태: 실행기 구현·테스트 완료, final holdout 미개봉. 연구실 Windows GPU 실행은 아직 하지 않았다.

## 무엇을 검증하나

Development에서 미리 선택한 `fine_features_top_1pct` **한 후보만** 새 품목 `pcb`와
`terminalblock`의 봉인된 final에서 평가한다. 후보·특징층(layer1+layer2)·상위 1% map
평균 점수·정상 calibration 95% 임곗값·seed 42/43/44는 고정했다. Final 이미지 총 710장
(PCB 360, terminalblock 350)은 이 실행 전까지 추출·디코딩·미리보기·점수 계산을 하지 않는다.

선택 잠금은 [`configs/realiad-followup-final-lock-v1.toml`](../configs/realiad-followup-final-lock-v1.toml),
근거는 [`reports/realiad-followup-development-selection-v1.md`](../reports/realiad-followup-development-selection-v1.md)에 있다.
선택 후보의 development warm p95는 약 10.3초/장이다. 최종 3-seed 평가와 나머지 두 seed의
fit은 연구실 GPU에서 장시간 걸릴 수 있다. 이 숫자는 최종 평가의 예상 완료 시간을 보증하지 않는다.

## 두 단계 실행

PowerShell에서 프로젝트 경로를 확인한다. 기존 development handoff ZIP, 두 ZIP 원본,
`data\manifests`의 두 sealed final manifest와 기존 development 체크포인트를 그대로 둔다.
`.venv`는 프로젝트의 기존 가상환경을 사용한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git status --short
Get-Item ".\artifacts\handoff\realiad-followup-development-v1.zip", `
  "$env:USERPROFILE\Downloads\pcb.zip", `
  "$env:USERPROFILE\Downloads\terminalblock.zip"

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_realiad_followup_final.ps1"
```

첫 실행은 잠금·체크섬을 확인하고 여섯 fit/normal calibration만 준비한다. **Final 이미지는
열지 않는다.** Seed 42는 검증된 development 모델을 재사용하고 seed 43·44는 정상 train
이미지로 새로 fit한다. 중단되면 같은 명령을 다시 실행해 완료된 fit을 재사용한다. 불완전한
staging 디렉터리는 자동 삭제하지 않는다.

여섯 fit이 끝나고 화면에 `Final holdout remains sealed`가 나온 뒤, 최종 평가를 실제로
시작하기로 결정했을 때만 아래 명령을 실행한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_realiad_followup_final.ps1" `
  -OpenFinalHoldout
```

이 옵션을 주면 두 원본 ZIP의 SHA-256을 다시 확인하고 final 이미지를 ZIP에서 직접 읽는다.
최종 파일을 디스크로 추출하지 않는다. 결과는 품목·seed·이미지별로 저장하며, 중단 후 같은
명령을 재실행하면 검증된 이미지 결과는 **재점수하지 않고** 재사용한다. 손상되거나 잠금과
다른 결과는 덮어쓰지 않고 중단한다. 이미지 추론 뒤 결과 파일이 원자적으로 저장되기 전
프로세스가 강제 종료되면 그 이미지 한 장은 다시 계산될 수 있다. 이것은 새 후보 실험이
아니라 동일 고정 프로토콜의 장애 복구이며, 결과 보고서에 중단 사실을 남긴다.

완료 시 `artifacts\handoff\realiad-followup-final-v1.zip`을 Codex 작업에 첨부한다.
ZIP에는 선택 잠금·manifest·품목/seed별 fit 메타데이터·점수·집계만 포함하고 원본
이미지·마스크·모델 체크포인트·anomaly map은 포함하지 않는다. 이 ZIP의 SHA-256도 함께
전달한다. 고장·오류가 나면 경로를 지우거나 임곗값을 바꾸지 말고 화면 출력을 공유한다.

## 결과 해석의 경계

- 품목별 세 seed 값과 평균, 두 품목 macro 평균, 정상 오검출률·불량 미탐률, AUROC/AP,
  이미지 수준 95% 신뢰구간, pixel AUROC/AUPRO를 보고한다.
- Final 결과를 본 뒤 후보, crop, 임곗값, 학습 설정을 변경하면 더 이상 확인적 final
  결과로 취급하지 않는다. 변경 실험은 별도 탐색적 결과로 분리한다.
- 이 데이터는 외부 공개 이미지이며 실제 생산 공정의 합격·폐기 기준이 아니다. **자동 PASS는
  계속 금지**하고 작업자 검토를 유지한다.
- Mac에서는 final 이미지·체크포인트가 없어 실제 GPU 평가를 수행하거나 결과를 미리
  확인할 수 없다. 이번 작업에서 확인한 것은 실행기 로직과 로컬 단위 테스트다.
