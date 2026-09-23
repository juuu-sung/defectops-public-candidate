# Real-IAD PCB final 저장 맵 정량 진단

상태: 2026-09-23에 **읽기 전용 구현 완료, 연구실 Windows 결과 미수신**.

앞선 [PCB final 시각 진단](realiad-followup-final-diagnostics.md)에서는 세 seed가 모두
놓친 불량 67장과 공통 오검출 정상 3장의 원본·정답 윤곽·히트맵을 확인했다. 표시용
히트맵은 pixel 임곗값으로 정규화·포화되므로 색상만으로 실제 점수 기여를 판단할 수 없다.
이번 단계는 이미 저장된 `.npz` 맵의 **원점수**를 읽어 그 차이를 수치로 설명한다.

## 계산과 한계

- 봉인된 final handoff, 이전 시각 진단 ZIP, PCB 원본 ZIP의 SHA-256을 검증한다.
  저장된 360장 × 3 seed = 1,080개 `.npz`도 final 평가 JSON의 파일 SHA-256·내부
  digest·이미지 ID·checkpoint·임곗값·원본 마스크와 대조한다. GPU 추론은 실행하지 않는다.
- 기존 후보의 이미지 점수는 256×256 맵 중 상위 1%인 **656픽셀의 평균**이다.
  저장 맵으로 이 점수를 다시 계산해 원래 기록과 `1e-5` 이내로 일치해야 계속한다.
- 결함 이미지에는 정답 영역 면적, 정답/비정답 픽셀 점수의 상대 순위(AUC), 정답
  픽셀의 상위 1% 포함률, 상위 1% 점수 합계 중 정답 영역의 몫, 고정 pixel 임곗값
  이상인 정답 픽셀 비율을 기록한다. 상위 1% 경계에서 동점이 있으면 정답 포함 수와
  점수 몫의 **최소·최대 범위**를 기록한다.
- 정상 오검출에는 정답 마스크가 없으므로 정답 관련 지표를 `null`로 두고 이미지
  점수 재구성 및 맵의 pixel 임곗값 이상 비율만 기록한다.
- 360장 전체의 상세 기록과 70장 검토 대상만 추린 기록을 함께 만든다. 집계는 세
  seed를 독립 이미지로 세지 않고 **이미지별 세 seed 평균의 중앙값**으로 표시한다.

이것은 이미 본 final의 **사후 설명적 분석**이다. GT 픽셀 순위가 높아도 이미지
판정 안전성을 뜻하지 않으며, 결과를 보고 현재 final의 모델·후보·임곗값을 바꾸거나
확인적 성능으로 재보고하지 않는다. 자동 PASS는 계속 금지한다.

## 연구실 PC에서 할 일

기존 `C:\개인 프로젝트\defectops\.venv`를 그대로 사용한다. PowerShell에서 아래만
실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_realiad_followup_map_attribution.ps1"
```

스크립트는 이전 결과를 덮어쓰지 않는다. 완료하면
`C:\개인 프로젝트\defectops\artifacts\handoff\realiad-followup-map-attribution-v1.zip`과
출력된 SHA-256을 이 작업에 첨부한다. ZIP에는 집계 JSON과 사례별 JSONL만 들어
있고 원본 이미지·맵·체크포인트는 포함하지 않는다. 오류가 나면 원본이나 기존
결과를 지우지 말고 오류 출력을 공유한다.
