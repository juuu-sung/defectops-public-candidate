# Real-IAD PCB final 결과 진단 — 연구실 Windows 실행

이 작업은 이미 완료된 final 평가의 **사후 설명적 검토**다. 학습·재추론·임곗값 조정·
final 재평가를 하지 않는다. 숫자 분석은 Mac에서 완료됐고
[`reports/realiad-followup-final-posthoc-errors-v1.md`](../reports/realiad-followup-final-posthoc-errors-v1.md)에
기록했다. 연구실 PC에서만 가능한 일은 기존 원본과 저장된 히트맵을 시각적으로 대조하는 것이다.

PowerShell에서 아래만 실행한다. 기존 `C:\개인 프로젝트\defectops\.venv`를 그대로 쓴다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_realiad_followup_final_diagnostics.ps1"
```

필요한 기존 파일은 `artifacts\handoff\realiad-followup-final-v1.zip`,
`$env:USERPROFILE\Downloads\pcb.zip`, 그리고
`artifacts\realiad-followup-final-v1\evaluations\pcb-seed{42,43,44}\samples\sample-####.npz`다.
스크립트가 파일 체크섬·manifest·예측 순서·세 seed의 저장된 맵·원본 이미지와 마스크를
검증한다. 하나라도 없거나 다르면 **덮어쓰거나 다시 추론하지 않고 중단**한다.
`run_realiad_followup_final.ps1`은 다시 실행하지 않는다. 이전 final 실행은 이전 commit에
잠겨 있으며, 이 진단 스크립트는 그 결과를 읽기만 한다.

완료하면 `C:\개인 프로젝트\defectops\artifacts\handoff\realiad-followup-final-errors-v1.zip`과
화면의 SHA-256을 Codex 작업에 첨부한다. ZIP에는 360장 전체의 판정표와 공통 미탐 67장·
공통 오검출 3장의 시트가 들어 있다. 시트는 결함 유형별로 5장씩 나뉘며, 각 사례에서
왼쪽은 원본+정답 윤곽(녹색), 오른쪽 세 칸은 seed 42/43/44의 저장된 anomaly map이다.
색상은 상대 강도를 보여 줄 뿐 판정 근거가 아니며 점수/이미지 임곗값은 별도로 표기한다.

Real-IAD 라이선스와 이미지 재배포 범위를 확인하기 전에는 이 ZIP을 공개 저장소나
공개 Notion 페이지에 올리지 않는다. 본 진단은 원인 가설을 세우기 위한 설명 자료다.
후속 개선이 필요하면 development 자료로 별도 탐색을 설계하고, 새 미관측 holdout에서
다시 검증한다. **자동 PASS는 계속 금지**한다.
