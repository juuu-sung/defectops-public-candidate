# Cable·Metal Nut 고정 평가와 3품목 집계

이 단계는 완료된 12개 fit을 변경하지 않고 Cable·Metal Nut 공식 test를 처음 평가한다.
모델·전처리·normal-only 임곗값·픽셀 임곗값·지표·후처리·후보 선택 규칙은
`configs/three-category-comparison-v1.toml`에 미리 고정돼 있다.

실행 범위는 다음과 같다.

- Cable: PaDiM/PatchCore × seed 42·43·44, test 150장
- Metal Nut: PaDiM/PatchCore × seed 42·43·44, test 115장
- 기존 Transistor 6개 평가와 합쳐 총 18개 결과 무결성 검증
- 품목에 같은 가중치를 주는 seed별 macro와 3-seed 평균·표본 표준편차 산출
- 고정 순서: macro escape rate → Image AUROC → overkill rate → 필요할 때만 warm p95
- 자동 PASS는 계속 금지

연구실 Windows에서 다음을 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only

powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File ".\scripts\run_additional_category_evaluations.ps1"
```

스크립트는 완료된 fit, manifest, 결과 상태와 `metrics.json` checksum을 확인한다. 중단 후 다시
실행하면 정상 완료된 평가만 재사용한다. 불완전한 출력은 삭제하거나 덮어쓰지 않는다.

완료되면 다음 파일을 이 작업에 첨부한다.

```text
artifacts\handoff\three-category-evaluation-v1.zip
```

ZIP에는 추가 품목 12개 평가의 JSON 메타데이터, 환경 기록, 18-run 집계 JSON/Markdown이 들어간다.
대용량 anomaly map은 원래 Windows 산출물에 보존하고 handoff에는 복제하지 않는다.
