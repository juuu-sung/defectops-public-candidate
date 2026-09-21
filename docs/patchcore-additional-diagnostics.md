# PatchCore 추가 품목 오류 분석과 warm 측정

3품목 고정 비교에서 PatchCore가 사전 고정한 1순위 기준인 macro escape rate로 포트폴리오 후보가
됐다. 이 단계는 결과를 개선하기 위한 튜닝이 아니라, 이미 고정된 seed 42 모델의 실패 사례와
실행 비용을 설명하기 위한 사후 진단이다.

대상은 Cable과 Metal Nut의 PatchCore seed 42다. seed 42는 Transistor 서비스 시연과 동일한
사전 기준 seed이며, 추가 품목 test 결과 중 가장 좋은 seed를 골라 선택한 것이 아니다.

연구실 Windows에서 다음을 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only

powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File ".\scripts\run_patchcore_additional_diagnostics.ps1"
```

품목별로 다음을 만든다.

- 고정 혼동행렬과 결함 유형별 recall/픽셀 겹침
- FN/FP contact sheet
- 모델 1회 로드 후 warmup 5회, 고정 test 입력 100장 측정
- 전체 sample/GPU 추론 p95, 처리량, peak GPU 메모리

모델·임곗값·ROI·후처리는 바꾸지 않는다. 완료 후 다음 파일을 이 작업에 첨부한다.

```text
artifacts\handoff\patchcore-additional-diagnostics-v1.zip
```

## 완료 결과

2026-09-21 연구실 Windows에서 완료했다. handoff ZIP과 내부 산출물 checksum, 기존 공식 평가와의
혼동행렬·manifest·checkpoint 연결을 재검증했다. Cable은 TP 87 / FP 4 / TN 54 / FN 5,
Metal Nut은 TP 92 / FP 3 / TN 19 / FN 1이었다. warm sample p95는 Cable 42.04 ms,
Metal Nut 25.97 ms였다.

오류 유형·contact sheet 판독·속도·무결성 검증은
[결과 보고서](../reports/patchcore-additional-diagnostics-v1.md)에 기록했다. 다음 단계는 현재 test를
재사용한 튜닝이 아니라 [새 미관찰 holdout 계획](unseen-holdout-plan.md)에 따른 독립 데이터 확보다.
