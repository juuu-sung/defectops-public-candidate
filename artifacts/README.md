# 실행 산출물 (Git 제외)

장비별 환경 점검 JSON, 실행별 모델·원시 점수 CSV·히트맵·모의 SQLite DB를 이곳에 저장한다.
PaDiM 및 탐색적 PatchCore fit bundle과 고정 test evaluation bundle 생성기를 구현했다.
모든 실제 산출물은 Git에서 제외한다.

```text
artifacts/<run_id>/
  run.json          # commit, dirty flag, config/split/checkpoint hash, versions, device
  config.toml
  metrics.json
  predictions.csv  # 상대 이미지 경로, 라벨, 원시 점수, 고정 판정
  calibration.json # 표본 수, 방식, 임곗값, 점수 공간
  model/           # weights + preprocessing/postprocessing contract
  heatmaps/
  checksums.sha256
```

현재 구현 경로는 `artifacts/runs/<fit_run_id>/`와 `artifacts/evaluations/<evaluation_id>/`다.
평가 bundle은 fit bundle을 수정하지 않고 checkpoint·manifest·sample ID·checksum을 확인한 뒤 생성한다.
PatchCore bundle에는 coreset memory-bank shape와 크기도 기록하며, PaDiM test 관찰 이후의
비교라는 탐색적 상태를 fit·평가 양쪽에 보존한다.
단일 이미지 검사는 `artifacts/inspections/<inspection_id>/`에 JSON·원본 복사·heatmap·overlay·
원시 anomaly map을 저장하며, fit·evaluation bundle과 commit·checksum을 먼저 검증한다.
탐색적 오류 분석은 `artifacts/analyses/<analysis_id>/`, 모델 1회 로드 속도 측정은
`artifacts/benchmarks/<benchmark_id>/`에 저장한다. 이 산출물도 Git에 넣지 않는다.

실행 ID에는 품목·모델·seed·시각 또는 UUID를 포함한다. 실행 디렉터리는 덮어쓰지 않는다.
결과 복사 후 체크섬을 검증하고, 요약 성능표·실험 결론만 검토하여 `reports/`에 기록한다.
SQLite 실행 중인 파일을 두 장비에서 공유/동기화하지 않는다.
