# 실행 산출물 (Git 제외)

장비별 환경 점검 JSON, 실행별 모델·원시 점수 CSV·히트맵·모의 SQLite DB를 이곳에 저장한다.
환경 점검 도구를 제외한 실행 번들 생성기는 후속 구현 대상이다.

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

실행 ID에는 품목·모델·seed·시각 또는 UUID를 포함한다. 실행 디렉터리는 덮어쓰지 않는다.
결과 복사 후 체크섬을 검증하고, 요약 성능표·실험 결론만 검토하여 `reports/`에 기록한다.
SQLite 실행 중인 파일을 두 장비에서 공유/동기화하지 않는다.
