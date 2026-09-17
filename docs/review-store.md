# 맥북 로컬 검사 등록·작업자 검토 이력

연구실 Windows에서 생성한 `artifacts/inspections/<inspection_id>/` **전체 폴더**를 안전한
오프라인 방법으로 맥북에 복사한 뒤 등록한다. Git에는 원본 이미지, 시각화 산출물, SQLite DB를
넣지 않는다. 등록과 검토는 현재 맥북 로컬 CLI에서만 가능하며 인증된 작업자 UI나 원격 연결은
아직 없다.

## 보존 규칙

- `inspection.json`의 모델 판정·점수·임곗값·실행 출처는 변경하지 않고 SQLite에 스냅샷으로
  보존한다. 등록 전 원본/heatmap/overlay/anomaly map의 SHA-256과 실패 시 통과 금지 계약을
  검증한다. 파일이 빠졌거나 바뀌면 등록·재검증·새 검토 기록을 거부한다.
- 원본 산출물 폴더는 등록 후에도 **그 자리에 유지**해야 한다. SQLite에는 파일 본문을 복사하지
  않는다. 검토 이력은 별도 표에 revision 1, 2, …로 추가되며 기존 기록은 덮어쓰지 않는다.
- 사람의 선택지는 `ANOMALY_VISIBLE`, `NO_ANOMALY_VISIBLE`, `UNCERTAIN`이다. 이는 화면을
  본 사람의 의견이며 실제 합격·불량 원인·안전 인증이 아니다. 모델의 `PASS`도 자동 합격으로
  바꾸지 않는다. 검사 실패에는 시각 검토 판정을 붙일 수 없다.
- 새 검토에는 화면에 표시된 `review_revision`을 `--expected-revision`으로 넘긴다. 다른 검토가
  먼저 기록됐다면 충돌로 거부해 덮어쓰기를 방지한다.
- `--reviewer`는 현재 사용자가 입력한 문자열일 뿐 인증된 신원이 아니다. 이 CLI/DB를 여러
  사람이 공유하는 운영 서비스로 공개하지 않는다.

## 실행 예시 — 맥북

전체 폴더가 맥북의 `artifacts/inspections/1977cc17d97c4c3f95d8cc2124e9285d/`에
복사된 경우:

```sh
python3 scripts/review_inspections.py init
python3 scripts/review_inspections.py import \
  --bundle artifacts/inspections/1977cc17d97c4c3f95d8cc2124e9285d
python3 scripts/review_inspections.py verify \
  --inspection-id 1977cc17d97c4c3f95d8cc2124e9285d
python3 scripts/review_inspections.py list
```

실제 이미지를 작업자가 확인한 뒤에만 검토를 입력한다. 아래 문장은 예시이며 실제 작업자
판정을 대신하지 않는다.

```sh
python3 scripts/review_inspections.py review \
  --inspection-id 1977cc17d97c4c3f95d8cc2124e9285d \
  --expected-revision 0 \
  --reviewer operator-1 \
  --verdict ANOMALY_VISIBLE \
  --note "왼쪽 리드의 굽힘이 이미지에서 보임"
python3 scripts/review_inspections.py history \
  --inspection-id 1977cc17d97c4c3f95d8cc2124e9285d
```

기본 DB는 `artifacts/review/reviews.sqlite3`다. DB와 원본 산출물은 모두 Git 제외 상태다.
실제 Windows bundle을 맥북으로 옮겨 등록하는 통합 검증은 아직 하지 않았다. 다음 단계는
이 저장소 위에 로컬 작업자 UI를 만들고, 이후 인증된 장비 간 전송 경로를 정하는 것이다.
