# 맥북 로컬 검사 등록·작업자 검토 이력

연구실 Windows에서 생성한 `artifacts/inspections/<inspection_id>/` **전체 폴더**를 안전한
오프라인 방법으로 맥북에 복사한 뒤 등록한다. Git에는 원본 이미지, 시각화 산출물, SQLite DB를
넣지 않는다. 등록과 검토 이력은 맥북 로컬 CLI에서 관리하며, 등록된 검사 확인과 명시적
의견 입력은 [로컬 작업자 화면](review-ui.md)에서도 가능하다. 인증된 다중 사용자 서비스나
원격 연결은 아직 없다.

2026-11-14: 같은 장비에서는 [API 자동 등록·재등록](api-review-integration.md)을 사용할 수 있다. 아래 수동 import는 다른 장비의 산출물을 가져오는 대안으로 유지한다.

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
- 모의 SOP 근거 연결은 최신 `ANOMALY_VISIBLE` 검토 revision에만 추가할 수 있다. 선택한
  유형과 문서 ID/개정판/해시/원문 스냅샷을 별도 append-only 표에 한 번만 남긴다. 이 기록은
  실제 원인 확정이나 조치 승인이 아니다.
- 저장된 근거 뒤에는 [로컬 모의 티켓 흐름](mock-ticket-workflow.md)을 연결할 수 있다. 조치
  초안, 승인/반려, 모의 티켓을 각각 별도 append-only 표에 남기며 같은 검토 revision만
  사용할 수 있다. 승인과 별도 발행 확인 없이는 티켓을 만들 수 없다.

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
2026-09-17 실제 Windows bundle `1977cc17d97c4c3f95d8cc2124e9285d`를 ZIP으로 받아
맥북에 보관·등록·재검증했다. `inspection.json` SHA-256은
`968b290d9c4d746810b8a4f0035256a5b632d71c5fde44b9f2f7e4aefd403f5f`, 입력 PNG는
맥북 원본과 같은 `82bf661fb868a80558fc45f4fc356c96888bf26827005a57aea6aa707e13483a`다.
등록 당시에는 `review_revision=0`, `human_verdict=null`이었다. 2026-09-18 이 저장소
위에 로컬 작업자 화면을 만들고 실제 등록 검사 이미지 표시를 확인했다. 이후 사람의 실제
검토 의견 1건이 로컬 DB에 추가되었다. 이름·의견 원문은 Git에 포함하지 않는다.
이후 [모의 SOP 검색](mock-sop.md)과 로컬 조치 초안·승인·모의 티켓 계약까지 구현했다.
2026-09-21에는 해당 사람 검토 revision 1에 모의 문서 `DEMO-TR-BENT` 개정 2의 근거
스냅샷 1건을 연결했다. 라이브 DB와 선택자 정보는 계속 Git 제외 상태다.
