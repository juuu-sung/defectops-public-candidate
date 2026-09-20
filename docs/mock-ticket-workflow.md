# 로컬 모의 조치·승인·티켓 흐름

사람이 저장한 검토 의견과 모의 SOP 근거를 이용해 **로컬에서만** 상태 전이를 시험한다.
실제 작업 지시나 외부 티켓 연동이 아니며, 생산품 합격·폐기·재작업·설비 변경을 수행하지
않는다. 검토자·승인자·발행자 표기는 인증된 계정이 아닌 데모용 문자열이다.

## 상태 전이

1. 최신 사람 검토가 `ANOMALY_VISIBLE`이고 같은 revision에 모의 SOP 근거가 저장되어야 한다.
2. 별도 확인란을 선택해 보수적인 `현장 상태 재확인 요청` 초안을 저장한다. 초안은 관찰 결과
   기록만 요청하며 공정 조치는 금지한다.
3. 담당자가 초안을 `APPROVED_FOR_MOCK_TICKET` 또는 `REJECTED`로 한 번 결정한다. 결정과
   근거는 변경·삭제되지 않는다. 반려된 초안에서는 티켓을 만들 수 없다.
4. 승인 후에도 별도 확인란을 선택해야 로컬 모의 티켓이 생성된다. 티켓에는 검사 기록,
   사람 검토, SOP 원문 스냅샷, 초안, 승인이 묶이고 SHA-256이 기록된다.

각 단계는 현재 `review_revision`과 일치해야 한다. 중간에 새 사람 검토가 추가되면 이전
revision의 초안·승인으로 새 티켓을 발행할 수 없다. 원본 검사 bundle의 checksum도 쓰기
직전에 다시 확인한다. 모든 표에는 update/delete 차단 trigger가 있으며 동일 요청 재시도는
같은 내용일 때만 중복 없이 성공한다.

## 화면 사용

`python scripts/serve_review_ui.py`로 [로컬 검토 화면](review-ui.md)을 열고 저장된 모의 SOP
근거 아래의 **모의 조치·승인·티켓** 영역을 사용한다. 화면에는 현재 상태가 다음 중 하나로
표시된다.

- 조치 초안 없음
- 승인 대기
- 모의 티켓 발행 승인
- 반려
- 모의 티켓 발행 완료

모든 결과는 `artifacts/review/reviews.sqlite3`에만 저장되고 네트워크로 전송되지 않는다.

## CLI 계약 확인

화면과 같은 상태 전이를 CLI에서도 실행·조회할 수 있다. 아래 값은 예시이므로 실제 현재
revision과 각 명령 결과의 event ID를 사용한다.

```sh
python scripts/review_inspections.py draft-action \
  --inspection-id <inspection-id> \
  --expected-review-revision 1 \
  --created-by operator-1
python scripts/review_inspections.py action-history --inspection-id <inspection-id>

python scripts/review_inspections.py decide-action \
  --inspection-id <inspection-id> \
  --expected-review-revision 1 \
  --action-draft-event-id 1 \
  --approver operator-1 \
  --decision APPROVED_FOR_MOCK_TICKET \
  --note "로컬 화면 흐름 시험용 승인"
python scripts/review_inspections.py approval-history --inspection-id <inspection-id>

python scripts/review_inspections.py issue-mock-ticket \
  --inspection-id <inspection-id> \
  --expected-review-revision 1 \
  --approval-event-id 1 \
  --issued-by operator-1
python scripts/review_inspections.py ticket-history --inspection-id <inspection-id>
```

개인정보·자유 입력 note·원본 경로·인용 원문을 제외한 Git-safe 결과는 다음 명령으로 확인한다.

```sh
python scripts/review_inspections.py ticket-summary --inspection-id <inspection-id>
```

이 단계가 완료되어도 실제 현장 적용 준비가 된 것은 아니다. 다음 확장 전에 인증·권한 분리,
실제 승인 SOP, 감사 로그 보관 정책, 외부 티켓 연동에 대한 별도 설계와 승인이 필요하다.
