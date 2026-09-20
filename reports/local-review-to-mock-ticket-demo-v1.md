# 로컬 작업자 검토 → 모의 티켓 시연 결과

검증일: 2026-09-21  
실행 환경: 맥북 로컬 Streamlit + SQLite  
상태: 전체 상태 전이 완료  
범위: 화면 검토·모의 SOP 근거·로컬 승인·모의 티켓 스냅샷

이 보고서는 로컬 DB의 `ticket-summary` 결과만 사용했다. 검토자·승인자·발행자 이름,
사람이 입력한 관찰/승인 원문, 원본 이미지 경로와 SOP 원문은 포함하지 않는다.

## 완료 결과

| 항목 | 값 |
|---|---|
| 실행 상태 | `completed_local_simulation` |
| 티켓 ID | `MOCK-11CD0B8CFAB2` |
| 검토 revision | 1 |
| 모델 | PatchCore |
| 모델 표시 | `REVIEW_REQUIRED` |
| 사람의 화면 관찰 | `ANOMALY_VISIBLE` |
| 모의 SOP 유형 | `bent_lead` |
| 모의 문서 | `DEMO-TR-BENT` 개정 2 |
| 승인 상태 | `APPROVED_FOR_MOCK_TICKET` |
| 자동 합격 허용 | `false` |
| 외부 티켓 전송 | `false` |

모델 원시 점수는 `52.699710845947266`, 고정 이미지 임곗값은
`40.7912712097168`이었다. 점수는 결함 확률·심각도·원인이 아니며, 이 시연에서도
모델 출력을 자동 합격이나 현장 조치로 변환하지 않았다.

## 상태 전이

```text
PatchCore REVIEW_REQUIRED
  → 사람의 화면 관찰 revision 1
  → 사람이 직접 선택한 모의 SOP 근거 스냅샷
  → [모의] 현장 상태 재확인 요청 초안
  → 모의 티켓 발행 승인
  → 로컬 모의 티켓 MOCK-11CD0B8CFAB2
```

각 단계는 별도 확인과 저장 동작을 요구했다. 승인 전 티켓 발행, 반려 후 티켓 발행,
다른 검토 revision의 근거 재사용은 서버 저장 계약에서 차단된다. 초안·승인·티켓은
SQLite append-only 표에 기록되며 update/delete trigger가 기존 이력 변경을 거부한다.

## 무결성 근거

- 검사 식별자 대신 사용한 fingerprint SHA-256:
  `a25d0986dbaf517fe95543264d431ff1d691e7e9c728ea8ae011ecb1c731f3a5`
- 모의 티켓 payload SHA-256:
  `c4462c8a2f8670d1b0e7f1772f9769d27fd50c7b1f21a65f2f4b5078dd3ccb74`
- PatchCore checkpoint SHA-256:
  `13e3cad9465c7cdbf8ca95acd9905637c42ae24140607d9d580e5ad00986ea8d`
- 고정 manifest SHA-256:
  `9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585`
- 모의 SOP catalog SHA-256:
  `f9cb50432f1599aa9516c20bb26075815eb7af17f5c1541e212af7af80f67481`

티켓 payload를 읽을 때도 저장된 SHA-256을 다시 계산한다. 값이 다르면 요약 생성을
거부한다. 실제 DB와 검사 산출물은 Git 제외 상태다. 이 기능을 포함한 저장소 전체 자동
테스트 75개가 통과했다.

## 재현 가능한 공개 요약

로컬 원본을 공개하지 않고 같은 필드만 다시 확인하려면 다음 명령을 사용한다.

```sh
python scripts/review_inspections.py ticket-summary \
  --inspection-id <로컬 검사 ID>
```

출력에는 이름, 자유 입력 note, 원본 경로, 인용 원문이 포함되지 않는다. 검사 ID도 원문
대신 SHA-256 fingerprint만 출력한다.

## 결론과 한계

단일 등록 검사에 대해 모델 결과 → 사람 검토 → 근거 연결 → 초안 → 승인 → 로컬 티켓의
기술적 흐름이 끝까지 동작했다. 이것은 워크플로우의 재현성과 실패 시 차단을 검증한 것이며,
실제 공정 승인·안전 인증·생산성 개선을 입증한 결과가 아니다. 문서는 모의 SOP이고 티켓은
외부 시스템에 전송되지 않았다. 실제 적용에는 인증된 사용자, 권한 분리, 승인된 현장 SOP,
감사 로그 보관 정책과 별도 운영 평가가 필요하다.
