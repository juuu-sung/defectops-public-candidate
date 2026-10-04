# SuperSimpleNet Phase2 실행 준비 검토

2026-10-05. 사용자 승인에 따라 정식300epoch 실행기와 연구실 PowerShell 명령을 준비했다.
상태: **CPU 검증 완료, 실제 정식 GPU 학습 미실행**.

## 고정한 실행

- 공식 가중치·환경·정상64/128/169 명단·성공probe ZIP 지문 검증.
- fresh seed42 초기화 지문 대조. probe checkpoint는 purpose가 달라 load 거부.
- batch8,300epoch,28,800업데이트. epoch마다 전체768타일을 한 번씩 사용.
- 240/270epoch 완료 후 양쪽 학습률0.4배; 최종300epoch 모델만 선택.
- 10epoch마다 저장. 50epoch마다 재로딩으로 학습 정상1사진12영역 확인.
- 개발128장 native logit 중앙값/Q90 폭, 보정169장 사진 최대점수 Q95(higher).
- 정상297장 전체 최종 재로딩: 원점수·지도·사진 점수 최대차≤1e-4, 판정변경0.
- 누적4시간, GPU예약6GiB, RSS20GiB. Ctrl+C 재작업 최대960회 포함29,760시도.
- JSON-only handoff; 원시 지도/모델은 로컬 유지. 불량 시험·서비스 전환 없음.

## 검증

- `.venv/bin/python -m pytest -q`: **819 passed,52 subtests passed**,112.38초.
- 기존 경고3개: Starlette/AnyIO alias1개, 중복 ZIP 방어 fixture2개.
- 신규 정식 fit 테스트20개와 기존 probe22개, 합계42개 집중 검사 통과.
- 학습률 경계, epoch별 전수 순열, 부호 보존/native score 선택, Q90/Q95 계산,
  입력 분리, 실제 수령ZIP 명단 준비, 새 초기화 거부, checkpoint purpose 분리,
  Ctrl+C 재개·누적예산, 실패/변경/강제종료 차단, 최종 정상 지도 drift 차단,
  handoff 재사용·손상 파일 덮어쓰기 거부를 확인.
- 구문 컴파일·CLI help·diff whitespace 검사 통과.

실제 성공한100회 GPU probe의 runtime을 재사용하며, 수정은 checkpoint purpose를
클래스 속성으로 분리한 것뿐이다. 정식 subclass가 학습률 일정과 단건 저장 형식을 추가한다.
CPU 실행기 테스트는 fake runtime으로 검증했다.300epoch CUDA 학습·PowerShell 실기동·
4시간 내 완료를 보증하지 않으며, 실제 결과 ZIP이 필요하다.

[연구실 명령·설정·복구 규칙](../docs/mpi-supersimplenet-fit-v1.md).
