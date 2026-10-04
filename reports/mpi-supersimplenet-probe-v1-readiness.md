# SuperSimpleNet Phase 0·1 준비 검토

2026-10-04. 상태: **코드·CPU 검증 완료 / 연구실 GPU 점검 대기**.
사용자 승인 범위는 실행기·검증 준비와 GPU 사전 점검이다. 정식 학습은 미실행이다.

## 구현

- 기존 정상64장 준비 ZIP·원본 archive·고정8사진96타일·순열 스케줄 지문 검증.
- 공식 Anomalib2.2.0 소스5개 SHA256 및 timm1.0.20 확인.
- 공식 timm 가중치의 고정 revision·전체 SHA256 확인, offline backbone 초기화.
- batch8(특징 단계16)·FP32·최대100 attempted updates.
- 정상0마스크·0라벨, 유한 손실/gradient/가중치, 정상·합성 클래스 발생 확인.
- 단건 raw logit/scalar 처리와 배치8 지도 검증. 음수값을 잘라내지 않음.
- 25회 checkpoint,50/100회 모델·optimizer·난수 복원과 점수/지도 차이≤1e-4.
- backbone 가중치/통계 불변, adaptor/두 head 변화 확인.
- 계산900초·GPU reserved6GiB·RSS20GiB, 실패·강제 종료 시 자동 축소/재시도 금지.
- 기존 성공 산출물 재사용, JSON-only 전달 ZIP. 기존 학습·서비스 파일 수정 없음.

## 검증 근거

- 전체: `.venv/bin/python -m pytest -q` → **799 passed,52 subtests passed**,113.85초.
- 경고3개: 기존 Starlette/AnyIO deprecated alias1개와 ZIP 중복항목 검사 fixture2개.
- 신규22개 테스트: 스케줄·출력·소스 지문·정상 전용 준비·재로딩·변경 거부·재개·실패 보존.
- 기존 EfficientAD 점검과 합쳐41개 테스트 통과.
- Python 구문 컴파일, CLI `--help`, `git diff --check` 통과.
- 실제 로컬 준비 ZIP에서8사진96타일의 고정 명단·체크섬 확인.
- 공식 가중치 원격 메타데이터와 작은 safetensors 헤더 확인. Mac에 가중치 전체를 받거나 학습하지 않음.

CPU orchestration 테스트의 runtime은 명시적인 fake다. 실제 torch/CUDA 실행,8GB 메모리 통과,
PowerShell 실행, 다운로드 후 GPU 가중치 적재는 연구실 결과가 필요하다.
위 테스트 수는 불량 검출 성능이나 정식 학습 준비 완료의 증거가 아니다.

## 다음

[연구실 실행 안내](../docs/mpi-supersimplenet-probe-v1.md)에 따라 GPU 점검을 실행하고
`mpi-supersimplenet-probe-v1.zip`을 검토한다. 자원/실행 검증 통과 뒤 정식300epoch 조건과
학습 예산을 잠근다. 자동 정식 학습·불량 평가·Q90 교체·서비스 전환은 없다.
