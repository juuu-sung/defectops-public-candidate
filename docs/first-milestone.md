# 첫 마일스톤

## 이번 착수에서 완료

- [x] 검토 결과 및 MVP 축소 범위 기록
- [x] 데이터·실험 프로토콜 초안과 파일럿 설정 작성
- [x] 장비 경로와 독립적인 manifest 생성/검증 도구
- [x] 재현성·누락·데이터 중복·파일 변경 테스트
- [x] 환경 점검 도구와 장비 간 전달 절차
- [x] 실제 transistor 이미지 313장·마스크 40장 무결성/디코딩 점검 (오류 0건)
- [x] train 170장 / calibration 43장 / test 100장 분할 고정 및 checksum 재검증
- [x] 정상 데이터 밝기·선명도 기초 통계, 데이터 점검 테스트 포함 총 17개 통과
- [x] Windows RTX 5060에서 PaDiM fit·normal calibration·고정 test 100장 평가
- [x] 단일 이미지 추론 JSON·heatmap·overlay·원본 좌표 영역·실패 기록 구현

## 다음 순서

1. 연구실 native Windows/RTX 5060 8GB/RAM 32GB/Python 3.12.0/드라이버 591.86 확인 완료.
   프로젝트 내부 `.venv`에서 PyTorch 2.8.0+cu128/CUDA runtime 12.8/Anomalib 2.2.0
   기본 GPU 점검을 통과했다. 저장소 전체 점검과 실제 학습 검증은 남아 있다.
2. 맥북 실제 데이터 준비·manifest·기초 EDA 완료. [보고서](../reports/transistor-data-audit-v1.md) 참고.
3. 비공개 원격 Git 저장소 연결 및 첫 push 완료. 연구실에서 코드·manifest 수신과 검증.
4. [Windows 설치 안내](windows-setup.md)의 초기 패키지 조합으로 설치 → GPU 연산/모델 로딩 점검 후 버전 확정.
5. manifest 고정 PaDiM fit·normal-only calibration 명령 구현 완료. 연구실에서 seed 42 첫 실행.
   RTX 5060에서 `fit_calibrated`, 임곗값 `56.027042388916016`, 20.225초로 완료했다.
6. normal-only pixel calibration·bbox 규칙과 공식 test 평가 완료. Image AUROC 0.9013,
   고정 임곗값 recall 0.55, Pixel AUROC 0.9689. 현재 임곗값은 자동 PASS에 사용하지 않는다.
7. 단일 이미지 검사 어댑터 구현 완료. 연구실 Windows에서 실제 fit·evaluation
   bundle로 산출물을 확인한다.

첫 완료 판정: 실제 transistor 입력에 대해 모델 버전·split hash·원시 점수·고정 임곗값·판정·
원본 좌표계의 히트맵/영역·지연시간을 기록한다. 같은 artifact로 맥북 재추론을 확인한다.
모델이 없는 상태의 모의 점수는 이 완료 조건에 포함되지 않는다.

## 이후 단계와 완료 기준

| 단계 | 결과 |
|---|---|
| 비교 | 사전 고정한 PaDiM/PatchCore × 3품목 × 3 seed, 성능/메모리/속도 표 |
| 검사 서비스 | FastAPI + SQLite, 실패 시 통과 금지, trace/model version 저장 |
| 작업자 UI | Streamlit 검사/검토, 변경 이력, PASS 표본 검토 설계 |
| 조치 지원 | 유효 SOP 검색, 원문 인용, LLM 장애 fallback, revision 기반 승인 |
| 데모 완성 | 티켓 중복 방지, 통합 테스트, Docker, 문서와 시연 영상 |

EfficientAD/AD 2/ONNX/OpenVINO는 MVP 통과 이후 비용과 효과를 보고 추가한다.
