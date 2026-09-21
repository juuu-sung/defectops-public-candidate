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
- [x] 연구실 Windows에서 실제 `bent_lead/000.png` 단일 이미지 검사와 산출물 checksum 확인
- [x] PatchCore 전용 단일 이미지 검사 명령·모델 bundle 차단·탐색 상태 추적 구현
- [x] 연구실 Windows에서 PatchCore 단일 이미지 smoke test와 산출물 확인
- [x] 모델 1회 로드·요청 재사용·실패 시 통과 금지 로컬 FastAPI 구현
- [x] 연구실 Windows에서 로컬 FastAPI 실제 GPU smoke test와 상주 RAM 측정
- [x] 맥북 로컬 SQLite 검사 등록·체크섬 검증·append-only 검토 이력 구현 및 단위 테스트
- [x] 실제 Windows inspection bundle을 맥북에 복사해 SQLite 등록·무결성 재검증
- [x] 맥북 로컬 작업자 UI에서 사람의 검토 의견을 명시적으로 입력·조회하는 흐름 구현·테스트
- [x] 작업자 선택 유형으로만 유효한 모의 SOP 원문·개정판을 검색하는 흐름 구현·테스트
- [x] 같은 검토 revision의 근거만 사용하는 로컬 조치 초안·승인/반려·모의 티켓 흐름 구현·테스트
- [x] 실제 로컬 전체 시연 완료 및 개인정보 제외 완료 보고서·포트폴리오 체크리스트 작성
- [x] PaDiM·PatchCore seed 43·44 실행·검증·3-seed 집계 도구 준비
- [x] 연구실 Windows에서 seed 43·44 GPU 실행 및 3-seed 결과 수집
- [x] cable·metal_nut 추가 품목과 3품목 비교 규칙을 test 열람 전에 사전등록
- [x] cable·metal_nut 공식 데이터 manifest 생성·무결성 audit·커밋
- [x] cable·metal_nut 두 모델 × 3 seed fit/calibration 완료
- [x] cable·metal_nut 고정 test 평가와 3품목 macro 비교 완료
- [x] PatchCore seed 42 추가 품목 FN/FP 분석과 warm 속도 측정 완료
- [x] 새 미관찰 development/final holdout 수집·검증 계획 작성

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
7. 단일 이미지 검사 어댑터와 연구실 Windows 검증 완료. `bent_lead/000.png`를
   `REVIEW_REQUIRED`로 판정하고 JSON·heatmap·overlay·원시 map과 checksum을 생성했다.
8. FN/FP·결함 유형별·픽셀 겹침 오류 분석과 warm 일괄 추론 측정 도구 구현 완료.
   연구실 Windows 실행 완료. `cut_lead` recall 0.20, `misplaced` recall 1.00,
   warm 처리량 19.83 images/s를 확인했다.
9. 동일한 분할·전처리·normal-only calibration·평가 지표를 쓰는 PatchCore fit·평가 도구를
   구현하고 연구실 Windows에서 실행했다. Image AUROC 0.9904, recall 0.925,
   TP 37 / FP 1 / TN 59 / FN 3이다. PaDiM test 관찰 이후 고정한 설정이므로 모든 산출물은
   탐색적 비교로 표시한다. [PatchCore 기준선 안내](patchcore-exploratory-baseline.md)를 따라
   FN/FP 시각화와 warm 속도 측정도 완료했다. 처리량은 12.56 images/s, 전체 sample p95는
   95.91 ms다. [최종 탐색 비교 보고서](../reports/padim-vs-patchcore-transistor-exploratory-v1.md)에
   정확도·오류·속도 판단과 contact sheet 판독을 기록했다. FN은 모두 임곗값에서 4.60 이내였고,
   FP는 제품 밖 배경 표식 반응으로 보인다.
10. [PatchCore 단일 이미지 검사](patchcore-single-image.md)를 구현했다. PaDiM bundle을 잘못 넘기면
    실패하도록 모델 계약을 고정했고, 결과 JSON에 `model`과 탐색 비교 상태를 기록한다. 연구실
    Windows에서 실제 checkpoint smoke test와 산출물 확인을 완료했다. 모델 로드 포함 총 25.44초,
    이 중 설정 검증·모델 로드가 24.13초였다. 입력/overlay SHA-256은 원본·기록과 일치했다.
11. [로컬 PatchCore 검사 API](patchcore-local-api.md)를 구현했다. 모델 1회 로드, 요청 직렬화,
    실패 기록·artifact checksum 조회를 자동 테스트로 검증했다. Windows 실제 GPU 검사와
    오버레이를 확인했다. 첫 요청 총 `902.63 ms`, 상주 RAM `1.891 GiB`를 관찰했다.
12. [맥북 로컬 검토 저장소](review-store.md)를 구현했다. 검사 결과와 4개 artifact의 checksum을
    등록 전에 검증하고, 모델 판정과 분리한 작업자 의견을 revision별로 추가한다. 실제 Windows
    산출물 `1977cc17d97c4c3f95d8cc2124e9285d`를 맥북에 전송·등록·재검증했다.
    사람의 판정은 자동 생성하지 않고 비워 두었다.
13. [맥북 로컬 작업자 화면](review-ui.md)을 구현했다. 실제 등록된 검사 1건의 원본·오버레이,
    모델 점수와 revision 0을 브라우저에서 확인했다. checksum 불일치 시 이미지 표시·의견
    저장을 막고, 명시적 체크가 있어야 검토 이력을 추가하는 테스트를 통과했다. 당시에는
    사람 의견이 없었으나 이후 로컬에서 1건을 기록했다.
14. [모의 SOP 근거 검색](mock-sop.md)을 구현했다. 사람의 최신 의견이 이상 관찰일 때만
    유형을 직접 선택하고, 정확한 품목·유형·유효 개정판의 원문 절을 인용한다. 선택 결과는
    사람 검토 revision과 문서 해시·원문 스냅샷에 묶인 append-only 이력으로 저장할 수 있다.
15. [로컬 모의 조치·승인·티켓 흐름](mock-ticket-workflow.md)을 구현했다. 근거가 저장된 현재
    검토 revision에서만 보수적인 현장 확인 초안을 만들고, 승인/반려와 발행 확인을 각각
    분리했다. 승인된 초안만 외부 전송 없는 로컬 티켓 스냅샷으로 만들 수 있다.
16. 실제 등록 검사 1건의 전체 상태 전이를 완료했다. 사람 입력 원문을 제외한 결과와 해시는
    [완료 보고서](../reports/local-review-to-mock-ticket-demo-v1.md)에, 시연 순서와 실패 시
    기대 동작은 [포트폴리오 체크리스트](demo-runbook.md)에 기록했다.
17. [3-seed 탐색 반복](transistor-multiseed.md)을 연구실 Windows에서 완료했다. PatchCore는
    세 seed 모두 TP 37 / FP 1 / TN 59 / FN 3으로 동일했고 평균 Image AUROC 0.9911,
    recall 0.925를 기록했다. PaDiM 평균은 Image AUROC 0.9175, recall 0.600이었다. 동일 test를
    재사용한 탐색 반복이므로 seed 44의 근소한 순위 지표를 보고 교체하지 않고, 현재 연결된
    PatchCore seed 42를 동결 유지한다. 상세 결과는
    [3-seed 보고서](../reports/transistor-multiseed-exploratory-v1.md)에 기록했다.
18. 초기 allowlist를 유지해 추가 품목을 cable·metal_nut으로 확정했다.
    [3품목 비교 사전등록](three-category-preregistration.md)에 두 모델 × 3 seed, 전처리,
    calibration, 평가 지표와 선택 규칙을 test 열람 전에 고정했다. 다음 체크포인트는 공식
    archive를 연구실 Windows에 받아 manifest·무결성 audit 결과를 이 저장소로 돌려받았다.
    두 품목 모두 오류·디코딩 픽셀 중복 0건이며, 고정 manifest와 보고서를 커밋한다.
19. 다음은 [추가 품목 fit/calibration](additional-category-fits.md)이다. 먼저 12개 fit을 모두
    완료했다. 이 단계의 스크립트는 cable·metal_nut 공식 test를 평가하지 않았으며, manifest와
    모델 상태를 모두 재검증했다.
20. 다음은 [추가 품목 고정 평가와 3품목 집계](additional-category-evaluation.md)다. 결과를 열기
    전에 18개 평가의 무결성 검사, 품목 동일 가중 macro, 후보 선택 순서를 코드와 테스트로 고정했다.
    PatchCore macro escape rate는 4.31%, PaDiM은 35.85%로 고정 선택 규칙에 따라 PatchCore가
    포트폴리오 후보가 됐다. [3품목 결과 보고서](../reports/three-category-padim-vs-patchcore-v1.md)를
    참고한다.
21. [PatchCore 추가 품목 진단](patchcore-additional-diagnostics.md)을 완료했다. Cable은
    TP 87 / FP 4 / TN 54 / FN 5, Metal Nut은 TP 92 / FP 3 / TN 19 / FN 1이었다.
    Cable 미탐은 `cable_swap` 4장과 `poke_insulation` 1장, Metal Nut 미탐은 `color` 1장이었다.
    warm sample p95는 각각 42.04 ms와 25.97 ms였으며 모델·임곗값을 변경하지 않았다.
22. 다음은 [새 미관찰 holdout 계획](unseen-holdout-plan.md)에 따라 development와 final holdout을
    물리 부품·촬영 세션 단위로 분리해 확보하는 단계다. 새 독립 데이터가 없으면 추가 튜닝을 하지 않는다.

첫 완료 판정: 실제 transistor 입력에 대해 모델 버전·split hash·원시 점수·고정 임곗값·판정·
원본 좌표계의 히트맵/영역·지연시간을 기록하고 artifact checksum으로 실행 추적을 확인한다.
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
