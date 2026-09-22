# DefectOps

정상 데이터 기반 제조 이상탐지 및 작업자 조치 지원 시스템.

이미지 이상탐지 → 작업자 검토 → 근거 SOP 검색 → 승인 후 모의 티켓 생성.
이미지로 공정 원인을 확정하거나 이상 점수를 심각도로 해석하지 않는다.
MVTec는 실제 공장 생산성·안전성 검증 자료가 아니다.

## 현재 상태

데이터 준비 단계 완료. 실제 transistor 이미지 313장·마스크 40장의 디코딩과 무결성을 확인하고
정상 train 170장 / calibration 43장 / 공식 test 100장으로 manifest를 고정했다.
데이터 관련 테스트 17개 통과. [데이터 점검 보고서](reports/transistor-data-audit-v1.md)를 참고한다.
고정 manifest의 train 170장으로 PaDiM을 fit하고 정상 calibration 43장으로 원시 점수 임곗값을
산출하는 첫 실행을 연구실 Windows RTX 5060에서 완료했다. seed 42 실행은 임곗값
`56.027042388916016`, fit+calibration 20.225초를 기록했다. 고정 test 평가 규칙과 실행 명령을
구현하고 연구실에서 공식 test 100장 파일럿 평가까지 완료했다. Image AUROC 0.9013,
고정 임곗값 recall 0.55, Pixel AUROC 0.9689를 기록했다. 낮은 recall 때문에 현재 모델을
실제 자동 PASS 용도로 사용할 수 없다. [파일럿 평가 보고서](reports/padim-transistor-seed42-pilot.md)를 참고한다.
같은 checkpoint·고정 임곗값으로 실제 `bent_lead/000.png` 한 장을 검사해
`REVIEW_REQUIRED`, heatmap·overlay·영역 JSON 생성과 실행 추적을 확인했다.
[단일 이미지 검증 보고서](reports/padim-single-image-smoke.md)를 참고한다.
고정 결과의 [오류 분석과 warm 추론 측정](docs/padim-error-analysis-and-benchmark.md)을 완료했다.
`cut_lead` recall 0.20이 가장 큰 취약점이었고, 모델 1회 로드 시 19.83 images/s를 기록했다.
[오류 분석 및 속도 보고서](reports/padim-error-analysis-warm-v1.md)를 참고한다.
같은 고정 manifest와 평가 규칙을 사용하는 [PatchCore 탐색적 기준선](docs/patchcore-exploratory-baseline.md)을
연구실 Windows에서 fit·평가했다. Image AUROC 0.9904, 고정 임곗값 recall 0.925,
Pixel AUPRO 0.9454였으며 TP 37 / FP 1 / TN 59 / FN 3을 기록했다. PaDiM test 결과를 이미
관찰한 뒤 설정했으므로 확증적 모델 선택이 아닌 탐색적 비교다. PatchCore 오류 시각화와
warm 속도 측정도 완료했다. PatchCore warm 처리량은 12.56 images/s, 전체 sample p95는
95.91 ms였다. 정확도 우선 주력 후보는 PatchCore, 가벼운 비교 baseline은 PaDiM으로 유지한다.
contact sheet 판독 결과 FN 3장은 모두 임곗값 근처였고, 두 `misplaced` 사례는 pixel 예측이
GT와 겹치는데도 image score 기준에서 탈락했다. FP 1장은 제품 밖 배경 표식 반응으로 보인다.
[PaDiM–PatchCore 최종 탐색 비교 보고서](reports/padim-vs-patchcore-transistor-exploratory-v1.md)를 참고한다.
동결된 PatchCore bundle을 사용하는 [단일 이미지 검사 명령](docs/patchcore-single-image.md)을
구현했으며 연구실 Windows smoke test를 남겨 두었다.
모델을 한 번만 로드하는 [로컬 PatchCore 검사 API](docs/patchcore-local-api.md)를 구현하고
연구실 Windows에서 실제 이미지·오버레이를 확인했다. 첫 요청 총 902.63 ms, 상주 RAM
1.891 GiB를 관찰했다. [맥북 로컬 SQLite 검토 이력](docs/review-store.md)도 구현했다.
실제 Windows 산출물의 맥북 등록·체크섬 재검증도 완료했다. [맥북 로컬 작업자 화면](docs/review-ui.md)은
원본·오버레이·히트맵, 모델 점수, 검토 이력을 보여주고 사람이 명시적으로 의견을 저장한다.
실제 등록 검사 1건의 화면 표시를 확인했고 사람의 실제 의견 1건은 로컬 DB에만 남겼다.
[모의 SOP 원문 검색](docs/mock-sop.md)은 사람이 직접 선택한 유형으로 유효 개정판을
조회하고, 선택한 원문 스냅샷을 사람 검토 revision에 묶어 저장한다. 실제 현장 SOP가
아니다. 이어서 [로컬 모의 조치·승인·티켓 흐름](docs/mock-ticket-workflow.md)을 구현했다.
저장된 근거와 같은 검토 revision에서만 보수적인 확인 초안을 만들 수 있고, 명시적 승인과
별도 발행 확인을 거쳐야 외부 전송 없는 모의 티켓 스냅샷이 생성된다.
실제 등록 검사 1건의 전체 로컬 시연도 완료했으며, 개인정보를 제외한 결과는
[완료 보고서](reports/local-review-to-mock-ticket-demo-v1.md), 재현 순서는
[포트폴리오 시연 체크리스트](docs/demo-runbook.md)에 정리했다.
[transistor 3-seed 탐색 반복](docs/transistor-multiseed.md)은 연구실 Windows에서 완료했다.
PatchCore는 seed 42·43·44 모두 TP 37 / FP 1 / TN 59 / FN 3으로 동일했고, 평균 Image AUROC
0.9911, recall 0.925, escape rate 7.5%를 기록했다. PaDiM은 평균 Image AUROC 0.9175,
recall 0.600, escape rate 40.0%였으며 seed에 따른 판정 변동도 더 컸다. 따라서 정확도 우선
주력 후보는 PatchCore로 유지한다. 다만 같은 test를 반복 사용한 탐색 결과이므로 seed 44의
근소한 순위 지표를 근거로 모델을 교체하지 않고, 이미 서비스에 연결한 seed 42 bundle을
동결해 유지한다. [3-seed 결과 보고서](reports/transistor-multiseed-exploratory-v1.md)를 참고한다.
Cable·Metal Nut을 포함한 3품목 고정 비교도 완료했다. 사전 고정한 품목 동일 가중 macro
escape rate는 PatchCore 4.31%, PaDiM 35.85%로 PatchCore가 정확도 우선 포트폴리오 후보가 됐다.
[3품목 결과 보고서](reports/three-category-padim-vs-patchcore-v1.md)를 참고한다. 이어서 seed 42
PatchCore의 [추가 품목 오류 분석과 warm 측정](reports/patchcore-additional-diagnostics-v1.md)을 완료했다.
Cable 미탐은 `cable_swap` 4장과 `poke_insulation` 1장, Metal Nut 미탐은 `color` 1장이었다.
warm sample p95는 Cable 42.04 ms, Metal Nut 25.97 ms였고 설정을 변경하지 않았다. 다음 변경은
[새 미관찰 holdout 계획](docs/unseen-holdout-plan.md)에 따라 독립 데이터에서만 수행한다.
직접 촬영을 대체할 외부 데이터로 Real-IAD 256의 `transistor1`과 `plastic_nut`을 확보했다.
실제 이미지를 확인한 결과 기존 MVTec 품목과 다른 물체이므로 동일제품 holdout으로 재표현하지 않고,
별도 `realiad_transistor1`·`realiad_plastic_nut` 카테고리로 등록한다. 공식 단일 시점 C1 분할,
3개 seed, test 이전 전체 fit 완료 조건은
[Real-IAD PatchCore 프로토콜](docs/realiad-patchcore-protocol.md)에 고정했다.
[Real-IAD 고정 평가](reports/realiad-patchcore-evaluation-v1.md)는 순위화·위치 검출은 유망하지만
현재 고정 image 임곗값의 평균 미탐률이 두 품목 모두 약 40%임을 보여 자동 합격 판정을 보류했다.
[3-seed 오류 진단](reports/realiad-patchcore-multiseed-diagnostics-v1.md)에서 공통 미탐은
`transistor1` 166/469장(35.39%), `plastic_nut` 48/118장(40.68%)으로, seed 불안정
49/469장(10.45%)·4/118장(3.39%)보다 많았다. seed 선택 문제가 아닌 구조적 개선 대상으로 보고,
새 development split과 미관찰 holdout을 고정하기 전에는 crop·특징·image score를 변경하지 않는다.
기존 두 품목의 다른 시점은 동일 표본 누수가 있어 제외하고, 아직 미관찰인 Real-IAD `pcb`와
`terminalblock`을 [후속 development/final 사전등록](docs/realiad-followup-preregistration.md) 대상으로
고정했다. 이미지 ZIP을 열기 전 annotation 경로를 그룹 단위로 봉인했고,
[후속 데이터 등록·감사](reports/realiad-followup-registration-v1.md)도 완료했다. Development 1,302장의
오류·경고·디코딩 픽셀 중복은 0건이며, final holdout 710장은 추출하지 않고 봉인을 유지한다.
가짜 추론 결과를 실험 성능으로 제시하지 않는다.

## 먼저 할 일

1. [계획 검토](docs/plan-review.md)와 [실험 규칙](docs/data-protocol.md)을 기준으로 진행한다.
2. 연구실 Windows의 드라이버/Python/실행 방식을 확인한 뒤 ML 의존성을 고정한다.
3. 맥북 데이터는 `data/transistor/`에 준비되어 있다. 원본은 Git 제외 상태다.
4. 비공개 원격 저장소 `https://github.com/juuu-sung/defectops`의 같은 commit을 연구실에서 받고,
   고정 manifest의 체크섬을 검증한다.
5. 단일 이미지 검사·탐색적 오류 분석·모델 1회 로드 warm 속도 측정 완료.
6. PatchCore fit·평가·오류 분석·warm 측정과 FN/FP contact sheet 판독 완료.
7. PatchCore 단일 이미지 검사 구현·연구실 Windows smoke test 완료.
8. 로컬 상주 검사 API의 연구실 Windows smoke test와 실제 산출물의 맥북 SQLite 등록 완료.
   맥북 로컬 검토 UI·실제 사람 의견 1건, 모의 SOP 검색, 저장된 근거 revision만 사용하는
   로컬 승인/모의 티켓 상태 전이와 실제 1회 시연까지 완료했다.
9. 연구실 Windows에서 PaDiM·PatchCore seed 43·44 추가 실행과 3-seed 무결성 검증·집계 완료.
   다음 품목은 초기 allowlist를 보존해 cable·metal_nut으로
   [사전등록](docs/three-category-preregistration.md)했다. 두 품목의 test를 보기 전에 모델·seed·전처리·
   calibration·평가·선택 규칙을 고정했다. 공식 데이터 등록·manifest audit도 완료했으며 오류와
   디코딩 픽셀 중복은 0건이었다. test를 열지 않는
   [두 품목 12개 fit/calibration](docs/additional-category-fits.md)도 완료했다. 다음 체크포인트는
   사전 고정한 [공식 test 평가와 3품목 macro 집계](docs/additional-category-evaluation.md)다.
   이 평가도 완료했고 PatchCore는 macro escape rate 4.31%로 PaDiM의 35.85%보다 낮아 정확도
   우선 포트폴리오 후보가 됐다. 상세 결과는
   [3품목 비교 보고서](reports/three-category-padim-vs-patchcore-v1.md)에 기록했다.
   [PatchCore 추가 품목 오류 분석과 warm 측정](docs/patchcore-additional-diagnostics.md)도 완료했고,
   Real-IAD 외부 카테고리 등록, test를 열지 않는 6개 fit/calibration, 고정 평가와
   [3-seed 오류 진단](reports/realiad-patchcore-multiseed-diagnostics-v1.md)까지 완료했다.
   미관찰 Real-IAD `pcb`·`terminalblock`의 그룹 단위 development/final 경로 계획과 후보 네 개를
   고정했다. 두 원본 ZIP의 development 등록·감사도 완료했고 final 이미지는 추출하지 않았다.
   [PatchCore 후보 네 개의 development 비교 실행 흐름](docs/realiad-followup-development.md)을 구현했다.
   연구실 Windows 실행 결과를 [검증·비교](reports/realiad-followup-development-selection-v1.md)해
   `fine_features_top_1pct`를 선택하고 final gate를 고정했다. 선택 후보는 warm p95가 약 10초/장으로
   느리고 PCB development 불량 224장 중 62장을 놓쳤다. 다음은 final 실행기와 재개 절차를 준비한 뒤
   봉인된 holdout에서 seed 42·43·44를 한 번씩 평가하는 것이다.

Python 3.11 이상. manifest/환경 점검 도구는 표준 라이브러리만 사용한다.
이미지 점검과 전체 테스트에는 Pillow/NumPy가 필요하다. 새 환경에서는 아래처럼 설치한다.
ML 학습·평가용 Windows 라이브러리는 별도 requirements에 고정했다. 아래는 맥북 데이터 도구 실행이다.

```sh
python3 -m pip install -e '.[data]'
python3 -m pip install -r requirements/api.txt
python3 scripts/check_environment.py --output artifacts/environment-mac.json
python3 -m unittest discover -s tests -v
python3 scripts/prepare_data.py verify --root data --manifest data/manifests/transistor-v1.json
```

Windows에서는 `python3` 대신 `python`, 데이터 경로는 해당 장비의 절대 경로를 사용한다.
`transistor-v1.json`은 이미 생성되어 있으므로 다시 create하지 않는다.
폴더 구조와 실행 절차는 [data/README.md](data/README.md), 장비 간 전달은
[작업 공간 운영](docs/workspaces.md)을 따른다.

## 첫 완료 목표

transistor 한 장 → 원시 이상 점수·고정 임곗값·PASS/REVIEW_REQUIRED·히트맵·영역 좌표를 JSON으로 반환.
추론 오류/입력 오류는 INSPECTION_FAILED. 모델·분할·설정 버전까지 추적 가능해야 한다.
추론·고정 판정·JSON·heatmap·영역·실패 기록을 구현했고 자동화 테스트를 통과했다.
연구실 Windows에서 실제 checkpoint로 한 장을 검사하고 산출물·checksum을 확인해 첫 기술 마일스톤을 완료했다.

최소 범위는 transistor + PaDiM, 이후 PatchCore 비교 → 3품목 → API/검토/SOP/티켓 순서다.
EfficientAD, AD 2, 모델 변환, embedding drift는 확장 단계로 둔다.

## 데이터·문서 출처

- [MVTec AD 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec AD 2 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
- [Real-IAD 공식 페이지](https://realiad4ad.github.io/Real-IAD/)
- 데이터 라이선스: CC BY-NC-SA 4.0. 원본 이미지·마스크·파생 시각화는 코드 라이선스와 별도로 취급한다.
- 이미지·모델·DB·비밀정보는 Git에 넣지 않는다. 파생물 공개 시에도 데이터 이용 조건을 확인한다.
- 향후 SOP는 워크플로우 구현용 모의 문서이며 실제 산업 안전 지침이 아니다.

구체적인 검토 근거와 후속 작업은 [계획 검토](docs/plan-review.md), [착수 체크리스트](docs/first-milestone.md)에 있다.
## Real-IAD 후속 final 검증

선택 후보 잠금과 연구실 Windows의 준비/명시적 final 평가·재개 절차는
[`docs/realiad-followup-final.md`](docs/realiad-followup-final.md)에 정리되어 있다.
