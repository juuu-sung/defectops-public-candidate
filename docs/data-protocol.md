# Protocol A: normal-only-v1

## 분할

- 공식 train/good만 train/calibration 80/20으로 나눈다. calibration 수는 ceil(n × 0.2).
- 상대 경로·split seed의 SHA-256 순서로 분할한다. 이미지·마스크 SHA-256을 manifest에 저장한다.
- 공식 test 전체는 최종 평가용. 모델 seed와 split seed를 구분한다.
- 경로 중복/이미지 바이트 중복·누락된 결함 마스크를 차단한다.
- 로더는 manifest 목록만 읽는다. glob 재분할이나 프레임워크 자동 test→validation 분할을 쓰지 않는다.
- train 이미지로 분포/메모리뱅크를 구성하고 calibration으로 임곗값을 구한다.
- checkpoint 및 통계 추정에 사용된 sample ID를 추적하여 test 유입을 검증한다.

## 고정해야 할 실험 조건

파일럿 설정은 `configs/pilot.toml`. 첫 PaDiM/resnet18, 256, seed 42 실행 후 파이프라인을 검증한다.
최종 비교 후보와 seed는 테스트 점수를 보기 전에 별도 실험 설정으로 동결한다.
전처리 interpolation·정규화·crop 여부·원본 좌표 복원은 모델 어댑터 구현 시 확정하고 저장한다.
아직 확정되지 않은 전처리/픽셀 규칙이 있으므로 현재 상태에서 최종 벤치마크를 실행하면 안 된다.

## 임곗값과 지표

- 이미지: 정상 calibration 원시 점수 95% 분위수(higher). 점수 >= 임곗값이면 REVIEW_REQUIRED.
- 표본 수·quantile 방법·score space·모델/전처리 hash를 저장한다. 95%는 모집단 오탐률 보장이 아니다.
- NaN/Inf/모델 없음/입력 오류는 INSPECTION_FAILED. 실패 건을 정상 판정이나 정상 점수로 대체하지 않는다.
- 픽셀: 원시 map 평가와 표시용 정규화 분리. 픽셀 threshold와 후처리 정책도 test 전에 고정한다.
- 품목별 image AUROC, AP, 고정 threshold F1/Precision/Recall, TP/FP/TN/FN, escape/overkill/review rate.
- pixel AUROC, AUPRO는 원시 map으로 계산. AUPRO는 FPR 적분 상한 0.30과 구간 정규화를 명시한다.
- 전체 결과는 품목별 표와 macro 평균. AP의 테스트 불량 비율 의존성을 명시한다.
- 분모가 0이거나 단일 클래스이면 해당 지표를 N/A로 보고한다. 실패율과 유효 평가 분모도 기록한다.
- seed별 값 + 평균/표준편차는 학습 변동성이다. 표본 불확실성과 구분한다.
- 비율의 불확실성은 이미지 단위 신뢰구간을 사용하고 픽셀을 독립 표본처럼 취급하지 않는다.
- 테스트 열람 후 수정한 실험은 exploratory로 표시하고 확증 평가와 구분한다.

## 첫 데이터 점검 산출물

품목별 train/calibration/test 정상·이상 수, 결함 유형별 수, 이미지/마스크 크기,
손상 파일, mask 값, train/calibration 밝기·블러 분포를 기록한다.
학습/후처리 설계용 EDA는 train/calibration으로 제한한다. test 이미지·오류 사례 시각화는 평가 동결 후 수행한다.
