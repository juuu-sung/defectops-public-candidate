# PaDiM transistor seed 42 파일럿 평가

평가일: 2026-09-14  
상태: 실제 연구실 Windows RTX 5060 실행 완료  
성격: 파이프라인 파일럿. 최종 모델 선택 또는 실제 공장 성능 추정이 아님.

## 실행 식별자

- Fit: `padim-transistor-seed42-retry1`
- Evaluation: `padim-transistor-seed42-eval-v1`
- Manifest SHA-256: `9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585`
- Fit: 정상 train 170장
- Calibration: 정상 43장
- Test: 정상 60장, 이상 40장
- 이미지 임곗값: `56.027042388916016`
- Fit+calibration 시간: `20.22541330003878`초

## 결과

| 지표 | 값 |
|---|---:|
| Image AUROC | 0.901250 |
| Image Average Precision | 0.881033 |
| 고정 임곗값 Precision | 0.880000 |
| 고정 임곗값 Recall | 0.550000 |
| 고정 임곗값 F1 | 0.676923 |
| Escape rate | 0.450000 |
| Overkill rate | 0.050000 |
| Review rate | 0.250000 |
| Pixel AUROC | 0.968896 |
| Pixel AUPRO @ FPR 0.30 | 0.900531 |

고정 임곗값 혼동행렬은 TP 22, FP 3, TN 57, FN 18이다.

| 비율 | 95% 구간 |
|---|---|
| Precision | 0.7004–0.9583 |
| Recall | 0.3983–0.6929 |
| Escape rate | 0.3071–0.6017 |
| Overkill rate | 0.0171–0.1370 |
| Review rate | 0.1755–0.3430 |

## 해석

원시 점수의 순위 구분력은 Image AUROC 0.901, AP 0.881로 파일럿 기준 유의미하다.
픽셀 위치화 순위도 Pixel AUROC 0.969, AUPRO 0.901로 높다. 즉 이상 이미지와 위치에
신호가 존재한다.

하지만 정상 calibration만으로 고정한 현재 이미지 임곗값에서는 이상 40장 중 18장을 PASS로
보내 recall이 55%, escape rate가 45%다. 정상 60장 중 REVIEW_REQUIRED는 3장으로 overkill은
5%였지만, 낮은 오탐을 얻는 대신 많은 불량을 놓치는 운영점이다. 따라서 이 모델·임곗값을
실제 자동 PASS 또는 안전 판단에 사용하면 안 된다.

## 실험 규율

이 test 결과를 보고 같은 test에 맞춰 임곗값·입력 크기·특징 수·후처리를 조정한 뒤 새 확증
성능처럼 보고하지 않는다. 변경 실험은 exploratory로 표시하고, 새로운 labeled development나
별도 holdout에서 선택한 뒤 독립 평가 데이터로 확인해야 한다.

이번 결과는 seed 42 한 번의 파일럿이다. 최종 비교에는 사전 고정한 여러 seed와 후보 모델,
결함 유형별 결과, 실패 사례를 포함해야 한다. MVTec의 이상 비율 40%는 실제 공장 유병률이
아니므로 Precision과 Review rate를 현장 수치로 옮기지 않는다.

## 다음 작업

1. 평가 bundle의 결함 유형별 recall, 지연시간, calibration pixel threshold를 보존한다.
2. 구현된 단일 이미지 검사기를 연구실 Windows의 같은 checkpoint·동결 임곗값으로 검증한다.
3. 입력/모델 오류를 `INSPECTION_FAILED`로 기록하고 자동 PASS를 금지한다.
4. 이후 변경은 test 재튜닝이 아니라 별도 development protocol을 먼저 설계한다.
