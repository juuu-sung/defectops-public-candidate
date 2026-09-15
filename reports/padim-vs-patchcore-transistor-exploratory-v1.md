# PaDiM vs PatchCore transistor 탐색적 비교

비교일: 2026-09-15

PaDiM 평가: `padim-transistor-seed42-eval-v1`

PatchCore 평가: `patchcore-transistor-seed42-eval-v1`

PatchCore 오류 분석: `patchcore-transistor-seed42-errors-v1`

PatchCore 속도 측정: `patchcore-transistor-seed42-warm-v1`

성격: PaDiM 공식 test 결과를 확인한 뒤 고정한 PatchCore와의 탐색적 비교. 새 확증 모델 선택
결과가 아님.

## 결론

동일한 transistor manifest·입력 크기·normal-only calibration·평가 지표에서
`PatchCore + Wide ResNet-50-2`가 `PaDiM + ResNet18`보다 이미지 판정 성능을 크게 개선했다.
Recall은 0.55에서 0.925로 증가했고 escape rate는 45.0%에서 7.5%로 감소했다. 특히 PaDiM이
취약했던 `cut_lead`와 `damaged_case`를 각각 10/10 탐지했다.

대신 PatchCore의 warm 처리량은 12.56 images/s로 PaDiM의 19.83 images/s보다 약 36.6% 낮고,
GPU 추론·map 전송 중앙값은 44.13 ms로 PaDiM의 5.27 ms보다 약 8.4배 길다. 그래도 전체 sample
p95는 95.91 ms로 현재 1.5초 목표 이내다. 정확도와 불량 유출 방지를 우선하는 다음 MVP 모델은
PatchCore로 두고, PaDiM은 가벼운 비교 baseline으로 유지한다.

PatchCore도 불량 40장 중 3장을 놓치므로 실제 자동 PASS에는 사용할 수 없다. 다음 모델 변경이나
임곗값 조정은 이 test가 아니라 새 미관찰 development/holdout 데이터에서 수행해야 한다.

## 비교 조건

| 항목 | PaDiM | PatchCore |
|---|---|---|
| 정상 학습 | 170장 | 170장 |
| 정상 calibration | 43장 | 43장 |
| 공식 test | 정상 60장 + 불량 40장 | 동일 |
| Backbone | ResNet18 | Wide ResNet-50-2 |
| 특징 | layer1/2/3, 선택 차원 100 | layer2/3, 차원 1,536 |
| 정상 표현 | 위치별 Gaussian 평균·공분산 | 10% coreset memory bank |
| 입력 | 256×256, ImageNet 정규화 | 동일 |
| 증강 / batch | 없음 / 1 | 동일 |
| 이미지 임곗값 | 정상 score 95% higher | 동일 규칙 |
| 픽셀 임곗값 | 정상 map pixel 99.5% higher | 동일 규칙 |
| Fit + calibration | 20.225초 | 141.677초 |
| 추가 저장 구조 | 위치별 통계 | 17,408 × 1,536, 약 102 MiB |

두 모델은 원시 점수 공간이 다르므로 PaDiM 임곗값 56.0270과 PatchCore 임곗값 40.7913의 숫자
크기를 비교하지 않는다. Backbone도 다르므로 이번 결과는 알고리즘만 분리한 비교가 아니라 두
전체 구성의 비교다.

## 이미지·픽셀 결과

| 지표 | PaDiM | PatchCore | 변화 |
|---|---:|---:|---:|
| Image AUROC | 0.9013 | **0.9904** | +0.0892 |
| Average Precision | 0.8810 | **0.9884** | +0.1073 |
| Precision | 0.8800 | **0.9737** | +0.0937 |
| Recall | 0.5500 | **0.9250** | +0.3750 |
| F1 | 0.6769 | **0.9487** | +0.2718 |
| Escape rate | 45.00% | **7.50%** | -37.50%p |
| Overkill rate | 5.00% | **1.67%** | -3.33%p |
| Review rate | 25.00% | 38.00% | +13.00%p |
| Pixel AUROC | 0.9689 | **0.9720** | +0.0031 |
| Pixel AUPRO | 0.9005 | **0.9454** | +0.0448 |

| 모델 | TP | FP | TN | FN |
|---|---:|---:|---:|---:|
| PaDiM | 22 | 3 | 57 | 18 |
| PatchCore | 37 | 1 | 59 | 3 |

PatchCore는 불량 15장을 추가로 검토 대상으로 보냈고 정상 오탐도 2장 줄였다. PatchCore
precision의 95% Wilson 구간은 약 0.865~0.995, recall은 약 0.801~0.974다. test 불량 비율
40%는 실제 공장 유병률이 아니므로 precision과 review rate를 현장 값으로 옮기지 않는다.

## 결함 유형별 오류와 위치화

| 결함 유형 | PaDiM recall | PatchCore recall | PaDiM IoU | PatchCore IoU | PaDiM pixel recall | PatchCore pixel recall |
|---|---:|---:|---:|---:|---:|---:|
| `bent_lead` | 0.70 | **0.90** | **0.2133** | 0.2035 | 0.6504 | **0.9550** |
| `cut_lead` | 0.20 | **1.00** | 0.0746 | **0.2154** | 0.1478 | **0.7270** |
| `damaged_case` | 0.30 | **1.00** | **0.3984** | 0.3301 | 0.6533 | **0.9428** |
| `misplaced` | **1.00** | 0.80 | **0.5090** | 0.4864 | 0.5907 | **0.6075** |

PatchCore의 FN 3장은 유형 기준으로 `bent_lead` 1장과 `misplaced` 2장이다. `cut_lead`와
`damaged_case`는 이미지 recall과 pixel recall이 모두 크게 개선됐다. `bent_lead`,
`damaged_case`, `misplaced`는 pixel recall이 증가했지만 평균 IoU는 소폭 감소했다. 이는 실제
결함을 더 넓게 덮는 대신 결함 외 영역까지 포함하는 과분할 가능성을 시사한다. Pixel AUPRO가
전체적으로 개선됐으므로 위치 순위 품질은 좋아졌지만, 정확한 공간 오류 원인은 contact sheet를
직접 확인해야 한다.

현재 대화에는 생성된 `false-negatives.png`와 `false-positives.png`가 첨부되지 않아 개별 파일명,
배경 반응, 임곗값 margin에 대한 시각 판독은 이 보고서에서 보류한다.

## Warm 속도 비교

조건: 모델 1회 로드, 워밍업 5회, batch size 1, 같은 고정 test 100장 1회 처리.

| 지표 | PaDiM | PatchCore | PatchCore / PaDiM |
|---|---:|---:|---:|
| 처리량 | 19.8264 images/s | 12.5621 images/s | 0.634× |
| GPU 추론·map 전송 평균 | 8.15 ms | 40.48 ms | 4.96× |
| GPU 추론·map 전송 중앙값 | 5.27 ms | 44.13 ms | 8.37× |
| GPU 추론·map 전송 p95 | 20.19 ms | 61.59 ms | 3.05× |
| 전체 sample 평균 | 49.04 ms | 78.29 ms | 1.60× |
| 전체 sample 중앙값 | 43.79 ms | 79.88 ms | 1.82× |
| 전체 sample p95 | 78.22 ms | 95.91 ms | 1.23× |
| 전체 sample 최대 | 125.40 ms | 125.85 ms | 1.00× |

PatchCore는 더 큰 backbone과 17,408개 정상 patch 최근접 이웃 검색 때문에 GPU 구간이 길다.
그러나 decode·checksum·전처리를 포함한 p95도 약 96 ms이고 처리량은 약 12.6 images/s다. 현재
MVP의 1.5초 목표에는 충분하지만 실제 생산 적용 가능 여부는 요구 line rate와 동시성 조건을
별도로 받아 검증해야 한다. 두 모델 모두 매 요청마다 checkpoint를 다시 읽지 않고 장기 실행
worker에서 모델을 한 번 로드해야 한다.

## 재현성

- PatchCore checkpoint 재로드 후 정상 calibration 43장의 최대 절대 점수 차이: 0.0
- 오류 분석에서 재계산한 TP/FP/TN/FN: 평가 bundle과 일치
- warm benchmark 100장: 저장된 평가 점수와 허용 오차 내 일치하도록 실행기가 검증
- 모델·임곗값·전처리 변경 없음

## 결정과 다음 단계

1. PatchCore seed-42 fit과 평가 결과를 현재 주력 후보 baseline으로 동결한다.
2. PaDiM은 속도가 빠른 비교 baseline과 fallback 후보로 보존한다.
3. 두 결과 모두 자동 PASS나 안전 판정에 사용하지 않는다.
4. PatchCore FN 3장·FP 1장 contact sheet를 사람이 확인해 실패 양상을 기록한다.
5. 새 미관찰 데이터 없이 현재 test 기반 임곗값·해상도·후처리를 튜닝하지 않는다.
6. 다음 구현은 PatchCore 단일 이미지 검사와 모델 상주 FastAPI worker를 우선한다.
7. 실제 line rate와 검토 비용 요구사항을 받은 뒤 속도·recall 운영점을 다시 설계한다.

## 한계

- PatchCore 설정은 PaDiM test 관찰 이후 고정했으므로 탐색적 결과다.
- seed 42 한 번과 공개 MVTec 한 품목의 결과이며 실제 공장 domain shift를 다루지 않는다.
- 각 결함 유형은 10장뿐이라 유형별 비율의 불확실성이 크다.
- 현재 비교는 알고리즘뿐 아니라 backbone 크기 차이도 포함한다.
- contact sheet 미첨부로 개별 PatchCore FN/FP의 시각 원인 분석은 남아 있다.
