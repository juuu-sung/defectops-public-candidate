# PatchCore 추가 품목 사후 진단 결과

실행 완료일: 2026-09-21

대상: Cable·Metal Nut PatchCore seed 42

결과 상태: `post_evaluation_exploratory_diagnostics`

검증한 handoff ZIP SHA-256:
`eee5ca50596b13be0501c48f28ea27e5a74b381d5ee3522a4959bf359f9d6c8c`

## 결론

진단 산출물은 앞서 고정한 공식 seed 42 평가와 정확히 일치했다. Cable은 불량 92장 중 5장을,
Metal Nut은 93장 중 1장을 놓쳤다. Cable 미탐은 `cable_swap` 4장과 `poke_insulation` 1장,
Metal Nut 미탐은 `color` 1장에 집중됐다. 모델·임곗값·ROI·후처리는 변경하지 않았다.

이 결과는 이미 관찰한 공개 test의 실패 사례를 설명하는 사후 진단이다. 현재 test에 맞춰 임곗값이나
후처리를 바꾸는 근거로 사용하지 않으며, 변경 후보는 반드시 새로운 development 데이터에서 설계하고
별도의 미관찰 final holdout에서 한 번만 검증한다.

## 고정 판정 결과

| 품목 | TP / FP / TN / FN | Recall | Escape rate | Overkill rate |
|---|---:|---:|---:|---:|
| Cable | 87 / 4 / 54 / 5 | 94.57% | 5.43% | 6.90% |
| Metal Nut | 92 / 3 / 19 / 1 | 98.92% | 1.08% | 13.64% |

### Cable 결함 유형

| 유형 | 표본 | 검토 / 미탐 | Recall | Mean IoU | Mean pixel recall |
|---|---:|---:|---:|---:|---:|
| bent_wire | 13 | 13 / 0 | 1.0000 | 0.2677 | 0.9995 |
| cable_swap | 12 | 8 / 4 | 0.6667 | 0.4042 | 0.6304 |
| combined | 11 | 11 / 0 | 1.0000 | 0.3913 | 0.9338 |
| cut_inner_insulation | 14 | 14 / 0 | 1.0000 | 0.3970 | 0.9396 |
| cut_outer_insulation | 10 | 10 / 0 | 1.0000 | 0.1732 | 0.7851 |
| missing_cable | 12 | 12 / 0 | 1.0000 | 0.6210 | 0.9951 |
| missing_wire | 10 | 10 / 0 | 1.0000 | 0.4527 | 0.9790 |
| poke_insulation | 10 | 9 / 1 | 0.9000 | 0.4381 | 0.8276 |

`cable_swap` 미탐 4장 중 3장은 이미지 임곗값보다 각각 0.015, 0.075, 0.532 낮았다.
contact sheet에서는 해당 도선 영역에 픽셀 반응이 보이지만 최종 image score가 고정 임곗값을 넘지
못했다. 나머지 `cable_swap` 1장은 −2.799, 작은 `poke_insulation`은 −2.400의 margin이었다.

### Metal Nut 결함 유형

| 유형 | 표본 | 검토 / 미탐 | Recall | Mean IoU | Mean pixel recall |
|---|---:|---:|---:|---:|---:|
| bent | 25 | 25 / 0 | 1.0000 | 0.3019 | 0.9829 |
| color | 22 | 21 / 1 | 0.9545 | 0.3377 | 0.9122 |
| flip | 23 | 23 / 0 | 1.0000 | 0.7876 | 0.9738 |
| scratch | 23 | 23 / 0 | 1.0000 | 0.4205 | 0.9299 |

유일한 미탐 `color/004.png`는 작은 색 변화 위치에 픽셀 반응이 있었지만 image score가 임곗값보다
0.865 낮았다. 정상 오탐 3장은 물체 중앙과 표면에 넓은 반응을 보였다. 조명·표면 질감 영향 가능성은
새 독립 데이터에서 확인해야 하며 현재 그림만으로 원인을 확정하지 않는다.

## Warm 실행 비용

모델을 한 번 로드한 프로세스에서 warmup 5회 후 고정 입력 100장을 batch size 1로 측정했다.

| 품목 | 처리량 | 전체 평균 / p95 | GPU 추론 평균 / p95 | Peak allocated / reserved |
|---|---:|---:|---:|---:|
| Cable | 25.03 images/s | 39.46 / 42.04 ms | 13.97 / 15.49 ms | 0.359 / 0.410 GiB |
| Metal Nut | 41.00 images/s | 23.98 / 25.97 ms | 13.37 / 14.97 ms | 0.354 / 0.406 GiB |

Cable과 Metal Nut의 GPU 추론 시간은 비슷하고 전체 시간 차이는 주로 이미지 decode/checksum에서
발생했다. 이전 Transistor warm 측정과는 실행 시점·입력 파일·시스템 부하가 다르므로 직접적인
우열이나 생산 처리량 보장으로 해석하지 않는다.

## 무결성 확인

- Cable 150개, Metal Nut 115개 case 행이 공식 평가 표본 수와 일치했다.
- Cable `87 / 4 / 54 / 5`, Metal Nut `92 / 3 / 19 / 1`이 기존 evaluation과 일치했다.
- 분석과 benchmark가 참조한 manifest·checkpoint SHA-256이 기존 seed 42 evaluation과 일치했다.
- `model_changed`, `threshold_changed`, `thresholds_changed`는 모두 `false`다.
- warm benchmark의 evaluation score 최대 절대 차이는 두 품목 모두 0.0이다.
- 각 contact sheet, summary, cases, samples의 SHA-256이 metadata 기록과 일치했다.
- warmup 5회, measured invocation 100회, label 미사용 조건을 확인했다.
- Windows식 ZIP 경로 때문에 macOS `unzip`이 경고했지만 14개 산출물은 모두 정상 추출·검증됐다.

## 다음 단계

[새 미관찰 holdout 계획](../docs/unseen-holdout-plan.md)에 따라 독립 촬영 데이터를 확보한다.
Cable은 `cable_swap`·작은 `poke_insulation`, Metal Nut은 작은 `color`와 정상 표면 변동을 우선
수집한다. 새 데이터가 없으면 현재 PatchCore seed 42와 고정 임곗값을 유지하고 추가 튜닝을 하지 않는다.

