# PaDiM–PatchCore 3품목 고정 비교 결과

실행 완료일: 2026-09-21

대상: `transistor`, `cable`, `metal_nut` × PaDiM/PatchCore × seed 42·43·44

결과 상태: `mixed_transistor_observed_additional_categories_preregistered`

검증한 handoff ZIP SHA-256:
`af4e90dc1d9fb3981873880b50c8c849cfbf0b13f7732621419573c3e5c9595f`

## 결론

사전 고정한 1순위 기준인 품목 동일 가중 macro escape rate에서 PatchCore가 4.31%, PaDiM이
35.85%를 기록했다. PatchCore가 불량 누락을 31.54%p 줄였으므로 warm 속도 동률 판정까지 가지 않고
정확도 우선 포트폴리오 후보로 선택됐다. 이 선택은 실제 제품의 자동 합격을 허용하지 않는다.

PatchCore는 macro Image AUROC 0.9908, recall 0.9569, F1 0.9602, Pixel AUPRO 0.9358로
PaDiM보다 높았다. 평가 과정 p95는 PatchCore 48.79 ms, PaDiM 30.59 ms였지만 이미지 decode와
전처리가 포함된 수치이며 별도 warm benchmark 선택 지표는 아니다.

## 품목 동일 가중 macro

각 seed에서 세 품목을 동일 가중 평균하고, 그 세 seed macro의 평균과 표본 표준편차를 계산했다.
품목별 test 이미지 수 차이가 macro 모델 선택을 지배하지 않는다.

| 지표 | PaDiM 평균 ± 표본 표준편차 | PatchCore 평균 ± 표본 표준편차 | PatchCore − PaDiM |
|---|---:|---:|---:|
| Image AUROC | 0.903491 ± 0.005216 | **0.990762 ± 0.000411** | +0.087271 |
| Average Precision | 0.926365 ± 0.004097 | **0.992500 ± 0.000699** | +0.066135 |
| Precision | 0.932988 ± 0.023078 | **0.963956 ± 0.003766** | +0.030968 |
| Recall | 0.641548 ± 0.070476 | **0.956923 ± 0.003623** | +0.315375 |
| F1 | 0.747590 ± 0.051885 | **0.960164 ± 0.001016** | +0.212574 |
| Escape rate | 35.8452% ± 7.0476%p | **4.3077% ± 0.3623%p** | −31.5375%p |
| Overkill rate | 8.1011% ± 2.0328%p | **8.0965% ± 1.2066%p** | −0.0046%p |
| Review rate | 42.9259% ± 6.1935%p | 60.8180% ± 0.5006%p | +17.8921%p |
| Pixel AUROC | 0.962171 ± 0.003829 | **0.980970 ± 0.000054** | +0.018799 |
| Pixel AUPRO | 0.876402 ± 0.006306 | **0.935839 ± 0.000553** | +0.059437 |
| 평가 p95 | **30.590 ± 4.571 ms** | 48.786 ± 15.084 ms | +18.196 ms |

## 품목별 평균

| 품목 | 모델 | Image AUROC | Recall | Escape rate | Overkill rate | Pixel AUPRO |
|---|---|---:|---:|---:|---:|---:|
| transistor | PaDiM | 0.9175 | 0.6000 | 40.00% | 2.78% | 0.9008 |
| transistor | PatchCore | **0.9911** | **0.9250** | **7.50%** | **1.67%** | **0.9450** |
| cable | PaDiM | 0.8436 | 0.5254 | 47.46% | 10.92% | 0.8603 |
| cable | PatchCore | **0.9839** | **0.9565** | **4.35%** | **7.47%** | **0.9291** |
| metal_nut | PaDiM | 0.9493 | 0.7993 | 20.07% | **10.61%** | 0.8681 |
| metal_nut | PatchCore | **0.9972** | **0.9892** | **1.08%** | 15.15% | **0.9335** |

Metal Nut에서 PatchCore는 불량을 거의 모두 잡았지만 정상 22장 중 seed에 따라 3~4장을 검토로
보내 overkill이 PaDiM보다 높았다. 이 22장의 공개 test 정상 표본으로 실제 공장 오탐률을 추정할
수는 없다. Cable PaDiM은 seed별 recall이 0.2826~0.6957로 크게 흔들린 반면 PatchCore는
0.9457~0.9674였다.

## 추가 품목 실행별 고정 판정

| 품목 | 모델 | Seed | TP / FP / TN / FN | Recall | Escape rate |
|---|---|---:|---:|---:|---:|
| cable | PaDiM | 42 | 64 / 10 / 48 / 28 | 0.6957 | 30.43% |
| cable | PaDiM | 43 | 26 / 2 / 56 / 66 | 0.2826 | 71.74% |
| cable | PaDiM | 44 | 55 / 7 / 51 / 37 | 0.5978 | 40.22% |
| cable | PatchCore | 42 | 87 / 4 / 54 / 5 | 0.9457 | 5.43% |
| cable | PatchCore | 43 | 89 / 5 / 53 / 3 | 0.9674 | 3.26% |
| cable | PatchCore | 44 | 88 / 4 / 54 / 4 | 0.9565 | 4.35% |
| metal_nut | PaDiM | 42 | 85 / 2 / 20 / 8 | 0.9140 | 8.60% |
| metal_nut | PaDiM | 43 | 76 / 3 / 19 / 17 | 0.8172 | 18.28% |
| metal_nut | PaDiM | 44 | 62 / 2 / 20 / 31 | 0.6667 | 33.33% |
| metal_nut | PatchCore | 42 | 92 / 3 / 19 / 1 | 0.9892 | 1.08% |
| metal_nut | PatchCore | 43 | 92 / 4 / 18 / 1 | 0.9892 | 1.08% |
| metal_nut | PatchCore | 44 | 92 / 3 / 19 / 1 | 0.9892 | 1.08% |

## 무결성 확인

- 추가 품목 평가 12개의 상태·모델·seed·manifest·표본 수를 재검증했다.
- `metrics.json`과 calibration contract의 SHA-256이 각 `evaluation.json` 기록과 일치했다.
- 모델 재로딩 후 calibration score 최대 절대 차이는 12개 실행 모두 0.0이었다.
- 추가 12개 실행별 행은 18-run 집계 JSON의 해당 행과 정확히 일치했다.
- 18개 실행 행에서 품목 동일 가중 macro를 독립 재계산해 집계 JSON과 일치함을 확인했다.
- 대용량 predictions/anomaly map은 Windows 원본 산출물에 보존되고 handoff ZIP에는 포함되지 않았다.

## 해석 한계와 다음 단계

- Transistor test는 추가 품목 사전등록 전에 이미 관찰했으므로 전체 결과는 혼합 증거다.
- Cable과 Metal Nut은 사전등록된 고정 규칙으로 평가했지만 공개 MVTec test를 사용한다.
- 공개 데이터 불량 비율과 review rate를 실제 생산 운영 비율로 해석하지 않는다.
- 결과를 근거로 모델·임곗값·ROI·후처리를 변경하지 않는다.
- 다음은 사전 선택한 seed 42 PatchCore의 Cable·Metal Nut FN/FP 유형 분석과 warm 측정이다.
- 이후 모델 변경 또는 자동 PASS 검증에는 새로운 미관찰 데이터가 필요하다.
