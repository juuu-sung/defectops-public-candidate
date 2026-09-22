# Real-IAD 후속 development 후보 선택 v1

상태: **사전등록 규칙에 따라 `fine_features_top_1pct` 선택, final holdout 미개봉**

검증일: 2026-09-23

- 연구실 handoff ZIP SHA-256:
  `373177ef8eaa0c63c632f58e2a32154542a6b121ccf2a1230f193bc30338b936`
- 선택 보고서 내부 canonical SHA-256:
  `1dd08ad5ea0aa3156240a9400f6fa8f7cbb5777eeea077cc448ffe94b3ea13d3`
- 실행 Git commit: `795e70eaa04d6a783909a7f44b44e431eadc3abd`
- 고정 분할 계획 SHA-256:
  `c68562288b378d44d969c3eda68a69a00c6c31e8dad0f4ff21e5c05073ac8f20`

## 검증 범위

압축 CRC와 27개 파일 구성, 계획·manifest·선택 보고서 자체 체크섬, 저장소 설정과 handoff 설정의
일치, 후보별 calibration·development JSONL 체크섬과 표본 순서를 확인했다. 원점수에서 95% higher
임곗값, 8개 후보·품목 조합의 혼동행렬·미탐률·오탐률·Image AUROC·Average Precision, 두 품목 macro
지표 및 사전등록된 선택 순위를 재계산했다. 총 1,542개 검사 항목이 통과했다.

Handoff에는 원본 이미지·final 이미지·모델 checkpoint가 없다. 따라서 이 검증은 checkpoint 바이트나
연구실 GPU의 warm 측정을 독립 재실행한 것은 아니다. 선택 보고서의 시간 수치는 연구실 실행 기록이다.

## Development 결과

아래는 고정된 seed 42, 정상 calibration 95% 임곗값을 사용한 development 결과다. Macro는 두
품목을 동일 가중 평균한 값이다.

| 후보 | PCB 미탐/불량 | Terminalblock 미탐/불량 | Macro 미탐률 | Macro Image AUROC | Macro 오탐률 | Macro warm p95 |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_raw` | 70/224 (31.25%) | 12/197 (6.09%) | 18.67% | 0.9309 | 6.91% | 28.26 ms |
| `map_top_1pct` | 75/224 (33.48%) | 9/197 (4.57%) | 19.03% | 0.9372 | 7.17% | 28.29 ms |
| `map_top_5pct` | 69/224 (30.80%) | 19/197 (9.64%) | 20.22% | 0.9321 | 7.24% | 30.39 ms |
| `fine_features_top_1pct` | 62/224 (27.68%) | 9/197 (4.57%) | **16.12%** | **0.9599** | **4.40%** | **10,300.68 ms** |

선택 후보의 세부 혼동행렬:

| 품목 | TP | FP | TN | FN | 미탐률 | 정상 오탐률 | Pixel AUROC | Pixel AUPRO |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PCB | 162 | 5 | 134 | 62 | 27.68% | 3.60% | 0.9966 | 0.9533 |
| Terminalblock | 188 | 8 | 146 | 9 | 4.57% | 5.19% | 0.9988 | 0.9894 |

네 후보 모두 품목별 정상 오탐률 10% 이하라는 제약을 만족했다. 선택 후보가 macro 미탐률이 가장
낮으므로, 사전등록된 규칙에 따라 `fine_features_top_1pct`가 선택됐다. 지연시간은 동률일 때만
사용하는 규칙이어서 이번 선택을 바꾸지 않는다.

## 계산 비용과 해석

선택 후보의 warm p95는 PCB 10,382 ms, terminalblock 10,219 ms/장으로, 기본 후보의 26~31
ms/장보다 약 360배 느리다. Layer1+layer2 memory bank는 PCB 91,340×768,
terminalblock 101,171×768이며, development에서 이 조합의 fit만 각각 약 1.82시간·3.16시간이
걸렸다. Final의 3개 seed를 같은 방식으로 처리하면 상당한 시간이 걸릴 수 있으므로 실행 전에
연구실 자원·재개 방법과 checkpoint 재사용을 확인해야 한다. 이것은 운영상 우려이며 선택 규칙을
사후 변경할 근거는 아니다.

PCB는 선택 후보로도 불량 224장 중 62장을 놓쳤다. 따라서 이 development 결과로 자동 PASS나
현장 사용을 허용하지 않는다. 새로운 final holdout에서의 성능은 아직 알 수 없다.

## Final gate 고정

| 필드 | 값 |
|---|---|
| 선택 후보 ID | `fine_features_top_1pct` |
| 선택 보고서 SHA-256 | `1dd08ad5ea0aa3156240a9400f6fa8f7cbb5777eeea077cc448ffe94b3ea13d3` |
| 선택 실행 Git commit | `795e70eaa04d6a783909a7f44b44e431eadc3abd` |
| PCB final manifest SHA-256 | `9b5a319600967c8e73bf3e4b53d0b31c9e7df9ff7261222c31db489b435478d2` |
| Terminalblock final manifest SHA-256 | `b2ff0bc8f52aeed8db97ce853869ecc9ec6a46d0bc939b1c4442a7d31507891b` |

이 값은 [`configs/realiad-followup-final-lock-v1.toml`](../configs/realiad-followup-final-lock-v1.toml)에
기록했다. Final 이미지는 아직 추출·평가하지 않았다. 다음은 이 잠금값을 검증하는 final 실행기를
준비하고, 선택된 한 후보를 seed 42·43·44로 정확히 한 번 평가하는 단계다.
