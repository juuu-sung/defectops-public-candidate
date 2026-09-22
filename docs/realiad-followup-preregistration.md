# Real-IAD 후속 development/final 사전등록 v1

상태: 이미지 미관찰 사전등록

기준일: 2026-09-22

## 목적과 한계

기존 `transistor1`·`plastic_nut` C1 test는 이미 평가와 오류 분석에 사용했으므로 다시 나누거나
C2–C5로 바꿔 새 holdout처럼 사용하지 않는다. 같은 물리 표본의 다른 시점이기 때문에 표본 누수를
막을 수 없기 때문이다.

후속 검증은 아직 이미지를 열거나 점수를 계산하지 않은 Real-IAD 256 `pcb`와 `terminalblock`의
공식 single-view C1 annotation을 사용한다. 이는 작은 국소 결함용 score와 특징 선택이 새 품목에도
전이되는지 확인하는 외부 검증이다. 기존 두 품목의 same-product 개선을 증명하지 않는다.

## 고정 분할

- 공식 정상 train은 `sample_group` 단위로 train 80%·calibration 20%로 나눈다.
- 공식 test는 결함 유형별 `sample_group` 단위로 development 50%·final holdout 50%로 나눈다.
- 정렬 키는 `SHA-256(seed:category:purpose:sample_group)`이고 seed는 `20260922`다.
- 같은 상대 부모 디렉터리의 연속 촬영 이미지는 항상 같은 split에 들어간다.
- C1만 사용하고 기존 관찰 품목과 그 품목의 C2–C5는 제외한다.
- 경로 계획은 이미지 ZIP을 열기 전에 Git에 고정한다. 이미지와 마스크 SHA-256은 원본 등록 때 추가한다.

고정 경로 계획 SHA-256은
`fefa8c3da53366180fd8d55890effb1d7f27894dae11c21f5352f6df7b761a79`이다.

| 카테고리 | Train 정상 | Calibration 정상 | Development 정상/불량 | Final 정상/불량 |
|---|---:|---:|---:|---:|
| realiad_pcb | 223 | 56 | 139 / 224 | 139 / 221 |
| realiad_terminalblock | 247 | 62 | 154 / 197 | 154 / 196 |

## Development 후보 네 개

모든 후보는 input 256, Wide ResNet-50-2, coreset 10%, 이웃 9개와 정상 calibration 95% 임곗값을
공유한다. Development에서는 seed 42만 사용한다.

1. `baseline_raw`: 기존 layer2·layer3와 Anomalib PatchCore raw image score
2. `map_top_1pct`: layer2·layer3 map의 상위 1% 픽셀 평균
3. `map_top_5pct`: layer2·layer3 map의 상위 5% 픽셀 평균
4. `fine_features_top_1pct`: layer1·layer2와 상위 1% 픽셀 평균

작은 결함이 넓은 map 또는 단일 최대값 기반 image score에서 희석되는 가설을 제한된 네 후보로
검증한다. Foreground mask는 안정적인 자동 생성 규칙을 아직 정의하지 못했으므로 이번 후보에서
제외한다. Development를 본 뒤 임의로 다섯 번째 후보를 추가하지 않는다.

## 선택과 final gate

- 1차: 두 품목 동일 가중 macro escape rate가 가장 낮은 후보
- 제약: 어느 품목도 정상 overkill rate 10%를 넘지 않아야 한다.
- 동률: macro Image AUROC, macro overkill rate, warm p95 순서
- 정확히 한 후보만 선택한다.
- final을 열기 전에 선택 후보 ID, selection report SHA-256, Git commit, final manifest SHA-256을 고정한다.
- 선택 후보는 final에서 seed 42·43·44를 한 번씩 실행한다.
- final 결과를 본 뒤 변경하면 그 결과는 탐색 결과로 강등하고 새 holdout 없이는 재확증하지 않는다.
- 자동 PASS는 계속 금지한다.

## 현재 단계

`realiad_jsons.zip`의 annotation metadata만 사용해 경로 분할을 고정한다. 이미지 ZIP 다운로드·추출·
디코딩·학습은 아직 하지 않는다. 고정 설정은
[`configs/realiad-followup-preregistration-v1.toml`](../configs/realiad-followup-preregistration-v1.toml)에,
생성기는 [`scripts/plan_realiad_followup.py`](../scripts/plan_realiad_followup.py)에 있다.
생성된 경로 목록은
[`data/manifests/realiad-followup-split-plan-v1.json`](../data/manifests/realiad-followup-split-plan-v1.json)에
봉인했다.

Windows 등록은 `scripts\\prepare_realiad_followup.ps1` 한 번으로 수행한다. 이 단계는 train,
calibration, development만 추출·디코딩 점검하고 final ZIP member는 SHA-256만 계산한다. Final 이미지는
선택 후보와 gate 근거가 고정되기 전까지 로컬 데이터 폴더에 추출하지 않는다.
