# 트랜지스터 검사 100장 비교

같은 PatchCore 모델로 MVTec AD 트랜지스터 공식 테스트 사진 100장(정상 60장, 불량 40장)을 검사했다. 연구실 GPU의 저장된 결과와 Mac CPU에서 다시 계산한 결과를 파일 체크섬으로 짝지었다. 아래 네 기준 모두 100장 각각의 경고 여부가 두 장비에서 일치했다. [사진별 점수·면적·정답](per-image.csv) · [점수 기준별 전체 집계](thresholds.csv)

| 사진 한 장의 경고 기준 | 불량 검출 | 미탐 | 오경보 | 정상 무경고 |
| --- | ---: | ---: | ---: | ---: |
| 기존 모델 점수 ≥ 40.7912712097168 | 37 | 3 | 1 | 59 |
| 점수 ≥ 38.70416 | 39 | 1 | 2 | 58 |
| 점수 ≥ 36.16603088378906 | 40 | 0 | 16 | 44 |
| 점수 ≥ 38.70416 또는 이상 면적 ≥ 2.5% | 40 | 0 | 2 | 58 |

이상 면적은 256 × 256 원시 이상 지도에서 픽셀 점수가 33.102806091308594 이상인 칸의 수다. 2.5%에 도달하려면 65,536칸 중 적어도 1,639칸이 필요하다. 정답 마스크는 검사 후 위치를 비교하는 데 사용했다. 점수·면적 결합 기준은 이 100장을 보면서 정했으며 별도 사진을 이용한 미탐·오경보 수는 아직 측정하지 않았다. 실제 추론 API의 기본 판정에는 기존 모델 기준을 사용한다. [고정한 실험 기준](frozen-rule.json)

## 어떤 사진에서 판정이 바뀌었나

- 기존 모델은 불량 `bent_lead/001.png`, `misplaced/003.png`, `misplaced/002.png`를 놓쳤다. 정상 `good/014.png`에 경고했다.
- 38.70416에서는 `misplaced/002.png` 한 장이 남았다. 두 장비에서 이 사진까지 점수만으로 잡는 가장 높은 공통 기준은 36.16603088378906이었다. 이때 정상 14장에 추가 경고가 떴다. 기존 모델 대비 새로 경고가 뜬 정상 사진은 총 15장이다.
- `misplaced/002.png`의 이상 면적은 CPU 1,774칸, GPU 1,776칸이었다. 추가 경고가 뜬 정상 14장 중 면적이 가장 큰 `good/044.png`도 CPU 1,435칸, GPU 1,428칸이었다. 2.5% 조건을 적용하자 마지막 불량을 잡았고, 정상 경고는 `good/014.png`, `good/039.png` 두 장이었다.
- 두 장비의 원시 지도 값 차이가 가장 큰 사진은 `damaged_case/002.png`였다(픽셀별 평균 절대 차이 0.011973). 픽셀 경고 여부가 가장 많이 바뀐 사진은 `misplaced/000.png`로, 65,536칸 중 45칸이 달랐다.

## 사진별 표 읽기

`ground_truth_label`은 정상이 0, 불량이 1이다. `cpu_`와 `gpu_` 열에는 각 장비의 이미지 점수와 픽셀 수를 기록했다. `area`는 전체 이상 면적, `right`는 오른쪽 절반의 이상 면적이다. `top_left` 등 네 구역은 위치별 면적이다. `baseline_outcome`, `score_outcome`, `lower_score_outcome`, `score_or_area_outcome`은 위 표 순서의 사진 판정이다. `score_or_right_outcome`은 점수 ≥ 38.70416 또는 오른쪽 이상 면적 ≥ 820칸을 사용한 별도 위치 실험이다.

`TP`는 불량 검출, `FN`은 미탐, `FP`는 정상 오경보, `TN`은 정상 무경고다. `raw_map_mae`는 두 장비 간 원시 지도의 평균 절대 차이, `different_mask_pixels`는 픽셀 경고 여부가 다른 칸의 수다. `gt_pixels_on_model_map`과 장비별 `overlap_pixels`는 모델 지도 크기로 맞춘 정답 영역과 그중 모델이 경고한 칸 수다. [100장 전체 표](per-image.csv)

## 저장한 비교 자료로 3D 실험 다시 열기

연구 장비에 보관한 MVTec 원본, 모델·평가 묶음, CPU 검사 폴더와 당시 분석 자료가 있으면 두 갤러리를 다시 만들 수 있다. `artifacts/analyses/transistor-patchcore-mac-gpu-test-20261119/`에는 장비 비교표와 원시 CPU 지도, `artifacts/analyses/transistor-patchcore-test-errors-20261119/`에는 정답 대조표와 요약이 있어야 한다.

CPU API를 켜고 프로젝트 루트에서 아래 순서대로 실행한다. 첫 명령이 새로 경고된 정상 사진의 검사 ID와 점수 실험을 준비한다. 두 번째 명령이 마지막 미탐 사례와 정상 14장의 비교 지도를 만든다. 실험별 `per-image.csv`의 주소를 같은 장비에서 열 수 있다. [3D 실험 보기](../../output/defectops-inspection-lab/README.md#실제-사진-검사)

```sh
artifacts/envs/patchcore-cpu/bin/python scripts/analyze_transistor_score_threshold.py
artifacts/envs/patchcore-cpu/bin/python scripts/analyze_transistor_map_area.py
```

## 별도로 수집한 사진 평가하기

정상과 불량을 모두 포함하는 독립 촬영분을 준비하고 CSV에 `image_path,label`을 적는다. 경로는 새 사진 폴더를 기준으로 하고, 정답에는 `normal` 또는 `defect`를 쓴다. 촬영 출처와 부품 단위로 학습·실험 자료와의 독립성을 확인한다. 검사기는 기존 313장(학습 170장, 보정 43장, 테스트 100장)과 새 배치 안에서 픽셀까지 같은 사진을 찾아 제외한다.

모델·평가 묶음과 원본 MVTec 트랜지스터 데이터를 프로젝트에 준비한다. 패키지를 설치한 뒤 첫 명령으로 입력을 확인한다. 실제 검사 때는 Mac CPU API와 3D 작업대를 켜고 두 번째 명령을 실행한다. `new-photos/`와 `labels.csv`에는 준비한 경로를 넣는다. [Mac 실행 안내](../../docs/patchcore-local-api.md#mac에서-기존-모델-실행하기) · [검사기](../../scripts/evaluate_transistor_holdout.py)

```sh
artifacts/envs/patchcore-cpu/bin/python scripts/evaluate_transistor_holdout.py new-photos/ labels.csv --check-only
artifacts/envs/patchcore-cpu/bin/python scripts/evaluate_transistor_holdout.py new-photos/ labels.csv
```

결과는 `artifacts/analyses/inspection-threshold-trials/transistor-holdout-*/`에 저장된다. `summary.json`에 미탐·오경보 수가, `per-image.csv`의 `error_map_url`에 틀린 사진의 3D 작업대 주소가 생긴다. [3D 검사 워크스페이스](../../output/defectops-inspection-lab/)
