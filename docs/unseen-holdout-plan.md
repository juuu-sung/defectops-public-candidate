# 새 미관찰 데이터와 holdout 계획

상태: 수집 전 계획 v1

기준일: 2026-09-21

## 목적

현재 공개 MVTec test는 모델 비교와 오류 분석에 이미 사용했다. 이 이미지를 다시 나누거나 이름만
바꿔 새 holdout으로 사용하지 않는다. 후속 변경은 독립적으로 새로 촬영하거나 별도 출처에서 확보한
이미지로만 설계·검증한다.

첫 우선순위는 [추가 품목 사후 진단](../reports/patchcore-additional-diagnostics-v1.md)에서 확인한
Cable `cable_swap`·작은 `poke_insulation`, Metal Nut의 작은 `color` 결함과 정상 표면 변동이다.
이 계획은 생산 합격 기준이나 안전 보증이 아니라 포트폴리오 단계의 누수 없는 검증 절차다.

## 데이터 집합을 두 개로 분리한다

| 집합 | 열람 가능 시점 | 허용 용도 | 금지 용도 |
|---|---|---|---|
| development | 등록·무결성 검사 후 | 후보 규칙·ROI·입력 크기·후처리 설계 | 최종 성능 주장 |
| final holdout | 프로토콜과 후보를 동결한 뒤 한 번 | 동결 후보의 최종 확인 | 재튜닝·재선택 |

동일 물리 부품, 동일 촬영 burst, 같은 생산 lot 또는 같은 세션의 근접 중복 이미지를 두 집합에
나누지 않는다. 파일 단위 무작위 분할이 아니라 `part_id`·`capture_session_id`·`lot_id` 중 확보 가능한
가장 강한 그룹 단위로 분리한다.

## 1차 수집 목표

아래 수는 최소 포트폴리오 검증 목표다. 생산 성능이나 희귀 결함률을 보장하기에는 부족하므로
95% 신뢰구간과 함께 보고한다.

| 품목·유형 | Development | Final holdout | 목적 |
|---|---:|---:|---|
| Cable 정상 | 60 | 60 | 도선 색·배치·반사 변화의 오탐 확인 |
| Cable cable_swap | 30 | 30 | 현재 가장 큰 미탐 유형 확인 |
| Cable 작은 poke_insulation | 30 | 30 | 작은 국소 결함 감도 확인 |
| Cable 기타 각 결함 유형 | 유형별 10 | 유형별 10 | 기존 장점의 회귀 감시 |
| Metal Nut 정상 | 60 | 60 | 표면·중앙부의 넓은 오탐 확인 |
| Metal Nut 작은 color | 30 | 30 | 현재 유일한 미탐 유형 확인 |
| Metal Nut 기타 각 결함 유형 | 유형별 10 | 유형별 10 | 기존 장점의 회귀 감시 |

물리적으로 결함 표본을 만들 수 없거나 라벨을 신뢰할 수 없으면 부족한 수를 합성 이미지나 기존 test
복제로 채우지 않는다. 확보 수와 결측 이유를 그대로 기록한다.

## 촬영·라벨 metadata

이미지마다 다음 값을 원본 기록에 남긴다.

- `sample_id`, `category`, `label`, `defect_type`
- `part_id`, `lot_id`, `capture_session_id`
- 촬영 시각, 카메라·렌즈 식별자, 해상도, 거리와 방향
- 조명 설정, 배경·고정구, 작업자 또는 수집 담당자
- 라벨 작성자, 라벨 근거, 마스크 유무와 검수 상태
- 원본 파일 SHA-256, decode 크기, 수집 출처와 이용 조건

결함 mask가 없으면 image-level 평가만 수행하고 Pixel AUROC/AUPRO를 억지로 계산하지 않는다.
조명이나 카메라 조건을 일부러 바꾸는 경우 운영 조건 변동 검증임을 별도 태그로 표시한다.

## 저장과 누수 차단

원본은 Git에 넣지 않는다. 권장 로컬 구조는 다음과 같다.

```text
data/custom_holdout_v1/
  cable/development/...
  cable/final_holdout/...
  metal_nut/development/...
  metal_nut/final_holdout/...
```

1. 결과를 보기 전에 그룹 단위 분할 목록과 예상 표본 수를 기록한다.
2. 파일 SHA-256과 decode pixel hash로 기존 MVTec·development·final holdout 중복을 차단한다.
3. 동일 부품·세션 그룹이 양쪽 split에 존재하면 등록을 실패시킨다.
4. manifest에 상대 경로·label·defect type·그룹 ID·checksum을 저장한다.
5. final holdout의 manifest SHA-256을 커밋하고 이미지 폴더는 봉인한다.
6. 봉인 후 후보 설정이 바뀌면 같은 final holdout 결과는 탐색 결과로 강등한다.

## 평가 순서

1. 현재 PatchCore seed 42 bundle과 임곗값을 그대로 새 development에 실행해 domain shift를 확인한다.
2. 변경이 필요하면 development에서만 후보를 만든다. 후보 수와 변경 범위를 문서에 먼저 고정한다.
3. 후보마다 입력 크기·ROI·score 결합·후처리·임곗값 출처를 저장한다.
4. 한 후보를 선택한 뒤 final holdout을 열기 전에 commit·manifest·checkpoint·설정을 고정한다.
5. final holdout은 한 번 실행하고 실패한 입력도 분모와 실패율에 포함한다.
6. 결과를 보고 다시 변경하면 새 final holdout을 수집하기 전까지 확증 결과로 부르지 않는다.

## 사전 성공 기준

1차 목표는 아래 조건을 모두 보고하는 것이다. 수집 수가 작으므로 수치 통과만으로 실제 생산 자동
합격을 허용하지 않는다.

- Cable 전체 escape rate ≤ 5%, `cable_swap` recall ≥ 90%, `poke_insulation` recall ≥ 90%
- Metal Nut 전체 escape rate ≤ 5%, `color` recall ≥ 90%
- 각 품목 정상 overkill rate ≤ 10%
- RTX 5060 batch size 1 warm sample p95 ≤ 100 ms, 실패율 0%
- 모델 재로딩 score 최대 절대 차이 0.0
- 모든 비율에 이미지 단위 95% 신뢰구간과 실제 분모 표시
- 자동 PASS는 계속 금지하고, 사람 REVIEW 흐름을 유지

후보가 recall을 개선해도 정상 overkill이 기준을 넘으면 채택하지 않는다. 기준에 미달한 결과를 숨기거나
표본을 제거하지 않고 원인과 다음 수집 계획을 함께 기록한다.

## 지금 필요한 사용자 작업

아직 연구실 Windows에서 실행할 명령은 없다. 먼저 실제로 새 이미지를 확보할 수 있는 출처와 촬영
조건을 정해야 한다. 새 독립 데이터 확보가 불가능하면 현재 모델을 탐색적 포트폴리오 결과로 동결하고
기존 공개 test에 대한 추가 튜닝은 중단한다.

