# Real-IAD PatchCore frozen evaluation v1

평가 ZIP SHA-256: `cbe64561e780bc11aa4e503e384ea90547323e3d269c214597f51874d4cdc183`

두 신규 외부 카테고리에서 seed 42·43·44의 사전 고정 PatchCore fit을 한 번씩 평가했다.
모든 평가의 상태는 `evaluated_preregistered`, 실패 수는 0, 모델 재로딩 최대 점수 차이는 0이었다.
`transistor1`과 `plastic_nut`은 기존 MVTec 품목의 동일제품 holdout이 아니다.

## 세 seed 평균

| 지표 | realiad_transistor1 | realiad_plastic_nut |
|---|---:|---:|
| Image AUROC | 0.9382 | 0.8696 |
| Average precision | 0.9654 | 0.7278 |
| Fixed-threshold precision | 0.9812 | 0.7312 |
| Fixed-threshold recall | 0.5935 | 0.5763 |
| Fixed-threshold F1 | 0.7390 | 0.6446 |
| Escape rate | 0.4065 | 0.4237 |
| Overkill rate | 0.0201 | 0.0566 |
| Review rate | 0.3865 | 0.1661 |
| Pixel AUROC | 0.9911 | 0.9904 |
| Pixel AUPRO | 0.9511 | 0.9050 |
| 평가 loop p95 | 17.96 ms | 24.60 ms |

순위화와 픽셀 위치 검출은 유망하지만 정상 calibration의 고정 95% image 임곗값에서 불량 약 40%를
통과시킨다. 따라서 현재 구성은 자동 합격 판정에 사용할 수 없다. 특히 `realiad_transistor1`의
`ZW` 평균 recall은 0.4104, `realiad_plastic_nut`의 `QS`는 0.4762, `AK`는 0.5000이었다.

`transistor1` seed 42가 test recall이 가장 높지만 test 관찰 후 그 seed만 고르는 것은 사후 선택이다.
공식 결과는 세 seed 분포와 평균을 유지한다. 다음은 모델이나 임곗값을 바꾸지 않고 세 seed 공통 미탐과
seed별 불안정 사례를 분리하는 탐색적 오류 진단이다. 현재 test에 맞춘 변경은 새 확증 결과로 보고하지 않는다.
