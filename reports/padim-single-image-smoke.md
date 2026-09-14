# PaDiM 단일 이미지 검사 검증

검증일: 2026-09-14  
장비: 연구실 Windows, NVIDIA GeForce RTX 5060  
상태: 실행·판정·추적·산출물 생성 성공

## 실행 식별자

- Inspection: `transistor-bent-lead-000`
- 입력: `transistor/test/bent_lead/000.png`
- 입력 라벨: anomaly, `bent_lead`
- 입력 SHA-256: `82bf661fb868a80558fc45f4fc356c96888bf26827005a57aea6aa707e13483a`
- 실행 commit: `8cb8f3edbafeb7fef380821fa6358a8eafd36dfb`
- Fit: `padim-transistor-seed42-retry1`
- Evaluation: `padim-transistor-seed42-eval-v1`
- Manifest SHA-256: `9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585`
- Checkpoint SHA-256: `7c45c84e74fe2416baa1e612ace519eb081ecee7afad102ecc175fcc4ea351ea`

입력 checksum은 고정 manifest의 해당 이미지 checksum과 일치한다. fit·evaluation·manifest·checkpoint와
실행 commit 추적도 모두 기록됐다.

## 판정 결과

| 항목 | 값 |
|---|---:|
| 상태 | `completed` |
| 판정 | `REVIEW_REQUIRED` |
| 원시 이미지 점수 | 56.807789 |
| 고정 이미지 임곗값 | 56.027042 |
| 임곗값 초과분 | 0.780746 (약 1.39%) |
| 고정 픽셀 임곗값 | 28.198650 |
| 이상 map pixel | 3,186 / 65,536 (약 4.86%) |
| 4연결 영역 수 | 7 |

이 입력은 실제 `bent_lead` 이상 이미지이므로 이미지 수준 판정은 true positive다. 점수가 임곗값에
가깝기 때문에 강한 여유를 둔 탐지 사례로 해석하지 않는다. 이 한 장의 성공으로 전체 recall이나
실제 현장 성능을 다시 추정하지 않는다.

## 위치 대조

- 정답 마스크 bbox: `[239, 829, 290, 947]`
- 가장 큰 예측 영역 bbox: `[200, 764, 352, 1024]`
- 가장 큰 영역: 1,651 map pixels, 전체 이상 map pixels의 약 51.82%
- 정답 마스크의 모든 픽셀은 가장 큰 예측 bbox 안에 위치
- 정답 bbox 대비 가장 큰 예측 bbox IoU: 약 0.152

주 예측 상자가 실제 결함 위치를 빠뜨리지 않고 포함했지만 정답보다 넓다. 이 값은 bbox 대조이며,
예측 픽셀과 정답 픽셀의 실제 겹침률은 `anomaly-map.npz` 또는 overlay를 직접 확인해야 한다.
추가로 아래쪽·테두리 중심의 영역 5개와 작은 내부 영역 1개가 있어 위치 정밀도와 경계 잡음은
후속 실패 사례 검토 대상으로 남긴다.

## 지연시간

| 구간 | 시간 |
|---|---:|
| 설정 검증·모델 로드 | 21,571.5 ms |
| PNG decode | 47.0 ms |
| 전처리 | 7.0 ms |
| GPU 추론·map 전송 | 508.1 ms |
| 후처리·산출물 저장 | 272.2 ms |
| 전체 | 22,407.1 ms |

전체 시간의 약 96.3%가 최초 설정 검증과 모델 로드다. 따라서 22.4초를 이미지 한 장의 순수 추론
속도로 해석하면 안 된다. GPU 추론 구간은 약 0.51초였으며, 운영 지연시간은 모델을 메모리에 유지한
상태의 warm 반복 실행을 별도로 측정해야 한다.

## 산출물과 결론

`inspection.json`, `input.png`, `heatmap.png`, `overlay.png`, `anomaly-map.npz`가 생성됐고 각 checksum이
기록됐다. `automatic_pass`와 `automatic_pass_allowed`는 모두 `false`다.

단일 이미지의 입력 검증 → 고정 모델 로드 → 원시 점수 → 고정 판정 → 픽셀 영역 → 추적 JSON·시각화
산출물 흐름은 실제 연구실 장비에서 동작했다. 따라서 첫 검사 파이프라인의 기술적 마일스톤은 완료다.
단, 전체 test recall이 55%이므로 현재 모델은 실제 자동 PASS나 안전 판단에 사용할 수 없다.

PowerShell 출력에서 한글 `source_path`가 깨져 보인 것은 UTF-8 BOM이 없는 JSON을 기본 문자셋으로
읽은 표시 문제다. 입력 checksum 일치로 실제 파일이 바뀐 것은 아님을 확인했다. 확인할 때는
`Get-Content -Encoding UTF8`을 사용한다.
