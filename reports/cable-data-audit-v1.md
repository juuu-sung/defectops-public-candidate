# Cable 데이터 준비 결과

점검 상태: **passed**

실제 로컬 파일을 검사한 결과다. 모델 학습·성능 평가 결과는 아니다.

- 이미지 디코딩: 374장
- 결함 마스크 디코딩: 92장
- 오류: 0건
- 디코딩 픽셀 중복: 0쌍

## 고정된 데이터 목록

| 구분 | 장수 |
|---|---:|
| cable/calibration/good | 45 |
| cable/test/anomaly | 92 |
| cable/test/good | 58 |
| cable/train/good | 179 |

목록만 나눴으며 원본 파일을 이동·수정하지 않았다.

## 이미지와 마스크 형식

- 이미지: 1024x1024/RGB, 374장
- 마스크: 1024x1024/L, 92장

## 유형별 수량

| 목록/유형 | 장수 |
|---|---:|
| cable/calibration/good | 45 |
| cable/test/bent_wire | 13 |
| cable/test/cable_swap | 12 |
| cable/test/combined | 11 |
| cable/test/cut_inner_insulation | 14 |
| cable/test/cut_outer_insulation | 10 |
| cable/test/good | 58 |
| cable/test/missing_cable | 12 |
| cable/test/missing_wire | 10 |
| cable/test/poke_insulation | 10 |
| cable/train/good | 179 |

## 정상 데이터 밝기·선명도 기초 점검

테스트는 무결성만 확인했다. 아래 통계는 train/calibration만 사용한다.

밝기는 0–255 회색조 평균, 선명도 지표는 원본 해상도의 Laplacian 분산이다.
이는 기술 통계이며 불량 점수나 품질 거절 기준이 아니다.

| 목록 | 지표 | 최소 | p05 | 중앙값 | p95 | 최대 |
|---|---|---:|---:|---:|---:|---:|
| cable/calibration | brightness_mean | 92.200 | 93.998 | 99.075 | 107.754 | 109.608 |
| cable/calibration | brightness_std | 44.155 | 46.796 | 50.061 | 52.708 | 53.217 |
| cable/calibration | laplacian_variance | 19.151 | 20.599 | 32.998 | 52.360 | 65.566 |
| cable/train | brightness_mean | 92.085 | 93.569 | 100.564 | 107.885 | 109.026 |
| cable/train | brightness_std | 42.056 | 46.124 | 50.095 | 53.298 | 57.051 |
| cable/train | laplacian_variance | 15.630 | 22.594 | 35.362 | 51.257 | 57.056 |

## 재현 정보

- Manifest SHA-256: `f2caf5c39f7fe53060278d2a000634db0c62781b4e81d8d9d984eb4abc73dd70`
- Pillow: 11.3.0, NumPy: 2.3.3
- 분할 seed 42, 정상 80/20, calibration 수 올림

## 한계와 다음 단계

파일 디코딩·마스크 이진값·크기·대응 관계를 점검했으나, 마스크가 실제 결함 경계를
정확하게 표시하는지는 판단하지 않았다. 체크섬은 로컬 데이터 버전을 고정하며 공식 원본 인증은 아니다.
정상 calibration 표본 수는 위 분할 표에 기록했다. 95% 분위수는 초기 경험적 운영점이며 오탐률 5%를 보장하지 않는다.
다음은 데이터 loader·전처리 규칙·PaDiM 학습 연결과 장비별 ML 환경 고정이다.
