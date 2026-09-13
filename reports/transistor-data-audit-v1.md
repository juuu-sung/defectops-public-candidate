기준일: 2026-09-12

# 데이터 준비 결과

점검 상태: **passed**

실제 로컬 파일을 검사한 결과다. 모델 학습·성능 평가 결과는 아니다.

- 이미지 디코딩: 313장
- 결함 마스크 디코딩: 40장
- 오류: 0건
- 디코딩 픽셀 중복: 0쌍

## 고정된 데이터 목록

| 구분 | 장수 |
|---|---:|
| transistor/calibration/good | 43 |
| transistor/test/anomaly | 40 |
| transistor/test/good | 60 |
| transistor/train/good | 170 |

목록만 나눴으며 원본 파일을 이동·수정하지 않았다.

## 이미지와 마스크 형식

- 이미지: 1024x1024/RGB, 313장
- 마스크: 1024x1024/L, 40장

## 유형별 수량

| 목록/유형 | 장수 |
|---|---:|
| transistor/calibration/good | 43 |
| transistor/test/bent_lead | 10 |
| transistor/test/cut_lead | 10 |
| transistor/test/damaged_case | 10 |
| transistor/test/good | 60 |
| transistor/test/misplaced | 10 |
| transistor/train/good | 170 |

## 정상 데이터 밝기·선명도 기초 점검

테스트는 무결성만 확인했다. 아래 통계는 train/calibration만 사용한다.

밝기는 0–255 회색조 평균, 선명도 지표는 원본 해상도의 Laplacian 분산이다.
이는 기술 통계이며 불량 점수나 품질 거절 기준이 아니다.

| 목록 | 지표 | 최소 | p05 | 중앙값 | p95 | 최대 |
|---|---|---:|---:|---:|---:|---:|
| transistor/calibration | brightness_mean | 70.723 | 73.440 | 77.166 | 81.498 | 82.688 |
| transistor/calibration | brightness_std | 36.020 | 37.967 | 39.644 | 41.402 | 42.910 |
| transistor/calibration | laplacian_variance | 20.663 | 21.412 | 25.703 | 32.054 | 32.320 |
| transistor/train | brightness_mean | 71.760 | 73.067 | 77.927 | 82.804 | 87.740 |
| transistor/train | brightness_std | 36.476 | 38.039 | 40.430 | 42.449 | 43.227 |
| transistor/train | laplacian_variance | 20.624 | 21.588 | 25.027 | 32.458 | 34.824 |

## 재현 정보

- Manifest SHA-256: `9695055446c1aafdb77dc28eb91d270b8b7335e650c7e74f400d386e9d8bf585`
- Pillow: 11.3.0, NumPy: 2.3.3
- 분할 seed 42, 정상 80/20, calibration 수 올림

## 한계와 다음 단계

파일 디코딩·마스크 이진값·크기·대응 관계를 점검했으나, 마스크가 실제 결함 경계를
정확하게 표시하는지는 판단하지 않았다. 체크섬은 로컬 데이터 버전을 고정하며 공식 원본 인증은 아니다.
정상 calibration 표본 수는 위 분할 표에 기록했다. 95% 분위수는 초기 경험적 운영점이며 오탐률 5%를 보장하지 않는다.
다음은 데이터 loader·전처리 규칙·PaDiM 학습 연결과 장비별 ML 환경 고정이다.
