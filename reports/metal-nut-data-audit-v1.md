# Metal Nut 데이터 준비 결과

점검 상태: **passed**

실제 로컬 파일을 검사한 결과다. 모델 학습·성능 평가 결과는 아니다.

- 이미지 디코딩: 335장
- 결함 마스크 디코딩: 93장
- 오류: 0건
- 디코딩 픽셀 중복: 0쌍

## 고정된 데이터 목록

| 구분 | 장수 |
|---|---:|
| metal_nut/calibration/good | 44 |
| metal_nut/test/anomaly | 93 |
| metal_nut/test/good | 22 |
| metal_nut/train/good | 176 |

목록만 나눴으며 원본 파일을 이동·수정하지 않았다.

## 이미지와 마스크 형식

- 이미지: 700x700/RGB, 335장
- 마스크: 700x700/L, 93장

## 유형별 수량

| 목록/유형 | 장수 |
|---|---:|
| metal_nut/calibration/good | 44 |
| metal_nut/test/bent | 25 |
| metal_nut/test/color | 22 |
| metal_nut/test/flip | 23 |
| metal_nut/test/good | 22 |
| metal_nut/test/scratch | 23 |
| metal_nut/train/good | 176 |

## 정상 데이터 밝기·선명도 기초 점검

테스트는 무결성만 확인했다. 아래 통계는 train/calibration만 사용한다.

밝기는 0–255 회색조 평균, 선명도 지표는 원본 해상도의 Laplacian 분산이다.
이는 기술 통계이며 불량 점수나 품질 거절 기준이 아니다.

| 목록 | 지표 | 최소 | p05 | 중앙값 | p95 | 최대 |
|---|---|---:|---:|---:|---:|---:|
| metal_nut/calibration | brightness_mean | 51.832 | 52.911 | 57.540 | 64.834 | 66.214 |
| metal_nut/calibration | brightness_std | 36.630 | 37.077 | 41.330 | 49.963 | 51.669 |
| metal_nut/calibration | laplacian_variance | 28.471 | 31.205 | 36.704 | 47.897 | 52.582 |
| metal_nut/train | brightness_mean | 50.913 | 52.131 | 57.992 | 65.739 | 70.126 |
| metal_nut/train | brightness_std | 36.407 | 37.020 | 42.085 | 52.163 | 55.586 |
| metal_nut/train | laplacian_variance | 21.607 | 27.681 | 35.933 | 49.162 | 61.340 |

## 재현 정보

- Manifest SHA-256: `6270aecb5f243662a1f6110078f34a5f76846f865510048d68391c0bcfee6be5`
- Pillow: 11.3.0, NumPy: 2.3.3
- 분할 seed 42, 정상 80/20, calibration 수 올림

## 한계와 다음 단계

파일 디코딩·마스크 이진값·크기·대응 관계를 점검했으나, 마스크가 실제 결함 경계를
정확하게 표시하는지는 판단하지 않았다. 체크섬은 로컬 데이터 버전을 고정하며 공식 원본 인증은 아니다.
정상 calibration 표본 수는 위 분할 표에 기록했다. 95% 분위수는 초기 경험적 운영점이며 오탐률 5%를 보장하지 않는다.
다음은 데이터 loader·전처리 규칙·PaDiM 학습 연결과 장비별 ML 환경 고정이다.
