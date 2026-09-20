# 3품목 비교 사전등록 — transistor, cable, metal_nut

등록일: 2026-09-21

설정: `configs/three-category-comparison-v1.toml`

## 범위와 증거 상태

최종 포트폴리오 비교 품목은 초기 코드 allowlist에 있던 `transistor`, `cable`, `metal_nut`로
고정한다. 새로 임의 선정한 품목을 추가하지 않는다. 품목별로 별도 모델과 별도 normal-only
calibration 임곗값을 만든다.

transistor test와 모델 결과는 이미 관찰했다. 반면 cable과 metal_nut의 test 이미지는 아직
열거나 시각적으로 확인하지 않았다. 따라서 전체 3품목 결과는 새 완전 독립 benchmark가 아니라
`mixed_transistor_observed_additional_categories_preregistered`로 표시한다. cable·metal_nut 결과는
이 문서와 설정을 커밋한 뒤에만 평가한다.

## test 전에 고정한 조건

| 항목 | 고정값 |
|---|---|
| 분할 | 공식 train/good의 80/20, SHA-256(seed:path), seed 42 |
| 모델 | PaDiM/ResNet18, PatchCore/Wide ResNet-50-2 |
| 모델 seed | 42, 43, 44 |
| 입력 | 원본 RGB PNG → 256×256 bilinear antialias stretch |
| 정규화 / 증강 | ImageNet mean/std / 없음 |
| 이미지 임곗값 | 정상 calibration score 95% `higher` |
| 픽셀 임곗값 | 정상 calibration map pixel 99.5% `higher` |
| morphology | 없음 |
| 지표 | AUROC, AP, 고정 임곗값 혼동행렬, Pixel AUROC/AUPRO |
| 자동 PASS | 금지 |

PaDiM은 layer1/2/3과 선택 특징 100개, PatchCore는 layer2/3, coreset 10%, 최근접 이웃 9개를
사용한다. test를 본 뒤 입력 크기·backbone·coreset·임곗값·ROI·후처리를 바꾸지 않는다.

## 후보 선택 규칙

3품목 × 3 seed를 모두 보고할 때 품목별 값과 macro 평균·표본 표준편차를 함께 제시한다.
포트폴리오 주력 후보는 다음 순서로 정한다.

1. 고정 임곗값 macro escape rate가 낮은 모델
2. 동률이면 macro Image AUROC가 높은 모델
3. 다시 동률이면 macro overkill rate가 낮은 모델
4. 다시 동률이면 warm end-to-end p95가 낮은 모델

선택된 결과도 실제 배포 성능의 불편 추정치는 아니다. MVTec 공개 test의 불량률은 공장 불량률이
아니며, 실제 자동 PASS를 허용하지 않는다.

## 데이터 입력 계약 변경

transistor는 1024×1024지만 metal_nut은 다른 원본 해상도를 사용한다. 그래서 PatchCore fit·평가의
1024×1024 전용 검사를 test 열람 전에 제거하고, RGB PNG이면 원본 크기를 기록한 뒤 동일한
256×256 stretch 전처리를 적용하도록 고정했다. 단일 이미지 검사 API는 현재 transistor 시연용
계약이므로 아직 1024×1024만 허용한다. 새 품목을 서비스에 연결할 때는 품목 선택과 bundle 검증을
별도로 구현한다.

## 데이터 확보와 manifest 체크포인트

공식 [MVTec AD 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)는 다운로드 전에
사용자 폼을 요구한다. cable과 metal_nut archive를 연구실 Windows에서 공식 페이지를 통해 받은 뒤
다음 구조로 압축 해제한다.

```text
C:\개인 프로젝트\defectops\data\cable\train\good\...
C:\개인 프로젝트\defectops\data\cable\test\good\...
C:\개인 프로젝트\defectops\data\cable\ground_truth\...
C:\개인 프로젝트\defectops\data\metal_nut\train\good\...
C:\개인 프로젝트\defectops\data\metal_nut\test\good\...
C:\개인 프로젝트\defectops\data\metal_nut\ground_truth\...
```

그다음 `scripts\prepare_additional_categories.ps1`을 한 번 실행한다. 이 스크립트는 archive가
남아 있으면 SHA-256을 확인하고, 두 manifest를 생성·재검증하며, 이미지/마스크를 디코딩해 형식과
대응 관계만 점검한다. test의 밝기·선명도 통계나 preview는 만들지 않는다.

스크립트가 만든 다음 파일을 이 작업에 첨부한다.

```text
artifacts\handoff\additional-category-data-registration-v1.zip
```

manifest와 점검 요약을 저장소에 커밋하기 전에는 학습을 시작하지 않는다. 이렇게 해야 Windows의
학습 실행이 깨끗한 Git commit과 고정 manifest를 함께 기록할 수 있다.

## 다음 단계

1. [완료] 공식 archive 다운로드·압축 해제
2. [완료] manifest 생성·무결성 audit
3. [완료] handoff ZIP checksum 검증과 manifest·데이터 보고서 반영
4. 연구실 Windows에서 cable·metal_nut의 두 모델 × 3 seed fit/calibration
5. 12개 fit 완료를 확인한 뒤 사전 고정한 규칙으로만 test 평가
6. 품목별 결과와 3품목 macro 비교, 오류 분석, warm 속도 측정
