# SuperSimpleNet Phase 0·1 — 실행 준비와 정상 전용 GPU 점검

2026-10-04. 사용자 승인 범위: Phase 0 실행기·검증 준비와 Phase 1 연구실 GPU 사전 점검.
**정식300epoch 학습·불량 평가·서비스 전환은 하지 않는다.** 이 문서는 PDF 설계의 실행 준비
상세이며, 향후 정식 조건은 점검 결과 검토 후 잠근다.

## 이번에 무엇을 확인하나요?

SuperSimpleNet은 고정 특징 추출기 위의 adaptor·위치 head·분류 head를 학습한다.
정상 특징 일부에 합성 이상을 추가해 구분하도록 학습한다. 현재 방식이 MPI 불량을 잘
검출한다는 근거는 아직 없다. 이번에는 다음 질문만 답한다.

> 연구실 RTX5060 8GB에서 계획한 배치8로 학습·추론·저장·재로딩이 정상 동작하는가?

입력 배치8은 **모델 내부 특징 단계에서16개로 복제**된다. 따라서 입력8장만으로
메모리 사용량을 추정하면 안 된다. OOM이나 상한 초과 시 자동으로 배치를 줄이지 않는다.

## 입력과 처리 순서

1. 기존 `Downloads\extracted.zip`과 정상64장 준비 ZIP의 SHA256·기존 명단을 검증한다.
2. 정상64장 중 `SHA256("mpi-supersimplenet-probe-v1" + NUL + photo_id)` 오름차순8장을 고른다.
   각12영역, 총96타일만 읽는다. 정상128/169장이나 시험 사진·위치 정답은 읽지 않는다.
   archive 전체 바이트 해시 확인은 시험 이미지 디코딩·모델 평가가 아니다.
3. 공식 timm의 `wide_resnet50_2.tv_in1k` 가중치 약276MB를 고정 revision으로 받고 전체 SHA256을 대조한다.
4. 정상96타일을 배치8로 최대100회 업데이트한다.12회마다96타일을 모두 한 번씩 사용하고,
   다음 순회는 독립적인 고정 순열로 진행한다. 총800타일 제시, **100epoch가 아니다.**
5. 25회마다 저장하고50·100회에는 재로딩한다. 원점수·지도 차이≤0.0001,
   모델 tensor 지문·optimizer step·CPU/CUDA/Python/NumPy 난수 복원을 확인한다.
6. 단건/배치8 추론, backbone 불변, 학습 계층 변화, 정상1사진12영역 처리 시간을 확인한다.
7. JSON만 담은 전달 ZIP을 만든다. 모델은 연구실에 유지한다.

점검 정상 ID는 `1619,0833,1605,1075,0203,1088,0465,0908`이다.
100회 합계에서 정상·합성 클래스가 모두 등장했는지도 확인한다.

## 고정한 설정

| 항목 | Phase 1 점검 계약 |
| --- | --- |
| 환경 | Windows Python3.12 / torch2.8.0+cu128 / torchvision0.23.0+cu128 / anomalib2.2.0 / timm1.0.20 |
| GPU | RTX5060 8GB; 시작 전 여유6GiB 이상 |
| 입력 | RGB256×256 → 0–1 tensor → ImageNet 정규화1회; crop·추가 증강 없음 |
| 특징 | layer2/3, 공식 upscaling·pooling; backbone 가중치와 BatchNorm 통계 고정 |
| 학습 | seed42, FP32, batch8; 내부 학습 특징 batch16 |
| 합성 | Perlin 기준0.6 / Gaussian 표준편차0.015 / 공식 약50% 합성 분기 |
| 정상 정답 | 메모리에서 생성한0마스크·0라벨. 결함 마스크 파일을 읽지 않음 |
| AdamW | adaptor lr1e-4 / weight_decay0.01, heads lr2e-4 / weight_decay1e-5 |
| 일정 | probe 고정 학습률, 최대100 attempted updates; 정식300epoch 일정 미실행 |
| 점수 | 부호 있는 공식 `pred_score` logit. 음수도 보존; 확률로 해석하지 않음 |
| 자원 상한 | 계산 누적900초 / PyTorch reserved peak6GiB / 프로세스 RSS20GiB |
| 저장 공간 | 실행 전 여유6GiB 이상; 최초 가중치와5개 checkpoint용 |

가중치 초기화만 offline으로 통제한다. 공식 `TimmFeatureExtractor(pre_trained=False)`로
동일 구조를 생성한 뒤 고정 safetensors를 strict-load한다. 사용하는 backbone key가 없거나
모르는 추가 key가 있으면 중단한다. 미사용 layer4와fc만 제외하며 forward·합성·손실은 공식 구현을 쓴다.
Anomalib 모델·합성·손실·특징 추출·Perlin 소스의 LF 기준SHA256도 v2.2.0과 대조한다.
환경을 자동 업그레이드하거나 로컬 site-packages를 수정하지 않는다.

다운로드·원본 해시 검증은 GPU900초와 별도로 시간을 기록한다. 계산 상한은 연산 경계에서
확인하는 제한이며 실행 중인 CUDA 커널을 강제 종료하는 하드 실시간 제한은 아니다.
RSS·전체 장치 사용량은100ms 표본 peak, PyTorch는 allocator peak다.
장치 전체 점유를 이 프로세스만의 점유라고 표현하지 않는다.

## 연구실 실행 명령

다른 GPU 작업을 정상 종료한 다음 실행한다. 기존 venv·모델·데이터를 유지한다.
PowerShell5.1에서도 경로에 공백·한글이 포함된 프로젝트를 지원한다.

```powershell
Set-Location 'C:\개인 프로젝트\defectops'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git update failed; stop here.' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\scripts\run_mpi_supersimplenet_probe.ps1'
```

필수 입력:

- `$env:USERPROFILE\Downloads\extracted.zip`
- `artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip`
- 프로젝트 내부 `.venv\Scripts\python.exe`

원본 경로가 다를 때만 wrapper의 `-Archive '실제 경로'`를 지정한다.
스크립트가 타 프로세스를 종료하거나 신규 의존성을 설치하지 않는다.

완료 후 전달:

```text
C:\개인 프로젝트\defectops\artifacts\handoff\mpi-supersimplenet-probe-v1.zip
```

ZIP은 환경·정상 명단 지문·설정·100회 손실·gradient·합성 비율·자원·재로딩 기록만 포함한다.
이미지·가중치·원시 지도·checkpoint를 포함하지 않는다.
checkpoint와 실패 기록은 `artifacts\mpi-supersimplenet-probe-v1`,
공식 가중치는 `artifacts\research-assets\supersimplenet-v1`에 별도 보존한다.

## 중단·재실행

완료 후 재실행은 동일 입력·코드·checkpoint 영수증을 검증하고 기존 ZIP을 재사용한다.
Ctrl+C는 검증된25회 저장점에서 남은900초·100 attempted updates 한도 안에서만 재개한다.
저장 이후 이미 소모된 업데이트도 차감하므로 중간에 끊으면100회 완료가 불가능할 수 있다.
강제 종료 기록 누락·OOM·검증 오류·상한 초과·변경된 소스/입력은 자동 재시도하지 않는다.
오류가 나면 **폴더를 삭제하거나 이름을 바꿔 다시 돌리지 말고 오류를 전달**한다.
이번 모델을 정식 학습의 초기값이나 서비스 모델로 사용하지 않는다.

## 결과 해석과 다음 단계

Mac CPU 테스트는 데이터 경계·해시·스케줄·출력 형태·중단/재개·실패 보존 검증이다.
가짜 runtime으로 수행한 테스트를 실제 CUDA 성공으로 해석하지 않는다.

100회 손실 감소는 불량 검출 개선이 아니며,12타일1회 처리 시간은 warm p95가 아니다.
실측 GPU 메모리·시간과 재로딩이 통과하면 Phase2 정식 조건을 확정한다.
제안300epoch는768타일/배치8=96회씩 총28,800 updates이며,
정상 개발128/보정169장은 학습에 섞지 않고 후속 점수 보정에만 사용한다.
논문의 batch32와 다른8GB용 실험이고4시간 내 완료는 아직 보장하지 않는다.
기존 Q90 기준선·EfficientAD 결과·transistor 서비스·자동합격 금지는 유지한다.

## 공식 출처

- [Anomalib2.2.0 모델](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/supersimplenet/torch_model.py)
- [공식 optimizer/전처리](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/supersimplenet/lightning_model.py)
- [공식 합성 이상 생성](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/supersimplenet/anomaly_generator.py)
- [고정 timm 가중치](https://huggingface.co/timm/wide_resnet50_2.tv_in1k/tree/343c5fa59ca8a131e0cdf42144a641b36a0387c2)

정상 타일과 가중치는 로컬 연구용이며 Git/전달 ZIP으로 재배포하지 않는다.
