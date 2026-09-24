# VisA 기판별 정상 memory bank 개발 실험 v1

## 왜 하는가

기존 VisA 외부 평가에서는 **Real-IAD PCB 정상 사진으로 만든 PatchCore**를 그대로 사용했다.
VisA 기판별 정상 56장은 임곗값 계산에만 썼다. 시험 불량 400장 중 기존 점수 방식은
30장만 검토 대상으로 잡았고, 새 상위 0.25% 방식은 17장만 잡았다. 이는 다른 기판으로
옮겨 쓸 때 정상 특징 기준이 맞지 않을 수 있다는 가설을 낳지만 원인을 입증하지는 않는다.

이번 개발 실험은 모델 구조와 점수 방식을 유지한 채 **정상 패치 기준 데이터의 출처**를
Real-IAD PCB에서 VisA 해당 기판으로 바꾼다. 사전등록된 미개봉 시험이 아니다. 이전에
VisA 시험 802장과 시각 사례를 이미 관찰했으므로 이번 fit과 후속 성능 비교는 모두
**사후 탐색적**이다. 현장 사용·자동 PASS·확인적 성능 주장에 사용할 수 없다.

## 정확히 무엇을 고정하는가

| 항목 | 값 |
|---|---|
| 모델 | PatchCore, Wide ResNet-50-2 사전학습 backbone |
| 특징 | layer1 + layer2 |
| 입력 | RGB, ImageNet 정규화, 256×256 bilinear/antialias 정사각형 변환 |
| 정상 패치 선택 | 10% coreset, 최근접 이웃 9개, seed 42 |
| 이미지 점수 | 이상 맵 상위 1%의 평균 |
| 이미지 임곗값 | 정상 calibration 점수의 95% higher 분위수 |
| 픽셀 임곗값 | 정상 calibration 맵의 99.5% higher 분위수 |

이전 v2와 **동일한 모델 구조·입력·점수 규칙**이다. 정상 memory bank의 학습 자료만
해당 VisA 기판의 공식 정상 train으로 바꾼다. 학습 표본 수가 달라 memory bank 크기도
달라질 수 있으므로 이를 완전히 단일 수치 변수의 실험이라고 과장하지 않는다.

| 기판 | 공식 정상 train | 이미 고정된 calibration | 이번 memory bank 학습 |
|---|---:|---:|---:|
| PCB1 | 904 | 56 | 848 |
| PCB2 | 901 | 56 | 845 |
| PCB3 | 905 | 56 | 849 |
| PCB4 | 904 | 56 | 848 |

위 분할은 [기존 원본 CSV와 고정 계획](../data/manifests/visa-pcb-external-plan-v1.json)을
매번 대조한다. 이전 calibration 56장을 학습 패치에 섞지 않는다. 새 학습은 공식 test의
JPEG/마스크를 열거나 점수화하지 않는다. 원본 tar, 기존 평가 ZIP 및 사전 점검 ZIP의
SHA-256도 실행 전에 다시 확인한다. 학습 중 정확한 픽셀 중복이 학습-학습 또는
학습-calibration 사이에서 발견되면 멈춘다.

## 1차 실행: PCB3만

PCB3는 상위 0.25% 방식으로 바꾸며 기존에 잡던 불량 17장을 놓친 기판이다. 우선
PCB3 하나만 실행해 학습 시간과 저장 모델의 재로딩 일치성을 확인한다. 같은 특징
구성의 앞선 Real-IAD PCB 학습은 약 **1.82시간** 걸렸지만, 이 시간은 PCB3 전체
849장의 4096개 패치/장에 적용할 수 있는 예상치가 아니었다. 아래 v1 시도는 실제로
정상 영상 특징 추출 849/849 직후 3,477,504개 후보에서 10% coreset 선택을 시작하다
연구실 RTX 5060 8GB의 CUDA 메모리 부족으로 멈췄다. 사진·기존 평가·기존 완료
모델은 삭제하지 않았다. v1을 그대로 재실행하지 않는다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_visa_pcb_normal_bank_pilot.ps1"
```

v1 명령은 이력 보존용이며 연구실 8GB 환경에서는 사용하지 않는다. 성공 시
`artifacts\handoff\visa-pcb-normal-bank-pcb3-v1.zip`과 SHA-256을 Codex 작업에
첨부한다. 출력 ZIP에는 fit 보고서, 학습 사진 감사 기록, 정상 calibration 점수만 있고
모델 checkpoint·원본 사진·시험 결과는 없다. checkpoint는 연구실 컴퓨터의
`artifacts\visa-pcb-normal-bank-development-v1\pcb3\model.pt`에 남는다. 삭제하거나
Git에 올리지 않는다. 같은 명령을 다시 실행하면 완료된 학습을 검증해 재사용하며,
유효하지 않은 기존 결과는 덮어쓰지 않고 오류를 낸다.

## 8GB 메모리 대응 v2: 모든 정상 사진에서 공간 균등 패치 표본 추출

v2는 학습 사진 849장을 모두 사용하되 각 이미지의 64×64 특징 격자에서 행·열을
하나씩 건너뛰어 32×32=1,024개 패치만 **coreset 후보로** 남긴다. 후보는
3,477,504개에서 869,376개, 10% coreset 목표는 347,750개에서 86,937개로
줄어든다. 현재 Anomalib 2.2.0은 전체 후보에 대한 특징 투영과 반복 거리 계산을
GPU에서 하므로 최종 10%만 낮추는 것보다 후보 자체를 제한하는 것이 중요하다.
향후 메모리 사용과 시간은 연구실 실측으로 확인해야 하며 성공을 보장하지 않는다.

이 변경은 정상 사진 수를 줄이지 않지만 공간 패치 밀도를 낮춘다. 작은 결함의
정상 참조가 누락될 수 있고, 기존 Real-IAD bank와의 비교에서도 데이터 출처와
후보 밀도가 함께 달라진다. 따라서 결과는 별도의 **사후 탐색 v2**로 기록하며
원본 v1 설정·출력은 덮어쓰지 않는다. 정상 calibration 56장은 계속 분리하고 공식
test 사진·마스크는 학습 중 열지 않는다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_visa_pcb_normal_bank_pilot_v2.ps1"
```

완료 시 `artifacts\handoff\visa-pcb-normal-bank-pcb3-v2.zip`을 첨부한다. ZIP은
모델·원본 사진 없이 fit 보고서·학습 감사 기록·정상 calibration 점수만 담는다.
checkpoint는 `artifacts\visa-pcb-normal-bank-development-v2\pcb3\model.pt`에만
남겨 둔다. v1의 실패 산출물이나 이전 실험 폴더를 지우지 않는다.

PCB3 v2는 연구실 Windows RTX 5060 8GB에서 정상 학습 849장·보정 56장으로
완료됐다. 기준표 `86,937×768`, 전체 실행 `8,123.5초`, 이미지 임곗값 `26.111570`이다.
이는 정상 기준 생성 완료이지 불량 검출 성능 결과가 아니다. 다음 단계는
[같은 PCB3 시험 201장 사후 탐색 비교](visa-pcb-normal-bank-pcb3-evaluation.md)다.

## 이 다음

PCB3의 학습/재로딩 근거와 자원 비용은 검토했다. 이제 별도의 명시적 평가 단계에서
이미 본 VisA PCB3 test와 비교한다. 결과를 검증하기 전에는 나머지 기판의 같은
정상 학습으로 확대하지 않는다. 그 시험을 이용한 변경·선택은 탐색적이다. 진짜 확인적 결론에는
새 미관측, 중복 통제된 holdout과 사전에 잠근 실패 허용 기준이 필요하다.
