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
구성의 앞선 Real-IAD PCB 학습은 약 **1.82시간** 걸렸다. 이번 시간은 장비 상태와
이미지 수에 따라 달라진다. PowerShell과 연구실 컴퓨터를 켠 채 충분한 시간을 둔다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass -File ".\scripts\run_visa_pcb_normal_bank_pilot.ps1"
```

완료 시 `artifacts\handoff\visa-pcb-normal-bank-pcb3-v1.zip`과 SHA-256을 Codex 작업에
첨부한다. 출력 ZIP에는 fit 보고서, 학습 사진 감사 기록, 정상 calibration 점수만 있고
모델 checkpoint·원본 사진·시험 결과는 없다. checkpoint는 연구실 컴퓨터의
`artifacts\visa-pcb-normal-bank-development-v1\pcb3\model.pt`에 남는다. 삭제하거나
Git에 올리지 않는다. 같은 명령을 다시 실행하면 완료된 학습을 검증해 재사용하며,
유효하지 않은 기존 결과는 덮어쓰지 않고 오류를 낸다.

## 이 다음

PCB3의 학습/재로딩 근거와 자원 비용을 먼저 검토한다. 그다음 나머지 기판의 같은
정상 학습을 순서대로 진행할지 결정하고, 별도의 명시적 평가 단계에서 이미 본 VisA
test와 비교한다. 그 시험을 이용한 변경·선택은 탐색적이다. 진짜 확인적 결론에는
새 미관측, 중복 통제된 holdout과 사전에 잠근 실패 허용 기준이 필요하다.
