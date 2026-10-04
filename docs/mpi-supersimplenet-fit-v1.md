# SuperSimpleNet Phase 2 — 정상64장 정식 학습과 정상 보정

2026-10-05. 사용자 승인: 새 초기화 학습 → 중간 저장/재개 → 별도 정상128/169장 보정 →
최종 모델 재로딩 검증. **불량12장·전체 시험·서비스 전환은 이번 실행에 포함하지 않는다.**

## 왜 지금 진행하나요?

[GPU 점검 수령 결과](../reports/mpi-supersimplenet-probe-v1-received-review.md)에서100회 학습,
GPU 예약5.14GiB, 재로딩 점수/지도 차이0을 확인했다. 실행 가능성 근거이지 검출 성능 근거는 아니다.
정식 실행기는 동일 모델 구조·전처리·batch8을 유지하고 정상64장 전체와 고정 학습 일정을 사용한다.

## 이번 실행의 네 단계

1. **새 초기화:** 공식 ImageNet backbone은 그대로 사용하고 학습할 adaptor/두 head를 seed42로 초기화한다.
   처음 가중치 지문이 probe의 학습 전 지문과 일치해야 한다. probe100회 모델은 읽지 않는다.
2. **300epoch 학습:** 정상64사진×12영역=768타일만 학습한다. batch8, epoch당96회, 총28,800회다.
3. **정상 보정:** 개발128사진으로12영역별 점수 중앙값·Q90 폭을 계산하고,
   보정169사진으로 사진 단위 경고선을 정한다. 두 자료 모두 가중치 학습에는 쓰지 않는다.
4. **재로딩:** 최종300epoch 모델을 다시 읽어 정상297사진 전체의 원점수·지도·사진 점수와 판정을 대조한다.
   성공하면 JSON-only 결과 ZIP을 만든다. 검사 성능 평가는 다음 단계다.

## 고정 학습 계약

| 항목 | 확정 조건 |
| --- | --- |
| 구현 | Anomalib2.2.0 ICPR형; 동일 공식 소스 SHA256과 GPU 환경 |
| 특징 | wide_resnet50_2.tv_in1k, layer2/3, 고정 backbone |
| 입력 | 원래 RGB256×256, ImageNet 정규화1회, crop·추가 증강 없음 |
| 학습 | seed42, FP32, batch8; 내부 특징 합성 배치16 |
| 합성 | Perlin0.6 / Gaussian std0.015 / 공식 약50% 합성 분기 |
| AdamW | adaptor lr1e-4·weight_decay0.01 / 두 head lr2e-4·weight_decay1e-5 |
| 학습량 | 300epoch,28,800 업데이트; 해시 기반 epoch별 전체768타일 순열 |
| 학습률 | 240epoch 완료 후0.4배,270epoch 완료 후 다시0.4배 |
| 저장 | 초기상태와10epoch마다 저장, 총31개; 자동 삭제 없음 |
| 중간 재로딩 | 50epoch마다 학습 정상 첫 사진12영역으로 원점수·지도·tensor지문 확인 |
| 선택 모델 | 최종300epoch 모델 하나. 손실/시험 결과로 중간 모델 선택 금지 |
| 계산 예산 | 학습·정상 보정·재로딩 누적14,400초(4시간) |
| 메모리 | PyTorch reserved peak≤6GiB, process RSS peak≤20GiB |
| 시작 조건 | GPU 여유6GiB, 디스크 여유20GiB, 다른 GPU 작업 정상 종료 |

정상 사진은 기존 손상·중복 검토가 완료된 명단을 그대로 사용한다. 사진 ID와 정확히 같은 픽셀의
교차 중복을 검사한다. 정식64장과128/169장은 서로 분리한다. 전체 archive 해시 검증과 ZIP 항목
목록 확인은 하지만 시험 이미지·결함 마스크는 디코딩하지 않는다.

## 점수와 경고 기준

공식 분류 head의 `pred_score`(부호 있는 logit)가 타일의 유일한 주 점수다. 음수를 자르거나
실제 불량 확률로 해석하지 않는다. 지도 최댓값/상위1% 점수로 바꾸지 않는다.

```text
영역 중심 = 정상 개발128장의 해당 영역 원점수 중앙값
영역 폭   = 같은 원점수의 Q90(higher) - 중앙값
영역 점수 = (타일 원점수 - 영역 중심) / 영역 폭
사진 점수 = 12영역 점수의 최댓값
경고 기준 = 별도 정상169장의 사진 점수 Q95(higher)
경고 여부 = 사진 점수 >= 경고 기준
```

폭≤1e-6 또는 비유한값이면 중단한다. 임의 epsilon·다른 보정식으로 대체하지 않는다.
원시 지도는 추후 위치 설명용으로 보존하지만 이미지 판정에는 사용하지 않는다.
이번 실행은 지도용 정상화/픽셀 임곗값을 만들지 않는다. 정상 Q95는 미래 오경보5% 보장이 아니다.

297사진 재로딩 시 원점수·지도·표준화 사진 점수 최대 절대 차이≤1e-4, 판정 변화0이어야 한다.
차이가 난다고 경고 기준을 다시 맞추지 않는다. checkpoint는 optimizer·CPU/CUDA/Python/NumPy
난수까지 복원하며, 학습률도 저장 step의 계약과 대조한다.

## 연구실 명령

전원을 연결하고 절전으로 계산이 중단되지 않도록 확인한 후, 다른 GPU 작업을 정상 종료한다.
스크립트가 전원 설정이나 타 프로세스를 임의 변경하지는 않는다.

```powershell
Set-Location 'C:\개인 프로젝트\defectops'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git update failed; stop here.' }
git rev-parse --short HEAD
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\scripts\run_mpi_supersimplenet_fit.ps1'
```

기본 원본: `$env:USERPROFILE\Downloads\extracted.zip`. 다르면 wrapper의 `-Archive '실제 경로'`를 지정한다.
다음 기존 파일을 재사용하며, 누락/변경 시 학습 전에 중단한다.

- `artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip`
- `artifacts\handoff\mpi-pcb-score-probe-v1.zip` (정상 개발 명단만 읽음)
- `artifacts\handoff\mpi-supersimplenet-probe-v1.zip`
- `artifacts\research-assets\supersimplenet-v1\wide_resnet50_2.tv_in1k.safetensors`

결과는 `artifacts\mpi-supersimplenet-fit-v1`에 별도 저장한다. 기존 점검·Q90·EfficientAD·서비스 모델을 덮어쓰지 않는다.
완료 후 아래 ZIP만 전달한다. 원본·모델·원시 지도는 로컬에 남긴다.

```text
C:\개인 프로젝트\defectops\artifacts\handoff\mpi-supersimplenet-fit-v1.zip
```

## 시간과 중단·재개

실측100회 평균 업데이트0.286초를28,800회로 늘리면 순수 업데이트 약2시간17분,
점검 전체 비용 비례값은 약2시간57분이다. **약3시간 전후는 계획용 추정이며 완료 보장이 아니다.**
원본 검증은 계산 예산 밖에 별도 수행한다. 학습·정상 보정·재로딩은4시간을 함께 사용한다.
epoch마다 진행을 출력하고10epoch마다 저장 완료를 출력한다.

Ctrl+C 정상 중단이면 같은 명령으로 검증된 마지막10epoch 저장점부터 재개한다.
저장 이후 잃어버린 작업도 계산 시간과 attempted updates에 포함한다. 재작업 예산은 최대960회,
전체29,760시도이며 최종 모델에 반영된 학습은 여전히28,800회다. 무제한 재시도하지 않는다.
여러 번 중단하면 재작업 또는 시간 예산을 다 쓸 수 있다.

강제 종료로 기록이 없거나 OOM·파일 변경·재로딩 오류·자원 상한 초과가 발생하면 자동 재개하지 않는다.
**결과를 삭제/이름 변경하거나 상한을 바꾸지 말고 오류를 전달한다.** 완성 파일은 보존한다.
완료 후 재실행하면 무결성을 확인해 기존 결과만 재사용한다.

## 다음 단계

정식 학습 ZIP의 무결성·정상 보정 폭·경고선·재로딩·시간/메모리를 검토한 뒤,
기존에 검출했던 고정 불량12장 회귀검사를 준비한다. 이미 관찰한 MPI 자료이므로
이후 성적도 탐색·회귀검증이며 독립 일반화 성능 주장이 아니다.
기존53/55 검출 보존과 정상 오경보2/54·12/256 이하 등 PDF의 후속 기준은 유지한다.
