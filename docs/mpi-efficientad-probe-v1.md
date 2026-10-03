# EfficientAD-S 정상 이미지 사전 점검

2026-10-03. 사용자가 보조 데이터·가중치 다운로드 절차와 짧은 사전 점검 준비에 동의했다.
이 실행은 **새 방식이 RTX5060에서 학습·저장·재로딩되는지와 비용을 확인하는 단계**다.
불량을 잘 잡는지, 정식 학습이 완료됐는지는 판단하지 않는다. [설계 근거](../reports/mpi-model-redesign-2026-10-03.md)

## 이번에 실행하는 일

1. 기존 MPI archive와 정상64장 준비 ZIP의 고정 해시·명단을 확인한다.
2. 공식 teacher 가중치와 Imagenette를 다운로드하고 Anomalib2.2.0 배포 코드의 SHA256과 대조한다. 완료 파일은 재사용한다.
3. 기존 정상64장에서 해시 순으로 고른8사진×12영역=96타일의 teacher 통계를 계산한다.
4. student와 autoencoder만 최대100회 업데이트한다. batch1이며 100epoch가 아니다.
5. 25회마다 저장한다. 50회와100회에는 모델·optimizer·CPU/CUDA 난수 상태를 다시 불러오고 정상1타일 지도의 차이≤0.0001을 확인한다.
6. 정상1사진의12타일 추론 시간을 예비 측정하고 JSON만 담은 전달 ZIP을 만든다.

학습 정상 ID는 `1075,1299,0386,1370,0195,0574,1150,0473`이다. 선정 규칙은
`SHA256("mpi-efficientad-probe-v1" + NUL + photo_id)` 오름차순 첫8개다.
각96타일을 한 번씩 사용한 뒤 앞4타일을 다시 사용해100회 업데이트한다.
손상·중복 대체를 새로 하지 않으며 원래 검증된64장 명단 밖의 제품 이미지는 사용하지 않는다.

보조 Imagenette는 `train`에서 파일명 해시로 선정한96JPEG만 사용한다. MPI 제품 정상·불량과 다른
보조 손실용 자료다. full-size archive를 해시 검증한 뒤 선택 파일만 로컬 임시 준비 경로로 복사한다.
경로 탈출·링크·중복 항목을 거부하며 archive 경로대로 전체 압축 해제하지 않는다.

## 고정 설정과 한계

| 항목 | 값 |
| --- | --- |
| 환경 | Windows Python3.12, torch2.8.0+cu128, torchvision0.23.0+cu128, anomalib2.2.0, RTX5060 8GB |
| 모델 | EfficientAdModelSize.S,384 teacher채널, padding=False, pad_maps=True |
| 전처리 | 제품 RGB256×256 → ToTensor만 적용. 모델 내부 정규화와 중복 금지 |
| 학습 | seed42, batch1, Adam lr0.0001, weight_decay0.00001, 이번 짧은 점검은 고정 학습률 |
| teacher | 사전학습 가중치 고정. 평균·표준편차는 점검 정상96타일만 사용 |
| 보조 이미지 | Resize512×512 → RandomGrayscale0.3 → CenterCrop256 → ToTensor |
| 업데이트 | 중단 시도까지 누적 최대100회. 자동 연장·재선정 없음 |
| 자원 | 계산 단계 누적900초, PyTorch reserved peak6GiB, 프로세스 RSS20GiB |
| 시작 조건 | GPU 여유6GiB 및 디스크 여유6GiB 이상 |
| 출력 지도 | 보정 전 지도. 정상128/169장·임곗값·불량 시험·위치 정답 접근 없음 |

상한은 각 계산 경계에서 확인한다. 진행 중인 CUDA 연산을 강제 종료하는 하드 실시간 제한은 아니다.
RSS와 장치 전체 사용량은100ms 간격 표본 peak, PyTorch 할당·예약량은 allocator peak다.
장치 전체 점유를 이 프로세스만의 GPU 메모리라고 표현하지 않는다.
다운로드·archive 검증과 준비 시간은900초 GPU 단계와 별개이며 콘솔에 따로 출력한다.
첫 다운로드는 네트워크에 따라 더 오래 걸릴 수 있다. 개별 파일 다운로드는 크기 상한과2시간 제한이 있다.

학습 손실과 gradient의 유한값·저장·재로딩 성공은 검출 성능이 아니다.12타일1회 시간은
정식 warm p95도, 전체 API 응답 시간도 아니다.100step 결과를4시간 정식 학습 완료의 보장으로 쓰지 않는다.
이번 checkpoint는 사전 점검용으로 표시하며 서비스나 정식 학습 초기값으로 재사용하지 않는다.

## 연구실에서 실행

다른 GPU 작업은 정상 종료한 뒤 실행한다. 서버나 프로세스를 이 스크립트가 임의로 종료하지 않는다.

```powershell
Set-Location 'C:\개인 프로젝트\defectops'
git pull --ff-only
if ($LASTEXITCODE -ne 0) { throw 'Git update failed; stop here.' }
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\scripts\run_mpi_efficientad_probe.ps1'
```

필수 입력은 `Downloads\extracted.zip` 및 프로젝트의
`artifacts\handoff\mpi-pcb-normal-bank-preparation-v1.zip`이다. 기존 PatchCore checkpoint는 읽거나 수정하지 않는다.
다운로드 캐시는 `artifacts\research-assets\efficientad-v1`, 실행 기록은
`artifacts\mpi-efficientad-probe-v1`에 별도로 보관한다. 완료 후 전달할 파일은 다음과 같다.

```text
C:\개인 프로젝트\defectops\artifacts\handoff\mpi-efficientad-probe-v1.zip
```

전달 ZIP에는 계획·환경·해시·손실·자원·재로딩 기록만 들어간다. 원본 이미지·보조 데이터·가중치·checkpoint는 포함하지 않는다.

## 중단과 재실행

완료 후 같은 명령을 실행하면 해시가 일치하는 기존 결과를 재사용하며 다시 학습하지 않는다.
다운로드 중 끊기면 완료 파일만 재사용하고 미완료 파일은 보존한 채 새 다운로드를 시작한다.

학습 중 Ctrl+C는 마지막 검증된25회 단위 저장점과 남은 시간·업데이트 한도 안에서만 재개할 수 있다.
저장 이후에 소모한 업데이트도 예산에서 차감하므로 재개해도100회 완료가 불가능할 수 있다.
강제 종료·OOM·검증 실패·자원 초과·변경된 소스/입력은 자동 재실행하지 않고 검토를 요구한다.
**오류가 나면 결과를 삭제하거나 한도를 바꾸지 말고 오류 메시지를 전달한다.**

다음은 ZIP 검토 후 정식 학습의 고정 step·학습률 일정·안정성 조건과 비용을 결정하는 단계다.
현재 실행기는 정식 학습이나 불량 평가로 자동 이어지지 않는다.

## 배포 출처와 이용 범위

- [Anomalib2.2.0 EfficientAD 구현과 파일 해시](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/efficient_ad/lightning_model.py)
- [모델 구현](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/src/anomalib/models/image/efficient_ad/torch_model.py)
- [Imagenette 제공처](https://github.com/fastai/imagenette)
- [ImageNet 제공처 이용 안내](https://www.image-net.org/)

이번 경로는 비상업 연구 점검용이다. Imagenette 저장소의 코드 라이선스를 개별 이미지의
상업 이용·재배포 허가로 해석하지 않는다. MPI와 보조 이미지·가중치를 Git이나 공개 문서로 배포하지 않는다.
