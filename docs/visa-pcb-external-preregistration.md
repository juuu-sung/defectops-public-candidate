# 새 VisA PCB 외부 평가 — 이미지 열람 전 고정 계획

상태: **공식 분할 metadata·후보·평가 기준을 고정했고, 원본 이미지와 점수는 미관찰** (2026-09-24).

## 쉽게 말하면

기존 Real-IAD PCB 개발 데이터에서는 상위 0.25% 점수 방식이 기존 1%보다 불량 4장을 덜
놓쳤다. 이 작은 차이가 우연인지, **다른 공개 PCB 사진**에서도 나타나는지 확인한다.
새 데이터의 정상·불량 시험 사진은 평가 규칙을 고정하기 전에 열거나 점수 내지 않는다.
같은 제품을 다시 촬영한 데이터가 아니므로 이것은 **서로 다른 기판으로의 전이** 검증이지
기존 Real-IAD PCB 제품의 성능 개선 증명이 아니다.

## 데이터 출처와 라이선스

- 원본: Amazon Science [VisA 프로젝트](https://github.com/amazon-science/spot-diff),
  [AWS Open Data 등록](https://registry.opendata.aws/visa/).
- 원본 묶음: `VisA_20220922.tar`, 공식 S3 URL은
  `https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar`.
  확인된 파일 크기는 1,929,840,640바이트(약 1.93 GB)다.
- Amazon Science가 게시한 one-class split CSV를 Git commit
  `2a692ab575001cbde74d402d897a7286086c6199`에 고정했다. CSV SHA-256은
  `a48557e6033318cb90556f706196bc9d247a776a23ea51aecee5a80dd0332995`다.
- 공식 프로젝트와 AWS 등록 페이지는 데이터 라이선스를 CC BY 4.0으로 안내한다.
  원본·가공 이미지는 Git이나 handoff ZIP에 넣지 않는다. 외부 공개 시 출처와 변형을 표기한다.

## 미리 고정한 데이터와 규칙

공식 `1cls.csv`의 PCB1–4만 사용한다. 각 품목의 공식 정상 train 중 경로 해시 순서로
정상 56장을 **임곗값 계산용**으로 정하고, 공식 test는 모두 외부 최종 시험으로 봉인한다.
선정 규칙·경로 목록은
[`visa-pcb-external-plan-v1.json`](../data/manifests/visa-pcb-external-plan-v1.json)에 고정했다.
계획 SHA-256은 `e134445218bfb71c3cbcfcad9da45548024ef6166a2acd8b0ab64ff23e5ba34a`다.

| 기판 종류 | 임곗값용 정상 | 봉인 시험 정상 | 봉인 시험 불량 |
|---|---:|---:|---:|
| PCB1 | 56 | 100 | 100 |
| PCB2 | 56 | 100 | 100 |
| PCB3 | 56 | 101 | 100 |
| PCB4 | 56 | 101 | 100 |
| 합계 | 224 | 402 | 400 |

- Real-IAD PCB seed 42의 기존 `layer1+layer2` PatchCore checkpoint를 그대로 사용한다.
  VisA 이미지로 모델을 **학습하지 않는다**.
- 두 이미지 점수만 비교한다: 기존 상위 **1% 평균**, 도전자 상위 **0.25% 평균**.
  결과를 본 뒤 후보를 추가하지 않는다.
- 256×256 입력·ImageNet 정규화를 유지한다. 각 기판 종류의 정상 56장으로 두 점수
  방식마다 95% higher 임곗값을 따로 계산한다. 이는 **정상 데이터만 이용한 기판별 임곗값
  적응**이지 완전한 zero-shot 평가가 아니다.
- 네 기판을 동일 가중한 미탐률 차이(기존 − 도전자)가 1차 비교값이다. 동일 이미지의
  짝지은 bootstrap 5,000회(고정 seed 20260924) 95% 구간의 하한이 0보다 크고,
  도전자의 모든 기판 정상 오검출률이 10% 이하이며, 어느 기판의 미탐률도 기존보다
  5%p 넘게 악화하지 않을 때만 **외부 전이에서 개선 근거가 있다**고 표현한다.
  통과해도 실제 생산 자동 PASS 근거는 아니다.
- 품목별 TP/FP/TN/FN, 불량 미탐률·정상 오검출률, Image AUROC/AP, 마스크 기반
  위치 지표, 실패 이미지 수와 분모를 함께 보고한다. 실패를 분모에서 빼지 않는다.

## 현재 가능한 등록 단계

이번에 만든 실행기는 **2 GB 데이터 묶음을 받아 구조와 체크섬만 등록**한다. 공식 CSV가
사전에 고정한 것과 같은지, 필요한 사진·마스크 파일 이름이 묶음 안에 있는지 확인한다.
시험 사진을 디코딩하거나 모델에 넣지 않는다. 사진의 손상·라벨·근접 중복 여부는 아직
검증한 것이 아니므로 등록 성공을 성능 검증으로 읽지 않는다.

연구실 Windows에서 기존 프로젝트와 `.venv`를 그대로 사용하고 PowerShell에서 실행한다.
스크립트가 공식 원본을 `Downloads`에 자동 다운로드하며, 중단 시 같은 명령으로 이어받는다.
필요한 여유 공간은 압축되지 않은 원본 약 1.93 GB와 추후 평가 산출물을 고려해 충분히
확보한다. 같은 이름의 완성 파일은 덮어쓰지 않고 검사하며, 다운로드가 중단된 부분 파일은
같은 명령으로 이어받는다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git status --short

powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\register_visa_pcb_external.ps1"
```

완료하면 `C:\개인 프로젝트\defectops\artifacts\handoff\visa-pcb-external-registration-v1.zip`과
출력된 SHA-256을 이 작업에 첨부한다. 원본 `.tar`은 첨부하지 않는다. 오류가 나면 파일을
삭제하지 말고 그대로 오류 출력을 공유한다.

## 다음 게이트

등록 ZIP의 CRC, 고정 계획·설정 SHA와 수량은 Mac에서 독립 대조해 일치했다. ZIP SHA-256은
`52e2f24e853e60738791d02e75f937521906da51a488eb37f66e7a6b0b9ec040`이다.
ZIP에는 원본 tar가 없으므로 Windows 보고서가 적은 tar SHA-256
`2eb8690c803ab37de0324772964100169ec8ba1fa3f7e94291c9ca673f40f362`은 Mac에서
다시 계산하지 못했다. 다음 사전 점검은 Windows 원본 tar의 SHA를 직접 다시 계산하고,
기존 Real-IAD PCB seed 42 checkpoint의 잠금 계약을 확인한다.

### 사전 점검: 정상 calibration만 디코딩

[사전 점검 설정](../configs/visa-pcb-external-preflight-v1.toml)에 등록 ZIP·tar SHA를 고정했다.
이 코드는 정상 calibration JPEG 224장만 열어 RGB 디코딩, 크기, 압축 바이트 및 픽셀
SHA-256, 정확한 픽셀 중복 여부를 검사한다. 시험 802장의 JPEG와 불량 마스크 400장의
payload는 **디코딩·점수 계산하지 않는다**. 원본 tar 전체를 해시하기 위해 파일 바이트를
순차적으로 읽는 것은 시험 사진을 추출·디코딩하는 것과 구분한다.

연구실 Windows에서 이전 등록에 사용한 원본 tar와 기존 `.venv`를 그대로 사용한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_visa_pcb_external_preflight.ps1"
```

완료하면 `artifacts\handoff\visa-pcb-external-preflight-v1.zip`과 출력 SHA-256을 첨부한다.
오류가 나면 원본·기존 결과를 삭제하거나 덮어쓰지 말고 메시지만 공유한다. 성공해도
**모델 성능 결과가 아니며**, 시험 이미지·마스크의 디코딩 가능 여부나 train/test 간
중복은 아직 확인되지 않았다. 이 파일을 검토한 뒤에야 후보 두 개와 지표를 고정한 GPU
평가 실행기를 열고, 시험 802장을 한 번 평가한다. 새 결과를 보고 후보나 임곗값을 바꾸면
그 시험은 탐색 결과로 강등한다. 자동 PASS는 계속 금지한다.

### 사전 점검 확인 결과와 시험 개봉 전 최종 잠금

연구실 사전 점검 ZIP SHA-256은
`c6568b6691d03c1a244da29a7505b0d6f56dc2970f618e8f7dacd9d35d4ce69d`다.
ZIP 내부 두 JSON의 CRC, 감사 기록의 정규화 SHA, 정상 이미지 224개 경로의 고정 계획
일치, 설정·등록·원본 tar SHA의 잠긴 값 일치와 checkpoint SHA 기록을 확인했다.
PCB1–4 정상 각 56장의 JPEG 디코딩이 성공한 것으로 보고됐고 정확한 픽셀 중복은 0이다.
시험 이미지 디코딩·점수 계산은 모두 0으로 기록됐다. Mac에는 원본 tar와 checkpoint가
없으므로 이 결과는 **전달 기록 검증**이지 원본 바이트의 독립 재검사가 아니다.

[최종 평가 설정](../configs/visa-pcb-external-evaluation-v1.toml)은 사전 점검 ZIP과 기존
Real-IAD PCB seed 42 checkpoint SHA를 고정한다. 모델을 VisA로 다시 학습하지 않는다.
평가기는 시작 전에 등록 ZIP·원본 tar·사전 점검 ZIP·checkpoint 잠금과 정상 JPEG 224장의
감사 기록을 재확인한다. 그 후 GPU 모델로 각 기판의 정상 56장을 점수화하고 **기존 상위
1%와 도전자 상위 0.25%의 95% higher 임곗값을 각각 고정**한다.

그 다음 같은 명령 실행 안에서 공식 test 802장의 JPEG와 불량 마스크 400장을 열어
전부 디코딩하고, 형식·크기·빈 마스크·정확한 이미지 중복을 검사한다. 하나라도 실패하면
그 이미지를 분모에서 빼지 않고 **성능 결과 생성 자체를 중단**한다. dHash 근접 중복은
참고 신호로 기록하되 동일 물리 기판의 증거로 단정하지 않는다. 데이터 감사가 모두 끝나면
두 고정 점수를 같은 802장에서 계산한다. 중단되면 검증된 샘플 캐시만 재사용하며 원본과
기존 결과를 덮어쓰지 않는다.

품목별 미탐·정상 오검출·AUROC/AP, 위치 지표(Pixel AUROC/AUPRO)를 기록한다. 불량에
대해서는 같은 이미지의 두 판정 차이를 기판별로 짝지어 5,000회 bootstrap한다. 95%
구간 하한이 0을 넘고, 모든 기판의 도전자 정상 오검출률이 10% 이하이며, 어느 기판도
미탐률이 기존보다 5%p 넘게 나빠지지 않을 때만 **다른 기판으로의 전이 개선 근거**라고
표현한다. 이는 같은 제품의 개선 확증·현장 배포 승인·자동 PASS 근거가 아니다.

연구실 Windows에서, 현재 Git 상태가 깨끗한지 확인한 뒤 아래 명령은 **공식 시험을
실제로 여는 명령**이다. 준비된 코드를 검토하지 않았다면 실행하지 않는다. 모델 추론에
시간이 걸릴 수 있으며, 중단 시 같은 명령으로 이어서 실행할 수 있다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git status --short
powershell.exe -NoProfile -ExecutionPolicy Bypass `
  -File ".\scripts\run_visa_pcb_external_evaluation.ps1" `
  -OpenFinal
```

완료 후 `artifacts\handoff\visa-pcb-external-evaluation-v1.zip`과 출력 SHA-256을 이 작업에
첨부한다. 원본 tar, 원본 사진, 모델 checkpoint, NPZ 샘플 캐시는 첨부하지 않는다.
오류가 나면 삭제·수정 없이 메시지를 공유한다. 같은 test를 본 뒤 설정을 수정해 재평가하면
그 결과는 사전 고정 시험으로 다시 주장할 수 없다.

### 2026-09-24 마스크 감사 중단 진단

첫 실행은 정상 224장으로 임곗값을 계산한 뒤 공식 test 감사에서
`pcb1/Data/Masks/Anomaly/000.png`가 축소 후 빈 마스크라는 오류로 중단됐다.
이 실행은 test 점수를 계산하거나 성능 지표를 만들지 않았다. 기존 부분 산출물은 보존한다.
공식 VisA 준비 코드는 원본 마스크를 **0이 아닌 모든 값을 결함**으로 이진화한다.
현재 평가기는 이 단계 없이 128 이상을 결함으로 간주하므로, 원본이 클래스 번호 1 등을
쓴다면 결함을 놓칠 수 있다. 실제 파일이 그런지는 아래 읽기 전용 진단으로 확인한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\diagnose_visa_pcb_mask.py" `
  --archive "$env:USERPROFILE\Downloads\VisA_20220922.tar"
```

이 진단은 **이미 오류가 난 마스크 한 장만** 열고 원본/기존 파서/공식 방식의 결함 픽셀
수를 출력한다. 원본 tar와 기존 평가 산출물을 수정하지 않으며 모델 추론도 하지 않는다.
출력을 확인하기 전에는 기존 평가 명령을 다시 실행하지 않는다. 마스크 파서를 수정해야
한다면 변경 이력과 별도 실행 ID를 기록하고, 수정 후 결과를 원래 사전 고정 평가와
구별해 보고한다.
