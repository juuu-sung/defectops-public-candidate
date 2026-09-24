# VisA PCB3 정상 기준표 v2 사후 탐색 평가

## 이번에 확인하는 질문

기존 Real-IAD PCB 정상 기준표를 사용했을 때와, VisA PCB3 정상 사진으로 만든
새 기준표를 사용했을 때 **같은 PCB3 공식 시험 사진 201장**의 미탐·정상 오검출·
이미지 순위·결함 위치 점수가 어떻게 달라지는지 본다. 새 기준표는 이미 정상 학습
849장과 정상 보정 56장으로 완료됐다. 이번 명령은 **다시 학습하지 않는다**.

두 결과 모두 이미 본 VisA 시험 데이터를 재사용한다. 새 v2는 정상 데이터 출처,
정상 학습 사진 수, 후보 패치 밀도를 함께 바꿨다. 따라서 개선이나 악화를 한 원인의 효과로 단정하거나,
확인적 일반화 성능·자동 PASS 근거로 사용하지 않는다.

## 비교 기준과 오류 방지

- 기존 기준: 완료된 `visa-pcb-external-evaluation-v2.zip`의 PCB3 상위 1% 점수·고정 임곗값
  `55.26094393613862`·같은 사진별 판정. 이전 모델을 재추론하지 않는다.
- 새 기준: `visa-pcb-normal-bank-pcb3-v2.zip`이 가리키는 연구실 checkpoint를 로드하고,
  정상 56장으로 이미 정해 둔 상위 1% 이미지 임곗값 `26.111570131487962`와 픽셀 임곗값
  `23.8155460357666`을 그대로 사용한다. 시험 정답으로 임곗값을 바꾸지 않는다.
- 공식 PCB3 test 정상 101장·불량 100장 전부. 지난 평가와 동일한 원본 경로·라벨·
  마스크 SHA-256과 비교한다. VisA 마스크는 `0이 아닌 값 → 255 → nearest 256×256`으로
  처리한다. 사진/마스크가 이전 감사 기록과 다르면 중단하며 표본을 제외하지 않는다.
- 코드가 시작 전에 원본 tar, 분할 계획, v2 fit ZIP, checkpoint, 이전 평가 ZIP과
  설정·내부 파일 체크섬을 확인한다. 시험 사진별 점수/맵/마스크는 연구실 로컬에
  원자적으로 저장해 중단 후 검증된 것만 재사용한다. 다른 설정의 캐시는 덮어쓰지 않는다.
- 비교는 같은 불량 사진에서 기존 미탐→새 검출, 기존 검출→새 미탐을 각각 센다.
  정상 오검출 변화, AUROC/AP, 픽셀 AUROC/AUPRO도 기록한다. 짝지은 bootstrap 구간은
  기술적 불확실성 참고값이며 이미 본 test의 확인적 유의성 근거가 아니다.

## 연구실 Windows에서 할 일

기존 v2 fit의 `model.pt`와 두 handoff ZIP, `Downloads\VisA_20220922.tar`를
그대로 보존한다. 다른 GPU 작업은 종료하고 다음을 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File ".\scripts\run_visa_pcb_normal_bank_evaluation.ps1" `
    -RunExploratory
```

완료 시 `artifacts\handoff\visa-pcb-normal-bank-pcb3-evaluation-v1.zip`을 이 작업에
첨부한다. ZIP에는 JSON 보고서·사진별 점수/판정·한국어 요약만 있고 원본 이미지,
마스크, 모델, anomaly map은 없다. 로컬 `artifacts\visa-pcb-normal-bank-pcb3-evaluation-v1\samples`
에는 재개용 맵이 남으므로 지우지 않는다. 실패 로그가 나오면 ZIP을 만들지 않거나
기존 결과를 덮어쓰지 않는다. 출력과 ZIP을 받은 뒤에만 결과를 검증·노션에 기록한다.

## 다음 판단

PCB3 한 품목에서 미탐이 줄더라도 정상 오검출 증가·위치 지표·시간 비용을 함께 본다.
PCB1·2·4로의 확대나 모델 교체는 이번 결과만으로 자동 결정하지 않는다. 진짜
확인적 결론에는 새로운 미관측·중복 통제 holdout과 사전 고정 기준이 필요하다.
