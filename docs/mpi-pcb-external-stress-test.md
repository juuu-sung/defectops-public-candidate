# MPI-PCB: 새 조립 기판 외부 시험 — 데이터 등록·정상 품질 게이트

상태: **v4 정상 보정 사진 169장 × 12영역 사전 점검 통과;
두 기존 모델의 checkpoint 신원·사진 점수·정상 임곗값 규칙을 코드로 잠금;
연구실 정상 전용 GPU 보정 결과·두 임곗값 재검증 완료;
외부 시험 실행 코드 준비, 시험 사진/마스크/모델 점수 미관찰**
(2026-09-26).

## 왜 이 자료를 보나

VisA PCB3 정상 기준표 v2는 이미 본 PCB3 시험에서 불량 100장 중 9장을 놓쳤다.
그 사진을 더 보고 기준을 바꾸면 새 데이터에서 잘 작동한다는 증거가 되지 않는다.
따라서 기존 VisA·Real-IAD·MVTec 사진과 출처가 다른 조립 기판을 찾았다.
MPI-PCB는 가스 펌프의 **한 기판**을 여러 조명·각도에서 촬영한 정상 사진 1,742장과
연구자가 변형을 추가해 촬영한 사진 55장, 위치 마스크를 공개한다.
출처: [데이터 원본](https://zenodo.org/records/8213098),
[논문](https://pmc.ncbi.nlm.nih.gov/articles/PMC9921794/),
[저자 저장소](https://github.com/Diulhio/pcb_anomaly/tree/main/dataset).

이것은 **다른 물체·다른 결함을 대상으로 한 외부 환경 스트레스 시험**이다.
VisA PCB3의 실제 생산 불량 미탐률이 줄었다는 검증이 아니다. 변형 사진은
제조 공정에서 자연 발생한 결함이 아니라, 원 저자가 부품 추가 등의 변형을
수동으로 만든 것이다. 따라서 현장 자동 합격이나 생산 투입 근거로 쓰지 않는다.

## 후보 선택과 권리 경계

- PCB-Bank PCB1–4는 이미 본 VisA, PCB5는 이미 본 Real-IAD의 재포장이므로
  새 시험 데이터에서 제외한다. VISION 기반 PCB6–7은 별도 접근 조건이 있다.
- DeepPCB는 새 출처지만 조립 기판이 아니라 **미조립 도체 패턴**의 쌍 사진이고
  인공 결함을 증강했다. 현 문제와 더 먼 영역이라 우선순위를 낮춘다.
- MPI-PCB 저자 저장소의 `License`는 `TODO`, Zenodo 기록의 `Rights/License`에는
  명시적 허락 문구가 보이지 않는다. 논문의 CC BY는 **논문**의 조건이지
  데이터 파일에 자동 적용되는 조건으로 해석하지 않는다. 연구실 내부 평가 외
  이미지/마스크 재배포, Git·Notion 업로드, 제품 사용은 하지 않는다. 범위를 넓히려면
  원 저자의 데이터 이용 조건을 별도로 확인한다.

## 이번에 고정하는 등록 단계

- Zenodo record `8213098`, version 1.0의 `extracted.zip` **2.8 GB**만 후보로 한다.
  원 저자가 4096×2816 전체 사진을 겹치는 1024×1024 영역으로 나눠 제공한
  자료다. 저자 논문은 전체 사진을 한 번에 줄이면 작은 변형이 사라질 수 있어
  영역별 처리했다고 설명한다. 우리는 전체 사진을 256으로 줄여 점수화하지 않는다.
- 원본 파일의 게시 MD5 `501d11d78a58ed0f26bc4d0baa34170e`와 실제 파일
  MD5를 비교하고 SHA-256을 별도로 기록한다. MD5는 게시 파일과의 동일성 확인용이다.
- ZIP 중앙 디렉터리의 경로·파일명 메타데이터만 읽어 12영역,
  영역별 `test/good` 55장·`test/defect` 55장·`ground_truth/defect` 55장과
  이름 일치를 점검한다. **시험 이미지/마스크 내용은 추출·디코딩하지 않는다.**
  ZIP 메타데이터만으로 압축 내부의 CRC 무결성, 이미지 손상이나 정답 품질은
  확인되지 않는다.
- 폴더 수, 이름, 원본 해시가 다르면 중단하고 계획을 수정하지 않은 채 원인을
  먼저 기록한다. 기존 데이터·모델·임곗값은 변경하지 않는다.

## 사진 개봉 전 잠글 후속 평가의 방향

등록이 통과해도 곧바로 모델을 실행하지 않는다. 사용자가 첫 MPI-PCB 시험은
**기존 두 모델만** 비교하기로 선택했다. 전체 패치 후보 모델을 추가 학습하거나
이번 시험에 넣지 않는다. 먼저 두 **동결 모델**
(기존 Real-IAD PCB 기준표와 VisA PCB3 v2 기준표)의 checkpoint 신원, 12영역
점수의 원본 사진 단위 결합법, 정상 보정 사진 선정·임곗값, 입력 전처리와
마스크 해석, 실패 처리, 1차 지표를 코드와 설정으로 고정한다.
같은 원본 사진의 12영역을 서로 독립 표본으로 세지 않으며, 정상/변형
**원본 사진 각 55장**을 평가 분모로 한다. 특히 각 모델 점수의 스케일이
다르므로 VisA PCB3 임곗값을 MPI-PCB에 그대로 적용한 수치를 공정한
동일 오검출률 비교로 주장하지 않는다. 원본의 train 정상 사진만으로
새 기판 보정을 한다면 이는 zero-shot이 아닌 **정상 데이터 적응**으로 명시한다.
시험 110장과 마스크는 이 규칙이 고정된 뒤 한 번만 연다. 손상·누락·중복이
드러나면 표본을 임의 제외하지 않고 감사 결과와 중단 여부를 먼저 보고한다.

## 연구실 Windows에서 할 일 — 등록만

이미 있는 `C:\개인 프로젝트\defectops\.venv`를 그대로 사용한다. Zenodo 페이지에서
`extracted.zip`을 **연구실 컴퓨터의 Downloads**에 직접 저장한다. Mac의 Downloads와
연구실 컴퓨터의 Downloads는 서로 다르다. 2.8 GB 다운로드 및 여유 공간을 확인한다.
압축을 풀거나 시험 사진을 열 필요는 없다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\audit_mpi_pcb_archive.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --output ".\artifacts\data-audit\mpi-pcb-registration-v1.json"
```

`status: registered_metadata_only`가 나오면 **JSON 파일과 출력된 SHA-256만** 이 작업에
첨부한다. 2.8 GB ZIP, 사진, 마스크는 첨부하거나 Git에 올리지 않는다.
오류가 나면 원본을 지우거나 이름을 바꾸지 말고 메시지를 그대로 공유한다.
이는 데이터 준비 결과일 뿐 모델 성능 결과가 아니다.

## 등록 출력 확인과 다음 정상 사전 점검

연구실 출력에 따르면 공식 파일 MD5 `501d11d78a58ed0f26bc4d0baa34170e`가
일치했고, SHA-256은
`0fa9b20fae64142a418487a1d898ee2ee824eee0bd564896a245f4d991112607`이다.
12개 영역 모두 `train/good` 1,687개, `test/good` 55개, `test/defect` 55개,
`ground_truth/defect` 55개로 기록됐으며 오류 목록은 비었다. 시험 사진/마스크
디코딩과 모델 점수 계산은 각각 0건이다. 이는 사용자가 붙여 준 등록 출력의
검토이며, Mac에는 2.8 GB 원본이나 JSON 파일이 없으므로 원본 SHA-256을
**독립 재계산한 결과는 아니다**.

다음 사전 점검은 원본 ZIP의 MD5·SHA-256을 연구실에서 재계산하고 등록 기록과
대조한다. `train/good`의 같은 사진 ID가 12영역에 모두 있는지 확인한 뒤,
사진 ID 문자열의 `SHA256("mpi-pcb-calibration-v1\0" + ID)` 순서에서
169개를 고른다. 이는 1,687개의 약 10%이며 이미지 내용·정답을 보지 않는
결정적 선택이다. **169개 원본 사진 × 12영역 = 정상 이미지 2,028장만**
압축을 풀어 RGB 디코딩하고, 픽셀 중복·크기·체크섬을 기록한다. 시험 사진과
마스크의 내용은 열지 않는다. 모델 추론도 하지 않는다. 원본 ZIP, 등록 JSON,
기존 모델과 임곗값은 덮어쓰지 않는다.

### v1 중단 원인과 v2 판정 규칙

첫 실행은 `grid1-1_s1024`의 사진 ID `0722`에서 **한 영역의 디코딩 픽셀이
이전 정상 이미지와 같다는 이유만으로** 중단했다. v1은 영역 한 장의 중복을
원본 사진 전체의 중복으로 취급했는데, 실제로 12영역 중 다른 영역이 같은지는
검사하지 않았다. 따라서 이것은 데이터 오류 또는 전체 원본 사진 중복을
입증하지 않는다. 중단 전에는 선택된 정상 훈련 사진만 읽었으며, 시험 사진,
마스크, 모델 점수는 열거나 계산하지 않았다.

v2는 **같은 169개 사진 ID**와 동일한 12영역을 디코딩한다. 단일 영역에서
동일한 타일은 영역별 요약에 기록하고 통과를 막지 않는다. 12영역의 이미지
크기와 RGB 픽셀 해시가 사진 ID 두 개에서 **모두 같을 때만** 사진의
12영역 표현이 완전히 동일하다고 판단해 `needs_review`로 중단한다. 이는
원본 카메라 파일 자체가 동일하다는 주장과는 다르다. 시험 개봉 전에 이
규칙을 고정하며, 사진 ID를 교체하거나 임곗값을 조정하지 않는다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\preflight_mpi_pcb_normals.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --registration ".\artifacts\data-audit\mpi-pcb-registration-v1.json" `
    --output ".\artifacts\data-audit\mpi-pcb-normal-preflight-v2.json"
```

`normal_preflight_passed_test_sealed`가 나오면 새 v2 JSON **파일만** 이 작업에
첨부한다. `normal_preflight_needs_review_test_sealed`가 나오면 12영역 전체가
동일한 사진 ID 그룹이 있는 것이므로 v2 JSON을 첨부하고 시험을 계속하지
않는다. 다른 오류가 나오면 ZIP·등록 결과·기존 모델을 삭제하지 말고 메시지를
보내 준다. 성공해도 아직 시험 110장의 디코딩·평가나 생산 성능 입증은 아니다.

### v2 중단 후: 선정된 정상 사진만 무결성 감사

연구실의 v2 재실행은 정상 타일
`extracted/grid2-1_s1024/train/good/0158.png`에서 중단됐다. 이 파일은 ZIP
CRC와 PNG 시그니처는 통과했지만 Pillow 11.3.0의 `verify()`에서 `IDAT`
CRC 불일치, `load()`에서 픽셀 디코딩 오류가 확인됐다. 해당 멤버 SHA-256은
`97459f5c9721adedc6739e63a8ac6a1de6fae61d72a084a7c2df94b14c68cd8e`다.
원본 ZIP은 공식 게시 MD5와 잠근 SHA-256을 통과했으므로, 동일 ZIP의 재다운로드만
권하지 않는다. 아직 **다른 정상 타일의 오류 범위는 모른다**.

아래 감사는 기존 해시 선정 규칙을 그대로 사용해 169개 사진 ID × 12영역 =
2,028개 **훈련 정상 타일만** 끝까지 조사한다. 각 멤버의 ZIP CRC, PNG chunk
CRC 및 픽셀 디코딩을 확인하고 실패 목록·해시만 JSON에 기록한다. 시험 이미지와
마스크 내용은 열지 않으며, 모델 점수·임곗값을 계산하지 않는다. 이 감사의
`completed`는 **데이터 통과를 뜻하지 않는다**. 오류가 있어도 원본 ZIP·이미지·
기존 v2 산출물은 수정·삭제·대체하지 않는다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\audit_mpi_pcb_selected_normals.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --registration ".\artifacts\data-audit\mpi-pcb-registration-v1.json" `
    --output ".\artifacts\data-audit\mpi-pcb-selected-normal-decode-audit-v1.json"
```

실행 뒤 JSON **파일만** 이 작업에 첨부한다. 원본 ZIP이나 이미지 자체는 첨부하지
않는다. 결과를 확인하기 전에는 v2 사전 점검을 다시 시도하거나 시험 사진을 열지
않는다. 발견한 파일을 조용히 건너뛰거나 Pillow의 손상 허용 옵션을 켜지 않는다.

### 선정 정상 감사 결과와 훈련 정상 전체 감사

연구실 출력은 선정된 2,028개 정상 타일을 모두 시도해 2,026개 통과, 2개 실패로
보고했다. 오류는 서로 다른 원본 사진 ID `0158`(`grid2-1_s1024`)과
`1303`(`grid3-1_s1024`)의 타일이다. 둘 다 PNG `IDAT` CRC와 픽셀 디코딩에
실패했다. 보고된 선정 사진 목록 SHA-256
`5410ae2851c3621f46a2e6d1a90b294ffaa41ed4c84129caad95da66333bdb41`은
잠근 코드에서 Mac으로 독립 계산한 값과 일치한다. 단, Mac에는 실제 2.8 GB ZIP이
없어 이미지 바이트를 독립 검증하지 못했다. **2개는 선정된 타일에서의 오류 수**이며
나머지 훈련 정상 전체의 오류 수로 해석하지 않는다.

시험을 열기 전에 `train/good` 전체 1,687개 사진 ID × 12영역 = 20,244개
타일을 읽기 전용으로 감사한다. 공식 ZIP 및 등록 기록의 해시를 다시 확인하고
각 타일의 ZIP CRC·PNG CRC·픽셀 디코딩을 검사한다. 오류가 하나라도 있는
사진 ID는 *새 후보 명단에서만* 제외한다. 원본 ZIP 파일을 수정하거나 삭제하지
않는다. **완전한 사진 ID가 169개 이상이면** 기존
`SHA256("mpi-pcb-calibration-v1\0" + photo_id)` 순서의 앞 169개를
*v3 정상 보정 후보*로 기록한다. 이는 정상 데이터 무결성만 사용한, 시험 개봉 전
버전 변경이다. 후보가 만들어져도 자동으로 승인하거나 모델 평가를 시작하지
않는다. 오류 범위와 보고서를 검토한 후 별도 v3 사전 점검을 통과해야 한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\audit_mpi_pcb_normal_pool.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --registration ".\artifacts\data-audit\mpi-pcb-registration-v1.json" `
    --output ".\artifacts\data-audit\mpi-pcb-normal-pool-integrity-v1.json"
```

20,244개 타일을 CPU로 검사하므로 앞선 2,028개 감사보다 오래 걸릴 수 있다.
GPU 학습은 하지 않는다. 완료 후 JSON **파일만** 첨부하고 원본 ZIP·사진·마스크는
첨부하지 않는다. `candidate_ready_for_review`는 후보 수가 충분하다는 뜻이지
사전 점검이나 시험 평가를 통과했다는 뜻은 아니다. 결과를 확인하기 전에는
기존 v2를 다시 실행하거나 시험 데이터를 열지 않는다.

### 훈련 정상 전체 감사 결과와 v3 정상 사전 점검

연구실 감사 보고서에는 1,687개 원본 사진 × 12영역 = 20,244개 정상 타일을
모두 시도해 20,228개 통과, 16개 실패로 기록됐다. 손상 타일은 서로 다른 사진
ID 16개에 하나씩 있으며, **12영역이 모두 온전한 원본 사진은 1,671개**다.
앞서 찾은 `0158`, `1303`도 이 16개에 포함된다. ZIP SHA-256, 등록 JSON
SHA-256, 16개 오류의 사진 ID·영역·경로, 완전한 ID 목록과 후보 명단 해시를
보고서에서 다시 계산·대조했다. 다만 이 검토는 사용자가 전달한 Windows
보고서를 대상으로 한 것으로, Mac에서 2.8 GB 원본 ZIP을 다시 디코딩한 것은 아니다.

v3는 *정상 데이터의 무결성만* 보고 16개 사진 ID를 후보에서 제외한 후 **기존
해시 순서**에서 169개를 다시 고른다. 원래 명단 중 손상된 `0158`, `1303`을
빼고 `0727`, `1257`을 추가한다. 후보 명단 SHA-256은
`af412abd25eac87c80ed97d83c7d9548ceb0fae1c7d513588377cf1590d5928c`다.
이는 **정상 보정 표본 수를 유지**하는 조치이지 성능 개선 또는 공정한 외부 시험
통과를 뜻하지 않는다. 손상 파일은 ZIP에서 삭제·수정하지 않는다.

다음 명령은 감사 보고서의 고정된 16개 손상 ID와 후보 명단 해시를 먼저 검증하고,
새 169개 원본 사진의 12영역 타일 2,028개만 ZIP에서 다시 읽는다. ZIP CRC,
PNG 내부 CRC, 픽셀 디코딩, 크기와 12영역 전체가 동일한 사진 ID 그룹을
점검한다. **시험 사진·마스크는 열지 않고 GPU 학습·모델 점수 계산도 하지 않는다.**
기존 `v2` 결과를 덮어쓰지 않도록 새 출력 파일을 사용한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\preflight_mpi_pcb_normals_v3.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --registration ".\artifacts\data-audit\mpi-pcb-registration-v1.json" `
    --pool-audit ".\artifacts\data-audit\mpi-pcb-normal-pool-integrity-v1.json" `
    --output ".\artifacts\data-audit\mpi-pcb-normal-preflight-v3.json"
```

출력이 `normal_preflight_v3_passed_test_sealed`이면 새 JSON **파일만** 작업에
첨부한다. `normal_preflight_v3_needs_review_test_sealed`이거나 오류가 나면
JSON 또는 오류 메시지를 공유하고 여기서 멈춘다. 어느 경우에도 아직 시험을
열지 않는다. 통과 후에도 두 기존 모델의 checkpoint 해시, 사진 단위 점수
결합법과 모델별 정상 임곗값을 실행 코드·설정으로 잠근 다음에만 외부 시험을
한 번 진행한다.

### v3 검토 결과와 중복 제거 v4 정상 사전 점검

연구실 v3 보고서는 정상 후보 169개 사진의 12영역, 총 2,028개 PNG를 모두
읽었지만 `normal_preflight_v3_needs_review_test_sealed`로 멈췄다. 사진 ID
`0879`와 `0722`는 **12영역 모두에서 디코딩 픽셀 해시뿐 아니라 PNG 멤버
바이트 해시까지 같다**. 다른 전체 12영역 중복 그룹은 보고되지 않았다. 받은
v3 JSON의 SHA-256은
`da5794ff706be9e91b87903626b0c82fedf63b5a41c7f229f9b9149dcaa98e3e`다.
Mac에서 2,028개 행의 경로·크기·명단 해시·중복 그룹을 재계산해 보고서와
일치함을 확인했지만, 원본 ZIP 바이트를 독립 재검증한 것은 아니다.
시험 이미지·마스크 디코딩과 모델 점수는 보고서상 모두 0건이다.

이 둘을 별개의 보정 사진 두 장으로 세지 않는다. **시험을 열기 전에** 기존
해시 순위 59위 `0879`를 유지하고 122위 `0722`를 새 명단에서만 제외한다.
같은 순서에서 아직 쓰지 않은 첫 번째 완전한 정상 사진 ID `1481`을 추가해
169장을 유지한다. 새 v4 명단 SHA-256은
`75768131852bfe1234f754c3b77054cce64e2a7c890a94db9724943b21faf11e`다.
이는 결과를 보고 임곗값이나 모델을 바꾸는 일이 아니라, 정상 보정 집합의
중복을 제거하는 **버전이 명시된 데이터 준비 단계**다. `1481`도 다른 후보와
같은지는 아직 모르므로 별도 사전 점검을 통과해야 한다. 원본 ZIP, 기존 v2·v3
JSON, 모델 checkpoint는 수정·삭제·덮어쓰지 않는다.

다음 명령은 연구실의 원본 ZIP과 등록·전체 정상 감사·v3 JSON의 잠근 SHA-256을
먼저 대조한다. v3의 12영역 동일 그룹을 다시 계산한 뒤, v4의 169개 정상
사진 × 12영역 = 2,028개 **훈련 정상 타일만** ZIP CRC·PNG CRC·픽셀 디코딩·
전체 중복 검사한다. GPU 학습, 모델 점수 계산, 시험 데이터 개봉은 하지 않는다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\preflight_mpi_pcb_normals_v4.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --registration ".\artifacts\data-audit\mpi-pcb-registration-v1.json" `
    --pool-audit ".\artifacts\data-audit\mpi-pcb-normal-pool-integrity-v1.json" `
    --v3-preflight ".\artifacts\data-audit\mpi-pcb-normal-preflight-v3.json" `
    --output ".\artifacts\data-audit\mpi-pcb-normal-preflight-v4.json"
```

`normal_preflight_v4_passed_test_sealed`가 나오면 새 v4 JSON **파일만** 작업에
첨부한다. `normal_preflight_v4_needs_review_test_sealed`나 다른 오류가 나오면
보고서 또는 오류 메시지를 공유하고 멈춘다. 성공하더라도 데이터 준비 게이트만
통과한 것이며, 두 기존 모델의 checkpoint와 평가 코드를 별도로 잠그기 전에는
시험 110장과 마스크를 열지 않는다. 자동 PASS도 허용하지 않는다.

사전 점검 보고서를 확인한 다음 두 checkpoint의 SHA-256과 평가 코드를 고정한다.
예정된 비교는 각 영역의 기존 상위 1% 이미지 점수를 구한 후 12영역의
**최댓값을 원본 사진 한 장의 점수**로 사용하는 것이다. 각 모델별 정상
보정 169장의 사진 단위 점수에서 95% `higher` 임곗값을 따로 정한다.
따라서 이후 결과는 55장 정상 시험과 55장 수동 변형 시험의 사진 단위
미탐·오검출·AUROC/AP 및 모델별 판정 역전으로 기록할 예정이다. 이 점수·임곗값
방식은 시험을 열기 전에 최종 실행 설정과 checkpoint 해시까지 잠가야 하며,
시험 결과를 보고 후보를 추가하거나 규칙을 바꾸지 않는다.

### v4 통과 확인과 두 모델 신원·보정 규칙 잠금

사용자가 첨부한 v4 JSON 파일의 SHA-256은
`65605ddcb5d89b2e4f5bcbcc8d2002e2996455d6235e37e35d1faf767dd5c93d`다.
169개 사진 ID의 SHA-256은
`75768131852bfe1234f754c3b77054cce64e2a7c890a94db9724943b21faf11e`이고,
169 × 12 = 2,028개의 256×256 정상 타일이 기록됐다. 12영역 전체가 같은
사진 ID 그룹은 보고되지 않았다. Mac에서 첨부된 JSON의 행·사진 ID·중복 그룹을
재계산해 일치함을 확인했다. **Mac에서 원본 2.8 GB ZIP을 다시 디코딩한 것은 아니다.**
v4 보고서상 시험 이미지/마스크 디코딩과 모델 점수 계산은 모두 0건이다.

이번에는 `configs/mpi-pcb-model-protocol-v1.toml`에 다음을 **시험 개봉 전에**
고정한다.

- 후보는 기존 Real-IAD PCB seed 42 layer1–layer2 checkpoint
  (`cd0e12d3d2daadfc1f30fde6e666002287af29cc610a870a6196899468c0de13`)와
  기존 VisA PCB3 v2 정상 뱅크 seed 42 checkpoint
  (`be8062541e2a14cb563284d34cfc9f60e3ca5e8cc1d75e0ccbd28d1287785e85`)
  두 개뿐이다. 둘 다 이 단계에서 다시 학습하지 않는다.
- 입력 영역은 256×256으로 bilinear antialias/ ImageNet 정규화한다. **영역별**
  이상 맵 값 중 상위 1%의 평균을 영역 점수로 사용하고, 같은 원본 사진의
  **12개 영역 점수 중 최댓값**을 사진 한 장의 점수로 사용한다.
- 두 모델 각각 v4 정상 사진 **169장의 사진 단위 점수**로 95% `higher` 분위
  임곗값을 별도로 정한다. 점수 ≥ 해당 모델 임곗값이면 검토 대상으로 표시한다.
  이는 정상 데이터 적응이며, 기존 VisA 임곗값을 MPI에 그대로 옮기는 시험이
  아니다. 아직 임곗값 숫자는 계산되지 않았다.
- 이 단계는 시험 사진/마스크를 읽지 않으며 생산 자동 PASS를 허용하지 않는다.
  후속 시험은 손상·중복 감사와 원본 사진 단위 55 정상/55 수동 변형의
  평가 규칙을 추가로 잠근 뒤에만 실행한다.

연구실 컴퓨터에서는 기존 두 checkpoint와 공식 ZIP의 파일 해시, v4 JSON 신원만
대조한다. 모델을 GPU에 올리거나 정상 사진에 대한 추론도 아직 하지 않는다.
출력 파일이 이미 있으면 덮어쓰지 않고 중단한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\preflight_mpi_pcb_models.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --preflight ".\artifacts\data-audit\mpi-pcb-normal-preflight-v4.json" `
    --config ".\configs\mpi-pcb-model-protocol-v1.toml" `
    --output ".\artifacts\data-audit\mpi-pcb-model-preflight-v1.json"
```

`mpi_pcb_model_protocol_locked_test_sealed`가 나오면 생성된 **JSON 파일만** 이
작업에 첨부한다. `normal_scores_computed`, `test_images_decoded`,
`test_masks_decoded`, `test_scores_computed`가 모두 0이어야 한다. 오류가 나면
원본 ZIP·checkpoint·v4 JSON을 옮기거나 고치지 말고 메시지를 공유한다.
이 결과를 확인한 다음 정상 169장만 모델로 점수화해 두 개의 임곗값을 만들고,
그때까지 시험 사진은 계속 봉인한다.

### 모델 신원 확인 결과와 정상 전용 GPU 보정

연구실의 `mpi-pcb-model-preflight-v1.json` SHA-256은
`b7d328daa3b2ca0e9c6fe50bffacae4d3bfddbf0947e39f1f8c27492b6bab2bf`다.
보고된 원본 ZIP·v4 명단·두 기존 checkpoint 해시는 잠근 값과 일치한다.
보고서의 정상 점수 계산, 시험 이미지·마스크 디코딩과 시험 점수 계산은 각각
0건이다. Mac에는 원본 ZIP·checkpoint가 없으므로 **실제 바이트의 해시를
Mac에서 독립 재계산한 것이 아니라 연구실 보고서를 코드·설정과 대조**했다.

다음 프로그램은 그 보고서와 파일 해시를 다시 확인한 뒤 **정상 훈련 사진
169장만** 두 기존 모델로 추론한다. 각 사진의 12개 영역은 모델마다 1번씩
점수화하므로 모델별 2,028개 영역, 두 모델 합계 4,056번의 정상 영역 추론이다.
원본 사진의 점수는 12영역 점수 중 최댓값이며, 각 모델의 169개 사진 점수로
95% `higher` 임곗값을 계산한다. 결과 JSON에는 12개 영역별 점수와 사진별
점수를 모두 보존해 임곗값을 독립 재계산할 수 있게 한다. 기존 모델 checkpoint,
원본 ZIP·v4 JSON·신원 확인 JSON은 수정하지 않는다. 결과 파일이 이미 있으면
덮어쓰지 않고 중단한다.

실행 중 선택된 `train/good` 타일의 ZIP/PNG CRC, 멤버 바이트 해시, RGB 픽셀
해시가 v4 기록과 다르면 **점수화를 중단**한다. ZIP 전체 해시와 중앙 디렉터리의
파일명은 확인하지만, `test` 멤버의 사진·마스크 내용을 읽거나 추론하지 않는다.
이는 새 기판의 정상 사진을 이용해 임곗값을 적응시키는 **정상 전용 보정**이지
모델 재학습이나 외부 성능 평가가 아니다.

연구실 Windows에서는 기존 `.venv`와 원본 ZIP을 그대로 사용한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
& ".\.venv\Scripts\python.exe" ".\scripts\calibrate_mpi_pcb_normals.py" `
    --archive "$env:USERPROFILE\Downloads\extracted.zip" `
    --v4-preflight ".\artifacts\data-audit\mpi-pcb-normal-preflight-v4.json" `
    --model-preflight ".\artifacts\data-audit\mpi-pcb-model-preflight-v1.json" `
    --config ".\configs\mpi-pcb-model-protocol-v1.toml" `
    --output ".\artifacts\calibrations\mpi-pcb-normal-calibration-v1.json"
```

출력이 `mpi_pcb_normal_calibrated_test_sealed`이면 생성된 **JSON 파일만** 이
작업에 첨부한다. 모델별 정상 사진 수 169, 영역 점수 수 2,028,
`test_images_decoded = 0`, `test_masks_decoded = 0`, `test_scores_computed = 0`을
함께 확인한다. 이 단계는 GPU를 사용하며 사진 4,056영역을 추론하므로 신원
검사보다 오래 걸린다. 오류가 나면 원본·checkpoint·기존 결과를 지우거나
명령을 임의 변경하지 말고 메시지를 공유한다. **보정 결과를 검증하고 시험
무결성·평가 규칙을 별도로 잠그기 전에는 시험 110장을 열지 않는다.**

### 정상 보정 확인 완료 — 다음은 동결 모델 외부 시험

수령한 `mpi-pcb-normal-calibration-v1.json`의 SHA-256은
`6fb949ae182453fc703770dba30342a644af4641eeb69ec57c89d10250019cbd`다.
모델별 정상 사진 169개 × 12영역 = 2,028개 점수에서 사진 최댓값과
95% `higher` 임곗값을 다시 계산해 일치함을 확인했다.

| 동결 모델 | 확정 임곗값 | 정상 보정에서 임곗값 이상 |
| --- | ---: | ---: |
| Real-IAD PCB | 67.42196179599297 | 9/169 |
| VisA PCB3 정상 뱅크 v2 | 66.18967593588481 | 9/169 |

이는 보정 결과이며 시험 오검출률이나 미탐률이 아니다. Mac에서는 첨부 JSON을
검산했고 실제 PNG·checkpoint·GPU 추론은 재현하지 않았다. 보고된 시험 이미지,
마스크 디코딩과 시험 점수 계산은 모두 0이다.

다음 [외부 시험 실행 안내](mpi-pcb-external-evaluation.md)에 보정 JSON 해시,
정밀도 그대로의 두 임곗값, 정상 재현 허용 오차, 시험 무결성·중복 중단 기준,
사진 단위 지표와 연구실 실행 명령을 고정했다. 기존 두 모델을 다시 학습하거나
시험 결과를 보고 임곗값을 고치지 않는다.
