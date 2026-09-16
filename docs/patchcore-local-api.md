# PatchCore 상주 검사 API — 로컬 smoke test

연구실 Windows에서 동결된 PatchCore fit/evaluation bundle을 서비스 시작 시 한 번 검증·로드하고,
요청마다 같은 모델을 재사용한다. 학습·임곗값 조정·새 성능 평가는 하지 않는다. 이 단계는
**127.0.0.1 로컬 접속 전용**이며 인증·외부 네트워크 공개·SQLite·작업자 UI는 아직 없다.

## 계약과 안전 범위

- 시작 시 기존 fit/evaluation의 commit·manifest·checkpoint·calibration 계약을 검증한다.
  오류가 있거나 PaDiM bundle이면 서버가 시작되지 않는다.
- worker 1개가 PatchCore 1개를 GPU에 유지한다. 요청은 잠금으로 한 번에 하나씩 처리한다.
- `POST /v1/inspections`: `Content-Type: image/png`인 1024×1024 RGB PNG 원본 바이트를
  최대 50 MiB까지 받는다. 모델 파일 경로·임곗값은 요청에서 받지 않는다.
- 기존 단일 이미지 검사와 같은 JSON·heatmap·overlay·원시 map을 각 inspection ID 아래에
  보존한다. 동일한 고정 임곗값과 영역 규칙을 사용한다.
- 잘못된 입력은 `INSPECTION_FAILED` 기록과 HTTP 413/415/422를 반환한다. 내부 검사 실패는
  상세 기록을 남기되 응답은 HTTP 500과 실패 상태만 보낸다. **모든 경우 자동 PASS 금지**다.
- `PASS`는 탐색적 모델 출력일 뿐 실제 합격 또는 안전 인증이 아니다. 현재 PatchCore도 공식
  test 불량 40장 중 3장을 놓쳤다.
- inspection ID는 서버가 생성한다. 파일 조회는 허용된 이름과 저장된 SHA-256 일치 여부를
  확인한다. 인증이 없으므로 포트를 다른 장비에 공개하거나 방화벽을 열지 않는다.

## 연구실 Windows 설치·시작

기존 `.venv`와 모델 산출물을 유지하고 코드만 업데이트한다. 첫 PowerShell 창:

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only
git status --short

$projectPath = "C:\개인 프로젝트\defectops"
$defectPython = "$projectPath\.venv\Scripts\python.exe"

& $defectPython -m pip install -r ".\requirements\api.txt"
& $defectPython -m unittest discover -s tests -v

& $defectPython scripts\serve_patchcore.py `
    --fit-run ".\artifacts\runs\patchcore-transistor-seed42" `
    --evaluation ".\artifacts\evaluations\patchcore-transistor-seed42-eval-v1" `
    --port 8000
```

서버가 시작될 때 검증·모델 로드 때문에 약 수십 초 걸릴 수 있다. `Uvicorn running on
http://127.0.0.1:8000`가 보이면 그 창을 그대로 둔다. `--reload`나 여러 worker는 사용하지
않는다. 종료는 첫 창에서 `Ctrl+C`다.

두 번째 PowerShell 창에서 로컬 상태와 이미지 한 장을 확인한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"

$health = Invoke-RestMethod "http://127.0.0.1:8000/healthz"
$health | ConvertTo-Json -Depth 5

$inspection = Invoke-RestMethod `
    -Method Post `
    -Uri "http://127.0.0.1:8000/v1/inspections" `
    -ContentType "image/png" `
    -InFile ".\data\transistor\test\bent_lead\000.png"

[ordered]@{
    inspection_id = $inspection.inspection_id
    status = $inspection.status
    model = $inspection.model
    model_reused = $inspection.model_reused
    decision = $inspection.decision
    raw_score = $inspection.raw_score
    image_threshold = $inspection.image_threshold
    setup_and_model_load_ms = $inspection.latency_ms.setup_and_model_load
    gpu_inference_ms = $inspection.latency_ms.gpu_inference_and_map_transfer
    total_request_work_ms = $inspection.latency_ms.total
    automatic_pass_allowed = $inspection.automatic_pass_allowed
} | ConvertTo-Json -Depth 5

Start-Process ".\artifacts\inspections\$($inspection.inspection_id)\overlay.png"
```

예상 결과: `healthz.status=ready`, `model=patchcore`, `model_reused=true`, 요청 결과의
`setup_and_model_load_ms=0`, `automatic_pass_allowed=false`. 동일 이미지는 기존 검사에서
`REVIEW_REQUIRED`, 점수 약 52.6997, 임곗값 40.7913이었다. 부동소수점 점수 차이는
검증된 환경 허용 오차 안에서 비교하며, 이 한 장으로 새 성능 지표를 만들지 않는다.

처음 요청의 GPU 시간이 warm benchmark보다 길 수 있다. JSON의 `total_request_work_ms`는
HTTP 업로드 시간은 제외하지만 PNG decode·추론·overlay/NPZ 작성·checksum은 포함한다.
앞서 측정한 약 80 ms warm sample 시간은 산출물 저장이 없는 벤치마크라 직접 비교하지 않는다.

## 상태·산출물·RAM 확인

```powershell
$inspectionId = $inspection.inspection_id
Invoke-RestMethod "http://127.0.0.1:8000/v1/inspections/$inspectionId" |
    ConvertTo-Json -Depth 12

$ram = Get-Process -Id $health.process_id
[ordered]@{
    process_id = $ram.Id
    ram_working_set_GiB = [math]::Round($ram.WorkingSet64 / 1GB, 3)
    gpu_allocated_GiB = [math]::Round($health.gpu_memory.allocated_bytes / 1GB, 3)
    gpu_reserved_GiB = [math]::Round($health.gpu_memory.reserved_bytes / 1GB, 3)
} | ConvertTo-Json

nvidia-smi
```

`healthz`의 GPU 값은 PyTorch 추적 메모리다. `nvidia-smi`의 프로세스 전체 점유량과 다를
수 있다. 앞선 warm benchmark의 PyTorch 피크는 allocated 0.35 GiB, reserved 0.39 GiB였다.
RAM working set은 아직 미측정이며, 실제 운영 예산은 여러 요청 후 다시 측정한다.

## 현재 미구현 범위

- 원격 접속 인증·TLS·네트워크 공개
- SQLite 검사/검토 이력 및 작업자 승인 상태 전이
- 장기 보관·삭제 정책, 다중 요청 부하 시험
- 새 holdout에서 ROI 마스크나 복합 REVIEW 규칙 검증

따라서 이 API는 같은 연구실 컴퓨터에서 GPU 상주 검사 흐름을 확인하는 기술 단계다. 맥북 UI가
접속하려면 먼저 인증·전송 경로와 운영 권한을 설계해야 한다.
