# 검사 API → 검토 목록 자동 등록

2026-11-14. 같은 컴퓨터에서 검사 API와 검토 화면이 같은 SQLite DB를 사용하는 흐름을 구현했다. API는 기존 모델·임곗값으로 검사 산출물을 저장한 뒤, 해시와 판정 계약을 확인해 검토 목록에 등록한다. API 응답의 등록 상태와 GPU 추론 성공 여부를 구분한다.

## 달라진 흐름

```text
이미지 요청 → 검사 결과·지도 저장 → 무결성 검사 → 검토 DB 자동 등록
                                      ↓ 등록 실패
                           원본 결과 보존 → 같은 ID로 재등록
```

- `serve_patchcore.py`의 기본 동작은 자동 등록이다. 기존 파일 전용 방식은 `--no-review-registration`으로 선택한다.
- 기본 DB는 `artifacts/review/reviews.sqlite3`다. `--review-db`로 지정했다면 UI에도 같은 경로를 전달한다.
- `REVIEW_REQUIRED`뿐 아니라 모델 기준 미만 `PASS`, 입력 오류와 추론 실패 기록도 등록한다. 실패 기록에는 사람의 시각 판정을 입력할 수 없다.
- 사람 의견·SOP·승인·티켓을 자동 작성하지 않는다. 자동 등록은 제품 합격이 아니다.
- UI를 이미 열었다면 브라우저를 새로고침해 목록을 다시 읽는다. 실시간 푸시 갱신은 아니다.
- Windows와 Mac 사이의 자동 전송이나 공유 DB는 이번 범위가 아니다. 같은 장비에서 먼저 확인한다.

## 연구실 실행 · PowerShell

기존 서버가 있다면 해당 창에서 정상 종료한 뒤 실행한다. 기존 모델·평가 폴더를 사용하며 재학습은 하지 않는다.

**첫 번째 창: 코드·화면 의존성 준비와 API 실행**

```powershell
Set-Location 'C:\개인 프로젝트\defectops'
git pull --ff-only
& '.\.venv\Scripts\python.exe' -m pip install -r requirements/api.txt -r requirements/ui.txt
& '.\.venv\Scripts\python.exe' scripts/serve_patchcore.py --fit-run artifacts/runs/patchcore-transistor-seed42 --evaluation artifacts/evaluations/patchcore-transistor-seed42-eval-v1 --review-db artifacts/review/reviews.sqlite3 --port 8000
```

DB를 준비하고 기존 UUID 검사 폴더를 점검한 뒤 GPU 모델을 로드한다. 모델이 준비되면 `/healthz`가 `ready`를 반환한다. 다른 용도의 DB이거나 DB 초기화가 불가능하면 시작을 중단한다.

**두 번째 창: 이미지 한 장 검사하고 검토 화면 실행**

```powershell
Set-Location 'C:\개인 프로젝트\defectops'
$health = Invoke-RestMethod 'http://127.0.0.1:8000/healthz'
$health.review_registration | ConvertTo-Json -Depth 5

$pngBytes = [System.IO.File]::ReadAllBytes((Resolve-Path '.\data\transistor\test\bent_lead\000.png'))
$inspection = Invoke-RestMethod -Method Post -Uri 'http://127.0.0.1:8000/v1/inspections' -ContentType 'image/png' -Body $pngBytes
$inspection | Select-Object inspection_id,status,decision,automatic_pass_allowed
$inspection.review_registration | ConvertTo-Json

& '.\.venv\Scripts\python.exe' scripts/serve_review_ui.py --db artifacts/review/reviews.sqlite3 --port 8501
```

같은 컴퓨터의 브라우저에서 `http://127.0.0.1:8501`을 연다. API 응답의 검사 ID가 목록에 있고 이미지가 표시되는지 확인한다. 별도 `import` 명령은 필요 없다. 이 동작이 실제 CUDA 모델과 함께 통과했는지는 연구실 실행 결과로 확인해야 한다.

## 등록 상태와 재시도

| `review_registration.status` | 의미 | 처리 |
| --- | --- | --- |
| `REGISTERED` | 동일 결과가 검토 DB에 등록됨 | UI 새로고침 후 확인 |
| `PENDING` | 아직 미등록 또는 DB 접근 실패 | DB 상태 확인 후 같은 검사 ID 재등록 |
| `BLOCKED` | 파일 누락·해시 불일치·ID 충돌 등 | 원본 보존, 원인 확인; 임의 삭제·덮어쓰기 금지 |

검사는 성공했지만 DB 등록이 실패해도 원래 HTTP 201과 검사 결과를 반환하고, 등록 상태를 `PENDING`/`BLOCKED`로 명시한다. 입력·추론 실패의 4xx/500 상태도 유지한다. 파일 생성 자체가 중단돼 완료 기록이 없는 폴더는 복구 점검에서 `BLOCKED`로 남는다.

**이미지를 다시 POST하지 말고 등록만 다시 시도한다.**

```powershell
$inspectionId = $inspection.inspection_id
Invoke-RestMethod -Method Post -Uri "http://127.0.0.1:8000/v1/inspections/$inspectionId/review-registration"
```

등록 API는 성공 200, DB 접근 실패 503, 파일·ID 충돌 409를 반환한다. 모델 추론은 실행하지 않는다. 동일 결과를 여러 번 재등록해도 검사 행과 사람 검토 이력을 복제하거나 초기화하지 않는다.

서버를 열지 않고 재등록하려면:

```powershell
& '.\.venv\Scripts\python.exe' scripts/register_api_inspection.py --inspection-id $inspectionId --review-db artifacts/review/reviews.sqlite3
```

`GET /v1/inspections/{id}`는 현재 등록 상태를 조회하며 DB에 새로 등록하지 않는다. 등록 상태는 응답에만 추가되므로 저장된 `inspection.json`과 기존 해시는 바뀌지 않는다. `elapsed_ms`는 해당 POST 요청에서 등록을 시도한 시간이며, 기존 검사 `latency_ms.total`에 포함되지 않는다.

## 재시작과 한계

- 과거 MVTec PatchCore 평가 JSON에는 `ground_truth_threshold`가 없을 수 있다. 당시 uint8 정답 마스크의 `> 0`은 현재 `>= 1`과 같으므로, 기존 탐색/사전등록 MVTec 프로토콜에서 이 항목만 누락된 경우 비교 시 기본값 1을 적용한다. 원본 JSON·TOML·해시와 모델 경고선은 수정하지 않는다. 명시된 값의 불일치, 다른 필드 차이, Real-IAD에는 이 예외를 적용하지 않는다.

- API 시작 시 출력 루트의 정규 UUID 이름을 가진 기존 항목을 순회해 재등록한다. `healthz.review_registration.startup_recovery`에 성공·대기·차단 건수를 표시한다. 이는 시작 시점 집계이며 실시간 대기 건수가 아니다.
- 완성된 검사 폴더가 복구의 근거다. 별도 백그라운드 자동 재시도 작업은 없다. DB가 다시 사용 가능해지면 같은 ID 재등록 또는 API 재시작으로 복구한다.
- 등록 도중 서버가 종료되어도 DB의 트랜잭션과 반복 등록 검사로 중복을 방지한다. 중간에 깨진 파일은 자동 수리하거나 재추론하지 않는다.
- 모든 기존 폴더의 무결성을 다시 확인하므로 자료가 많으면 시작 시간이 늘어난다.
- 검사 POST 자체의 요청 중복 방지는 아직 없다. 이미지를 다시 POST하면 새 검사 ID와 새 추론이 생성된다. 이번에 보장하는 것은 **같은 저장 결과의 등록 재시도**다.
- 로컬 단일 worker와 동일 장비의 파일·SQLite DB가 대상이다. 인증·외부 네트워크 공개·분산 작업 큐를 제공하지 않는다.
- Mac 로컬 검증은 합성 runner와 실제 SQLite·Streamlit UI를 사용했다. 실제 모델 추론 시간이나 Windows CUDA 통과를 대신하지 않는다.

## 코드 검증

```sh
python -m pytest tests/test_review_registration.py tests/test_inspection_api.py tests/test_review_store.py tests/test_review_ui.py tests/test_synthetic_demo.py -q
```

성공/기준 미만/입력 실패/추론 실패의 자동 등록, 실제 검토 화면 표시와 의견 저장, DB 오류 후 재등록, 재시작 복구, 원본 JSON 보존, 중복·변조·부분 파일·다른 DB 차단을 확인한다.
