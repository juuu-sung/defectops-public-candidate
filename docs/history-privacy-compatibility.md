# 개인정보 이력 정리 후 기존 모델 연결 — 로컬 시험

이 변경은 정리된 이력의 별도 복사본에서만 검증한다. 원격 반영이나 공개 전환을
의미하지 않는다. 기존 모델·평가 묶음과 연구실 저장소는 보존한다.

## 허용 범위

기존의 직접 조상 커밋 검증을 우선한다. 직접 연결되지 않는 과거 커밋에만
`src/defectops/history_migrations/privacy-20261116.json`의 고정 대응표를 적용한다.

- 정리 전후 Git 트리 전체가 동일한 85개 커밋만 등록했다.
- 당시 문서·테스트 내용이라도 변경된 나머지 60개 커밋은 이 대응표로 허용하지 않는다.
- 실제 transistor seed42 학습 커밋은 허용 범위에 포함된다.
- 원본과 정리 복사본을 대조하여 만든 대응표의 SHA-256을 코드에 고정했다.
- 대응표가 누락되거나 수정되면 차단한다. 모델 묶음에 들어 있는 대응표나 사용자가
  지정한 임의 파일을 받아들이지 않는다.
- 정리된 기준 커밋과 대응 커밋이 현재 HEAD의 조상이어야 하며, 대응 커밋의 트리가
  사전 확인한 트리와 같아야 한다. Git replace 참조로 이 검사를 바꾸지 않는다.
- 이 경로에는 전체 Git 이력이 필요하다. 얕은 복제본에서 확인이 안 되면 차단한다.

이는 기존 체크섬·설정·보정값·체크포인트 메타데이터 검증을 대체하지 않는다.
원래 `run.json`, `evaluation.json`, 체크포인트의 커밋 번호를 변경하지 않는다.
새 검사 결과의 `trace.git_provenance`와 `/healthz`에 원래 번호, 대응 번호,
대응표 식별자·체크섬을 남긴다. `trace.fit_git_commit`도 원래 번호를 유지한다.

## 검증과 한계

로컬 테스트는 실제 Git 이력과 실제 묶음 검증 함수를 사용한다. 묶음 테스트의
체크포인트 바이트와 API 추론 런타임은 합성이므로 실제 GPU 추론 성공을 주장하지 않는다.
대응표는 코드와 함께 신뢰하는 고정 승인 자료다. 코드 자체를 바꿀 수 있는 공격자를
방어하는 서명 검증 체계는 아니다.

2026-11-17 연구실에서 transistor seed42 모델·평가 묶음으로 GPU 로딩·단일 이미지
검사·검토 등록을 완료했고, 기존 점수·임곗값·지도 값이 정확히 일치했다.
[수신 결과와 검증 범위](../reports/history-privacy-lab-validation-2026-11-17.md)를 참고한다.
다른 모델 묶음이나 입력에 대해서는 해당 자료가 있는 환경에서 별도로 확인한다. 미지원 커밋이면 검사를
끄거나 원본을 수정하지 말고, 해당 묶음의 출처를 별도로 확인한다.

## Windows 줄바꿈과 전체 이력

`.gitattributes`에서 대응표 JSON을 `-text`로 지정해 체크아웃 시 줄바꿈을 변환하지
않는다. Windows의 `core.autocrlf=true`를 포함한 설정에서도 고정 SHA-256이 유지돼야
한다. 이 설정은 대응표 내용을 변경하거나 검증 허용오차를 넓히지 않는다.
CI는 전체 Git 이력을 받아 커밋 대응 통합 검사를 실행한다.

## 연구실 확인 순서

이 복사본이 비공개 파일 전달 등으로 준비된 뒤에만 실행한다. 원격 업로드 전에는
기존 저장소에서 `git pull`해도 이 코드를 받을 수 없다.

1. 전체 정리 이력과 이 호환 커밋이 들어 있는 별도 폴더를 준비한다.
2. 원래 fit/evaluation 폴더를 새 폴더의 동일한 artifacts 하위 위치로 **복사**한다.
   원본은 이동·편집하지 않는다. 복사 전후 각 파일 SHA-256을 비교한다.
3. 검증된 연구실 Python 환경으로 아래 CPU 검증을 실행한다. 기존 환경에 새 editable
   설치를 덮어쓰지 않는다. 실행 스크립트가 이 복사본의 src를 직접 선택한다.

```powershell
python scripts/check_inspection_provenance.py `
  --fit-run artifacts/runs/patchcore-transistor-seed42 `
  --evaluation artifacts/evaluations/patchcore-transistor-seed42-eval-v1
```

`validated_bundles_only`는 모델 파일의 체크섬과 묶음 검증 성공이다.
`checkpoint_deserialized: false`, `gpu_inference_performed: false`를 함께 표시한다.

4. 다음으로 별도 API를 실행한다. 이 폴더의 새 검사 출력과 새 검토 DB를 사용한다.

```powershell
python scripts/serve_patchcore.py `
  --fit-run artifacts/runs/patchcore-transistor-seed42 `
  --evaluation artifacts/evaluations/patchcore-transistor-seed42-eval-v1 `
  --review-db artifacts/review/privacy-trial.sqlite3 --port 8001
```

5. 동일한 원본 이미지로 원점수·임곗값·판정·이상 지도를 기존 결과와 비교한다.
   검토 등록과 새 검사 기록의 커밋 대응 정보를 확인한다. 모델·평가 파일의 체크섬이
   실행 전후 그대로인지도 확인한다. 차이가 생기면 원본을 수정하지 않고 조사한다.

원본 환경은 이 시험의 성공 여부와 무관하게 그대로 보존한다.
