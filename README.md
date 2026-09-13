# DefectOps

정상 데이터 기반 제조 이상탐지 및 작업자 조치 지원 시스템.

이미지 이상탐지 → 작업자 검토 → 근거 SOP 검색 → 승인 후 모의 티켓 생성.
이미지로 공정 원인을 확정하거나 이상 점수를 심각도로 해석하지 않는다.
MVTec는 실제 공장 생산성·안전성 검증 자료가 아니다.

## 현재 상태

데이터 준비 단계 완료. 실제 transistor 이미지 313장·마스크 40장의 디코딩과 무결성을 확인하고
정상 train 170장 / calibration 43장 / 공식 test 100장으로 manifest를 고정했다.
데이터 관련 테스트 17개 통과. [데이터 점검 보고서](reports/transistor-data-audit-v1.md)를 참고한다.
모델 학습, FastAPI·Streamlit·SQLite 앱은 아직 구현/실행하지 않았다.
가짜 추론 결과를 실험 성능으로 제시하지 않는다.

## 먼저 할 일

1. [계획 검토](docs/plan-review.md)와 [실험 규칙](docs/data-protocol.md)을 기준으로 진행한다.
2. 연구실 Windows의 드라이버/Python/실행 방식을 확인한 뒤 ML 의존성을 고정한다.
3. 맥북 데이터는 `data/transistor/`에 준비되어 있다. 원본은 Git 제외 상태다.
4. 첫 commit과 원격 저장소를 준비해 고정 manifest를 공유하고, 연구실에서 같은 파일의 체크섬을 검증한다.
5. 다음 개발 단계는 manifest를 직접 읽는 PaDiM 학습·추론 어댑터다.

Python 3.11 이상. manifest/환경 점검 도구는 표준 라이브러리만 사용한다.
이미지 점검과 전체 테스트에는 Pillow/NumPy가 필요하다. 새 환경에서는 아래처럼 설치한다.
ML 학습용 라이브러리는 아직 포함하지 않는다. 저장소 루트에서 실행한다.

```sh
python3 -m pip install -e '.[data]'
python3 scripts/check_environment.py --output artifacts/environment-mac.json
python3 -m unittest discover -s tests -v
python3 scripts/prepare_data.py verify --root data --manifest data/manifests/transistor-v1.json
```

Windows에서는 `python3` 대신 `python`, 데이터 경로는 해당 장비의 절대 경로를 사용한다.
`transistor-v1.json`은 이미 생성되어 있으므로 다시 create하지 않는다.
폴더 구조와 실행 절차는 [data/README.md](data/README.md), 장비 간 전달은
[작업 공간 운영](docs/workspaces.md)을 따른다.

## 첫 완료 목표

transistor 한 장 → 원시 이상 점수·고정 임곗값·PASS/REVIEW_REQUIRED·히트맵·영역 좌표를 JSON으로 반환.
추론 오류/입력 오류는 INSPECTION_FAILED. 모델·분할·설정 버전까지 추적 가능해야 한다.
이는 **다음 단계의 목표**이며 현재 구현된 기능은 아니다.

최소 범위는 transistor + PaDiM, 이후 PatchCore 비교 → 3품목 → API/검토/SOP/티켓 순서다.
EfficientAD, AD 2, 모델 변환, embedding drift는 확장 단계로 둔다.

## 데이터·문서 출처

- [MVTec AD 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec AD 2 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
- 데이터 라이선스: CC BY-NC-SA 4.0. 원본 이미지·마스크·파생 시각화는 코드 라이선스와 별도로 취급한다.
- 이미지·모델·DB·비밀정보는 Git에 넣지 않는다. 파생물 공개 시에도 데이터 이용 조건을 확인한다.
- 향후 SOP는 워크플로우 구현용 모의 문서이며 실제 산업 안전 지침이 아니다.

구체적인 검토 근거와 후속 작업은 [계획 검토](docs/plan-review.md), [착수 체크리스트](docs/first-milestone.md)에 있다.
