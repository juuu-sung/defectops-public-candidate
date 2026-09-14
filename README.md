# DefectOps

정상 데이터 기반 제조 이상탐지 및 작업자 조치 지원 시스템.

이미지 이상탐지 → 작업자 검토 → 근거 SOP 검색 → 승인 후 모의 티켓 생성.
이미지로 공정 원인을 확정하거나 이상 점수를 심각도로 해석하지 않는다.
MVTec는 실제 공장 생산성·안전성 검증 자료가 아니다.

## 현재 상태

데이터 준비 단계 완료. 실제 transistor 이미지 313장·마스크 40장의 디코딩과 무결성을 확인하고
정상 train 170장 / calibration 43장 / 공식 test 100장으로 manifest를 고정했다.
데이터 관련 테스트 17개 통과. [데이터 점검 보고서](reports/transistor-data-audit-v1.md)를 참고한다.
고정 manifest의 train 170장으로 PaDiM을 fit하고 정상 calibration 43장으로 원시 점수 임곗값을
산출하는 첫 실행을 연구실 Windows RTX 5060에서 완료했다. seed 42 실행은 임곗값
`56.027042388916016`, fit+calibration 20.225초를 기록했다. 고정 test 평가 규칙과 실행 명령을
구현하고 연구실에서 공식 test 100장 파일럿 평가까지 완료했다. Image AUROC 0.9013,
고정 임곗값 recall 0.55, Pixel AUROC 0.9689를 기록했다. 낮은 recall 때문에 현재 모델을
실제 자동 PASS 용도로 사용할 수 없다. [파일럿 평가 보고서](reports/padim-transistor-seed42-pilot.md)를 참고한다.
같은 checkpoint·고정 임곗값으로 실제 `bent_lead/000.png` 한 장을 검사해
`REVIEW_REQUIRED`, heatmap·overlay·영역 JSON 생성과 실행 추적을 확인했다.
[단일 이미지 검증 보고서](reports/padim-single-image-smoke.md)를 참고한다.
고정 결과의 [오류 분석과 warm 추론 측정](docs/padim-error-analysis-and-benchmark.md) 도구를 구현했으며
연구실 Windows 실행을 기다리고 있다.
FastAPI·Streamlit·SQLite 앱도 아직 구현/실행하지 않았다.
가짜 추론 결과를 실험 성능으로 제시하지 않는다.

## 먼저 할 일

1. [계획 검토](docs/plan-review.md)와 [실험 규칙](docs/data-protocol.md)을 기준으로 진행한다.
2. 연구실 Windows의 드라이버/Python/실행 방식을 확인한 뒤 ML 의존성을 고정한다.
3. 맥북 데이터는 `data/transistor/`에 준비되어 있다. 원본은 Git 제외 상태다.
4. 비공개 원격 저장소 `https://github.com/juuu-sung/defectops`의 같은 commit을 연구실에서 받고,
   고정 manifest의 체크섬을 검증한다.
5. 단일 이미지 검사까지 완료. 고정 test의 탐색적 오류 분석과 모델 1회 로드 warm 속도를 측정한다.

Python 3.11 이상. manifest/환경 점검 도구는 표준 라이브러리만 사용한다.
이미지 점검과 전체 테스트에는 Pillow/NumPy가 필요하다. 새 환경에서는 아래처럼 설치한다.
ML 학습·평가용 Windows 라이브러리는 별도 requirements에 고정했다. 아래는 맥북 데이터 도구 실행이다.

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
추론·고정 판정·JSON·heatmap·영역·실패 기록을 구현했고 자동화 테스트를 통과했다.
연구실 Windows에서 실제 checkpoint로 한 장을 검사하고 산출물·checksum을 확인해 첫 기술 마일스톤을 완료했다.

최소 범위는 transistor + PaDiM, 이후 PatchCore 비교 → 3품목 → API/검토/SOP/티켓 순서다.
EfficientAD, AD 2, 모델 변환, embedding drift는 확장 단계로 둔다.

## 데이터·문서 출처

- [MVTec AD 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)
- [MVTec AD 2 공식 페이지](https://www.mvtec.com/research-teaching/datasets/mvtec-ad-2)
- 데이터 라이선스: CC BY-NC-SA 4.0. 원본 이미지·마스크·파생 시각화는 코드 라이선스와 별도로 취급한다.
- 이미지·모델·DB·비밀정보는 Git에 넣지 않는다. 파생물 공개 시에도 데이터 이용 조건을 확인한다.
- 향후 SOP는 워크플로우 구현용 모의 문서이며 실제 산업 안전 지침이 아니다.

구체적인 검토 근거와 후속 작업은 [계획 검토](docs/plan-review.md), [착수 체크리스트](docs/first-milestone.md)에 있다.
