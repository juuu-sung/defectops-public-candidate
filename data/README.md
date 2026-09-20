# 데이터 준비

공식 [MVTec AD](https://www.mvtec.com/research-teaching/datasets/mvtec-ad)에서 대상 품목을 받는다.
다운로드·압축 해제 후 아래 구조를 확인한다. `--root`는 transistor의 **부모 폴더**다.

```text
mvtec_ad/
└── transistor/
    ├── train/good/*.png
    ├── test/good/*.png
    ├── test/<defect>/*.png
    └── ground_truth/<defect>/*_mask.png
```

MVP는 원본 PNG 형식만 지원한다. 원본은 Git 제외 폴더에 저장하고 Git에는 manifest만 넣는다.
현재 맥북 원본은 이 저장소의 `data/transistor/`에 있으며 `--root data`로 읽는다.
이미 정상 train 170장 / calibration 43장 / test 정상 60장·이상 40장으로 v1 목록을 고정했다.
연구실에서는 이 manifest를 전달받아 verify만 실행한다.
`.env`에 쓰는 것만으로 현재 스크립트에 반영되지 않으므로 `--root`를 명시한다.

```sh
python3 scripts/prepare_data.py verify --root /absolute/path/mvtec_ad --manifest data/manifests/transistor-v1.json
```

`create`는 `configs/pilot.toml`의 품목·split_seed·calibration_fraction을 읽는다.
정상 파일의 상대 경로와 seed를 SHA-256으로 정렬하고 ceil(n × 0.2)개를 calibration으로 지정한다.
절대 경로·시간을 제외하므로 같은 파일과 설정이면 장비에 관계없이 동일한 manifest를 생성한다.
이미지/마스크 체크섬과 공식 test 목록까지 고정한다. test 파일 내용 해싱은 무결성 확인만 수행하며
테스트 이미지 내용·라벨·마스크를 모델 튜닝에 사용한다는 의미가 아니다.

기존 출력은 덮어쓰지 않는다. 변경이 필요하면 새 버전 파일을 생성하고 변경 이유를 기록한다.
검증은 원본에서 전체 manifest를 다시 계산하므로 파일 누락·추가·수정·분할 변조를 감지한다.
동일 이미지 바이트가 여러 경로에 있으면 중복 데이터로 중단한다. 원본 중복이 발견되면
임의 삭제하지 말고 조사 후 그룹 단위 분할 정책을 새 프로토콜로 작성한다.

manifest 도구는 구조와 바이트 무결성을 확인한다. 별도 `audit_data.py`는 실제 디코딩·해상도·
마스크 이진값·이미지와 마스크 크기 일치·디코딩 픽셀 중복을 확인한다. 테스트는 무결성 점검만
수행하고 밝기/선명도 통계는 train/calibration에만 계산한다. 마스크의 의미적 정확성은 판단하지 않는다.

```sh
python3 -m pip install -e '.[data]'
python3 scripts/audit_data.py --root data --manifest data/manifests/transistor-v1.json --output artifacts/data-audit/transistor-v1-recheck
```

출력 경로는 새 디렉터리여야 한다. 최초 결과는 `artifacts/data-audit/transistor-v1/`에 있으며
Git에 포함할 요약은 [점검 보고서](../reports/transistor-data-audit-v1.md)에 저장했다.
새 품목/새 데이터 버전에서만 설정과 새 출력 파일명으로 create를 실행한다.
Windows에서 심볼릭 링크 생성 권한이 없으면 경로 탈출용 링크 fixture 테스트 1개만 skip될 수 있다.
나머지 점검은 실행하며, skip은 실제 데이터 점검 성공을 대신하지 않는다.
실제 데이터가 없는 상태에서 생성한 가짜 manifest는 저장소에 넣지 않는다.

## cable·metal_nut 추가 등록

3품목 비교의 추가 품목은 cable과 metal_nut으로 test 열람 전에 고정했다. 두 archive는 공식
MVTec 페이지의 폼을 통해 연구실 Windows에서 받고 `data/cable/`, `data/metal_nut/`에 압축
해제한다. 자세한 순서와 사전 고정 조건은
[3품목 비교 사전등록](../docs/three-category-preregistration.md)을 따른다.

원본을 준비한 뒤 다음 스크립트를 한 번 실행한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File ".\scripts\prepare_additional_categories.ps1"
```

생성된 handoff ZIP을 이 작업으로 전달해 manifest와 데이터 점검 보고서를 먼저 커밋한다.
그 전에는 추가 품목의 학습이나 test 평가를 시작하지 않는다.
