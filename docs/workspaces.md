# 맥북 개발 + 연구실 계산 서버

## 원본과 역할

Git에 저장된 코드·문서·설정·manifest가 기준이다. 맥북은 개발, API/UI, 모의 SQLite DB를 담당한다.
연구실은 같은 commit을 checkout하여 학습·평가하고 실행별 artifact를 생성한다.
원격 Git 저장소 URL은 아직 정하지 않았으므로 clone/push는 아직 실행하지 않았다.

| 항목 | 전달 방법 |
|---|---|
| 코드·문서·설정·manifest·테스트·SOP | Git |
| MVTec 원본 | 장비별 공식 다운로드 + manifest checksum 검증 |
| 학습 결과·모델·히트맵 | run_id별 압축/복사 + checksum 검증 |
| SQLite DB | 맥북 로컬 유지; 필요 시 일관된 백업본만 별도 보관 |
| API 키·장비별 절대 경로 | 로컬 환경 설정, Git 제외 |

## 실행 절차

1. 맥북에서 코드와 설정을 검증하고 commit한다. 원격 저장소를 연결한 뒤 push한다.
2. 연구실에서 동일 commit을 가져온다. 학습 실행 시 변경 파일이 없는지 확인한다.
3. `check_environment.py`로 OS/CPU/Python/GPU 드라이버/설치 패키지 버전을 기록한다.
4. 연구실 데이터 경로를 `--root`로 지정해 맥북에서 만든 manifest를 verify한다.
5. 이후 구현할 train/evaluate 명령에 commit·manifest·config를 명시하여 실행한다.
6. run_id 디렉터리에 체크포인트, 원시 점수, calibration 통계, 평가, 환경, checksum을 함께 둔다.
7. 결과를 맥북으로 복사하고 checksum 검증 후 앱에 등록한다. 모델 등록 검증은 후속 구현한다.

연구실에서 수정이 불가피하면 브랜치에 commit해서 맥북에 반영한다. 원본 코드를 파일 복사로만 전달하지 않는다.
Git은 대용량 artifact 백업이 아니므로 모델/실험 결과에는 별도 백업 위치가 필요하다.
GPU를 포함한 환경 정보 파일에는 컴퓨터 정보가 있을 수 있으므로 공개 전 내용을 검토한다.

## 환경 고정 시점

확인된 연구실 사양: **Windows / RTX 5060 / VRAM 8GB / RAM 32GB** (사용자 제공).
사용자가 2026-09-12 17:25에 제공한 Windows PowerShell 출력에서 Python 3.12.0,
NVIDIA 드라이버 591.86, WDDM을 확인했다. 현재 명령은 native Windows의 기존 `(venv)`에서 실행했다.
Windows 세부 버전·DefectOps 전용 환경의 설치/작동 결과는 아직 미확인이다.
2026-09-13 프로젝트 폴더의 새 PowerShell에서는 Python 3.11.9를 확인했다. 사용자 요청에 따라
DefectOps 전용 환경은 `py -3.12`로 Python 3.12를 명시적으로 선택해 만든다. 설치 패키지 버전은 그대로다.
GPU 메모리는 당시 7,767/8,151 MiB 사용(95.3%, 차이 384 MiB)이었다. 이는 당시 상태이며
GPU 사용률 2%만으로 메모리가 비어 있다고 판단할 수 없다. 사용자가 다른 학습을 실행 중임을 확인했으며,
그 학습 종료 후 DefectOps를 시작한다. 추가 점유 프로세스 조사는 필요 없다.
`nvidia-smi`의 CUDA 13.1은 드라이버 지원 상한이며 설치된 Toolkit/PyTorch runtime 버전이 아니다.
다음 확인 명령과 해석은 [Windows 환경 준비](windows-setup.md)를 따른다.
맥북은 ARM64/Python 3.11.9를 확인했으며 칩 모델/RAM은 미확인이다.
첫 실행은 PaDiM/resnet18, 256 입력, 품목 하나로 한정한다. 학습 batch는 처음 1로 시작하고
실측 후 늘린다. PatchCore는 feature 수집/coreset 구성 시 peak GPU/RAM을 별도로 측정하고,
기본 0.1 coreset이 메모리를 초과하면 train-only 자원 실험으로 비율을 정한 뒤 비교 설정을 동결한다.
coreset을 줄여도 축소 이전 특징 수집 단계의 peak 메모리가 줄어든다는 보장은 없다.

RTX 50 계열을 고려해 Blackwell 지원 CUDA build를 선택해야 한다.
PyTorch는 2.7에서 CUDA 12.8 기반 Blackwell 지원을 도입했다.
초기 설치 후보는 torch 2.8.0+cu128 / torchvision 0.23.0+cu128 / anomalib 2.2.0이다.
주요 의존성은 `requirements/ml-windows-cu128.txt`로 지정했고 Windows GPU 점검 통과 후 확정한다.
2026-09-13 사용자가 확인한 Windows 프로젝트 폴더는 `C:\개인 프로젝트\defectops`다.
전용 venv는 프로젝트 내부의 `C:\개인 프로젝트\defectops\.venv`를 사용한다.
프로젝트 내부 `.venv`에서 Python 3.12.0, PyTorch 2.8.0+cu128, CUDA runtime 12.8,
RTX 5060, Anomalib 2.2.0 import와 기본 GPU 행렬 연산의 `BASIC_GPU_CHECK_OK`를 확인했다.
저장소 전체 점검과 실제 학습 검증은 아직 하지 않았다.
공백이 포함된 프로젝트 경로는 PowerShell 명령에서 따옴표로 감싼다.
[PyTorch 공식 발표](https://pytorch.org/blog/pytorch-2-7/)

manifest/환경 도구는 Python 3.11+ 표준 라이브러리만 쓴다. 추가한 이미지 점검 도구와 전체 테스트는
`pip install -e '.[data]'`로 Pillow/NumPy를 설치한다. 연구실 실행 환경 확인 후
공통 라이브러리 버전과 macOS/연구실 OS별 설치·lock 파일을 추가한다.
Anomalib·PyTorch를 무작정 최신 버전으로 각각 설치하지 않는다.
먼저 최소 모델 import/weights load/한 장 추론을 확인한 조합을 기록한다.
최종 CPU p95 목표 장비는 실제 맥북 사양 확인 후 고정한다.
