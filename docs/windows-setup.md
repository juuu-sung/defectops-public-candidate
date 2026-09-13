# 연구실 Windows 설치 안내

수정일: 2026-09-13. **다른 학습이 끝난 뒤 DefectOps를 시작**한다.
GPU 메모리 점유 원인은 사용자 실행 중인 다른 학습으로 확인됐다. 추가 프로세스 조사는 필요하지 않다.

## 이번 설정의 목표

DefectOps 전용 가상환경에서 GPU와 Anomalib가 작동하는 상태를 만든다.
가상환경은 프로젝트마다 라이브러리를 따로 보관하는 공간이다.
연구실 PC에서 Python 3.12.0, PyTorch 2.8.0+cu128, CUDA runtime 12.8, RTX 5060,
Anomalib 2.2.0 import와 기본 GPU 행렬 연산을 확인해 `BASIC_GPU_CHECK_OK`를 통과했다.
저장소의 전체 점검 스크립트와 실제 학습 검증은 아직 하지 않았다.

## 사용할 초기 조합

| 항목 | 설정 |
|---|---|
| OS / Python | native Windows / Python 3.12 (사용자 지정) |
| GPU / RAM / 드라이버 | RTX 5060 8GB / RAM 32GB / 현재 591.86 유지 |
| PyTorch / torchvision | 2.8.0+cu128 / 0.23.0+cu128 |
| Anomalib | 2.2.0 |
| Lightning / pytorch-lightning | 2.5.5 / 2.5.5 |
| TorchMetrics / timm | 1.8.2 / 1.0.20 |
| NumPy / Pillow / OpenCV | 2.3.3 / 11.3.0 / 4.11.0.86 |
| Requests | 2.34.2 |

공식 Windows/Python 3.12 CUDA wheel과 주요 의존 조건을 확인한 초기 기준 조합이다.
전체 전이 의존성을 잠근 lock 파일이나 실제 Windows GPU 검증 결과는 아니다.
설치 후 pip check와 GPU 점검을 통과한 조합을 확정한다.

## 1. Python 3.12 선택 및 전용 환경 만들기

사용자 요청에 따라 연구실 DefectOps 환경은 **Python 3.12**로 고정한다.
현재 `python --version`이 3.11.9인 것은 기본 명령이 다른 Python을 선택한다는 뜻이다.
기본 설정을 바꾸지 않고 `py -3.12`로 설치된 3.12를 명시적으로 선택한다.

먼저 새 PowerShell에서 확인한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
py -0p
py -3.12 --version
```

`Python 3.12.x`가 나오면 다음으로 진행한다.
3.12를 찾지 못하거나 py 명령이 없으면 [Python 3.12.10 공식 배포 페이지](https://www.python.org/downloads/release/python-31210/)의
Windows installer (64-bit)를 설치하고 Python Launcher 옵션을 포함한다. PowerShell을 다시 열어 위 명령을 확인한다.
이미 3.12가 확인되면 추가 설치할 필요가 없다.

환경 경로는 프로젝트 내부의 `C:\개인 프로젝트\defectops\.venv`로 정한다.
프로젝트 경로에 공백이 있으므로 PowerShell에서는 항상 따옴표로 감싼다.

```powershell
$defectEnv = "C:\개인 프로젝트\defectops\.venv"
$defectPython = "$defectEnv\Scripts\python.exe"
if (-not (Test-Path $defectEnv)) {
    py -3.12 -m venv $defectEnv
    if ($LASTEXITCODE -ne 0) { throw "Python 3.12 environment creation failed." }
}
if (-not (Test-Path $defectPython)) { throw "Environment is incomplete. Check the folder." }
& $defectPython -c "import sys; print(sys.executable); print(sys.version); assert sys.version_info[:2] == (3, 12)"
if ($LASTEXITCODE -ne 0) { throw "This environment must use Python 3.12." }
& $defectPython -m pip install --upgrade pip
```

이후 모든 설치·실행은 전용 Python 경로를 사용한다. `Activate.ps1`이나 실행 정책 변경은 필요 없다.
PowerShell을 새로 열 때는 `$defectPython = "C:\개인 프로젝트\defectops\.venv\Scripts\python.exe"`를 다시 실행한다.
각 블록에서 오류가 나면 다음 블록을 실행하기 전에 오류를 확인한다.

## 2. GPU용 PyTorch 설치

```powershell
& $defectPython -m pip install "torch==2.8.0" "torchvision==0.23.0" --index-url https://download.pytorch.org/whl/cu128
```

CUDA 12.8 빌드로 설치한다. nvidia-smi의 CUDA 13.1은 드라이버 지원 상한이다.
이 설정에서는 CUDA Toolkit·cuDNN을 별도 설치하거나 현재 드라이버를 변경할 필요가 없다.
GPU 패키지 용량이 커서 다운로드에 시간이 걸릴 수 있다.

## 3. Anomalib와 공통 라이브러리 설치

Windows에 아직 Git 저장소가 없어도 아래 명령은 실행할 수 있다.

```powershell
$defectPackages = @(
    "torch==2.8.0+cu128", "torchvision==0.23.0+cu128",
    "anomalib==2.2.0", "lightning==2.5.5", "pytorch-lightning==2.5.5",
    "torchmetrics==1.8.2", "timm==1.0.20",
    "numpy==2.3.3", "Pillow==11.3.0", "opencv-python==4.11.0.86",
    "requests==2.34.2"
)
& $defectPython -m pip install $defectPackages
& $defectPython -m pip check
```

성공 문구는 `No broken requirements found.`다.
저장소를 받은 뒤에는 동일 목록인 `requirements/ml-windows-cu128.txt`를 -r로 설치해도 된다.
PyTorch 설치를 먼저 완료해야 한다. 뒤에서 torch 버전이나 CUDA build가 바뀌지 않도록 목록에 포함했다.

## 4. 기본 GPU 작동 확인

아래 전체 블록을 PowerShell에 붙여 넣는다. 실제 이미지나 학습 데이터는 쓰지 않는다.

```powershell
@'
import sys
import torch
import torchvision
import importlib.metadata as metadata
assert sys.version_info[:2] == (3, 12), sys.version
print("Python:", sys.version)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("GPU available:", torch.cuda.is_available())
assert torch.cuda.is_available(), "GPU is unavailable in this environment"
print("GPU:", torch.cuda.get_device_name(0))
x = torch.ones((64, 64), device="cuda")
y = x @ x
torch.cuda.synchronize()
assert bool(torch.all(y == 64).item()), "Unexpected GPU operation result"
from anomalib.models import Padim
print("Anomalib:", metadata.version("anomalib"))
print("BASIC_GPU_CHECK_OK")
'@ | & $defectPython -
```

예상: Python 3.12.x / PyTorch 2.8.0+cu128 / CUDA runtime 12.8 / GPU available True / RTX 5060 / Anomalib 2.2.0 /
`BASIC_GPU_CHECK_OK`. 이것은 기본 연산·import 성공이며 PaDiM 학습 완료가 아니다.

## 5. 코드·데이터 연결

| 종류 | Windows 권장 위치 |
|---|---|
| 가상환경 | `C:\개인 프로젝트\defectops\.venv` |
| 프로젝트 폴더 | `C:\개인 프로젝트\defectops` — 2026-09-13 사용자 확인, Git 연결은 별도 |
| 원본 데이터 | `C:\개인 프로젝트\defectops\data\transistor` |

프로젝트 폴더는 사용자가 확인한 `C:\개인 프로젝트\defectops`를 사용한다.
전용 가상환경은 위에서 만든 Python 3.12 환경을 사용한다.
프로젝트 경로에 공백이 있으므로 PowerShell에서는 따옴표로 감싼다.
Git 연결 전에는 해당 폴더의 기존 파일을 확인한다. 비어 있지 않은 폴더에 바로 clone하지 않는다.
맥북이 코드 기준이며, 환경 준비를 위해 연구실에서 별도 원본 코드를 새로 만들지 않는다.
비공개 원격 Git URL은 `https://github.com/juuu-sung/defectops.git`다.
원본 데이터도 프로젝트의 Git 제외 `data\transistor` 폴더에 둔다.

코드와 데이터가 준비된 후 저장소 루트에서 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
$defectPython = "C:\개인 프로젝트\defectops\.venv\Scripts\python.exe"
& $defectPython -m pip install -e ".[data]"
& $defectPython -m pip check
& $defectPython scripts/check_ml_environment.py --output artifacts/environment-lab-gpu.json
& $defectPython scripts/prepare_data.py verify --root "C:\개인 프로젝트\defectops\data" --manifest data/manifests/transistor-v1.json
& $defectPython -m pip freeze | Out-File -Encoding utf8 artifacts/environment-lab-freeze.txt
```

전체 GPU 점검은 행렬곱·torchvision CUDA 연산·PaDiM 특징 추출까지 확인한다.
검사에서는 사전학습 가중치를 내려받지 않으며, 실제 학습은 사전학습 가중치와 고정 manifest로 별도 실행한다.
GPU 점검이 passed인 뒤에 freeze 기록을 확정한다. freeze는 환경 기록이며 CUDA 설치 인덱스까지 대신하지는 않는다.
첫 실행 설정은 PaDiM/resnet18, 입력 256, batch 1, num_workers 0, model seed 42다.
이미 고정한 train 170 / calibration 43 / test 100 명단을 사용한다.
실행 명령과 산출물은 [PaDiM 첫 실행](padim-first-run.md)을 따른다.

## 확인 근거

- [Python 3.12 Windows 설치·런처 공식 안내](https://docs.python.org/3.12/using/windows.html)

- [PyTorch 공식 설치 조합](https://pytorch.org/get-started/previous-versions/)
- [CUDA 12.8 Windows/Python 3.12 공식 wheel](https://download.pytorch.org/whl/cu128/torch/)
- [Anomalib 2.2.0](https://pypi.org/project/anomalib/2.2.0/)
- [Anomalib 2.2.0 의존성 선언](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/pyproject.toml)
- [NVIDIA 드라이버 하위 호환성](https://docs.nvidia.com/datacenter/tesla/drivers/cuda-toolkit-driver-and-architecture-matrix.html)
