# 외부 코드·의존성·데이터 고지

DefectOps의 자체 코드·문서는 별도 표시가 없으면 [Apache-2.0](LICENSE)을 적용합니다.
외부 코드의 기존 저작권과 라이선스는 유지합니다. 이 문서는 상위 프로젝트의 허가 조건을
대체하거나 데이터·모델 가중치의 재배포 권한을 새로 부여하지 않습니다.

## 저장소에 포함한 외부 코드

| 항목 | 내용 |
| --- | --- |
| 원본 프로젝트 | [SuperSimpleNet](https://github.com/blaz-r/SuperSimpleNet) |
| 기준 버전 | `9c6867c3e1ef36d9e9fa7722a7dfdc8f9e41e301` |
| 원본 파일 | [model/supersimplenet.py](https://github.com/blaz-r/SuperSimpleNet/blob/9c6867c3e1ef36d9e9fa7722a7dfdc8f9e41e301/model/supersimplenet.py), [common/loss.py](https://github.com/blaz-r/SuperSimpleNet/blob/9c6867c3e1ef36d9e9fa7722a7dfdc8f9e41e301/common/loss.py) |
| 이 저장소의 파일 | [scripts/mpi_supersimplenet_parity_reference.py](scripts/mpi_supersimplenet_parity_reference.py) |
| 저작권 | Copyright (c) 2024 Blaž Rolih |
| 적용 라이선스 | MIT — [보관 원문](licenses/SuperSimpleNet-MIT.txt), [상위 버전 원문](https://github.com/blaz-r/SuperSimpleNet/blob/9c6867c3e1ef36d9e9fa7722a7dfdc8f9e41e301/LICENSE) |

변경 범위: 특징 변환부·검출부와 손실 계산을 독립 대조 구현으로 옮겼습니다. 특징과 합성
잡음을 외부에서 동일하게 주입해 계산 경로를 비교하는 용도이며 논문 전체의 재현 구현을
뜻하지 않습니다. 파일 안의 MIT 고지 전문을 보존하며 루트 Apache-2.0으로 대체하지 않습니다.

두 3D 시연은 Three.js r170 (0.170.0)을 포함하고, 초기 흐름 시연은 같은 버전의 OrbitControls도 사용합니다. MIT 라이선스의 저작권 고지를 각 배포 폴더에 보존합니다. [Three.js r170 원본](https://github.com/mrdoob/three.js/tree/r170) · [검사 워크스페이스의 MIT 고지](output/defectops-inspection-lab/vendor/LICENSE-three.txt) · [초기 시연의 MIT 고지](output/defectops-workflow-3d/vendor/LICENSE-three.txt)

## 별도로 설치하는 주요 의존성

아래 라이브러리 전체를 이 저장소에 복사해 배포하지 않습니다. 설치되는 각 패키지와 그
전이 의존성에는 해당 배포물의 LICENSE·NOTICE가 적용됩니다. 이 표는 주요 구성요소의
출처 안내이며 전체 의존성 목록이나 SBOM은 아닙니다.

| 구성요소 | 역할 | 상위 라이선스 원문 |
| --- | --- | --- |
| Anomalib 2.2.0 | PaDiM·PatchCore·EfficientAD·SuperSimpleNet 등 연구 모델 | [Apache-2.0](https://github.com/open-edge-platform/anomalib/blob/v2.2.0/LICENSE) |
| PyTorch 2.8.0 / torchvision 0.23.0 | GPU 텐서 연산·특징 추출 | [PyTorch](https://github.com/pytorch/pytorch/blob/v2.8.0/LICENSE), [torchvision](https://github.com/pytorch/vision/blob/v0.23.0/LICENSE) |
| FastAPI 0.117.1 | 로컬 검사 API | [FastAPI](https://github.com/fastapi/fastapi/blob/0.117.1/LICENSE) |
| Streamlit 1.64.0 | 사람 검토 화면 | [Streamlit](https://github.com/streamlit/streamlit/blob/1.64.0/LICENSE) |

설치 조건은 [pyproject.toml](pyproject.toml), [API](requirements/api.txt),
[UI](requirements/ui.txt), [연구실 ML 환경](requirements/ml-windows-cu128.txt)에 기록합니다.
PyTorch 등의 배포 패키지에 포함된 외부 구성요소 고지도 별도로 유지해야 합니다.

## 데이터·가중치·검토 기록

MVTec AD, Real-IAD, VisA, MPI-PCB의 이미지·마스크, 외부 사전학습 가중치,
연구실 모델 체크포인트와 실제 검토 DB는 소스 배포에 포함하지 않습니다.
데이터 이용 조건은 [데이터 안내](data/README.md)와 각 실험의 출처 문서에서 구분합니다.
특히 [MPI-PCB](docs/mpi-pcb-external-stress-test.md)는 기존 권리 확인 범위에 따라
이미지·마스크를 재배포하지 않습니다. 코드 라이선스를 데이터나 가중치의 라이선스로
해석하지 않습니다.

README의 [합성 시연](docs/synthetic-demo.md)은 저장소의 생성기로 만든 예시입니다.
실제 데이터셋 이미지나 실제 모델 추론 결과를 공개 시연 이미지로 배포하지 않습니다.
