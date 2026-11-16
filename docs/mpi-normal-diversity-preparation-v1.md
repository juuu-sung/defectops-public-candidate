# MPI 정상 학습 사진64장 선정 준비 실행 안내

2026-10-01. [MPI-C1-02 설계](mpi-normal-diversity-v1.md)의 B단계 실행기다.
**정상 후보1,086장에서 학습할64장을 고르는 작업이며, 새 모델을 학습하거나 불량 점수를 계산하지 않는다.**
선정 결과를 검토한 다음 별도 학습 단계로 넘어간다.
후속:2026-10-02 [전달물 수령 검산](../reports/mpi-normal-diversity-preparation-v1-received-review.md)을 완료했다.
아래는 완료한 B단계의 실행 기록이다. 이미 완료했다면 다시 실행할 필요가 없다.

## 이번에 하는 일

1. 기존 전달물·정상 풀·보정/검증 명단·원본ZIP·모델의 해시를 확인한다.
2. 고정 후보 순서의 정상16장으로 특징 추출 경로,반복 계산,시간·메모리를 점검한다.
3. 통과하면 정상1,086장×12영역=13,032타일의 특징을 추출한다. 사진마다 요약 벡터를 저장한다.
4. 정상 외관의 다양성을 기준으로64장 명단을 고정하고,기존64장과의 겹침 및 특징 거리 요약을 기록한다.
5. 전달ZIP을 만든다. 이미지·모델은 넣지 않으며,명단·특징 벡터·검증 기록만 담는다.

특징 추출에는 기존 모델의 동결 가중치를 사용한다. 정상 기준표를 만드는 `fit`이나 최근접 검색은
호출하지 않는다. 기존 학습 경로와 일치하는지 확인할 때만 특징을 반환하는 분기를 사용하며,
신경망/배치 정규화는 eval 상태를 유지하고 임시 특징은 즉시 비운다. 가중치나 기존 기준표를 변경하지 않는다.
[Anomalib2.2.0 원 구현](https://github.com/openvinotoolkit/anomalib/blob/v2.2.0/src/anomalib/models/image/patchcore/torch_model.py)의
특징 추출·평균풀링·융합 순서를 따르며,실제GPU에서 원래 특징 반환 경로와 최대차1e-4 이내인지 확인한다.

정상 개발128장·보정169장·관찰 정상256장은 학습 후보에서 제외한다. 시험 이미지·마스크는 디코딩하지 않는다.
이전 점수 전달물은 고정 명단·기록 검산용으로 읽지만 선정 거리나 순위에 불량 점수·라벨을 사용하지 않는다.

## 연구실 Windows 실행

다른 GPU 작업을 정상적으로 종료한 뒤 기존 PowerShell에서 실행한다.
새 가상환경을 만들거나 PyTorch·Anomalib을 다시 설치하지 않는다. 추가 설치는 자원 측정용 psutil뿐이다.

```powershell
Set-Location 'C:\개인 프로젝트\defectops'
git pull --ff-only
& '.\.venv\Scripts\python.exe' -m pip install -r '.\requirements\mpi-normal-diversity-monitor.txt'
```

설치가 성공했으면 다음 명령을 실행한다.

```powershell
powershell.exe -NoProfile -ExecutionPolicy Bypass -File '.\scripts\run_mpi_normal_diversity_preparation.ps1'
```

원본은 기본적으로 `$env:USERPROFILE\Downloads\extracted.zip`을 사용한다(실제 사용자 홈 기준).
다른 곳에 있다면 실행 명령 뒤에 `-Archive '실제 전체 경로\extracted.zip'`을 붙인다.
기존 모델 `artifacts\mpi-pcb-normal-bank-pilot-v1\fit\model.pt`와 과거 `artifacts\handoff` ZIP들을 그대로 사용한다.
과거 전달물은 정상 준비·fit·외부v2·score-probe·score-validation·spatial-score가 필요하다.
누락 파일이나 계약 불일치가 나오면 새 파일을 임의 생성하지 말고 오류를 전달한다.

완료하면 다음 파일을 현재 대화에 첨부한다.

```text
C:\개인 프로젝트\defectops\artifacts\handoff\mpi-normal-diversity-preparation-v1.zip
```

결과 폴더는 `artifacts\experiments\mpi-normal-diversity-v1\prepare\result`다.
`report.json`은 실행 요약,`selection.json`은 선정64장과 기존 명단 겹침,`pool.json`은 후보/배제 근거,
`descriptors.npy`는 재검산용 특징 벡터,`preflight.json`은 정상16장 사전 점검이다.
벡터와 사진 자료를 공개 저장소나 Notion에 올리지 않는다.

## 완료 상태의 의미

- `normal_diversity_selected_ready_for_review`: 새64장 선정 완료. 아직 학습 승인이나 성능 개선 결과가 아니다.
- `selection_unchanged_stop`: 기존64장과 같아 변경 없음. 다른 명단이 나올 때까지 다시 뽑지 않는다.
- 오류: 완료 조각과 원본을 보존하고 메시지를 전달한다. 임곗값 수정이나 재학습으로 우회하지 않는다.

이번 과정에서 모델의 검출률·미탐률·오경보율을 새로 계산하지 않는다. 특징 거리 요약이 좋아져도
정상 후보를 더 넓게 대표한다는 의미일 뿐,불량 검출이 좋아졌다는 근거는 아니다.

## 중단과 비용

사진별 특징을 원자적으로 저장한다. 일반 오류나Ctrl+C 뒤 같은 명령을 실행하면 코드·설정·환경·파일을
검증하고 완료 사진을 재사용한다. 정상16장 점검은 재개 때도 다시 한다. 완전히 끝난 결과는 새GPU 실행 없이 검산한다.
프로세스 강제 종료나 전원 종료로 시도별 시간 기록이 사라졌다면 비용을0으로 초기화하지 않고 멈춘다.
이때도 폴더를 삭제하지 말고 오류와 남은 파일을 전달한다. Git commit/실행 소스/환경이 바뀐 경우에도 조용히 이어가지 않는다.

선정 계산의 누적 시간1시간,RSS20GiB,GPU reserved6GiB를 검사한다. 처음16장의 보수적 시간 추정이
상한을 넘으면 전체 선정을 시작하지 않는다. 시간 상한은 소스 검산·ZIP 해시·최종 압축 시간을 제외한
runtime 준비와 특징/선정 계산 범위다. 실제 소요 시간을 보장하는 예상치가 아니다.
RSS와 GPU 전체 점유는100ms 표본 최대,PyTorch 할당/예약은 allocator peak를 기록한다.
PyTorch peak는 모델 로드 후 기준이며 전체 GPU 순간 최대와 구분한다. 연산 사이에 예산을 검사하므로
한 번의 큰 연산까지 사전에 차단하는 hard limit는 아니다. OOM·초과·측정 실패 시 자동 축소하지 않는다.

## 구현 검증과 다음 단계

구현 시점 검증: 관련 테스트24개,전체490개 테스트·52개 subtest 통과. 기존 경고3개는 유지된다.
명령행 도움말/인자 전달과 문서 링크도 검사했다. 이는 아래 범위의 CPU 검증 결과다.

CPU 합성 테스트에서 수동 선정 대조·동점·순서 불변성·0/NaN 특징·자료 누수·부분 중복·파일 변조·
중단/재개·환경/특징 변화·예산 초과·기존 명단 동일 시 중단·CLI 인자 전달을 확인한다.
실제 수령 메타데이터에서도1,086장·13,032타일·고정 명단 해시를 재검산한다.
CPU 모의 특징 경로 검사는 실제 Anomalib/CUDA 실행의 대체가 아니다. Mac에는 torch와 PowerShell이 없어
이번 도구의 실제 Windows/CUDA 실행은 아직 검증하지 않았다. 연구실16장 preflight가 그 검증 단계다.

전달물 확인 후 선정 명단·특징·누수·비용을 검산하고,통과하면 C단계의64장 학습·128/169장 보정 실행기를 준비한다.
이번 명령은 그 학습을 자동으로 시작하지 않는다.
