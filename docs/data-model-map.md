# 데이터와 모델 연결 관계

기준일: 2026-09-27. 검증된 연구실 결과와 현재 구현 상태를 구분했다.
**MPI-PCB 전용 모델은 학습·정상 보정·저장 재검증 완료이며, 새 모델의 시험 성능은 아직 없다.**

## 먼저 세 가지를 구분한다

| 데이터 역할 | 쉬운 설명 | 모델과의 관계 |
| --- | --- | --- |
| 정상 학습 | 정상 제품의 생김새를 기준으로 만든다 | PatchCore는 정상 특징의 대표 모음인 memory bank를 만든다. PaDiM은 정상 특징의 통계 분포를 만든다 |
| 정상 보정 | 어느 점수부터 사람에게 경고할지 정한다 | 학습에 쓰지 않은 정상 사진으로 임곗값을 계산한다. 기존 bank에 새 특징을 넣는 작업은 아니다 |
| 평가 | 불량을 얼마나 놓치고, 정상을 얼마나 경고하는지 확인한다 | 정상·불량 시험 사진과 위치 정답을 사용한다. 결과로 설정을 고치면 해당 시험은 이후 탐색·회귀용이다 |

**같은 PatchCore는 같은 알고리즘이라는 뜻이지 같은 모델 파일이라는 뜻이 아니다.**
현재 PatchCore는 ImageNet 사전학습 특징 추출기를 고정해 사용하고, 데이터별 정상
기준표를 따로 만든다. 여러 데이터셋을 하나의 모델에 계속 누적 학습한 구조가 아니다.
PaDiM도 이번 프로젝트에서는 사전학습 특징을 사용하며 정상 특징의 통계를 적합한다.

## 전체 연결표

| 정상 기준을 만든 데이터 | 만들어진 모델 | 보정·평가에 연결된 데이터 | 현재 위치 |
| --- | --- | --- | --- |
| **MVTec AD**: transistor / cable / metal_nut | 품목마다 별도 PaDiM·PatchCore, seed 42·43·44 반복 | 각 모델을 같은 품목의 분리된 정상 보정·공식 시험에 연결 | 3품목 비교 완료. 검사 API·검토 화면에서 실제 확인한 흐름은 transistor PatchCore 중심 |
| **Real-IAD**: transistor1 / plastic_nut | 품목마다 별도 PatchCore, seed 반복 | 각 품목의 정상 보정·시험 자료 | 평가·반복 및 오류 진단 완료. MVTec 모델을 그대로 시험한 것이 아님 |
| **Real-IAD**: pcb | 개발 비교 후 선택한 세밀한 특징 PatchCore, seed 42·43·44 | 자체 개발·최종 시험. 그중 PCB seed 42 모델을 **VisA PCB1–4와 MPI-PCB에도 그대로 적용**, 각 대상의 정상 자료로 경고 기준만 다시 보정 | 자체 평가와 외부 스트레스 시험 완료. 외부 시험 때 해당 제품의 bank를 새로 학습하지 않았음 |
| **Real-IAD**: terminalblock | pcb와 별도로 만든 세밀한 특징 PatchCore, seed 42·43·44 | terminalblock 자체 개발·최종 시험 | 평가 완료. VisA/MPI 외부 비교에 사용한 모델은 이것이 아니라 위 PCB 모델 |
| **VisA**: pcb3 | **PCB3 정상 기준표 v2 PatchCore** | PCB3 정상 보정·같은 PCB3 시험. 이후 **MPI-PCB에 그대로 적용**하고 MPI 정상으로 경고 기준 보정 | 학습·사후 평가·MPI 스트레스 시험 완료. PCB1·2·4 전용 bank까지 학습한 상태는 아님 |
| **MPI-PCB**: 정상 원본 64장 | **MPI 전용 PatchCore 하나**: 12영역 공용 bank 78,643×768 | 별도 정상 원본 169장으로 경고 기준 46.8394 산출 → 기존 109장 시험에서 탐색 비교 예정 | **실제 학습·정상 보정·저장 재검증 완료 / 시험 실행기 준비, 실제 평가 대기** |

`transistor`와 `transistor1`, `metal_nut`과 `plastic_nut`, 다른 데이터셋의 `pcb`는
같은 제품의 다른 촬영 세션으로 간주할 수 없다. 이름이 비슷하다고 모델을 공유하지 않는다.
seed 반복은 무작위 선택의 영향을 확인하는 반복 실험이지, 새 품목이거나 누적 학습 횟수가 아니다.

## 지금 MPI-PCB 실험을 확대해서 보면

세 모델 모두 PatchCore이지만 **어느 기판의 정상 패턴을 참조하는지**가 다르다.

| MPI-PCB에 적용하는 모델 | 정상 기준표의 출처 | MPI 정상 169장의 역할 | MPI 시험 109장 결과 |
| --- | --- | --- | --- |
| 기존 Real-IAD PCB 모델 | Real-IAD PCB 정상 학습 자료 | 기준표 고정, MPI용 경고 임곗값만 계산 | 불량 55장 중 검출 2 / 미탐 53, 정상 54장 중 오경보 2 |
| 기존 VisA PCB3 v2 모델 | VisA PCB3 정상 849장 | 기준표 고정, MPI용 경고 임곗값만 계산 | 검출 3 / 미탐 52, 정상 오경보 4 |
| **새 MPI-PCB 전용 모델** | MPI 정상 원본 **64장 × 12영역 = 768개** | 기존 명단 169장으로 **새 모델의** 임곗값 계산 | **아직 결과 없음** |

이번 질문은 “같은 제품의 정상 패턴을 참조하면 나아지는가?”다. 임곗값만 낮추는 작업도,
Real-IAD/VisA 기존 모델을 덮어쓰는 작업도 아니다. 고정 특징층은 layer1 + layer2,
입력 256×256, 공간 패치 간격 2다. 후보 786,432개에서 10%인 78,643개를 선택한다.
VisA와 학습량·bank 크기도 달라지므로 개선을 데이터 출처 하나의 인과 효과로 주장하지 않는다.

사진 하나가 12영역으로 나뉘므로 **768개 학습 영역은 독립된 기판 768장이 아니다.**
영역별 맵 상위 1% 평균을 계산한 뒤 12영역 중 가장 높은 점수를 원본 사진 점수로 쓴다.
정상 원본 169장의 사진 점수에서 95% 분위수(higher)를 경고 기준으로 정한다.
기준 이하라고 자동 합격하는 것은 아니다. 이번에는 이진 픽셀 판정 임곗값을 만들지 않는다.

기존 시험을 이미 본 뒤 설계한 실험이므로 성적이 좋아져도 사후 탐색 결과다.
기존 예외(정상 중복 제외, 복구 사진 제외 108장 보조 분석, 위치 불량 54장)를 유지한다.
자동 PASS와 현장 생산 성능 보장은 허용하지 않는다.

## 검사 서비스와 연구 모델은 별개다

현재 실제로 확인한 운영 시연은 **MVTec transistor PatchCore → Windows 검사 API →
저장된 검사 결과를 Mac에 전달 → 검토 화면·사람 의견·모의 SOP·승인·모의 티켓**이다.
단일 장비의 자동 연결 흐름이나 다품목 모델 선택 기능까지 완성한 상태는 아니다.

Real-IAD·VisA·MPI 모델은 연구 실험용이다. 새 MPI 학습을 실행해도 검사 서버의 모델,
기존 검토 이력, SOP, 티켓은 변경되지 않는다. 같은 PatchCore라고 실험 모델을 서비스에
자동 교체하지 않는다. 별도의 품목 연결·입력 조건·검증·승인이 필요하다.

## 근거와 다음 행동

- 이번 실행 안내: [MPI 전용 정상 기준표 비교 평가](mpi-pcb-normal-bank-evaluation.md).
- 이번 학습 결과: [수령·검증 보고서](../reports/mpi-pcb-normal-bank-fit-v1.md).
- MVTec 3품목: [PaDiM–PatchCore 비교](../reports/three-category-padim-vs-patchcore-v1.md).
- Real-IAD 초기: [평가](../reports/realiad-patchcore-evaluation-v1.md), [반복·진단](../reports/realiad-patchcore-multiseed-diagnostics-v1.md).
- Real-IAD 후속: [개발 모델 선택](../reports/realiad-followup-development-selection-v1.md), [최종 평가](realiad-followup-final.md).
- VisA: [기존 모델 외부 평가](visa-pcb-external-preregistration.md), [PCB3 자체 bank 평가](visa-pcb-normal-bank-pcb3-evaluation.md).
- MPI: [기존 모델·정상 보정](mpi-pcb-external-stress-test.md), [외부 평가 v2](mpi-pcb-external-evaluation-v2.md), [맵 진단](mpi-pcb-map-diagnostics.md).
- 운영 시연: [검사 API](patchcore-local-api.md), [검토부터 모의 티켓까지](../reports/local-review-to-mock-ticket-demo-v1.md).

다음 순서: **새 프로세스에서 저장 모델 로드 → 짧은 정상 점수·속도 점검 → 같은 시험의
탐색 비교 → 채택 여부 판단**. 평가 실행기는 준비됐지만 실제 평가 결과는 아직 없고
서비스 모델은 교체하지 않았다.
