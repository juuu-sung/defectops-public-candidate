# DefectOps

정상 데이터 기반 제조 이상탐지 및 작업자 조치 지원 시스템.

이미지 이상탐지 → 작업자 검토 → 근거 SOP 검색 → 승인 후 모의 티켓 생성.
이미지로 공정 원인을 확정하거나 이상 점수를 심각도로 해석하지 않는다.
MVTec는 실제 공장 생산성·안전성 검증 자료가 아니다.

## 지금 진행 중 (2026-10-08)

**[정상 대표 균형 비교](docs/mpi-patchcore-balanced-bank-v1.md)를 준비했다.**
정상 학습256장·저장 특징·대표 총78,643개를 유지하고, 대표 선정만12영역별6,553~6,554개로 바꾼다.
검사 시에는 여전히 전체 기준표를 검색하며 같은영역 제한 검색이 아니다. 오경보·불량·검증 사진은 대표 선정에 넣지 않는다.
`scripts/run_mpi_patchcore_balanced_bank.ps1` 한 명령으로 선정·정상개발256/보정169·재로딩·불량55/정상54/검증256·지도·속도·600회 안정성을 비교한다.
기존 개발256 보정 모델(TP55,FP1/54·15/256)이 직접 비교 기준이다. 아직 새 후보의 연구실 실행 결과는 없으며 서비스·이전 경고선은 유지한다.

**[전체 후보 지도 비교 v2 결과](reports/mpi-patchcore-full-map-comparison-v2-received-review.md)는 수치 검증을 통과했다.**
지도 TF32 영향은 확인됐고 최종CPU64 지도는 독립 기준과 최대2.487e-13 차이였다. 정상0706의 영역 점수는11.53% 낮아졌지만 다른 정상 반응은 남았고 일부 결함 위치 구분은 나빠졌다.
전체 후보를 서비스에 적용하지 않는다. 기존 수치 진단은 완료됐으므로 v2 명령을 다시 실행할 필요가 없다.

**[개발자료 확대 결과](reports/mpi-patchcore-development-scaling-v1-received-review.md)는 55/55 검출 유지·오경보 16→15/256이며 전체 목표는 미달이다.**
개발자료 구성에 대한 안정성은 좋아졌지만 경고선만 바꾸어서는 55/55와 오경보 ≤12/256을 동시에 달성할 수 없다.
[정상14장과 불량0024 대조 결과](reports/mpi-patchcore-score-overlap-v1-received-review.md)를 수령했다.
[대표 압축 진단 결과](reports/mpi-patchcore-candidate-coverage-v1-received-review.md)는 정상과 결함의 거리 모두 감소해, 한 점의 개선을 오경보 감소로 해석할 수 없었다.
[동일20타일 전체지도 비교 v2](docs/mpi-patchcore-full-map-comparison-v2.md)는 각 타일4,096개 위치를 대표78,643개와 보관후보3,145,728개에서 조회한 제한 진단이다.
새 사진 판정은 계산하지 않았으며 기존55/55와오경보1/54·15/256은 이전 모델 결과다. 후속 대표 균형 후보는 별도 실험이다.

### 이전 개발자료 확대 준비 기록

**모델은 고정하고 [정상 개발자료128→256장 비교](docs/mpi-patchcore-development-scaling-v1.md)를 준비했다.**
[PatchCore 학습256장 확대](reports/mpi-patchcore-scaling-v1-received-review.md)는 불량55/55 검출에 도달했지만
정상 오경보1/54·16/256 및 보정 안정성은 미해결이다.
[정상23타일 검토](reports/mpi-patchcore-scaling-normal-review-v1-received-review.md)에서는 여러 정상 부위의 반응과
보정 경계의 영향을 구분했다. 추가 정상128장만 추론하고 기존 시험 점수를 재사용해 검출·오경보·600회 민감도를 비교한다.
새 학습·bank 변경·시험 이미지 재열람은 없다. 실제 새128장 명단은 연구실의 고정 풀과 원본을 검증한 뒤 확정한다.
서비스와 이전 경고 기준은 유지한다. 아래는 이전 단계의 기록이며 재학습 명령을 다시 실행하지 않는다.

### 이전 PatchCore 학습 확대 준비 기록

**[SuperSimpleNet 확대 학습 전체 비교](reports/mpi-supersimplenet-scaling-evaluation-v1-received-review.md)를 완료하고 A·C·D 모두 비채택으로 종결했다.**
불량55장 검출은 A4 / C1 / D9장으로 Q90의53장보다 낮다. 정상54/256장 오경보는
A3/17, C1/4, D3/14장이다. 위치 반응·반복 속도·600회 보정 민감도까지 모두 평가했다.
동일 설정의600epoch 연장·추가 확대는 하지 않는다. [다음 설계 기준](docs/mpi-q90-next-design-2026-10-07.md)은
Q90의 미탐0019·0024, 정상 오경보3/54·14/256, 보정 불안정을 개선 대상으로 고정한다.
후속 방향은 같은 bank 크기의 PatchCore 정상64→256장 확대다.
[합성 규모 점검](reports/mpi-patchcore-scaling-probe-v1-received-review.md)은 공식 소규모 선택 대조와
전체314만 후보100회 점검을 통과했다. [실제 확대 실행기](docs/mpi-patchcore-scaling-v1.md)를 준비했다.
원특징 디스크 저장→전체 후보 대표78,643개 선택·재개→정상128/169 보정·재로딩→
전체 검출·오경보·위치·속도·600회 보정 민감도를 한 명령으로 진행한다. 실제 결과는 연구실 실행 후 확인한다.
시간 상한은 없고 메모리·무결성 제한은 유지한다. 기존 Q90은 연구 기준선이며 서비스 교체·자동 PASS는 없다.

[직전 특징 변환부 시작 상태 대조](docs/mpi-supersimplenet-adaptor-start-v1.md)는 한 사진8타일의 제한 진단이다.
[첫 업데이트 점검](docs/mpi-supersimplenet-update-audit-v1.md)의 두 난수에서 동일 구현 반복과
같은 손실을 쓰는 독립 경로가 일치했지만 전체 모델·자료 조건에서의 충분한 학습이나 일반화를 보장하지 않는다.

[검출부 초기화 대조 v2](docs/mpi-supersimplenet-head-reset-v2.md)는500회 완료했으나
같은 위치의 정상·합성 구분 AUROC는0.4996으로 개선 근거가 없다. 후속 시작 상태 대조와 확대 학습은 별도 실험이다.
장시간 재학습 전에 저장된 학습 시작 전·완료 모델에서 동일 입력·가중치·합성 잡음으로
공식 경로와 독립 대조 구현의 출력·손실·기울기·AdamW1회 변화량을 비교했다.
원 논문과의 알려진 보간 차이는 별도 기록한다. backbone은 공유하므로 원 논문 전체 재현이 아니다.
해당 GPU 비교는 위 결과까지 완료했다. 기존 모델·경고 기준·시험 자료·서비스는 유지한다.
아래 항목은 직전 진행 이력이다.

**SuperSimpleNet [고정12장 평가](reports/mpi-supersimplenet-evaluation-v1-received-review.md)는0/12 검출로 실패했다.**
정상 재현 차이0이며 전체 비교·속도·민감도는 미실행이다. 전체55장 검출률0%라는 뜻은 아니다.
저장 지도 진단에서도 실제 결함 위치 반응이 약했다.
[고정 모델 합성 이상 점검](docs/mpi-supersimplenet-synthetic-check-v1.md)에서도 사진·픽셀 구분력이 약했다.
[작은 고정 합성 문제 500회 제한 학습](docs/mpi-supersimplenet-learnability-v1.md)은 완료했으며,
위치 성적이 높아져도 정상 대조에서 같은 위치에 반응하는 문제가 남았다.
이후 [위치와 잡음을 바꾸는 500회 결과](reports/mpi-supersimplenet-varying-synthetic-v1-received-review.md)도 수령했다.
평가용 네 난수 평균 픽셀 AUROC 0.5035, 같은 위치 정상/합성 구분 AUROC 0.4999로 학습 구분력이 약했다.
이후 [잡음 강도 0.03·0.06 각각 500회 제한 비교](docs/mpi-supersimplenet-noise-strength-v1.md)를 진행했다.
같은 정상 한 사진의 8타일·학습률·난수를 유지하고 모든 모델을 공통 세 강도에서 검사한다.
기존 0.015 모델은 재학습하지 않으며 더 강한 합성 문제의 성적만으로 개선을 선언하지 않는다.
새 불량 시험·경고 기준 변경·서비스 전환은 없다.

**SuperSimpleNet [Phase 0·1 실행 준비](docs/mpi-supersimplenet-probe-v1.md)를 추가했다.**
기존 정상64장 중8장×12영역만 사용하는 batch8·100회 GPU 점검이며 정식 학습/불량 평가는 아니다.
연구실 결과 수령 전 CUDA 통과·성능 개선을 주장하지 않는다. 기존 Q90과 서비스는 유지한다.

**[EfficientAD 고정 평가 결과](reports/mpi-efficientad-evaluation-v2-received-review.md)는 기존 검출12장 중0장 유지다.**
정상8장 재현은 차이0이고 실행 오류는 해결됐지만 검출 보존 기준을 실패했다. 전체 비교·반복 속도·
600회 민감도는 미실행이다. 전체 불량의 검출률0%라는 뜻은 아니며 기존 Q90·서비스는 유지한다.
[저장 지도 진단 전체 ZIP](reports/mpi-efficientad-map-diagnostics-v1-received-review.md)을 수령·검산하고 대표 카드를 확인했다.
일부 정상 외관에 ST가 강하게 반응하고 일부 결함에는 약한 반응이 남지만 원인은 미확정이다.
[저장10k·30k·70k 비교 결과](reports/mpi-efficientad-trajectory-v1-received-review.md)를 수령했다.
70k 기존20사례 재현 차이0·입력 감사60건 차이0이며 일부 정상 반응의 후반 증가를 확인했다.
이전 모델의 판정 성능은 아직 미검증이다. 후속 동의로 [10k·30k 별도 보정·회귀검사](docs/mpi-efficientad-checkpoint-evaluation-v1.md)를 준비했다.
각각 정상128/169장으로 보정하고12장 모두 유지한 모델만 전체 비교, 성적·속도 통과 시600회 민감도를 실행한다.
두 저장 시점만 확인하며 새 학습·자동 모델 채택·서비스 교체는 없다. 연구실 신규 결과 대기 중이다.

### 직전 학습·검증 이력

**[EfficientAD 정식 학습 결과](reports/mpi-efficientad-fit-v1-received-review.md)를 수령·검산했다.**
70,000회 업데이트·정상128/169사진 보정을1시간50분21초에 완료했다. 최종297사진 재로딩의
원시 지도·점수·판정 차이는0이며 RAM3.875GiB·GPU reserved0.471GiB로 자원 상한 내다.
학습 손실은 전반적으로 감소했지만 학습 종료 당시에는 불량 정확도를 측정하지 않았다.
[고정 모델 검증 실행기](docs/mpi-efficientad-evaluation-v1.md)를 준비했다. 정상 재현→기존 검출12장→
통과 시 전체55불량·정상54/256 비교와 반복 속도→성적·속도 통과 시600회 보정 민감도를 한 명령으로 진행한다.
첫 실행의0054 복구 이미지 로드 순서 오류를 수정했다. 기존v1을 보존하고 실패 시간까지 합산해
별도v2에서 같은12장을 재검사했고 위 결과를 수령했다. CPU 코드 검증을 CUDA 성능으로 해석하지 않는다.
기존 Q90·서비스는 유지하며 학습 연장·임곗값 변경은 하지 않는다.

**[EfficientAD-S 정상 이미지 사전 점검 결과](reports/mpi-efficientad-probe-v1-received-review.md)를 수령·검산했다.**
정상8사진×12영역·100회 업데이트를11.826초에 완료했다. GPU reserved peak0.473GiB,
RAM peak1.957GiB이며 두 저장·재로딩 검사에서 정상 참조 지도 차이는0이다.
D2 자원·호환성 점검 통과이지 정확도·수렴 검증은 아니다. 후속 D3 실행 조건·도구는 위 안내에 고정했다.
정식 학습·고정12장 시험은 위 결과까지 완료했고 서비스 교체는 미실행이다. 아래 최근접3은 종료된 이전 후보다.

**[최근접 3개 회귀 검사 결과](reports/mpi-knn3-screen-v1-received-review.md)를 수령·검산했고 후보를 종료했다.**
고정 기존 검출12장 중10장만 유지하고0025·0034를 놓쳤다. 309장 원 모델 점수 재현 차이0,
정상 보정·12장 판정 재계산 일치. 실행7분47초·reserved2.916GiB로 자원 내 완료했지만 검출 보존을 실패했다.
후보 재로딩·전체 평가·오경보·보정 민감도는 사전 규칙대로 미실행이다. 기존 Q90과서비스는 유지한다.
이후 [누적 결과와 다음 모델 설계](reports/mpi-model-redesign-2026-10-03.md)를 정리했다.
EfficientAD-S 실행 가능성 점검을 다음 우선안으로 제안하되 성능 우위는 미확인이다.
후속 동의로 사전 점검과 정식 학습을 완료했다. 모델 교체는 미실행이다.

**[패치 특징 정규화 결과](reports/mpi-unit-feature-v1-received-review.md)를 수령·검산했고 후보는 비채택했다.**
불량 검출53→8/55, 정상 오경보3→4/54 및14→13/256이다. 기존 검출45장을 잃고 새 검출은0이다.
959개 사진 기록과 정상297장 재로딩 차이0을 확인했다. 실행19분54초·reserved2.928GiB로 자원 내 완료했지만
고정 성적을 실패해 추가 재표집은 생략했다. 기존 Q90은 연구용으로 보존하며 서비스와 원본 checkpoint는 바꾸지 않는다.
[직전 영역 경고 구조 비교](reports/mpi-region-structure-v1-review.md)의 정상-only 순위 후보는
40/55 검출로 기존 검출13장을 잃어 비채택했다. 같은 원점수 보정 공식의 추가 탐색은 중단한다.

### 직전 비교 이력

**MPI 사이클1 E단계 정리 완료: 두 개선 후보 모두 비채택, 연구 기준선 보존.**
기준선/후보1/후보2의 불량 검출은53/42/50장(55장 중), 정상256장 오경보는14/12/19장이다.
두 후보 모두 보정 안정성 조건을 실패했다. 후속 [원점수·보정 영향 분리 진단](reports/mpi-diversity-score-decomposition-v1-review.md)도 완료했다.
새 미탐3장은 원점수와 정상 보정의 복합 하락이고, 새 오경보13장은 원점수 상승·폭 축소 영향이 함께 있었다.
[Qn 정상 폭 후보 MPI-C2-01](reports/mpi-qn-scale-v1-review.md)도 CPU 비교·검산을 완료하고 비채택했다.
검출43/55·정상 오경보2/54와12/256으로 기존 검출10장을 잃었고 안정성 조건을 실패했다.
본 계산5.937초·RSS55.09MiB, 전체 테스트600개·52 subtest 통과. 새 학습·서비스 교체는 없다.
사이클2는 후보1개에서 종료하며, 다음은 영역 보정과 공통 사진 경고 기준 구조에 대한 설계 검토 제안이다.
[사이클1 결론](reports/mpi-model-cycle-1-conclusion.md)을 보존한다.

### 연구 진행 이력

**[정상 보정 부분 공유 실험](reports/mpi-scale-shrinkage-v1-review.md)을 완료했고 후보는 비채택이다.**
진행 동의 후 [사이클1 기준선 원장·진단 카드](docs/mpi-model-cycle-1.md)를 검산했다.
검출95%·오경보5%·사진 warm p95 2초는 초기 연구 작업 목표다. 세부 자원·안정성 조건은
새 후보 실행 전에 확정한다. 미탐6장 진단은 완료했고, 새 학습이나 모델 교체는 하지 않는다.

MPI-PCB의 [성능 개선 검토](reports/performance-improvement-review-2026-09-29.md)에서 기존 모델
출력의 영역별 보정을 바꾸면 이미 관찰한 불량 검출이 기준 `33/55`에서 Q90 후보 `53/55`,
원시 점수+MAD 후보 `46/55`로 늘어남을 확인했다. 세 방법의 기존 정상 오경보는 모두 `3/54`다.
시험 결과를 보고 선정한 탐색 후보이므로 독립 성능으로 주장하지 않는다.

연구실 [재현·정상 검증 결과](reports/mpi-score-validation-v1-received-review.md)를 수령했다.
기존406장 점수 차이는0이며 새 정상256장 오경보는 A13장·Q9014장·원시 점수18장이다.
세 방법 모두 관측5% 목표를 초과했다. [다섯 사례 지도 검토](reports/mpi-score-case-review-v1-received-review.md)에서
잔여 미탐은 결함에 반응하지만 경고 기준에 못 미침을 확인했다. [26×26 공간 집계 결과](reports/mpi-spatial-score-v1-received-review.md)는
검출53→49장·정상 오경보3→5장으로 악화되어 비채택이다. 기존 모델·Q90 기준은 유지한다.
[미탐6장 진단 수령 검토](reports/mpi-spatial-miss-review-v1-received-review.md)를 완료했다.
72영역 점수·고정 판정 차이0이며 주요 길쭉한 결함의 포함률이 top1 약70~78%에서 창 평균 약37~41%로
줄어드는 양상을 확인했다. 다른 결함 부위의 약한 반응도 남아 있다. Phase2 제한 진단은 완료하고,
[정상 변동 폭 50% 부분 공유 후보](docs/mpi-scale-shrinkage-v1.md)를 저장 점수로 비교했다.
검출53→42장, 정상 오경보3→2장 및14→12장으로 기존 검출11장을 잃었고 재표집 안정성도 실패했다.
기존 Q90을 연구 기준선으로 보존하되 안정성 문제는 미해결이다.
[정상 표본 영향 분석](reports/mpi-normal-influence-v1-review.md)에서3,000회 반복을 추적했다.
grid3-2 정상 폭의 표본 민감성과 다른 영역의 정상 반응이 사진 경고 기준에 미치는 영향을 확인했다.
[정상10장 원본 검토 결과](reports/mpi-normal-visual-review-v1-received-review.md)를 수령했다.
120타일의 파일·픽셀 해시와10개 모음 이미지를 검증했고 색조·반사·각도·선명도 차이를 관찰했다.
원본 검토 당시0290의 큰 검은 영역과 최고 점수 타일은 일치했지만 픽셀 반응 위치는 미확정이었다.
[보정 정상5장 지도 수령 검토](reports/mpi-normal-map-review-v1-received-review.md)를 완료했다.
60타일의 과거 점수·판정 차이는0이고 카드15개도 재구성 픽셀이 일치했다.0290의 최고 영역에서
상위656개 중451개가 검은 부분/바깥2픽셀 띠와 겹쳤다. 다른 사진은 커넥터 핀·콘덴서 옆면에 반응했다.
이번 제한 진단을 종결하고 [MPI-C1-02 정상64장 다양성 선정 설계](docs/mpi-normal-diversity-v1.md)를 작성했다.
보존 정상553장과 정확 중복을 제외한 후보1,086장을 메타데이터로 확인했다. 같은64장·bank 크기에서
사진 선정만 바꾸고 검출 보존·오경보·보정 민감도·자원 조건을 비교할 계획이다.
[선정 결과 수령 검토](reports/mpi-normal-diversity-preparation-v1-received-review.md)를 완료했다.
1,086장 중64장(기존과3장 겹침)의 명단·순서를 저장 특징으로 재현했다. 최대 제곱거리는65.19% 줄었지만
평균은20.91% 늘어 대표 범위의 장단점이 있다. 준비 계산98.854초,RSS2.243GiB,GPU reserved0.330GiB를 기록했다.
[C단계 학습·정상 보정 결과](reports/mpi-normal-diversity-fit-v1-received-review.md)를 수령·검산했다.
새64장 기준표와128/169장 보정이 완료됐고 재로드297장 점수·판정 차이는0이다. 첫 시도의GPU예약7.715GiB 제한 위반은 유지한다.
저장 모델로 재개한 보정·재로드는2.928GiB에서 완료됐다.
후속 동의에 따라 [D단계 고정 성적·민감도·비용 비교 실행기](docs/mpi-normal-diversity-evaluation-v1.md)를 준비했다.
관찰109장·정상256장,3,000회 정상 재표집,별도 프로세스warm비용의 [D결과를 검산](reports/mpi-normal-diversity-evaluation-v1-received-review.md)했다.
성적·안정성 실패로 비채택이며 C단계 자원 위반도 유지한다. 재학습·자동PASS·서비스 교체는 하지 않는다.

별도 서비스 트랙의 중심은 **동결된 transistor PatchCore 검사 → 사람 검토 → 모의 SOP → 승인 → 모의 티켓**을
반복 가능하고 복구 가능한 하나의 시스템으로 만드는 것이다. 구체적인 범위, 중단 판단, 단계별
완료 기준은 [v1 완성 로드맵](docs/v1-completion-roadmap.md)에 고정했다. 전체 연구와 현재 서비스의
관계는 [데이터 → 모델 → 보정·시험 연결표](docs/data-model-map.md)에서 볼 수 있다.

v1 첫 단계인 재현성 정리에서는 개발·테스트 의존성을 한 설치 명령으로 제공하고,
전체 CPU 테스트를 `python -m pytest -q`로 통일했으며 같은 검사를 GitHub Actions에서도 실행한다.
서비스 트랙의 다음 단계는 기존 모델이나 임곗값을 건드리지 않고, 검사 bundle을 같은 장비의 SQLite 검토
저장소에 안전하게 등록하는 단일 진입점을 만드는 것이다.

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
고정 결과의 [오류 분석과 warm 추론 측정](docs/padim-error-analysis-and-benchmark.md)을 완료했다.
`cut_lead` recall 0.20이 가장 큰 취약점이었고, 모델 1회 로드 시 19.83 images/s를 기록했다.
[오류 분석 및 속도 보고서](reports/padim-error-analysis-warm-v1.md)를 참고한다.
같은 고정 manifest와 평가 규칙을 사용하는 [PatchCore 탐색적 기준선](docs/patchcore-exploratory-baseline.md)을
연구실 Windows에서 fit·평가했다. Image AUROC 0.9904, 고정 임곗값 recall 0.925,
Pixel AUPRO 0.9454였으며 TP 37 / FP 1 / TN 59 / FN 3을 기록했다. PaDiM test 결과를 이미
관찰한 뒤 설정했으므로 확증적 모델 선택이 아닌 탐색적 비교다. PatchCore 오류 시각화와
warm 속도 측정도 완료했다. PatchCore warm 처리량은 12.56 images/s, 전체 sample p95는
95.91 ms였다. 정확도 우선 주력 후보는 PatchCore, 가벼운 비교 baseline은 PaDiM으로 유지한다.
contact sheet 판독 결과 FN 3장은 모두 임곗값 근처였고, 두 `misplaced` 사례는 pixel 예측이
GT와 겹치는데도 image score 기준에서 탈락했다. FP 1장은 제품 밖 배경 표식 반응으로 보인다.
[PaDiM–PatchCore 최종 탐색 비교 보고서](reports/padim-vs-patchcore-transistor-exploratory-v1.md)를 참고한다.
동결된 PatchCore bundle을 사용하는 [단일 이미지 검사 명령](docs/patchcore-single-image.md)을
구현했으며 연구실 Windows smoke test를 남겨 두었다.
모델을 한 번만 로드하는 [로컬 PatchCore 검사 API](docs/patchcore-local-api.md)를 구현하고
연구실 Windows에서 실제 이미지·오버레이를 확인했다. 첫 요청 총 902.63 ms, 상주 RAM
1.891 GiB를 관찰했다. [맥북 로컬 SQLite 검토 이력](docs/review-store.md)도 구현했다.
실제 Windows 산출물의 맥북 등록·체크섬 재검증도 완료했다. [맥북 로컬 작업자 화면](docs/review-ui.md)은
원본·오버레이·히트맵, 모델 점수, 검토 이력을 보여주고 사람이 명시적으로 의견을 저장한다.
실제 등록 검사 1건의 화면 표시를 확인했고 사람의 실제 의견 1건은 로컬 DB에만 남겼다.
[모의 SOP 원문 검색](docs/mock-sop.md)은 사람이 직접 선택한 유형으로 유효 개정판을
조회하고, 선택한 원문 스냅샷을 사람 검토 revision에 묶어 저장한다. 실제 현장 SOP가
아니다. 이어서 [로컬 모의 조치·승인·티켓 흐름](docs/mock-ticket-workflow.md)을 구현했다.
저장된 근거와 같은 검토 revision에서만 보수적인 확인 초안을 만들 수 있고, 명시적 승인과
별도 발행 확인을 거쳐야 외부 전송 없는 모의 티켓 스냅샷이 생성된다.
실제 등록 검사 1건의 전체 로컬 시연도 완료했으며, 개인정보를 제외한 결과는
[완료 보고서](reports/local-review-to-mock-ticket-demo-v1.md), 재현 순서는
[포트폴리오 시연 체크리스트](docs/demo-runbook.md)에 정리했다.
[transistor 3-seed 탐색 반복](docs/transistor-multiseed.md)은 연구실 Windows에서 완료했다.
PatchCore는 seed 42·43·44 모두 TP 37 / FP 1 / TN 59 / FN 3으로 동일했고, 평균 Image AUROC
0.9911, recall 0.925, escape rate 7.5%를 기록했다. PaDiM은 평균 Image AUROC 0.9175,
recall 0.600, escape rate 40.0%였으며 seed에 따른 판정 변동도 더 컸다. 따라서 정확도 우선
주력 후보는 PatchCore로 유지한다. 다만 같은 test를 반복 사용한 탐색 결과이므로 seed 44의
근소한 순위 지표를 근거로 모델을 교체하지 않고, 이미 서비스에 연결한 seed 42 bundle을
동결해 유지한다. [3-seed 결과 보고서](reports/transistor-multiseed-exploratory-v1.md)를 참고한다.
Cable·Metal Nut을 포함한 3품목 고정 비교도 완료했다. 사전 고정한 품목 동일 가중 macro
escape rate는 PatchCore 4.31%, PaDiM 35.85%로 PatchCore가 정확도 우선 포트폴리오 후보가 됐다.
[3품목 결과 보고서](reports/three-category-padim-vs-patchcore-v1.md)를 참고한다. 이어서 seed 42
PatchCore의 [추가 품목 오류 분석과 warm 측정](reports/patchcore-additional-diagnostics-v1.md)을 완료했다.
Cable 미탐은 `cable_swap` 4장과 `poke_insulation` 1장, Metal Nut 미탐은 `color` 1장이었다.
warm sample p95는 Cable 42.04 ms, Metal Nut 25.97 ms였고 설정을 변경하지 않았다. 다음 변경은
[새 미관찰 holdout 계획](docs/unseen-holdout-plan.md)에 따라 독립 데이터에서만 수행한다.
직접 촬영을 대체할 외부 데이터로 Real-IAD 256의 `transistor1`과 `plastic_nut`을 확보했다.
실제 이미지를 확인한 결과 기존 MVTec 품목과 다른 물체이므로 동일제품 holdout으로 재표현하지 않고,
별도 `realiad_transistor1`·`realiad_plastic_nut` 카테고리로 등록한다. 공식 단일 시점 C1 분할,
3개 seed, test 이전 전체 fit 완료 조건은
[Real-IAD PatchCore 프로토콜](docs/realiad-patchcore-protocol.md)에 고정했다.
[Real-IAD 고정 평가](reports/realiad-patchcore-evaluation-v1.md)는 순위화·위치 검출은 유망하지만
현재 고정 image 임곗값의 평균 미탐률이 두 품목 모두 약 40%임을 보여 자동 합격 판정을 보류했다.
[3-seed 오류 진단](reports/realiad-patchcore-multiseed-diagnostics-v1.md)에서 공통 미탐은
`transistor1` 166/469장(35.39%), `plastic_nut` 48/118장(40.68%)으로, seed 불안정
49/469장(10.45%)·4/118장(3.39%)보다 많았다. seed 선택 문제가 아닌 구조적 개선 대상으로 보고,
새 development split과 미관찰 holdout을 고정하기 전에는 crop·특징·image score를 변경하지 않는다.
기존 두 품목의 다른 시점은 동일 표본 누수가 있어 제외하고, 아직 미관찰인 Real-IAD `pcb`와
`terminalblock`을 [후속 development/final 사전등록](docs/realiad-followup-preregistration.md) 대상으로
고정했다. 이미지 ZIP을 열기 전 annotation 경로를 그룹 단위로 봉인했고,
[후속 데이터 등록·감사](reports/realiad-followup-registration-v1.md)도 완료했다. Development 1,302장의
오류·경고·디코딩 픽셀 중복은 0건이며, final holdout 710장은 추출하지 않고 봉인을 유지한다.
가짜 추론 결과를 실험 성능으로 제시하지 않는다.

## v1 마무리 순서

1. 전체 CPU 테스트의 설치·실행·자동 검사를 통일한다.
2. transistor 검사 결과를 같은 장비의 검토 저장소에 안전하게 연결한다.
3. 실패·재시도·중복 요청·검토 revision 변경을 통합 시험한다.
4. 최초 모델 준비와 상주 모델 재사용을 나눠 전체 응답시간과 자원을 측정한다.
5. 시연, DB 백업·복구, 모델 복구, 결과 재생 모드와 문서를 마감한다.

모델 연구는 [재개 조건](docs/v1-completion-roadmap.md#모델-연구-재개-조건)에 따라 범위를 제한한다.
현재 허용한 MPI 보정 후보 검증을 넘는 bank 재학습·새 모델 탐색은 결과 검토 후 결정한다.

Python 3.11 이상. manifest/환경 점검 도구는 표준 라이브러리만 사용한다.
이미지 점검과 전체 테스트에는 Pillow/NumPy가 필요하다. 새 환경에서는 아래처럼 설치한다.
ML 학습·평가용 Windows 라이브러리는 별도 requirements에 고정했다. 아래는 맥북 데이터 도구 실행이다.

```sh
python3 -m pip install -e '.[test]'
python3 scripts/check_environment.py --output artifacts/environment-mac.json
python3 -m pytest -q
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
- [Real-IAD 공식 페이지](https://realiad4ad.github.io/Real-IAD/)
- 데이터 라이선스: CC BY-NC-SA 4.0. 원본 이미지·마스크·파생 시각화는 코드 라이선스와 별도로 취급한다.
- 이미지·모델·DB·비밀정보는 Git에 넣지 않는다. 파생물 공개 시에도 데이터 이용 조건을 확인한다.
- 향후 SOP는 워크플로우 구현용 모의 문서이며 실제 산업 안전 지침이 아니다.

구체적인 검토 근거와 후속 작업은 [계획 검토](docs/plan-review.md), [착수 체크리스트](docs/first-milestone.md)에 있다.
## Real-IAD 후속 final 검증

선택 후보 잠금과 연구실 Windows의 준비/명시적 final 평가·재개 절차는
[`docs/realiad-followup-final.md`](docs/realiad-followup-final.md)에 정리되어 있다.
