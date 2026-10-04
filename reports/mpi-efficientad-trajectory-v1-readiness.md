# EfficientAD 저장 시점 비교 실행 준비 검증

2026-10-04. [실행 계획·PowerShell 안내](../docs/mpi-efficientad-trajectory-v1.md).

## 구현과 검증

- 추가 학습 없이70k 재현→10k·30k 동일20사례 원시 지도 비교. 이전 checkpoint 경고 판정은 계산하지 않는다.
- 학습64사진의 전체768타일 모음48장과 시점 지도40장, 요약·receipt·HTML을 한 ZIP으로 전달한다.
- 실제 수령 fit/evaluation ZIP에서20사례 선정이 이전 지도 진단과 같음을 확인했다.
- 이전 학습 소스137개, 평가 소스141개 SHA256 변경0. 기존 frozen 문서를 수정하지 않았다.
- 신규 CPU 테스트14개 통과. 원시 통계·동점·0006 제외·최종 재현 실패 차단·checkpoint 지문 변조 거부·
  Ctrl+C 재개·오류 후 자동 재시도 차단·실제 카드/ZIP 생성·완료 캐시 변조 검출·768타일 전체 모음·CLI를 검사했다.
- 가짜 GPU runtime으로20사례×3시점의 전체 저장→그림→ZIP→재개 흐름을 실행했다.
  가짜 자료는 테스트용이며 정확도·속도 측정이 아니다. 실제 렌더러의 공통 색상·배치도 확인했다.
- 전체 `python -m pytest -q`: **761 passed,52 subtests passed**,115.33초.
  경고3개는기존 Starlette/AnyIO 폐기예정 안내와 중복ZIP 거부용 테스트2개다.
- `git diff --check` 통과. 사용자가 만든 별도 health review2개는 수정·추적하지 않았다.

## 한계

Mac에는 CUDA·PyTorch·PowerShell이 없어 실제 Windows 모델 로드·GPU tensor 비교·원시 지도 재현은 실행하지 않았다.
이를 연구실 실행기의 초반 필수 검사로 두었다. 내부 PDN의 정규화는 고정 공식2.2.0 소스를 확인했고,
실제 설치 파일은 fit 당시 소스 지문과 일치해야 실행된다.

현재는 준비 완료이지10k·30k의 검출 성능이나 과적합 여부가 확인된 상태가 아니다.
새 임곗값·최적 checkpoint·새 학습·서비스 배포는 이번 범위에 없다.
