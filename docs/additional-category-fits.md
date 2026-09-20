# Cable·Metal Nut 학습/보정 단계

Cable과 Metal Nut 데이터 등록은 완료됐다. 두 manifest와 데이터 점검 보고서는 Windows에서 만든
handoff ZIP의 SHA-256을 확인한 뒤 저장소에 반영했다.

| 품목 | train 정상 | calibration 정상 | test 정상 | test 이상 | Manifest SHA-256 |
|---|---:|---:|---:|---:|---|
| cable | 179 | 45 | 58 | 92 | `f2caf5c39f7fe53060278d2a000634db0c62781b4e81d8d9d984eb4abc73dd70` |
| metal_nut | 176 | 44 | 22 | 93 | `6270aecb5f243662a1f6110078f34a5f76846f865510048d68391c0bcfee6be5` |

첫 실행은 fit/calibration만 수행한다. 품목별 PaDiM과 PatchCore를 seed 42·43·44로 실행해 총 12개
모델과 normal-only 임곗값을 만든다. 이 단계에서는 공식 test 이미지를 모델에 전달하지 않는다.
모든 fit이 끝난 것을 확인한 다음 별도 고정 평가 단계에서 test를 한 번 연다.

연구실 Windows에서 저장소를 최신 commit으로 받은 뒤 다음만 실행한다.

```powershell
Set-Location "C:\개인 프로젝트\defectops"
git pull --ff-only

powershell.exe -NoProfile -ExecutionPolicy Bypass `
    -File ".\scripts\run_additional_category_fits.ps1"
```

스크립트는 다음을 자동으로 확인한다.

- Git 작업 폴더가 깨끗한지
- Python/CUDA/Anomalib GPU 기본 연산이 정상인지
- 로컬 원본이 커밋된 두 manifest와 완전히 같은지
- 기존 결과가 정상 완료본인지
- 12개 fit이 모두 존재하는지

중간에 중단돼도 완료된 결과는 재사용한다. 단, 일부 파일만 남은 불완전한 출력은 자동 삭제하거나
덮어쓰지 않는다. 그 경우 해당 경로를 보존해 원인을 확인하고 이름을 바꾼 뒤 다시 실행한다.

완료 문구는 `Completed 12 fits: 2 categories x 2 models x 3 seeds.`다. 이 문구가 나온 뒤 콘솔
마지막 부분을 이 작업에 전달한다. 다음 단계는 고정 평가 스크립트 실행과 3품목 macro 비교다.
