# Real-IAD 후속 데이터 등록·감사 v1

상태: **통과 — development 사용 가능, final holdout 봉인 유지**

검증일: 2026-09-22

등록 handoff ZIP SHA-256:
`d2617f84ae1a6ddc76b2fec6c16383678787f0ed082efdf05616cadcf9742797`

## 검증 결론

연구실 Windows에서 Real-IAD `pcb.zip`과 `terminalblock.zip`을 고정 분할 계획에 따라 등록했다.
첨부된 handoff ZIP의 압축 무결성, 정확한 9개 파일 구성, JSON 자체 체크섬, 고정 계획과의 레코드
대응, development 감사 결과, final 봉인 계약을 재검증했다. 총 3,062개 조건이 모두 통과했다.

- handoff에는 계획·manifest·감사 보고서만 있고 이미지 파일은 없다.
- development 이미지 1,302장과 불량 마스크 421장을 모두 디코딩했다.
- 디코딩 오류, 경고, 픽셀 중복은 모두 0건이다.
- 모든 이미지가 `1024x1024/RGB`, 모든 마스크가 `1024x1024/L`이다.
- final holdout 710장은 원본 ZIP member 체크섬만 기록됐고 추출되지 않았다.
- 고정 분할 계획 SHA-256은
  `c68562288b378d44d969c3eda68a69a00c6c31e8dad0f4ff21e5c05073ac8f20`과 일치한다.

## Development 등록 결과

| 카테고리 | Train 정상 | Calibration 정상 | Development 정상 | Development 불량 | 합계 | 감사 |
|---|---:|---:|---:|---:|---:|---|
| `realiad_pcb` | 223 | 56 | 139 | 224 | 642 | passed |
| `realiad_terminalblock` | 247 | 62 | 154 | 197 | 660 | passed |
| 합계 | 470 | 118 | 293 | 421 | 1,302 | 오류·경고·중복 0 |

| 카테고리 | 원본 ZIP SHA-256 | Development manifest SHA-256 |
|---|---|---|
| `realiad_pcb` | `8fbcb2f20000501800d0baf3880fd0df5f5b53a51d4b5a244a185bdb839059ab` | `2d16e21b2e665defbf54266c1d5e8b08f41838dfae679fff8f65dff615b49ed4` |
| `realiad_terminalblock` | `6e3efd9a7588cbf9689b113cb67a2fa0ded4ddd6a11af710f7eb0e3d5f9cef7f` | `a5f6e45935e2a16f64a23f4757fa383f569efec84b25e7ed3ddf197456c2b338` |

## Final holdout 봉인 결과

| 카테고리 | 정상 | 불량 | 합계 | Final manifest SHA-256 |
|---|---:|---:|---:|---|
| `realiad_pcb` | 139 | 221 | 360 | `9b5a319600967c8e73bf3e4b53d0b31c9e7df9ff7261222c31db489b435478d2` |
| `realiad_terminalblock` | 154 | 196 | 350 | `b2ff0bc8f52aeed8db97ce853869ecc9ec6a46d0bc939b1c4442a7d31507891b` |

두 manifest 모두 `sealed=true`, `images_extracted=false`이고 다음 네 값이 고정되기 전에는 final을
열 수 없다.

1. 선택 후보 ID
2. 선택 보고서 SHA-256
3. 선택 시점 Git commit
4. final manifest SHA-256

## 해석 한계와 다음 단계

이번 통과는 데이터 파일의 디코딩, 마스크 형식·대응, 체크섬, 분할 누수 방지 계약을 확인한 것이다.
마스크 경계의 의미 정확성이나 모델 성능을 검증한 결과는 아니다. 다음 단계는 final을 열지 않은 채
development에서 사전등록한 PatchCore 후보 네 개를 구현하고 비교하는 것이다.
