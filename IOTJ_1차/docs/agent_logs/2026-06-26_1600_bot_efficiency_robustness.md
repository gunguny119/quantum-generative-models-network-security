# Bot 양자 효율 우위의 견고성 검증 — 216p QCBM 우위가 best 운인지 전형인지 (재학습 없음)

- **작성일시**: 2026-06-26 16:00 (로컬)
- **작성 주체**: 로컬 Claude Code agent (plan→default)
- **작업 성격**: 저장된 16 restart AUC 분포를 고전 곡선과 정밀 비교. **QCBM 재학습 없음. 기존 소스/데이터/결과 무수정.**

## 1. 작업 목적
직전 작업에서 "Bot 216p에서 QCBM best(0.7458) > 고전 최고(비트그룹 6+6, 0.706)"가 나왔으나, 이는
**best 1개 기준**이다. QCBM 16 restart는 median 0.665(고전 최고 0.706보다 낮음)·std 0.071. 따라서
"216p 우위"가 **(a) best 선택의 운인지, (b) 분포적으로도 성립하는지** 미확정. (no_psh 전례: best만
상한 초과, median은 아래.) 본 작업은 16 분포를 고전선과 비교해 **견고성을 사실로 규명**한다. 판정은 웹 AI.

## 2. 시작 git status / 보존 대상
```
?? docs/  ?? qcbm/  ?? qcbm_share_20260626.zip  (추적 파일 변경 0)
```
- 새 생성만: `tools/verify_bot_efficiency_robustness.py`, `results2/bot_efficiency_robustness.json/.png`, 본 `.md`.
- 무수정: 모든 기존 소스/데이터/results2 기존 결과.

## 3. 재현/정합 (스크립트가 수행)
- npz `qcbm_bot_restarts.npz` keys: seeds, p_finals(16,4096), aucs, mmd2, tv, weights, q_train. (states 미저장 → `TBOT.build_bot()` 재현, q_train allclose + occ 361 게이트.)
- 16 p_final로 AUC 재계산 → npz aucs 및 qcbm_bot.json 분포(median 0.665)와 정합 검증. 불일치 시 중단.
- (직전 작업서 재계산 diff 0.0 확인된 경로 재사용.)

## 4. 고전 비교 기준 (param_efficiency_bot.json 로드, 재생성 안 함)
- **216p 이하 고전 최고 = 비트그룹 6+6 (126p, AUC 0.7064).** (그 외 SVD/NMF r1=128p 0.593.)
- 인접 참고점: 8+4(270p, 0.796), SVD/NMF r3(384p, 0.740), r4(512p, 0.806/0.803), emp_joint(361p, 0.875).

## 5. 확실한 사실 / 추정 / 확인 불가 (사전 확인값)
- **확실**: QCBM 16 AUC = [0.524,0.594,0.604,0.613,0.614,0.620,0.635,0.656,0.674,0.677,0.706(s6=0.7059),0.721,0.726,0.736,0.746,0.817].
  분포 min0.524/median0.665/75th0.722/90th0.741/max0.817/std0.071. best(min-MMD² seed7)=0.7458, TV 0.4933.
  corr(TV,AUC)≈**−0.01**(거의 0). 고전 216p-이하 최고 0.7064.
- **추정(스크립트로 확정)**: 고전 최고(0.706) 초과 restart ≈ 5/16; median<고전최고 → "median 우위" 불성립.
- **확인 불가**: 없음(정합 통과 예정).

## 6. 견고성 지표 (실측)
### 정합
- 16 p_final로 AUC 재계산: **max|Δ| = 0.0e+00** vs npz aucs; median 재0.6650=json 0.6650, max 재0.8167=json 0.8167. occ 361. → 통과.

### 핵심 지표
- **고전 216p-이하 최고 = 비트그룹 6+6 (126p, AUC 0.7064).**
- **이를 초과한 QCBM restart: 5/16 (31%)** — 0.7207, 0.7259, 0.7359, 0.7458, 0.8167.
  (s6=0.7059는 0.7064 바로 아래로 미달.)
- QCBM AUC 분포 vs 고전최고(0.7064):

| 통계 | 값 | 고전최고(0.7064) 대비 |
|------|----|----------------------|
| min | 0.5235 | 아래 |
| p25 | 0.6135 | 아래 |
| **median** | **0.6650** | **아래** |
| p75 | 0.7220 | **위** |
| p90 | 0.7409 | **위** |
| max | 0.8167 | 위 |
| best(min-MMD², seed7) | 0.7458 | **위** (순위 상위 2/16) |
| std | 0.0709 | — |

- **best 우위: 예. p90/p75 우위: 예. median 우위: 아니오(아래).**
- **(d) 고전이 QCBM median(0.665) 도달 필요 params**: any-최소 **126p (비트그룹 6+6, <216!)** / SVD·NMF 384p.
  → 중앙값 기준으론 고전(비트그룹)이 **더 적은 params(126<216)**로 도달 = median에서 양자 우위 없음.
- (참고) 고전이 QCBM best(0.746) 도달: 비트그룹 8+4 **270p(>216)**.

### (e) 모델선택 견고성 — 중요
- **corr(MMD², AUC) = 0.077**, **corr(TV, AUC) = −0.014** (둘 다 거의 0).
- 실제 선택규칙 = **min MMD²(라벨 무사용)** → 이번엔 seed7(AUC 0.746, 상위 2/16)를 운좋게 골랐으나,
  MMD²↔AUC 상관이 ~0이라 **선택규칙이 고AUC restart를 보장 못 함**. 상관 0이면 기대 픽 ≈ median(0.665, 고전최고 아래).

## 7. TV-AUC 해리 (restart 전반)
- QCBM TV 범위 **0.493~0.703**(전부 나쁨). corr(TV, AUC)=**−0.014** → **분포적합(TV)과 탐지(AUC) 무관.**
- 같은 params 고전은 TV가 훨씬 낮음(SVD r4 512p TV 0.19)인데 AUC는 QCBM 상위권과 비슷 → QCBM 우위는 "적합도"가 아니라 "분리"에서, 그리고 그마저 상위 일부 restart에서만.

## 8. 사실 관찰 (판정 아님 — 웹 AI 몫)
- **216p 우위는 best 전용이 아니라 "상위 약 1/3(5/16, p75·p90 포함)"에서 성립**하나, **분포 중앙값(median 0.665)에서는 불성립**(고전 6+6 0.706이 더 높고 params도 126<216으로 더 적음).
- **추가로 선택 취약**: 라벨없는 실제 선택규칙(min MMD²)이 AUC와 거의 무상관(corr 0.08)이라, 우위 restart를 고르는 것이 보장되지 않음(이번은 운으로 상위 2/16 선택).
- **한줄 요약**: Bot 216p 양자 우위는 **"상위 분위에서만 성립하는 부분적/취약(best·운 의존)"** 우위다. 전형적(median) restart로는 고전이 더 효율적이며, 무라벨 모델선택이 우위를 신뢰성 있게 포착하지 못한다. (no_psh 전례 — best만 우위, median 열위 — 와 동일 패턴.)

## 9. 생성 파일 / 실행 명령
- 생성: `tools/verify_bot_efficiency_robustness.py`, `results2/bot_efficiency_robustness.json`(5.8KB), `results2/bot_efficiency_robustness.png`(152KB), 본 `.md`.
```bash
python3 -m py_compile tools/verify_bot_efficiency_robustness.py     # PY_COMPILE_OK
python3 tools/verify_bot_efficiency_robustness.py                   # DONE
#  정합 max|Δ|=0.0 | 고전최고 0.7064 초과 5/16 | median 0.665 아래 | corr(MMD²,AUC)=0.077
```
- QCBM 재학습 없음. 방향반전 없음(raw MWU). 새 dependency 없음. 고전 곡선은 param_efficiency_bot.json 로드(재생성 안 함).

## 10. git diff --stat (기존 무변경)
```
git diff --stat → (빈 출력 = 추적 파일 변경 0)
```
- 기존 소스/데이터/기존 results2 무수정. 새 untracked만 추가.

## 11. 성공 기준 충족 / 남은 불확실성 / 다음 추천
- 충족: 16 AUC 재계산 정합 ✓ / 고전최고 초과 수 5/16 보고 ✓ / 분포(min~max·분위) vs 고전선 기술 ✓ /
  best·상위·median 구분 ✓ / median 도달 고전 params(126) 역산 ✓ / json·png 저장 ✓ / 무수정·무재학습·무설치 ✓.
- 남은 불확실성:
  - 16 restart는 표본이 작음(std 0.071) — 더 많은 restart면 5/16 비율·median 추정이 흔들릴 수 있음.
  - best 선택을 min-MMD²로 했으나 그 규칙이 AUC와 무상관 → "실전에서 우위를 뽑을 수 있는가"는 별도 문제.
- 다음 추천:
  1. (우위가 부분적/취약으로 확인됨) **무라벨 모델선택 규칙 개선** 탐색: MMD² 외에 AUC와 상관 높은
     무라벨 지표(예: validation benign-only likelihood gap)가 있는지 — 있으면 우위가 실전화 가능.
  2. 또는 **restart 수 늘려** median 자체를 고전최고 위로 올릴 수 있는지(앙상블/더 깊은 ansatz).
  3. 웹 AI 최종 판정: 현 사실(상위 1/3만 우위, median 열위, 선택 무상관)로 후보 A의 "견고한 양자 우위" 성립 여부.
