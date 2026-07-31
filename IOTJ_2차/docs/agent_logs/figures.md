# 논문 그림 생성 — Fig 3/4/5/6/7 (Fig 8 HOLD)

- 실행: 2026-07-29 · HEAD `d7540ec` (= origin/main, clean) · matplotlib 3.10.9
- 신규 코드: `tools_fig/_style.py` + 그림당 1 스크립트 5개. **`tools/` 무수정, 새 계산 없음.**
- 산출: `figures/*.pdf` (본문용) + `figures/*.png` (300 dpi 검토용) + `figures/manifest.json`
- 그림 텍스트는 **전부 영문**이다 — 지시서가 지정한 DejaVu Sans 에 한글 글리프가 없다.

---

## [측정된 사실]

### Phase 0 게이트

| 게이트 | 결과 |
|---|---|
| 0-1 git | HEAD `d7540ec` = origin/main, 미추적 0 |
| 0-2 키 실측 | 아래 §키 확인 표 — Fig 8 을 제외한 전 필드 존재 확인 |
| 0-3 matplotlib | 3.10.9, 스타일 블록 경고 없이 적용 |
| 0-4 제외 목록 | 각 스크립트 상단 상수로 하드코딩, 필터 통계를 stdout 출력 |
| 0-5 결정성 | **PNG 5/5 비트 동일**, PDF 는 `/CreationDate` 제외 시 5/5 동일 |

### 키 확인 (경로:키 실측)

| 그림 | 확인된 경로 | 결과 |
|---|---|---|
| 3 | `gap_cf.json → sweeps.inf[].{lam,cf,kl_nc_floor,kl_quantum,gap_contextual,gap_matched,gap_seedwise,gap_ctx_seedwise}`, `self_check_cf[].cf_lp`, `self_check_maxdiff` | 전부 존재 |
| 4 | `gate_e.json → results.<scn>.<dir>.{points[].cf, points[].ncfloor, alpha, c_fit, c_pred_limit, r2_fit, regime, smooth}` | 존재. **단 `c_pred` 는 없고 `c_pred_limit`** |
| 5 | `r2.json → anchor_placement.<k>.{cf,gap,curve_pred,rel_dev}`, `track_A_kcbs.cf_ci`, `track_B_delft.cf_ci`, `reference_sweep[]`; `r3.json → cells[].{cf,gap_upper,definition}` | 전부 존재 (편차 3.7/5.3% 은 `rel_dev` 로 저장돼 있었다) |
| 6 | `s3b.json → cells[].{plugin,ns,quantum}.{bias,mse,ci,ci_contains_true}`, `cells[].cf_true` | 전부 존재 |
| 7 | `s8_phaseB1.json → results[].{V,CF,h_S,h_full,dH,status_S,status_full}` | 전부 존재 |
| 8 | `s6.json → e3_collapse[]`, `s7.json → n_star` | **불충분 → HOLD** (아래 참조) |

### 지시서 인용값 재확인 (전부 일치)

- Fig 3: λ=0.8 → NCfloor **0.07246**, quantum **0.009539** (지시서 0.0725 / 0.0095) ✅
- Fig 4: α = **2.0057 / 2.0033 / 2.0066 / 2.0079** (quadratic), **1.0021 / 1.0065** (linear);
  R² = **0.99997~0.99999** / **0.78392 / 0.79157** ✅ 전부 지시서와 일치
- Fig 5: KCBS CF **0.2629** gap **0.012325** rel_dev **3.68%**; Delft CF **0.21125** CI **[0.11506, 0.41849]**
  gap **0.008463** rel_dev **5.28%**; N-BaIoT **15 + 3 = 18** 셀 ✅
- Fig 6: n=245 delft plugin **+0.147958**, projected **+0.040266** (지시서 +0.148 / +0.040) ✅;
  V=1.00 quantum bias **−0.070128**, `ci_contains_true=False`, projected **+0.005538** `True` ✅
- Fig 7: 상대격차 **7.78 / 9.31 / 7.63 / 2.94** (4행). 운용점 CF=0.210567 에서 **+0.019395 bits (9.31%)** ✅

### 제외 규칙 적용 결과

| 그림 | 사용 | 제외 | 사유 |
|---|---|---|---|
| 3 | 16 | 16 | `sweeps['5000']` 유한표본 행 |
| 4 | 6 | 0 | V=1.000·generic_23 규칙이 공허(gate_e 에 해당 행 없음) |
| 5 | r2 앵커 2 + 참조 21, r3 18 | 0 | — |
| 6 | (a) 5, (b) 2 | 55 | V=1.00 은 (b) 전용, 그 외 V·uniform 미사용 |
| 7 | **4** | 44 | generic_23 / x\*=2 / level 1+AB / V=1.000 / **V=0.750(status inaccurate)** |

---

## [탐색의 한계]

### ★ Fig 7 — 지시서가 예상한 5행이 아니라 4행이다

지시서는 `box=CANON_23 ∧ x_star=0 ∧ level=2 ∧ V≤0.96 → 5개 행` 이라고 썼으나,
같은 지시서의 제외 규칙 *"status 에 `optimal_inaccurate` 포함 행도 제외"* 가 **V=0.750**
(`status_S=optimal`, `status_full=optimal_inaccurate`)을 추가로 떨어뜨린다. 따라서 **4행**이다.
지시서가 나열한 상대격차 5개 중 **4.61%(=V=0.750)** 는 그리지 않았다.

B-1 의 `recheck_multisolver` 가 CLARABEL 로 같은 셀을 재해해 `h_full=0.049540`, status `optimal`
을 얻은 기록은 있으나, **그 셀의 행으로 저장돼 있지 않다.** "새 계산 없음" 규칙을 지켜
사용하지 않았다. 되살리려면 그 값을 정식 행으로 저장하는 별도 작업이 필요하다.

### ★ Fig 7 — 최대 구간 괄호가 지시서보다 넓다

지시서는 최대가 **CF 0.301~0.358** 에 있다고 했다. 실측 dH 는 CF=0.301076(V=0.920)에서
**최대(0.027741)** 이고 양옆이 0.019395(0.210567)·0.015236(0.357645) 이다. 단봉 함수의 연속
최대는 이산 최대점의 **양옆 이웃**으로 괄호되므로 옳은 구간은 **[0.210567, 0.357645]** 이다.
지시서의 좁은 구간은 dH(0.920) < dH(0.960) 일 때만 성립하며 실측은 그 반대다.
넓은 쪽을 그렸고, 지시대로 **최대점 마커·peak 라벨은 넣지 않았다.**

### ★ Fig 5 — 예측곡선 정의가 편차 수치와 어긋난다

지시서는 `CF²/6` 을 주 예측곡선으로 지정했다. 그런데 r2.json 의 `rel_dev`(3.68% / 5.28%)는
`curve_pred` 기준이고, `curve_pred` 는 **수치 `reference_sweep` 보간값**이다. 그 곡선의 국소
계수 NC/CF² 는 0.171 → 0.288 로 드리프트하며 앵커 위치에서 **0.185(KCBS) / 0.180(Delft)** 다.

`CF²/6`(=0.1667·CF²) 기준으로 읽으면 편차가 **7.0% / 13.8%** 가 되어 인용값과 다르다.
따라서 두 곡선을 모두 그렸다 — `reference_sweep`(실선, rel_dev 의 기준)과 `CF²/6`(얇은 회색
파선, gate_e 균등방향 경계극한). 캡션에서 3.7/5.3% 가 **실선 기준**임을 명시해야 한다.

### 기타

- **Fig 3 seed 산포**: `gap_contextual`·`gap_matched` 에만 seedwise 배열이 있다.
  `kl_nc_floor`·`kl_quantum` 은 seedwise 미저장이라 **band 없이 선만** 그렸다(지시서 규칙 준수).
- **Fig 3 x축 결정**: CF 단일축으로 갔다. CF=0 인 7행(λ≤0.50)이 x=0 에 겹치며, 그 지점에서
  parameter-matched 계열만 0 → 0.0654 로 세로로 퍼진다. 이것이 "CF=0 에서 matched gap>0"
  의 시각 증거다. λ 보조축은 **달지 않았다** — CF=0 구간은 CF 축에 표현 불가하다.
- **Fig 6 에러바 변환**: s3b 의 `ci` 는 CF 추정치의 2.5/97.5 분위수다. bias 축에 올리려고
  `ci − cf_true` 로 변환했다. 마커 채움/비움은 `ci_contains_true` 를 직접 따른다.
- **Fig 4 기준선**: 기울기 1·2 기준선은 임의 위치가 아니라 실측 스케일에 정박했다
  (slope-2 → quadratic 4방향 `c_fit` 중앙값 0.1489, slope-1 → linear 2방향 NC/CF 중앙값 0.1907).
- **Fig 5 CI 상단**: 지시서는 KCBS CI 상단을 0.278 로 적었으나 저장값은 **0.27954**
  (0.280 으로 반올림)다. 저장값을 썼다.

### ★ Fig 8 — HOLD, 그리지 않았다

지시서의 조건부 규칙(*"개별 값이 없으면 그림을 만들지 말고 보고"*)을 따랐다. 다만 사유가
"개별값 미저장" 보다 복잡하다:

1. `s7.json → n_star` 에는 `true_template.n_star = 980`, `fitted_template.n_star = 980` 둘뿐이고
   **KL 이 붙어 있지 않으며 세 번째 경로가 없다.**
2. `s6.json → e3_collapse` 에는 5행이 있고 각각 `M`·`kappa`·`n_star`·`nM_star` 를 갖는다.
   그런데 이 5개는 **경로가 아니라 가시도 V 5점**이다.
3. 인용된 세 곱 **9.06 / 9.87 / 10.31** 은 저장된 삼중항이 아니다. 9.06·9.87 은 s6 의
   `nM_star` (V=0.856, V=0.80) 이고, **10.31 은 s6·s7 에 없다** — `s7_phase1.json phase1D` 의
   상수 A (N\* ≈ A/KL) 로 **다른 양**이다.
4. 기준선 **N\* = 9.30/KL** 의 9.30 은 어디에도 없고 저장값으로 유도되지 않는다
   (s6 `nM_star` = 8.987, 9.867, 9.060, 11.119, 11.547).
5. **s6 자신이** `e3_summary.formula = "붕괴 미확립 — N*(V) 테이블로만 보고"` 로 결론했다.
   1/KL 기준선을 그리는 것은 출처의 결론과 모순이다.
6. `KL_max ∝ CF^1.87` 인셋은 `s7_phase2c.json` 이 필요한데 이 그림의 선언된 소스가 아니다.

지시서대로 **표로 대체 권고**한다.

---

## [미검증·추정]

- **Fig 7 의 4행이 논문에 충분한가**: 4점으로 내부 최대를 갖는 곡선을 주장한다. 최대의 위치는
  격자 해상도로 특정 불가이며, **곡선 형태(단봉성) 자체도 4점으로는 가정**이다. 더 촘촘한 V
  격자가 필요하다.
- **Fig 5 두 곡선 중 어느 것을 논문 주 예측선으로 쓸지**: `reference_sweep` 은 수치이고
  `CF²/6` 은 해석적 경계극한이다. 앵커 편차를 인용하려면 전자, 이론 서술과 맞추려면 후자다.
  이 선택은 논문 서술에 달렸고 **본 작업에서 결정하지 않았다.**
- **Fig 6 (a) 에서 projected 와 quantum 이 거의 겹치는 것**: n≥500 에서 소수 4자리까지 같다.
  Gate S3b 로그의 "151/300 rep 정확 동일" 과 정합하나, 겹침의 원인(사영점이 이미 양자집합
  안) 은 **미검증 가설**이다.
- **Fig 8 대체 그림의 정당성**: `N*` vs `M` (log-log, 5점) + `nM_star` 라벨 + κ 드리프트
  (1.750 → 0.763) 표기가 저장값만으로 가능하다. 그러나 이는 지시된 주장과 **다른 주장**이라
  승인 없이 그리지 않았다.

---

V=1.000 행과 generic_23, plugin 유령비트는 전 그림에서 제외했다.
