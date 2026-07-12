# CF² 논문 V장 핵심 수치 5개 추출 (조사 전용, 코드/결과 무수정)

- 작성: 2026-07-08 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **읽기 전용** — 코드·결과 무수정·무재계산. 본 .md 1개만 신규.
- 목적: 논문 V장 플레이스홀더 5개를, 이미 저장된 실험결과(`gap_cf`·`kink`·`gate_e`)에서 **출처와 함께** 추출. 추측·보간 금지.

## 시작 git status
`git -C /home/elicer/IOTJ status --short` = 변경항목 6(최근 R3/R4/R5 untracked 산출물; 본 조사와 무관). 본 조사로 인한 파일 변경 없음(로그 1개만 추가).

## 확인한 파일 구조
`results2/gap_cf/{gap_cf.json, gate0.json, gap_cf.png, run_full.log}`, `results2/kink/{kink.json, kink.png, run.log}`, `results2/gate_e/gate_e.json`. 코드: `tools/kink_origin_v1.py`, `tools/gate_e_coefficient.py`.

---

## [값 1] CF=0.6 (λ=0.8) triple
출처: `results2/gap_cf/gap_cf.json` → `sweeps.inf[λ=0.8 셀]`의 각 key.
| 항목 | 값 | json key |
|---|---|---|
| **NCfloor** (비맥락 floor) | **0.0725** | `kl_nc_floor` |
| **KL_quantum** (QCBM 도달) | **0.0095** | `kl_quantum` |
| **gap** (= NCfloor − KL_q) | **0.0629** | `gap_contextual` (=0.0725−0.0095) |

## [값 2] HMM hidden-state 사다리 KL
출처: `gap_cf.json` → `sweeps.inf[i].per_model["H{h}"]["mean"]` (held-out KL 평균, seed 5). NCfloor=`kl_nc_floor`.
- ★ 실제 실행 H = **{1, 2, 3, 4, 6, 8}** (`gap_vs_cf_v0.py` `H_LADDER`). **H5·H7 은 미실행 → 저장 없음**(추측 금지).

**CF=0.6 (λ=0.8) 지점** (요청):
| H | 1 | 2 | 3 | 4 | 6 | 8 | (NCfloor) |
|---|---|---|---|---|---|---|---|
| KL | 0.3681 | 0.1840 | 0.1129 | 0.1079 | 0.0823 | 0.0777 | 0.0725 |

**다른 CF 지점(저장분 전체, N=∞):**
| CF(λ) | H1 | H2 | H3 | H4 | H6 | H8 | NCfloor |
|---|---|---|---|---|---|---|---|
| 0.0 (0.5) | 0.1308 | 0.0654 | 0.0163 | 0.0040 | 0.0014 | 0.0000 | 0.0000 |
| 0.2 (0.6) | 0.1927 | 0.0964 | 0.0367 | 0.0363 | 0.0113 | 0.0070 | 0.0070 |
| 0.4 (0.7) | 0.2704 | 0.1352 | 0.0702 | 0.0502 | 0.0358 | 0.0298 | 0.0298 |
| **0.6 (0.8)** | **0.3681** | **0.1840** | **0.1129** | **0.1079** | **0.0823** | **0.0777** | **0.0725** |
| 0.8 (0.9) | 0.4946 | 0.2473 | 0.1689 | 0.1759 | 0.1483 | 0.1509 | 0.1441 |
| 1.0 (1.0) | 0.6931 | 0.3466 | 0.3144 | 0.2969 | 0.2954 | 0.2892 | 0.2877 |
- (유한표본판 N=5000 은 `sweeps["5000"]` 에 별도 저장.)

## [값 3] CF² fit 지수 α (+ fit 방법)
**두 실험에서 방식이 다름 — 둘 다 log-log 기울기이고, 비선형 최소제곱(자유 α) 아님.**

**(a) kink_origin_v1 — 국소 차분:** `local_exponent` = **log-log 중심차분**(수치미분), 창 2/4. 코드 `tools/kink_origin_v1.py:74-83`.
- 저장값: `kink.json verdicts.H1_geometry.alpha_boundary_median` = **2.202338441039515**.
- ※ 이 "median" 은 넓은 경계구간 평균이라 2.20; **가장 작은 CF에서의 국소지수**(`gateA.local_exponent["2"]`)는 CF=0.06→**2.012**, 0.10→2.023 (경계극한 α→2에 근접).

**(b) gate_e_coefficient — 전역 회귀:** `alpha = np.polyfit(log CF, log NC, 1)[0]` (경계점 CF≤0.1 전역 log-log 선형회귀). 코드 `tools/gate_e_coefficient.py:159`. (c는 별도로 α=2 고정 후 `NC=c·CF²` 적합.)
- 저장값 `gate_e.json results.<scen>.<dir>.alpha` (방향별):
  - CHSH/uniform **2.0057**, CHSH/det_vertex_0101 **2.0033**, KCBS5/uniform **2.0066**, KCBS5/det_vertex_all0 **2.0079** (CF² 레짐)
  - CHSH/det_vertex_0000 **1.0021**, CHSH/local_perfect_corr **1.0065** (선형 레짐)

→ **CF² 레짐 α ≈ 2.00–2.01(gate_e 전역회귀) / 경계 국소 2.01–2.02(kink)**; kink median 2.20은 더 넓은 구간 포함값. 선형 레짐 α≈1.00.

## [값 4] 곡선 fit family AIC 비교
출처: `kink.json` → `gateA.fit_family_full.aic` (와 `fit_family_boundary.aic`). AIC 낮을수록 우수.
**전체(full):**
| 모델 | linear | quadratic | power_law | piecewise_linear | quad_to_linear |
|---|---|---|---|---|---|
| AIC | −509.3 | −586.8 | −469.7 | −710.4 | **−907.8 (승)** |
**경계(CF≤0.414):**
| 모델 | linear | quadratic | power_law | piecewise_linear | quad_to_linear |
|---|---|---|---|---|---|
| AIC | −238.0 | **−373.2 (승)** | −333.3 | −289.6 | −357.7 |
- **승자: 전체 = quad_to_linear, 경계 = quadratic** (둘 다 **매끄러운** 모델). `verdicts.H4_artifact.piecewise_beaten = True`.
- **"0.25 kink 착시" 근거**: piecewise_linear(꺾은선)이 전 구간서 매끄러운 모델에 **AIC로 패배**(full −710.4 > quad_to_linear −907.8; boundary −289.6 > quadratic −373.2) → 꺾임은 매끄러운 볼록곡선에 구간선형을 억지 적합한 산물.

## [값 5] CF² fit R² (결정계수)
출처: `results2/gate_e/gate_e.json` → `results.<scen>.<dir>.r2_fit` (`NC=c·CF²` 원점통과 적합의 R²). **저장돼 있음.**
| 방향 | r2_fit | 레짐 |
|---|---|---|
| CHSH/uniform | **0.9999744** | CF²(α≈2) |
| CHSH/det_vertex_0101 | **0.9999882** | CF²(α≈2) |
| KCBS5/uniform | **0.9999684** | CF²(α≈2) |
| KCBS5/det_vertex_all0 | **0.9999697** | CF²(α≈2) |
| CHSH/det_vertex_0000 | 0.7839172 | linear(α≈1) |
| CHSH/local_perfect_corr | 0.7915730 | linear(α≈1) |
- **4개 smooth-facet 방향 R² ≈ 0.99997–0.99999**(≈1.000). **선형 레짐 2방향 R² 0.784/0.792**(c·CF² 강제부적합 신호). ※ `gap_cf.json` 의 곡선형태는 AIC만 저장(R² 없음) → CF² fit R²는 gate_e 값이 해당.

## 확인 불가한 값과 이유
- **HMM H5·H7 KL**: `H_LADDER=[1,2,3,4,6,8]`로 미실행 → 저장 없음(추측 금지로 "없음").
- **kink 곡선형태 R²**: `kink.json`/`gap_cf.json`은 AIC 저장(R² 미기록) → R²는 gate_e(값5)로만 존재.
- 그 외 값1~5는 모두 파일에서 확인됨(확인 불가 없음).

## 실행한 명령 (읽기 수준만)
`git status` / `find results2/gap_cf results2/kink -type f` / `python3 -c "json.load..."`(값 읽기) / `grep -n`(코드 fit 라인 확인). **값 변경·재계산 명령 없음.**

## 생성한 .md 로그 경로
`docs/agent_logs/2026-07-08_1030_extract_paperV_batch1.md` (본 파일, 유일한 신규).
