# 논문 ↔ 실제 실험 정합성 검증 (IOTJ_INIT_QNN_Security, Donggun Lee 초고)

- 작성: 2026-07-09 (KST). 작업 디렉터리 `/home/elicer/IOTJ/IOTJ_2차`. **읽기 전용** — 코드·결과 무수정. 본 .md 1개만 신규.
- 목적: 제출 초고(PDF)의 각 서술을 `results2/`에 저장된 **실제 실험값·코드**와 1:1 대조하여, 부정확·불일치·모호 서술을 등급별로 정리.
- 대조에 사용한 원본: `results2/gate_r2/r2.json`, `results2/gate_r3/r3.json`, `results2/gate_e/gate_e.json`, `results2/gap_cf/gap_cf.json`, `tools/gap_vs_cf_v0.py`, `tools/gate_r1–r5*.py`.
- 원칙: 저장값이 논문 서술을 **지지**하면 "일치", 저장값과 **다르면** "불일치(등급)", 저장은 맞으나 **본문 표현이 오해 소지**면 "모호". 추측·보간 없음.

---

## 0. 총평
- **수치의 압도적 다수가 저장값과 정확히 일치**한다: Table I(8개), Table II(α·c_pred·c_fit·rel_err·R² 전부), Table III의 KCBS·Delft 전 항목, V-B의 AIC 4개, self-check(좌절3-cycle=1.000/정합=0.000), N-BaIoT MD5·기기 행수·def-b 상한(2.9×10⁻⁹)까지 **문자 그대로 재현**된다.
- **실질적 오류 1건**(F-1: Table III def-a gap 상한 숫자), **모호/오해/방법론 엄밀성 6건**(F-2~F-5, F-8, + §7 시나리오 서술), **경미/문구 3건**(F-6, F-7, F-9)을 발견했다. 어느 것도 핵심 주장(gap≈c·CF², CF=0 필연, 앵커 on-curve)을 뒤집지 않으나, **F-1은 저장 데이터가 논문 숫자를 지지하지 않는** 사실 오류이므로 수정 권장.
- 가장 실질적인 서술 이슈 3개: **§7**(IoT 감사를 CHSH "(2,2,2)/setting roles"로 서술 — 실제 n-cycle), **F-8**(IV-A "marginals held fixed"가 V-E 비교 방향엔 미성립), **F-2**(앵커 dev % 기준 곡선 불명확).

---

## 1. 정확히 일치 확인된 항목 (수정 불필요)

### 1.1 QCBM (IV-B, Fig 2a, Appendix C)
| 논문 서술 | 저장/코드 | 판정 |
|---|---|---|
| 2큐빗, `Ry(2θ)`+CNOT로 `cosθ\|00⟩+sinθ\|11⟩` 준비, 1 파라미터 | `psi=[cosθ,0,0,sinθ]` (`gap_vs_cf_v0.py:145`), θ 1개 | ✅ |
| 측정 회전 8 파라미터(설정당 (θ,φ) 1쌍), 총 9 | `PA`(x∈{0,1})+`PB`(y∈{0,1})=4설정×2각=8, `QPARAM=9` (`:69,148–149`) | ✅ |
| 단일 상태를 전 설정으로 측정 → 상관은 CNOT 얽힘에서만 | `e[ci,a,b]=Tr[ρ·PA⊗PB]` 하나의 ψ (`:154`) | ✅ |
| PennyLane analytic와 maxdiff `7×10⁻¹⁶`(본문)/`6.66×10⁻¹⁶`(App C) | 실측 6.66e-16(200 param) | ✅ (본문은 반올림 표기) |

### 1.2 고전 latent-class 사다리 (IV-B, Fig 2b)
- "혼합(latent-class), h를 π에서 **1회** 추출, **전이 없음·시퀀스 없음**, 설정조건부 emission으로 (a,b) 방출, H∈{1,2,3,4,6,8}" → `classical_behavior`(`einsum('act,bct,t->cab')`, transition 無)와 정확히 일치. ✅
- App C "EM이 아니라 직접 MLE(Powell)" → `fit_classical`=Powell, EM 미사용. ✅ (과거 HMM/EM 오기재를 올바르게 교정한 판본)

### 1.3 스윕/최적화 설정 (IV-A, App C)
- 16 λ(0.1간격 + {0.45,0.475,0.5,0.525,0.55,0.575,0.6}), master seed **20260703**, seed 5개, N∈{모집단,5000}, Powell(QCBM 3 restart=CHSH최적1+랜덤2 / 고전 2 restart / NCfloor 3 restart, maxiter 2000/–/6000) → 코드와 전부 일치. ✅

### 1.4 Table I (CF=0.6, CHSH)
- H1..H8 = 0.3681/0.1840/0.1129/0.1079/0.0823/0.0777, NCfloor 0.0725, QCBM 0.0095, gap 0.0629 → `gap_cf.json` sweeps.inf[λ=0.8]과 정확 일치. ✅

### 1.5 Table II (계수, gate_e.json 대조)
| Scenario/Dir | α | c_pred | c_fit | rel_err | 논문 | 판정 |
|---|---|---|---|---|---|---|
| CHSH/uniform | 2.0057 | 0.16667(=1/6) | 0.16995 | 0.0193 | 2.006 / 1/6 / 0.170 / 1.9% | ✅ |
| CHSH/det_0101 | 2.0033 | 0.19921 | 0.20191 | 0.0134 | 2.003 / 0.199 / 0.202 / 1.3% | ✅ |
| KCBS/uniform | 2.0066 | 0.12500(=1/8) | 0.12776 | 0.0216 | 2.007 / 1/8 / 0.128 / 2.2% | ✅ |
| KCBS/all0 | 2.0079 | 0.12500 | 0.12772 | 0.0213 | 2.008 / 0.125 / 0.128 / 2.1% | ✅ |
| CHSH/det_0000 | 1.0021 | 48.76 | 2.483 | R²0.7839 | 1.002 / →48.8 / R²0.784 | ✅ |
| CHSH/local_perf | 1.0065 | 48.80 | 2.158 | R²0.7916 | 1.007 / →48.8 / R²0.792 | ✅ |
- 4 smooth-facet R² = 0.99997/0.99999/0.99997/0.99997 → 논문 각주와 정확 일치. ✅

### 1.6 Table III 앵커 (r2.json 대조)
| | witness | CF | CI | KLq | gap | dev |
|---|---|---|---|---|---|---|
| KCBS | C=2.5258→**2.526** | 0.2629→**0.263** | [0.2482,0.2795]→[0.248,0.278] | 1.4e-12→**0.0000** | 0.01233→**0.0123** | 0.0368→**3.7%** | ✅ |
| Delft | S=2.4225→**2.422** | 0.21125→**0.211** | [0.1151,0.4185]→[0.115,0.418] | **0.004457→0.0045** | 0.008463→**0.0085** | 0.0528→**5.3%** | ✅ |
- Delft no-signalling residual `no_sig_L1=0.08947` → 본문 "L1=0.0895" ✅. Delft n=245, per-ctx [53,79,62,51], DOI/MD5(342f29f8…) 전부 일치. ✅

### 1.7 self-check / provenance
- App C "좌절 3-cycle CF=1.000, 정합 0.000" → r3.json `self_check={frustrated:1.0, consistent:0.0}` 정확. ✅
- N-BaIoT UCI#442, MD5 `56bf8e22debda482596f6d2be1260afd`, Danmini 49,548행 → r3.json data_source와 일치. ✅
- def-b 상한 `≤2.9×10⁻⁹` → Ecobee 셀 `gap_upper=2.909e-9`(cf_ci 상단 1.525e-4, c=1/8) **정확 일치**. ✅
- def-b max CF `7.6×10⁻⁵` → r3 max=7.627e-5. ✅ / def-a 15셀 CF=0 → 실측 max cf=7.8e-16(수치영). ✅
- KLq(λ)는 λ=0.7까지 0, λ=0.8부터 lift → "Tsirelson λ_Q=1/√2≈0.707 이후 lift-off" 서술과 일치. ✅

---

## 2. 발견된 불일치·모호 (등급별)

### 🔴 F-1 [사실 오류] Table III def-a gap 상한 `≤1.5×10⁻³¹`
- **논문**: N-BaIoT (def-a) gap ≤ **1.5×10⁻³¹**.
- **저장값**: r3.json의 def-a 15셀 중 **max `gap_upper = 5.56×10⁻³¹`** (실측).
- **판정**: 논문 숫자(1.5e-31)가 데이터의 실제 최댓값(5.56e-31)보다 **작다 = 데이터가 지지하지 않는 더 tight한 값**. 보수적 상한이 되려면 `≤5.6×10⁻³¹`(또는 `≤10⁻³⁰`)로 적어야 함.
- **영향**: 결론("10⁻⁹ 이하, 실용 무관")에는 무영향(둘 다 ~10⁻³¹). 그러나 표에 박힌 유효숫자가 틀림 → **수정 권장**.

### 🟠 F-2 [모호/그림-지표 불일치] 앵커 "within 3.7% / 5.3%"가 가리키는 곡선
- **논문**: 앵커가 "예측 곡선의 3.7%/5.3% 이내". Fig 5(=우리 fig3)는 곡선을 **"predicted gap = CF²/6"**(c=1/6)로 그림·표기.
- **저장값**: rel_dev는 `curve_pred` 대비 계산이며 KCBS `curve_pred=0.01280`, Delft `0.00804` = **경험 reference_sweep**(국소 c≈0.18) 기준. CF²/6(c=1/6) 곡선값은 KCBS에서 0.01153 → 이 곡선 대비 편차는 **≈6.7%**(3.7% 아님).
- **판정**: "dev %"의 기준 곡선(경험 스윕)과 **그림에 그린/이름 붙인 곡선(CF²/6)**이 서로 다름. 독자가 CF²/6로 재보면 3.7%가 안 나옴.
- **권고**: (a) dev를 "수치 측정된 gap–CF 스윕 곡선 기준"으로 명시하거나, (b) 그림 곡선을 경험 스윕으로 바꾸거나, (c) 두 편차를 병기. 우리 `fig3`의 "dev 3.7%" 주석도 그린 1/6 곡선과 시각적으로 어긋나므로 동일 수정 필요.

### 🟠 F-3 [서술 오해 소지] "KLquantum은 Tsirelson까지 0"과 Delft KLq=0.0045
- **논문**: Fig 3 캡션·V-B "KLquantum은 Tsirelson점(λ=1/√2, CF=0.414)까지 0을 유지".
- **저장값**: 이는 **합성 스윕(PR-box 등방혼합)**에서만 성립. 실제 **Delft 앵커는 CF=0.211(<0.414)인데 KLq=0.0045≠0**(Table III). 원인: 실데이터의 no-signalling 잔차(L1=0.0895)·비이상성으로 양자실현영역 매니폴드에서 벗어남.
- **판정**: 조작 아님(값은 r2.json에 실재). 단, "Tsirelson 아래선 KLq=0"이라는 무조건적 문구가 실앵커에는 성립하지 않음 → **합성 구성에 한정**임을 1문장 스코핑 권장.

### 🟠 F-4 [그림-캡션 계수 불일치] Fig 4 캡션 c=1/6 vs 그린 곡선 c=0.170
- **논문**: Fig 4 캡션 "gap≈c·CF² with **c=1/6**".
- **우리 그림(fig2a)**: 실제 그린 곡선·범례는 **c_fit=0.1699** 사용(제목 α=2.006, R²=0.99997). 1/6=0.1667.
- **판정**: 캡션(예측값 1/6)과 그림(적합값 0.170)이 다른 c. 오차 1.9%라 시각적으론 미미하나 **문자 불일치**. "c≈1/6 (예측; 적합 0.170)"로 통일 권장.

### 🟠 F-5 [그림 회로 단순화] Fig 2(a)의 측정 회전 개수
- **논문 본문**: "설정당 (θ,φ) 1쌍, **8 측정 파라미터 = 4 설정**".
- **Fig 2(a)**: 큐빗당 측정회전 `U(θ₁,φ₁)`,`U(θ₂,φ₂)` **2개만** 표시(첨자 1,2가 큐빗을 지칭). 실제 모델은 큐빗당 **2 설정(A₀,A₁ / B₀,B₁)** → 4 설정·8 파라미터.
- **판정**: 그림은 "설정당 다른 측정회전"을 축약해 큐빗당 1개만 그림 → "8 파라미터/4 설정" 본문과 시각적으로 어긋남. 회로도에 설정 인덱스가 큐빗이 아니라 **측정설정**임을 캡션에 명시하거나, 큐빗당 2개(파선/스위치)로 표기 권장.

### 🟡 F-6 [경미/반올림] 초록 "below ∼10⁻⁹" vs 실제 상한 2.9×10⁻⁹
- 전체 상한 = max(def-a 5.6e-31, def-b **2.9e-9**) = 2.9×10⁻⁹ ≈ **3 parts per billion**. "one part in a billion(10⁻⁹) 이하"는 엄밀히는 초과. 틸드(∼) 근사이나 "a few ×10⁻⁹" 또는 "below ~3×10⁻⁹"가 정확.

### 🟡 F-7 [경미/문구] robustness "2 definitions × 5 feature subsets" & "gate r5 pending"
- IV-E "2 정의 × 5 부분집합"은 **def-a에만 5 부분집합**이 해당(def-b는 기기당 1 시간창). def-b까지 ×5로 읽히면 과대. → "def-a: 5 부분집합; def-b: 시간창 1" 정도로 정밀화 권장.
- App C "gate r5 pending" → 실제 `results2/gate_r5/`는 **완료**(9기기 54셀). 논문은 3기기 gate_r3(18셀)을 채택했으므로 내용은 정합하나, "pending" 문구는 stale → 갱신/삭제 권장.

### 🟠 F-8 [방법론 엄밀성] IV-A "marginals held fixed"가 V-E 비교 방향엔 성립 안 함
- **논문**: IV-A "λ chosen so that **all marginals are held fixed** and only the correlation structure varies". V-E는 이를 근거로 "같은 CF, 다른 방향 → 다른 gap"(CF 충분통계량 아님)을 주장하며 **대표 비교로 CHSH/uniform vs CHSH/det_vertex_0101**(0.0070 vs 0.0130)을 제시.
- **저장/계산 검증**: PR-box와 혼합 시 marginal 고정 여부 —
  - **uniform, local_perfect_corr** → 전 λ에서 0.5/0.5 **고정** ✅
  - **det_vertex_0000, det_vertex_0101** → marginal이 **λ에 따라 변함**(λ=0.5서 0.75/0.75, 0101은 컨텍스트별 상이) ❌
- **판정**: V-E의 헤드라인 비교 상대인 **det_vertex_0101은 marginal이 고정 안 됨** → 그 비교는 "marginal 고정, 방향만 다름"이 아니라 **marginal도 함께 변하는** 비교. 즉 IV-A의 "marginals held fixed"는 **메인 uniform 스윕에만 엄밀히 참**이고, Table II vertex 방향 2개(det_0000/0101)엔 적용 안 됨.
- **왜 그렇게 됐나**: marginal-고정인 다른 방향(local_perfect_corr)은 **선형(α=1) 레짐**이라 "quadratic vs quadratic" 비교가 불가 → 부득이 marginal이 변하는 det_0101을 quadratic 상대로 씀.
- **영향**: "CF가 gap을 유일하게 결정하지 않는다"는 **결론 자체는 유효**(gap은 전체 기하 의존). 단 "marginal 고정 하에 순수 방향 효과"로 읽히면 과함 → **"이 방향쌍은 marginal이 λ에 따라 변함"을 각주 명시**하거나 marginal-고정 예를 병기 권장.

### 🟡 F-9 [미공개 근사] IoT 감사 gap 상한의 c=1/8은 5-cycle 계수 차용
- **논문**: Eq(10)/Algorithm 1 "gap ≤ **c·CF²** ≤ 2.9×10⁻⁹".
- **코드**: `C_GATE_E=0.125`=**1/8**(=5-cycle KCBS 균등혼합 경계값, App A Step4에서 유도). 그러나 감사 시나리오는 **8-cycle(def-a)·3-cycle(def-b)** → **5-cycle 상수를 8/3-cycle에 빌려 쓴 경계근사**(코드 주석도 "경계근사 주의"). 
- **판정**: CF≈0이라 **숫자엔 무영향**(c가 O(1)이면 gap~10⁻⁹ 불변). 단 논문은 이 차용을 밝히지 않음 → **"c=1/8은 5-cycle 값을 차용한 보수적 근사(감사 시나리오는 8/3-cycle)"** 각주 권장.

---

## 3. 확인했으나 문제 없는(오해하기 쉬운) 지점
- **KCBS 앵커 = 4-cycle chained(CHSH-동치) 4상관 매핑**(footnote 1): C=2.5258 보존, CF=(C−2)/2=0.2629 교차확인 Δ=0. "5측정 중 4선택"이 아니라 4-cycle이라는 서술은 코드와 일치. ✅
- **N-BaIoT benign만 사용**: 공격 트래픽(.rar) 접근 불가로 benign 9 CSV 사용 → 논문도 "benign traffic"으로만 서술(정직). ✅
- **"3 commercial devices"**: gate_r3 채택(Danmini/Ecobee/Ennio). gate_r5(9기기)는 미채택이나 존재. 내부 정합. ✅
- **abstract/Positioning "supremacy 아님·simulation 기반·모델클래스 분리"**: 전 구간 일관 명시(우리 원칙과 일치). ✅

---

## 4. 수정 우선순위 요약
1. **F-1 (🔴 필수)**: Table III def-a `1.5×10⁻³¹` → `≤5.6×10⁻³¹`(저장 max) 로 정정.
2. **§7 (🟠)**: IoT 감사 시나리오 서술 정정 — Algorithm/(2,2,2)·IV-E "setting roles" → n-cycle(def-a 8-cycle, def-b 3-cycle). (문구는 §7.3)
3. **F-8 (🟠)**: V-E 방향비교(uniform vs det_0101)가 marginal 미고정임을 각주 명시 or marginal-고정 예 병기.
4. **F-2 (🟠)**: 앵커 dev %의 기준 곡선(경험 스윕 vs CF²/6) 명시 — 본문·Fig 5·우리 fig3 주석 정합.
5. **F-3 (🟠)**: "Tsirelson 아래 KLq=0"을 합성 구성 한정으로 스코핑(Delft KLq=0.0045 각주).
6. **F-4/F-5 (🟠)**: Fig 4 캡션 c 표기 통일, Fig 2(a) 측정설정 수 명확화.
7. **F-6/F-7/F-9 (🟡)**: 초록 10⁻⁹ 표기, "2×5"/"gate r5 pending" 문구, 감사 c=1/8 차용 각주.

## 5. 실행한 명령 (읽기 수준만)
`python3 -c "json.load..."`(r2/r3/gate_e.json 값 읽기) / `grep -n`(self-check·QCBM 구조 라인). **값 변경·모델 재실행 없음.** 기존 코드·결과 무수정.

---

## 7. [추가 검증] IoT 감사 시나리오 구조 — 코드 직접 확인 (def-a/def-b/"5 subsets")
계기: 외부 에이전트가 "def-a=8축 n-cycle(단일자, Alice/Bob 없음)"이라 단정 → **요약/기억이 아니라 `gate_r3_iot_cf.py`·`kink_origin_v1.py` 코드로 재검증**.

### 7.1 코드가 실제로 하는 것 (확정)
| 항목 | 코드 근거 | 실제 구조 |
|---|---|---|
| **CF-LP 시나리오** | `_cyclic_setup(n)`(`kink:195–208`): `V=2ⁿ 전역배정`, 컨텍스트 = 인접쌍 `{M_i, M_{i+1}}`(cyclic). `cyclic_cf`(`:211`)=ABM LP | **순수 n-cycle**(단일자, **Alice/Bob 분할 없음**) |
| **def-a** | `feature_cycle_model`(`:145–156`): `e[i]=P(f_i,f_{i+1})`, i=0..k−1, `j=(i+1)%k`; `n=len(idx)=8`; `cyclic_cf(e,8)` (`:194–199`) | **8 feature = 8축 = 8측정, 인접쌍 8개가 컨텍스트 → 8-cycle** |
| **def-b** | `time_cycle_model`(`:159–168`): `col=한 feature`, `x0,x1,x2=col[t],col[t+1],col[t+2]`, 쌍 `(t,t+1),(t+1,t+2),(t,t+2)`; `cyclic_cf(e,3)` (`:204–208`) | **단일 feature를 3개 시간지연(t,t+1,t+2)으로 = 3측정, 그 3쌍이 컨텍스트 → 3-cycle**(행순서=시간대용) |
| **"5 feature subsets"** | `methods=["highvar","targeted","random1","random2","random3"]`(`:258`); def-a만 `for m in methods`, def-b는 `"highvar",k=1` 고정(`:260`) | **8축을 채울 feature 8개를 고르는 5가지 선택전략**(def-a 전용). def-b는 부분집합 변주 없음(기기당 1) |
| **셀수** | `def-a = 기기3×5 = 15`, `def-b = 기기3×1 = 3` (`:259–260`) | 18셀 ✅ |
| **gap 상한 c** | `C_GATE_E=0.125`(`:49`)=**1/8**(cyclic/KCBS 경계) | **이미 cyclic 계수 1/8 사용**(CHSH 1/6 아님) → 코드는 내부적으로 cyclic임을 반영 |

→ **외부 에이전트의 "def-a=8축 n-cycle, 단일자" 주장은 코드와 정확히 일치**(재검증 통과). 나 자신의 이전 답변("def-a=8-cycle")도 코드로 확증됨. **단, def-b는 이전에 상술 안 했었음 → 이제 확정: "3개의 다른 feature"가 아니라 "한 feature의 3 시간지연".**

### 7.2 논문과의 충돌 (확정)
| 논문 위치 | 서술 | 실제 | 판정 |
|---|---|---|---|
| Algorithm 1 Require | "scenario (**e.g., (2,2,2)**)" | 감사는 def-a **8-cycle** / def-b **3-cycle** (2,2,2 아님) | 🟠 오해 소지 — (2,2,2)를 감사 예시로 제시 |
| IV-E | "assigned to **setting roles** (def-a: **feature blocks**; def-b: time windows)" | def-a: 각 feature가 축, 인접쌍이 컨텍스트(k-cycle). "setting roles"(A₀A₁B₀B₁ CHSH 배정) 아님 | 🟠 부정확 — CHSH 이분 틀 암시 |
| IV-D "Why the (2,2,2) Scenario" | 전 작업이 (2,2,2)+KCBS 5-cycle | IoT 감사는 **8-cycle·3-cycle**(미언급 시나리오) | 🟠 커버리지 누락 — 감사 시나리오 미기재 |
| IV-E / VI-C | "2 definitions × **5 feature subsets**" | 5 subsets는 **def-a 전용**(def-b 1) | 🟡 def-b까지 ×5로 읽히면 과대(§F-7과 동일) |

**★ 결론 불변**: Proposition 1은 축 개수·시나리오와 무관하므로 CF=0(15셀)·상한(2.9×10⁻⁹)·앵커 전부 그대로. **바뀌는 건 "감사가 어떤 시나리오냐"의 서술뿐.** 게다가 코드는 이미 c=1/8(cyclic)을 써서 상한을 냈으므로 숫자는 cyclic 기준으로 정합 — **문구(Algorithm의 (2,2,2), IV-E "setting roles")만 CHSH 잔재.**

### 7.3 권장 수정 문구 (초고엔 .tex 소스가 리포에 없음 → 문구만 제시)
- **IV-E**: "…binarizing selected features at their medians (±1) and treating **each binarized feature as a two-outcome measurement in a cyclic (n-cycle) scenario** — not the bipartite (2,2,2) scenario used for the synthetic sweep and quantum anchors. In **def-a**, k=8 binarized features form an **8-cycle**: each feature is a measurement M_i and each adjacent pair {M_i, M_{i+1}} (cyclically) is a context. In **def-b**, a single high-variance feature sampled at three consecutive rows (X_t, X_{t+1}, X_{t+2}) forms a **3-cycle** (row order as a time proxy), its contexts being the three lag-pairs. Robustness is assessed across the 2 definitions and, **for def-a, 5 feature-selection strategies** (highest-variance, a higher-order-correlation-targeted greedy selection, and three random draws) — 15 def-a and 3 def-b cells over 3 devices. By Proposition 1, CF = 0 holds for any def-a-type construction **regardless of the number of measurements**…"
- **Algorithm 1 Require/step2**: "scenario (e.g., (2,2,2))" → "**a cyclic scenario (def-a: k-cycle over k features; def-b: 3-cycle over one feature's time lags)**"; step 2 "assign features to measurement settings" → "**arrange binarized features as the cyclic scenario's measurements**".
- **IV-D**: 1문장 추가 — "The IoT audit instead runs cyclic n-cycle scenarios (8-cycle, 3-cycle); Proposition 1 makes the specific scenario immaterial to its CF = 0 conclusion."

### 7.4 def-b가 CF>0(유한표본) 될 수 있는 이유 — Remark 1과 정합
def-b 3측정은 **같은 정상시계열의 3 지연**이라 점근적으론 단일 결합 `P(X_t,X_{t+1},X_{t+2})`이 존재(→CF→0). 유한표본에선 세 2-주변확률이 약간 어긋난 지지에서 추정돼 CF가 O(N^{-1/2})로 양수 가능 → **논문 Remark 1 서술과 정확히 일치**(def-b max 7.6×10⁻⁵). 이 부분은 정확.

## 8. [검증 커버리지 경계] 무엇을 검증했고 무엇을 아직 안 했나 (정직)
**독립 대조 완료(원본 파일 기준):** 모든 표(I/II/III)·수치·스윕/최적화 설정·self-check·provenance(MD5·기기 행수 Danmini49548/Ecobee13113/Ennio39100 전부 일치)·QCBM/고전 구조·def-a/def-b/subset 시나리오 구조·def-b 상한. c_pred(1/6,1/8)은 코드가 `chi2_curvature`로 **수치 계산**(하드코드 아님, 0.16667/0.125)임까지 확인.

**아직 독립 검증 못 함(단정 불가):**
1. **Appendix A 해석적 유도**(Step1–4, Eq8–14, Remark2): 결과 숫자만 코드 ½χ²와 일치 확인, **손유도 재검 안 함**.
2. **III-B "예비 QCBM 우위 없음"**: IOTJ_1차 결과 대상 주장 — **1차 결과로 미대조**.
3. **Section II 문헌 특성화**([1]–[9] 요약의 공정성): 원논문 미독 → 검증 불가.
4. **참고문헌 서지 정확성**([11][12][13][4][18] 등): 내부 정합만 확인, 개별 서지 미확인.
5. **KCBS 4-cycle(앵커) vs 5-cycle(합성 교차검증)**: 둘 다 논문에 존재·내부 정합, 오류 아님(독자 혼동 소지만).
6. **Fig 1/Fig 6 도식**: 개념도(수치 대상 아님).

→ **§2·§7의 발견은 "정량 서술 전수" 결과이며, 위 6개(특히 1·2)는 미검증 영역**임을 명시. 추가 요청 시 ① App A 손유도 ② III-B↔IOTJ_1차 대조부터 수행 권장.

## 6. 생성한 .md 로그 경로
`docs/agent_logs/2026-07-09_paper_vs_experiment_verification.md` (본 파일, 유일한 신규).
