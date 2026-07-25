# Gate S 시리즈 — DI-QKD 탐지 실험 (S1 시뮬·재현 / S2 스코어 / S2b 재설계공격·상보성)

- 작성: 2026-07-24 (KST). 작업 디렉터리 `IOTJ_2차/`. **측정+훈련+탐지**, 결정적(seed=20260724), 무설치(numpy·scipy).
- 재현: `python3 tools/sim_diqkd.py` · `tools/gate_s2_detection.py` · `tools/gate_s2b_isoS.py` · `tools/gate_s2c_prbox_blindshell.py`.
  산출: `results2/gate_s1/{s1.json,s1.png}`, `results2/gate_s2/{s2.json,s2_auc.png,s2_p2_monotone.png}`, `results2/gate_s2b/{s2b.json,s2b_auc.png,s2b_geometry.png}`, `results2/gate_s2c/{s2c.json,s2c.png}`.
  논문 서술 초안: `docs/paper_draft_diqkd_detection.md`.
- 재사용(무수정): `gap_vs_cf_v0`(=G, CHSH CF-LP·nc_floor·fit_quantum·quantum_behavior), `gate_r2`(=R2, build_box·chsh_S·no_signalling_L1). ★**모델클래스 분리이지 supremacy 아님. 키보안 아닌 장비무결성 맥락.**
- 정합규칙: **R1** bipartite G 트랙만(K.cyclic_* 미사용) / **R2** 16셀 변환기=R2.build_box 하나 / **R3** 훈련·평가 분리(신규) / **R4** G.fit_shape(quadratic 없음) 미사용 / **R5** 결정성·JSON(note/params/self_check/limits).
- 신규함수(원본 무수정): `sim_diqkd.simulate_diqkd`·`nc_floor_weights`, `gate_s2.fit_quantum_params`·`fit_nc_params`(래퍼 원본과 **1e-9 일치 검증**), `gate_s2b.find_iso_S_attack`.

---

## STEP 1 — DI-QKD 시뮬레이터 + Delft 재현 (gate_s1) — **PASS**

시뮬레이터: 모든 공격(none/intercept_resend/blinding/mimic_ncfloor/partial)을 16셀 behavior 로 환원 → R2.build_box 로 (a,b,xo,yo)→box. 핵심 단순화: 측정 = Bloch `P(o|û)=½(1+(−1)^o û·s)` (정확·고속). Werner = `V·quantum_behavior(p)+(1−V)·uniform` (측정이 ρ에 선형, _proj trace=1).

### self-check — 4/4 통과
| 항목 | 결과 | 기대 |
|---|---|---|
| blinding CF | 0.00e0 (S=2.000) | 0 (국소결정론, 고전한계 포화) |
| mimic_ncfloor CF | 0.00e0 | 0 (NC 혼합) |
| none n=1e5 S | 2.8245 | 2√2=2.8284 |
| intercept_resend | CF=0, S=0.938 | ≤2 (무작위기저 intercept가 상관 파괴) |

### Delft Hensen2015 재현 STOP 게이트 — **3/3 통과** (1000 reps)
raw Delft 데이터는 gitignore 부재 → 검증된 `r2.json` 요약통계를 참조 타깃으로 사용(정직 기록). 피팅: **visibility V=0.85648**(canon 각도 고정, S 는 V 에 선형)로 S=2.4225 매칭, per-ctx N=[53,79,62,51] 고정.
| 지표 | sim (1000 reps) | 실측 | 판정 |
|---|---|---|---|
| CF | median 0.236, 95%밴드 [0.107, 0.421] | 0.211 | ⊇ ✓ |
| gap = NCfloor−KL_q | median 0.00783 (=해석 0.00784), 밴드 [0, 0.030] | 0.0085 | ⊇ ✓ |
| CHSH S | median 2.419 | 2.4225 (|Δ|=0.004<0.05) | ✓ |
- (참고) sim no-sig median 0.148 = 유한표본 겉보기 signalling. 이상적 등방 sim 은 KL_q≈0(box 정확히 양자실현) — 실측 KL_q=0.0045(no-sig 잔차 0.089 반영)와 다르나 gap 밴드로 비교. → **STEP 2 진입 허용.**

---

## STEP 2 — 탐지 실험 (gate_s2) — **P1(naive NLL) FAIL = 방법론적 발견 / deviance 교정으로 gap→마진**

훈련(정상 V=0.856, CF≈0.211) → 양자탐지기 q_model=`quantum_behavior(fit_quantum_params)`, 고전탐지기 c_model=`fit_nc_params`(NCfloor 도달점). 평가 6클래스×100반복. **주가설 P1: mimic_ncfloor 에서 AUC_quantum > AUC_classical.**

### 스코어 규약 (사람 결정 2026-07-24 반영)
- **탐지 스코어 = deviance/G-통계량**: `KL(emp‖model) = CE − H(emp)`, **2N·KL ~ χ²** (포화모델 대비 우도비 검정통계량). 표본엔트로피 교란 제거·경험박스만으로 계산.
- **CE(=naive NLL) 결과는 보존**, 라벨 "naive NLL 은 batch entropy 에 교란됨(방법론적 발견)".

### 결과 (mimic_ncfloor, N_train=10000)
| 스코어 | AUC quantum | AUC classical | AUC S-test | Δ(q−c) |
|---|---|---|---|---|
| **naive NLL (CE)** | 0.994 | 0.994 | 0.994 | +0.000 → **FAIL** |
| **deviance (KL)** | 0.925 | **0.075** | — | **+0.850** |
- **naive NLL FAIL**: 세 탐지기 동률. 원인 — mimic 은 비맥락(CF=0,S=2)이라 정상보다 엔트로피 높음(H 5.02 vs 4.76)·S 낮음 → NLL 이 H(emp)에 지배. 즉 mimic 은 이미 고전수단(S-검정)만으로 탐지됨. **NLL 의 엔트로피 교란을 드러내는 방법론적 발견.**
- **deviance 교정**: gap→탐지마진 실재. 고전탐지기 AUC<0.5(mimic 을 정상보다 덜 이상으로 인증), 양자>0.5.
- 모집단 진단: KL(정상‖c_model)=0.0081(=NCfloor), KL(mimic‖c_model)=0.0004(mimic 은 고전 모델클래스 안), KL(mimic‖q_model)=0.0092.

### P2 — gap↑ ⇒ ΔAUC(deviance)↑ **완벽 단조 (Spearman = 1.000)**
| V | CF | gap | ΔAUC(deviance) |
|---|---|---|---|
| 0.70 | 0.00 | 0.000 | 0.000 |
| 0.80 | 0.131 | 0.003 | 0.288 |
| 0.856 | 0.211 | 0.008 | 0.893 |
| 0.92 | 0.301 | 0.016 | 0.991 |
| 1.00 | 0.414 | 0.032 | 1.000 |
- CF=0(자원 없음)→마진 0, CF↑→마진↑. **c·CF² 함수형은 주장 안 함**(AUC 는 KL 의 함수 아님). 단조성만.
- **한계 진단**: mimic 은 S=2(고전적)라 S-검정 자체를 통과 못 함 → "양자모델이 S-검정보다 정밀한가"를 격리 못 하는 부적합 공격. → STEP 2b 로 승계.

---

## STEP 2b — 재설계 공격 (gate_s2b): S-검정 통과 공격 · **상보성 발견 + blind_shell 불가능성**

S-검정을 통과(S≈2.42)하지만 정상 지문과 다른 공격을 `quantum_behavior(p)` 이미지(물리 구현 가능)에서 탐색. 스코어=deviance(KL). 탐지기는 **정상 장비의 고정 템플릿 대조**(재적합 아님)=장비지문.

### 공격 설계 (penalty 최적화, 24 restarts)
| 공격 | S(dev) | CF | KL→정상(양자축) | KL→고전(NCfloor=0.0078) | marg_dev | feasible |
|---|---|---|---|---|---|---ㄴㅅ|
| **iso_S** | 2.419(0.002) | 0.210 | **0.353** | 0.361 | 1.13 | ✓ |
| **marginal_matched** | 2.420(0.001) | 0.210 | **0.207** | 0.215 | 0.000 | ✓ |
| **blind_shell** | 2.421(0.0001) | 0.210 | **0.0001** | 0.0079(≈NCfloor) | 0.000 | 정상으로 붕괴 |

### 탐지 AUC (deviance KL, N_train=10000)
| 공격 (S) | 양자 템플릿 | 고전 NC 템플릿 | S-검정 (N=245 / 1000) |
|---|---|---|---|
| **iso_S** (2.42) | **1.00** | **1.00** | 0.53 / 0.57 |
| **marginal_matched** (2.42) | **1.00** | **1.00** | 0.50 / 0.59 |
| blind_shell (2.42) | 0.49 | 0.49 | 0.52 |
| partial_05 (2.35) | 0.46 / 0.45 | 0.42 / 0.29 | **0.58 / 0.72** |
| partial_10 (2.27) | 0.49 / 0.60 | 0.34 / 0.17 | **0.75 / 0.89** |
| partial_20 (2.12) | 0.53 / 0.74 | 0.30 / 0.08 | **0.83 / 0.98** |

### ★ 핵심 발견 — 상보성 (complementarity)
- **(a) S-보존 지문변조**(iso_S/marginal_matched): 모델 템플릿 AUC=**1.00**, **S-검정 완전 blind(≈0.5)**. marginal_matched 는 marginal 일치(QBER 모니터 우회)해도 잡힘 → **생성모델 지문이 CHSH S-검정 사각지대(S-보존 변조)를 포착.**
- **(b) S-저하 공격**(partial=intercept 소량혼합): **S-검정이 잡음**(0.58→0.98), 모델 KL 은 ≤0.5 (intercept 가 CF↓ 로 데이터를 더 '고전적'으로 만들어 고전템플릿엔 덜 이상 → AUC<0.5).
- → **두 탐지 계층이 disjoint 공격을 커버 = defense-in-depth. "S-검정 보완 계층" 실증.**

### ⚠️ blind_shell 불가능성 (P1′ FAIL, 원리적)
blind_shell 최적화가 **정상으로 붕괴**(KL→normal≈0.0001). **정리**: 같은 S 에서 등방(정상) 상자가 고전거리 KL(‖c_model)=NCfloor 를 **유일하게 최소화** → 비등방 이탈 시 KL(‖c)↑(iso_S 0.36) → 고전탐지기가 잡음. 따라서 '고전 껍질(NCfloor)에 머물기'='정상에 고정'→양자도 못 잡음. **geometry 좌상단(고전-blind & 양자-visible)이 공집합.** → 물리(양자) 같은-S 공격으로는 **양자 템플릿이 고전 NC-템플릿보다 우월 불가**. 모델기반 우위(a)는 **양자 특유 아님**.
- **P3′ FAIL**: partial 은 S-저하라 S-검정 홈그라운드 — 상보성의 이면(실패 아님).
- **기하 연결**: 정상=고전껍질 반지름 **NCfloor=0.00778 ≈ c·CF²=0.00739** (c=1/6, CF=0.211). 이 껍질의 유일 최소점이 정상.

---

## STEP 2c — blind_shell 불가능성 **강한 정리** (gate_s2c): 무신호 폴리토프 전체 공집합

공격집합을 quantum_behavior(p)(물리적 양자)에서 **무신호(no-signalling) 상관 파라미터화**로 확장. 균등 marginal 무신호 상자 `P(a,b|x,y)=¼(1+(−1)^(a+b)E_xy)`, E∈[−1,1]⁴. 양자성 = **TLM(Tsirelson–Landau–Masanes)** `max_k|Σasin E−2asin E_k|≤π`.

### 검증 (연속 최적화 + 전수 그리드)
| 방법 | 결과 |
|---|---|
| quantum_only 탐색 | 정상 붕괴 KL→normal=0.00009 (TLM=−0.54, 양자) |
| unrestricted(초양자 허용) 탐색 | **정상 붕괴** KL→normal=0.00009 (초양자 blind_shell **부재**) |
| 전수 그리드(19,600 상자, S=2.42) | **MIN KL→c = NCfloor**(정상), min−NCfloor=6.7e-5; 껍질 위 정상과 먼 상자 **0개** |
| 참조 탐지(최적해=정상 근방) | 3 탐지기 전부 ≈0.5 (탐지대상 없음) |

### 정리 (강한 형태)
> **S=S_normal 인 모든 무신호 상자**(양자∪초양자, PR-box 포함) 중 **정상이 고전템플릿 거리 KL(‖c_model)=NCfloor 를 유일 최소화.** 동치로, 정상은 고전탐지기 **blind 공** {KL(‖c_model)≤NCfloor} 안에서 **CHSH-S 를 유일 최대화**.
- ⟹ (고전 blind) ∧ (S≥S_normal, S-검정 blind) ∧ (≠정상) 은 **무신호 폴리토프 전체에서 동시 불가.**
- ⟹ 초양자(비물리 PR-box) 공격자에게조차 '고전엔 안 걸리고 양자엔 걸리는' 변조 부재 → **양자 템플릿 고유 탐지역=공집합.** 모델기반 우위(iso_S)는 전적으로 **비양자적.**
- **우위 정리가 아니라 탐지가능성 경계 정리.** blind 공 반지름 = NCfloor = c·CF²(층위 연결).

---

## 종합 판정
1. **STEP 1 PASS**: 시뮬레이터가 Delft 실측(CF/gap/S)을 유한표본 밴드로 재현.
2. **STEP 2**: naive NLL 은 batch entropy 교란으로 세 탐지기 동률(방법론적 발견) → **deviance(2N·KL~χ²)** 교정 시 gap→탐지마진, P2 완벽 단조(Spearman=1.0).
3. **STEP 2b 상보성**: 모델 지문탐지(S-보존 변조 AUC=1.0) ⊥ S-검정(S-저하 공격 0.58→0.98) = defense-in-depth. 단 모델기반 우위는 **양자 특유 아님**.
4. **STEP 2c 강한 불가능성**: blind_shell 이 무신호 폴리토프 **전체**(양자∪초양자)에서 공집합 — 정상이 고전 blind 공 안에서 S 유일 최대. → 양자>고전템플릿 우위는 **물리·비물리 공격자 모두에게 불가**. blind 공 반지름 = NCfloor = c·CF²(층위 1↔2 연결).

## 한계 (필수)
- (2,2,2) 소규모·시뮬·메모리매칭·finite-sample. raw Delft 부재(요약통계 사용). 이상적 등방 sim 은 no-sig 잔차 미재현.
- iso_S/marginal_matched 는 특정 정상 운용점의 **존재증명**(모든 공격 커버 아님). 탐지기는 고정 템플릿(재적합 아님).
- 모델기반 우위=양자 특유 아님(고전 NC-템플릿 동급). partial 은 S-저하라 S-검정 홈그라운드.
- **관측/모델클래스 분리이지 quantum supremacy 아님. 키보안 개선 아닌 장비무결성/변조모니터링 맥락.**

## 미해결 후보
- 비-양자 no-signalling box(초양자/일반 NS) 공격으로 blind_shell 재시도(물리성 완화 시 좌상단 채워지는지).
- marginal_matched 를 주력 공격으로 정식화(QBER 모니터 우회 서사).
- iso_S 를 sub-attack 으로 한 partial(S-보존)에서 유한통계 우위 재검(P3′의 S-보존 버전).

## [해석입력용 요약 — 웹 AI 전달]
1. **STEP1 PASS**: DI-QKD 시뮬(모든 공격→behavior→build_box)이 Delft(V=0.856로 S=2.42 매칭, n=245)를 1000 reps 로 CF/gap/S 밴드 재현.
2. **STEP2**: 주가설 P1(mimic 에서 양자>고전 AUC)은 **naive NLL(CE)로 FAIL**(세 탐지기 동률=엔트로피 교란, 방법론적 발견). **deviance(KL=CE−H(emp), 2N·KL~χ²)** 교정 시 성립(고전 AUC 0.075<0.5, 양자 0.925). **P2 gap↑⇒마진↑ Spearman=1.000.**
3. **STEP2b 상보성**: S-보존 지문변조(iso_S/marginal_matched, S=2.42)는 모델템플릿 AUC=1.0·S-검정 blind(0.5); S-저하 공격(partial=intercept)은 S-검정 0.58→0.98·모델 KL≤0.5. → **defense-in-depth.** 단 **blind_shell 불가능성**(정상이 같은-S 고전거리 유일 최소 → 양자>고전템플릿 우월 불가): 모델우위는 양자 특유 아님.
4. **기하**: 정상=고전껍질 반지름 NCfloor=0.0078≈c·CF²=0.0074(gap 법칙 연결).
5. **비단정·한계**: 소규모·시뮬·존재증명·장비무결성 맥락. supremacy 아님.
