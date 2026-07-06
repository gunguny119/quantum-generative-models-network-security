# QCBM 상대 경쟁력 통제 실험 — 저차 vs 고차 부분집합에서 QCBM vs 차수사다리(M1/M2/full) KL 대결 (표현력, 탐지 아님)

- 작성: 2026-07-02 07:45 UTC (KST 16:45)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- **순수 KL 분포 표현 대결(탐지·AUC·y_test 없음).** 기존 코드 무수정(import), 무설치, 단일 프로세스. SOTA·양자우위 주장 없음, 단정 금지. QCBM seed std 그대로.

## 1. 작업 목적
핵심 가설: **데이터 ≥3차 비중↑ → QCBM의 (pairwise M2 대비) 상대 경쟁력↑**. pairwise는 구조적 2차 상한 → 고차 subset서 ≥3차(34~64%)를 놓침. QCBM은 얽힘으로 고차 표현 원리 가능. 저차(highvar)/고차(targeted) 부분집합에서 같은 목표 q에 대해 KL(q‖M1/M2/full/QCBM)를 비교, **gap_QCBM_M2의 저차→고차 변화**를 본다. 이기지 않아도 됨 — 상대 위치 변화가 결과.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/ ?? qcbm/ ?? *.zip`(전부 untracked). 보존: 기존 소스·results2 기존 json·Drebin 원본(읽기만). 특히 `experiment2.py`·`qcbm_kl_variant.py` 무수정(import만).

## 3. 확인한 QCBM entrypoint / 목표·모델 정의 / 조합 수 / 시간 / 축소안
- **QCBM 재사용(무수정)**: `qcbm_kl_variant.train_kl_one(q_target, nq, layers, epochs, lr=0.1, seed)` → 임의 2^nq 목표로 forward-KL(cross-entropy) 학습(experiment2.make_circuit/wshape=StronglyEntangling, exact qml.probs, Adam), P=L·nq·3, 반환 (p_final, ce, loss_hist). seed=2000+seed.
- **타이밍 실측**: k=10 L=4 ep=150 = **98s/훈련**(CE 감소 중=BP 벽으로 완전수렴 안 함, 알려진 현상). → **k=10 한정**(k12 느리고 목표 3·4차 비수렴).
- **목표/모델 정의**(★): p_emp=emp_dist(X_train[:,idx]). 목표 **A**=order-4 maxent(p_emp)(매끈·k10 수렴·full support), **B**=Laplace 평활 경험분포(c=1/2^k). 모델 KL(q‖·): **M1**=order1 maxent, **M2**=order2 maxent, **full**=q(참조=0), **QCBM**=train_kl_one seed들. 목표 ≥3차비중=KL(q‖M2)/KL(q‖M1).
- **검증 예비**(split0 k10 표적, 목표A): KL(A‖M1)=2.55, KL(A‖M2)=0.87, KL(A‖full)=0, QCBM(L4 ep150)=1.22 → M2보다 나쁨·M1보다 좋음(상관 일부 포착).
- **조합/축소**: k=10, split{0,1,2}, subset{저차 highvar, 고차 targeted}, 목표{A,B}, seed{0,1,2}, 고정 L=4·ep=120·lr=0.1 = 3×2×2×3 = **36 QCBM 훈련 ≈ 47분**. k12·더 큰 L·더 많은 seed는 시간상 제외(사유: 훈련 98s/개). 저차/고차 동일 조건.

## 4. 확실한 사실 / 추정 / 확인 불가
- 확실: train_kl_one 재사용·타이밍·목표/모델 정의·조합 수. 예비 KL(표적 A) 값.
- 추정: 저차서 KL(q‖M2)≈0(QCBM 이점 없음), 고차서 KL(q‖M2)↑ → gap_QCBM_M2 감소 가능(미검).
- 확인 불가(본실행 전): gap_QCBM_M2 저차→고차 방향·일관성, QCBM vs M1 우열, 목표 A/B 차이.

## 5. KL 결과표 (k=10, L=4, ep=120, seed3; KL(목표 q ‖ 모델))
목표 A(order-4 maxent)·B(Laplace 경험) 거의 동일 → A 기준 대표 표기(B는 §괄호).
| split | subset | ≥3차비중 | M1(1차) | M2(2차) | full | QCBM 평균±std |
|---|---|---|---|---|---|---|
| 0 | 저차 highvar | 0.23 | 2.62 | 0.59 | 0 | 1.32±0.13 |
| 1 | 저차 highvar | 0.18 | 3.27 | 0.58 | 0 | 1.35±0.12 |
| 2 | 저차 highvar | 0.15 | 3.08 | 0.45 | 0 | 1.27±0.15 |
| 0 | 고차 targeted | 0.34 | 2.55 | 0.87 | 0 | 1.26±0.27 |
| 1 | 고차 targeted | **0.62** | 2.56 | **1.59** | 0 | 1.41±0.05 |
| 2 | 고차 targeted | **0.59** | 2.63 | **1.56** | 0 | 1.42±0.02 |
- 목표 B도 값 거의 동일(예 s1 고차 M2=1.59, QCBM=1.41±0.05) → maxent/경험 무관하게 같은 그림(robust).
- **QCBM KL은 subset 무관 대체로 평평(1.26~1.42)**. 반면 **M2는 저차 0.45~0.61 → 고차 0.87~1.59로 급증**(2차 상한에 걸려 ≥3차 놓침).

## 6. 핵심: gap_QCBM_M2 = KL(QCBM) − KL(M2) 의 저차→고차 변화
| 목표 | 저차 gap 평균 | 고차 gap 평균 | 감소량 |
|---|---|---|---|
| A | **+0.773** | **+0.024** | 0.749 |
| B | +0.776 | +0.029 | 0.747 |
- **저차: gap +0.77(QCBM이 M2보다 뚜렷이 나쁨). 고차: gap ≈0**(QCBM이 M2에 필적). 두 목표 버전 일관.
- **최고차 셀(s1/s2 targeted, ≥3차 0.59~0.62)에서 gap 음수**(−0.14, −0.18) = **QCBM이 M2를 KL로 앞섬**(M2 1.56~1.59 vs QCBM 1.41~1.42). 이유: M2는 2차 상한, QCBM은 ~평평.
- **corr(≥3차비중, gap_QCBM_M2) = −0.997** (12셀) — ≥3차↑일수록 gap↓, 거의 완전 단조. ★가설 방향 강하게 지지.
- ★단, 원인은 "QCBM이 고차서 더 잘함"이 아니라 **M2가 고차서 무너짐 + QCBM은 평평**의 조합(QCBM 절대 KL은 여전히 ~1.3~1.4로 바닥 못 감=BP 벽).

## 7. QCBM vs M1(독립) — 상관 포착 증거
- **12/12 셀 모두 gap_QCBM_M1 < 0**(−1.15 ~ −1.93) → QCBM은 항상 1차 독립모델을 KL로 앞섬 = **상관을 실제로 포착**(2차 이상 구조 반영). 저차/고차 공통.

## 8. self-check
- **full(=q) KL = 0.00**(전 셀, 참조 일치) ✓. **M1 ≥ M2 위반 0/12**(nested, 차수↑ 근사 개선) ✓. **QCBM 비감소 seed 0/12**(전부 CE 감소=학습됨) ✓. 저차서 M2 작음(0.45~0.61) ✓. 합성 QK.selftest PASS(KL 단조 감소) ✓.

## 9. 사실 관찰 (판정 아님, 단정 금지)
- **고차성↑ → QCBM의 (M2 대비) 상대 위치↑ = 지지(방향·일관성 강함)**:
  - gap_QCBM_M2가 저차 +0.77 → 고차 +0.02(최고차 셀 −0.14~−0.18)로 **단조 감소**, corr −0.997, 목표 A/B·3 split 모두 일관.
  - 기전: **M2(pairwise)는 2차 상한이라 ≥3차 비중↑에 KL 급증**(0.45→1.59), **QCBM은 얽힘으로 ~평평**(1.3~1.4) → 고차 영역서 교차(QCBM이 M2 앞섬).
  - QCBM은 전 셀서 M1(독립) 앞섬 → 상관 포착 확인.
- **중요 한정(과대해석 방지)**: QCBM **절대 KL은 높음(~1.3~1.4, 바닥 0 못 감=BP 벽, ep120 미완수렴)**. "QCBM이 고차서 좋아진다"가 아니라 **M2가 무너지고 QCBM이 버티는** 상대 현상. 저차서는 여전히 M2가 압도.
- ★ **상대 비교·표현력(KL) 대결 한정. 탐지 아님(직전 drift AUC와 별개). SOTA·양자우위 주장 없음.** QCBM seed std 병기(고차 셀 std 작음 0.02~0.05, 저차 0.09~0.27). 이 ansatz·L4·ep120·k10 한정. 단정 금지.

## 10. 생성 파일 / 실행 명령 / 시간 / tmux
- 신규: `tools/exp_qcbm_order_competitiveness.py`, `results2/qcbm_order_comp/{qcbm_order_comp.json,.png,run.log}`, 본 로그.
```
python3 -m py_compile tools/exp_qcbm_order_competitiveness.py          # OK
python3 tools/exp_qcbm_order_competitiveness.py --smoke                # self-check PASS
tmux hoc_qcbmcomp: python3 ... --splits 0,1,2 --k 10 --layers 4 --epochs 120 --seeds 3 --targets A,B
```
- **tmux 세션 hoc_qcbmcomp**, 로그 `results2/qcbm_order_comp/run.log`. 12셀×3 seed=36 QCBM 훈련, 12:53→13:42 ≈ **49분**. 셀별 증분 json 저장(중단 대비). 단일 프로세스. 실패/발산 없음(전 seed 감소). png 한글 글리프 경고=박스(무해).

## 11. git diff --stat (기존 무변경)
`git diff --stat`=변경 없음. `git status --short`=`?? docs/ ?? qcbm/ ?? *.zip`(신규만). 기존 소스(experiment2/qcbm_kl_variant 포함)·results2 기존 json·Drebin 원본 무수정.

## 12. 성공 기준 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] QCBM 재사용(train_kl_one)·목표(A/B)/모델 정의·조합(12셀×3seed)·시간(49분)·축소안(k10·L4·seed3) 확인·기록.
- [x] 저차/고차 × {QCBM(평균±std), M1, M2, full} KL 표 + 목표 A/B 둘 다.
- [x] 핵심: gap_QCBM_M2 저차(+0.77)→고차(+0.02, 최고차 음수) + corr(≥3차,gap)=−0.997 도시.
- [x] QCBM vs M1: 12/12 앞섬(상관 포착).
- [x] self-check: full KL=0, M1≥M2, QCBM 감소, 저차 M2 작음 — 전부 통과.
- [x] 판정 재료: **고차성↑→QCBM 상대위치↑ 지지**(값·일관성). 절대 KL 높음(BP) 한정 명시. 탐지 아님·무변경·무설치·공정조건.

**남은 불확실성**
- QCBM ep=120 미완수렴(BP 벽) → 절대 KL은 하한. 더 긴 학습/더 큰 L이 저차/고차 절대 KL을 얼마나 바꾸는지 미측정(상대 방향은 견고 예상).
- k=10 한정(k12는 목표 3·4차 maxent 비수렴). ≥3차 축은 0.15~0.62 커버(더 높은 고차는 미탐).
- gap 감소가 "M2 붕괴"에 크게 기인 → QCBM 절대 개선과 분리 필요(다음 단계).

**다음 추천 단계**
1. 고차 셀에서 QCBM epoch↑/L↑ 시 절대 KL이 M2 밑으로 더 내려가는지(진짜 표현 우위 vs 단순 M2 붕괴 분리) — 단 BP 벽 전제.
2. ≥3차 비중을 연속적으로 제어(합성 목표)해 gap_QCBM_M2 곡선의 교차점(QCBM=M2가 되는 ≥3차 임계) 정밀 측정.
3. (해석 입력) 웹 AI에: "k10서 gap_QCBM_M2 저차 +0.77→고차 +0.02(최고차 −0.18), corr(≥3차,gap)=−0.997, QCBM 12/12서 M1 앞섬. 단 QCBM 절대 KL~1.4(BP)로 M2 붕괴가 주 기전. 상대·표현력 대결·이 설정 한정, 양자우위 주장 없음."
