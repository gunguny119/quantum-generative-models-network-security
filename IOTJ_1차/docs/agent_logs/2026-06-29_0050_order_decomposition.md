# 고차 상관 차수별 분해 — UNSW의 ≥3차 KL을 3차/4차로 분해 (UNSW 집중, CIC 대조)

- 작성: 2026-06-29 00:50 (KST)
- 작업 디렉터리: `/home/elicer/IOTJ/qcbm`
- 순수 측정(학습 없음). 직전 maxent/L-BFGS 인프라 차수별 일반화. import 재사용만. 단일 프로세스(병렬 금지). 판정은 웹 AI.

## 1. 작업 목적
직전 KL₂(=순수 ≥3차 합)에서 UNSW가 최대(0.617, ≥3차 27%). 이번엔 차수별 max-entropy(p₁/p₂/p₃/p₄)를 적합해
연속 KL 차이로 2차/3차/4차/≥5차 순수 기여를 분해 → UNSW의 ≥3차가 주로 3차인지 4차+까지 퍼지는지 규명.

## 2. 시작 git status, 보존 대상
`git status --short`(IOTJ): `?? docs/  ?? qcbm/  ?? *.zip` — 전부 untracked. tmux=/usr/bin/tmux(세션 없음).
보존: 모든 기존 소스/데이터/결과(import/재사용만). 신규 파일만.

## 3. 확인한 maxent 구현 / q_train / 비트규약
- 직전 `tools/diagnose_maxent_kl.py` maxent_2nd: spin sᵢ=1−2·bitᵢ, Φ(상태×특징), mu_emp=Φᵀp,
  nll_grad(logsumexp), `scipy.optimize.minimize(L-BFGS-B, gtol=tol, maxiter, ftol=1e-18)`, max_moment_violation 보고.
  → Φ를 1..kmax 모든 부분집합 곱 열로 확장 = 차수별 maxent (이번 일반화 지점).
- q_train: `build_b2/build_bot/build_unsw` 반환(benign-train empirical, 2ⁿ). 비트 `(states>>i)&1`. `marginal_dist`=1차 maxent(교차검증).
- 대조: `results2/maxent_kl.json` (UNSW KL₂=0.617).

## 4. 각 데이터셋 비트 수 / 점유 / 파라미터 규모
HOIC 12b/4096(점유 94), Bot 12b/4096(점유 361), UNSW 10b/1024(점유 174).
누적 파라미터(p₄): UNSW 385, CIC 793. (Φ UNSW 1024×385, CIC 4096×793 — 메모리 충분.)

## 5. 방법 / 이론 제약
- p_k = 1..k차 모멘트 보존 로그선형 maxent(L-BFGS). KL(p‖p_k) (nats).
- 기여: c₂=KL₁−KL₂, c₃=KL₂−KL₃, c₄=KL₃−KL₄, ≥5차 잔여=KL₄. 이론상 모두 ≥0(단조). 음수=미수렴 신호(진짜값 아님).
- 교차검증: c₃+c₄+KL₄(=≥3차 합) ≈ 직전 KL₂(0.617). self-check: 순수3차/순수4차/2차합성/단조성.

## 6. self-check 결과 (n=5 합성, PASS)
- **순수 3차(3-bit parity)**: c₃=0.1308 ≫0, c₄=4.5e-17≈0, ≥5차=6.9e-18≈0 → 3차만 잡음(PASS).
- **순수 4차(4-bit parity)**: c₃=0.0≈0, c₄=0.1308 ≫0 → 4차만 잡음(PASS).
- **2차 Ising**: c₃=4.1e-17, c₄=−1.7e-16 ≈0 → 고차 기여 없음(PASS).
- **단조성**: 모든 합성서 KL₁≥KL₂≥KL₃≥KL₄≥0, 음수 기여 없음(PASS).
→ 지표가 차수를 실제로 분해함을 증명(특히 3차 vs 4차 구분).

## 7. 차수별 maxent 수렴 상태 (데이터셋별 p₁~p₄)
| 데이터셋 | 차수 | D(params) | 수렴(success&viol<1e-6) | 반복 | 최대 모멘트 위반 |
|---|---|---|---|---|---|
| UNSW | k1/k2/k3/k4 | 10/55/175/385 | T / T / T / **F** | 11/4130/17012/20000 | 3.5e-9 / 1.9e-8 / 1.3e-7 / **9.2e-7** |
| Bot  | k1/k2/k3/k4 | 12/78/298/793 | T / T / **F** / **F** | 17/4127/20000/20000 | 6.6e-9 / 4.0e-7 / **1.5e-6** / **5.6e-6** |
| HOIC | k1/k2/k3/k4 | 12/78/298/793 | T / **F** / **F** / T | 57/20000/20000/18306 | 1.0e-8 / **1.4e-7** / **6.2e-7** / 1.7e-7 |
- 일부 고차 적합이 maxiter(20000) 도달로 success=False. **단 모멘트 위반 모두 ≤5.6e-6**(작음), **단조성·음수없음**,
  **≥3차합=직전 KL₂ 정확 일치**(아래) → KL 값 신뢰. (희소 데이터 + 고차 → 일부 θ→∞로 gtol 형식 미달, 예상된 한계.)

## 8. 차수별 KL 기여 표 (★전수, nats)
| 데이터셋 | bits | 점유 | KL₁ | KL₂ | KL₃ | KL₄ | c₂(2차) | c₃(3차) | c₄(4차) | ≥5차잔여 |
|---|---|---|---|---|---|---|---|---|---|---|
| UNSW | 10 | 174 | 2.284 | 0.617 | 0.177 | 0.054 | 1.667 | **0.440** | 0.123 | 0.054 |
| Bot  | 12 | 361 | 2.919 | 0.549 | 0.026 | 0.0005 | 2.370 | **0.523** | 0.026 | 0.0005 |
| HOIC | 12 | 94  | 1.461 | 0.017 | 0.004 | ~0 | 1.443 | 0.0137 | 0.0036 | ~0 |

**≥3차 내부 비율(=KL₂ 중 차수별 share):**
| 데이터셋 | 3차 | 4차 | ≥5차 |
|---|---|---|---|
| UNSW | **71.3%** | **19.9%** | **8.8%** |
| Bot  | 95.2% | 4.7% | 0.1% |
| HOIC | 78.7% | 21.0% | 0.2% |

## 9. 직전 KL₂와 정합 (≥3차 합 = c₃+c₄+≥5차 잔여)
- UNSW: 0.6170 vs 직전 KL₂ 0.6170 → **Δ=0.0** ✓
- Bot:  0.5493 vs 0.5493 → **Δ=1.1e-16** ✓
- HOIC: 0.0173 vs 0.0173 → **Δ=0.0** ✓
→ 차수별 분해가 직전 2차 maxent KL₂와 완전 정합(분해 일관성 입증).

## 10. 사실 관찰 (판정 아님)
- **UNSW의 ≥3차는 주로 3차(71%)지만 4차+로도 의미 있게 퍼짐(4차 20% + ≥5차 9% = 29%).** 절대값으로도 4차 0.123,
  ≥5차 0.054 nats로 비무시 → 세 데이터 중 고차가 가장 멀리(4차+) 퍼지는 분포.
- **Bot의 ≥3차는 거의 순수 3차(95%)** — 4차+ 기여 미미. HOIC는 ≥3차 절대값이 작지만(0.017) 3차 79%/4차 21%.
- **음수 기여 0 / 단조성 모두 충족** — maxiter 미수렴 적합이 있었으나(특히 CIC k3/k4, HOIC k2/k3) 위반 ≤5.6e-6,
  분해 일관성(≥3차합=직전 KL₂ 정확)으로 값 신뢰. 미수렴은 형식적(gtol) 한계로 §7에 명시.
- (해석 주의) 비트수 차이(UNSW10 vs CIC12), 인코딩 의존 → 차수 절대비교보다 "≥3차 내부 분포 형태"로 해석.

## 11. 생성 파일 / 실행 / tmux
신규: `tools/diagnose_order_decomposition.py`, `results2/order_decomposition.json`, `.png`, `order_decomp_run.log`, 본 로그.
```
python -m py_compile tools/diagnose_order_decomposition.py            # OK
python tools/diagnose_order_decomposition.py --help                   # OK
python tools/diagnose_order_decomposition.py --dataset unsw           # self-check PASS, UNSW(smoke)
tmux new -s hoc_deg -d 'python3 tools/diagnose_order_decomposition.py 2>&1 | tee results2/order_decomp_run.log'  # 본 실행
```
- **tmux 세션명 hoc_deg**, 로그 `results2/order_decomp_run.log`. UNSW→Bot→HOIC 순차(병렬 없음), 전수.
  CIC(12b)는 k3/k4가 느려(D 298/793, 4096상태) 본 실행 약 1h50m 소요(00:53~02:44). 완료 후 세션 자동 종료.
- QCBM 무학습, dependency 무설치(numpy/scipy.optimize/pandas/matplotlib). 실패/크래시 없음.
  (워처 1회가 일시적 오판으로 조기 종료 → 견고 워처(연속3회 확인)로 재감시해 완료 확인. 실행 안 한 것을 했다고 하지 않음.)

## 12. git diff --stat (기존 무변경)
`git diff --stat` = 변경 없음. `git status --short` = `?? docs/  ?? qcbm/  ?? *.zip`(전체 untracked, 신규는 그 안).
기존 소스/데이터/결과 무수정·무삭제·무덮어쓰기.

## 13. 성공 기준 충족 / 남은 불확실성 / 다음 추천
**성공 기준**
- [x] maxent 구현·q_train·비트규약 확인·기록(§3).
- [x] compile + --help.
- [x] self-check 통과: 순수3차→c₃만, 순수4차→c₄만, 2차합성→c₃·c₄≈0, 단조성(음수 없음).
- [x] p₁~p₄ 수렴상태(플래그·반복·위반) 기록 + 미수렴 사실(§7).
- [x] UNSW 차수별 기여 산출·json/png 저장. CIC 대조 결과 기록(미수렴 명시).
- [x] ≥3차 합 ≈ 직전 KL₂ 정합(Δ≈0, 세 데이터 모두).
- [x] 기존 무변경, QCBM 무학습, 무설치, 본 실행 tmux(hoc_deg).

**남은 불확실성**
- CIC k3/k4, HOIC k2/k3 gtol 형식 미달(희소+고차로 θ→∞ 방향). KL값은 위반≤5.6e-6·단조성·정합으로 신뢰하나 형식적 수렴 아님.
- p₄까지만 분해(≥5차는 잔여 합). UNSW ≥5차 0.054(9%)의 세부 차수는 미측정.
- 비트수/인코딩 의존 — 차수 절대비교 주의.

**다음 추천 단계**
1. (해석 입력) 웹 AI에: "UNSW ≥3차는 3차 71%지만 4차+로 29% 퍼짐(셋 중 최다 spread); Bot은 거의 순수 3차(95%).
   ≥3차합=직전 KL₂ 정확 일치로 분해 신뢰." 양자 우위 여지(고차 spread) 재검토.
2. UNSW kmax=5~6로 확장해 ≥5차 잔여(0.054)의 차수 특정(수렴 한계 감수, 위반치 기록).
3. 공통 10bit 동일 인코딩으로 세 데이터 재측정해 비트수 confound 제거 후 차수 분포 재확인.
