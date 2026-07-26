#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP E (gate_s6) — 마진 법칙 M(CF) 측정: (2,3,2) 미감시 껍질의 크기가 CF 를 어떻게 따르는가.

전제(gate_s5 C-2b): (2,3,2) 정렬 기하에서 iso-S ∧ 고전껍질 을 만족하며 정상과 떨어진 e′ 가 존재.
  δ 를 1e-3→1e-5 로 조여도 KL→normal 유지(0.0088~0.0099), 등식제약 SLSQP 는 잔차 5.9e-13 에서 0.010182.

★기하 선택 — **정렬(aligned) 기하가 주(主)** 다:
  실제 DI-QKD 는 키생성 설정 A₂ 를 상관 최대화를 위해 CHSH 설정과 **정렬**시킨다(A₂=A₀, B₂=B₀).
  이것이 gate_s5.CANON_23 이고, 마진은 여기서 측정해야 물리적으로 의미가 있다.
  일반각(GENERIC, A₂·B₂ 를 임의 방향)에서는 마진이 사라진다 — E2-c 에서 정량화한다(경계 규명).

구성:
  E1  앵커 — 최강 공격점 탐지 AUC(양자템플릿 / 고전NC / S-검정), N ∈ {245, 1000}.
  E2-a NCfloor₂₃(V) ~ c₂₃·CF^α  … Appendix A 차원무관 c·CF² 주장의 (2,3,2) 교차검증
  E2-b M(V) = max KL(e′‖normal) s.t. **등식제약** KL(e′‖c_model)=NCfloor ∧ S=S_t (SLSQP, 잔차<1e-6)
       판정 분기: |α_M − α_NCfloor| < 0.1 → κ 상수 → M ≈ κ·c₂₃·CF² 로 보고
                  아니면 κ(CF) 테이블로만 보고(상수 주장 금지). 앵커: κ ≈ 1.7 @ V=0.856
  E2-c 정렬 의존성 — A₂/B₂ 를 정렬에서 이탈시키며 M 측정(무엇이 마진을 만드는가)
  E3  붕괴 — AUC(N,V) → N*(AUC≥0.9) → N*·M 상수성 → "required trials ≈ N₀/(κ·c₂₃·CF²)"

★왜 SLSQP 인가: 벌점 Powell 은 δ 가 작아질수록 조건수가 나빠지고, 정상점이 −KL(e′‖normal) 의 **정류점**이라
  섭동 없는 초기점에서 즉시 멈춘다. 등식제약 SLSQP 는 잔차 1e-13 수준으로 수렴하며 허용오차 자체가 없다.
  (계보: gate_s5 C-2b 는 12~13 restart 중 1개만 비섭동 초기점이었고 나머지는 ±π 섭동 → 결론 영향 없음.)

가드레일:
  - R6 유지: (2,2,2) 탐지 커버리지 불가능성 정리는 그대로다. M>0 은 **기하가 다른 (2,3,2)** 의 결과이며
    (2,2,2) 커버리지 우위 주장이 아니다.
  - AUC 0.99+ 는 **최악사례 공격점** 하나의 값이다. 클래스 전체 커버는 E3 붕괴곡선이 담당한다.
  - 컨텍스트 분해상 변조의 64%가 미감시 컨텍스트에 산다 — **"A₂만 건드리는 공격" 서사는 금지**(36%는 감시 쪽).
  - 장비 무결성 맥락. 키 보안 아님. 존재 증명 ≠ 실현 가능성. supremacy 아님.

사용: python tools/gate_s6_margin.py --jobs 14 [--restarts 13] [--reps 400]
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G                       # noqa: F401  (S5 가 사용)
import gate_s2_detection as D2                 # auc
import gate_s5_multifacet as S5
import fastq

FASTQ23 = fastq.install_23(S5)                 # 수치동일(<1e-12) 고속화. gate_s5 원본 로직 무변경.

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

OUTDIR = os.path.join("results2", "gate_s6")
OUT_JSON = os.path.join(OUTDIR, "s6.json")
OUT_LAW = os.path.join(OUTDIR, "s6_margin_law.png")
OUT_COL = os.path.join(OUTDIR, "s6_collapse.png")

SEED = 20260725
RESID_TOL = 1e-6                               # 등식제약 잔차 허용(판정용; 실측은 1e-13 수준)
V_GRID = [0.75, 0.80, 0.856, 0.92, 1.00]       # → CF₂₃ ≈ 0.061, 0.131, 0.211, 0.301, 0.414
N_GRID = [245, 1000, 4000, 16000, 64000]
N_E3 = [245, 1000, 4000]
AUC_TARGET = 0.90
ALPHA_MATCH_TOL = 0.10                         # |α_M − α_NCfloor| < 0.1 → κ 상수 주장 허용

ALIGNED_23 = S5.CANON_23.copy()                # 주 기하: A₂=A₀, B₂=B₀ (실제 키생성 정렬)
GENERIC_23 = np.array([np.pi / 4,              # 대조 기하: A₂·B₂ 일반각(비정렬)
                       0.0, 0.0, np.pi / 2, 0.0, 0.7, 0.9,
                       np.pi / 4, 0.0, -np.pi / 4, 0.0, 1.1, 2.0])

JOBS = 1


def log(m):
    print(m, flush=True)


def _pmap(fn, tasks):
    if JOBS <= 1 or len(tasks) < 2:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        return list(ex.map(fn, tasks))


def _ensure_fast():
    if S5.quantum_behavior_23 is not fastq.quantum_behavior_23:
        fastq.install_23(S5, verbose=False)


# ===========================================================================
# 워커
# ===========================================================================
def _floor_worker(t):
    """base13, V → (CF₂₃, S_t, NCfloor₂₃, c_model)."""
    base13, V, restarts, seed = t
    _ensure_fast()
    base13 = np.array(base13)
    e = S5.werner_23(base13, V)
    nc, c_model = S5.nc_floor_23(e, e, np.random.default_rng(seed), restarts)
    return dict(V=float(V), cf=float(S5.contextual_fraction_23(e)), S_t=float(S5.chsh_S_sub(e)),
                ncfloor=float(nc), c_model=c_model.tolist())


def _margin_one(t):
    """등식제약 SLSQP 1 restart: max KL(e′‖normal) s.t. KL(e′‖c_model)=NCfloor, S(e′)=S_t.
       ★초기점은 항상 섭동한다(정상점은 목적함수의 정류점)."""
    key, base13, V, e_normal, S_t, c_model, nc, wide, seed = t
    _ensure_fast()
    base13 = np.array(base13); e_normal = np.array(e_normal); c_model = np.array(c_model)
    rng = np.random.default_rng(seed)
    x0 = np.concatenate([base13, [V]])
    if wide:
        x0[1:13] += rng.uniform(-np.pi, np.pi, size=12)
        x0[13] = rng.uniform(0.6, 1.0)
    else:
        x0[1:13] += rng.uniform(-0.2, 0.2, size=12)
    bnds = [(-4 * np.pi, 4 * np.pi)] * 13 + [(0.0, 1.0)]
    cons = [dict(type="eq", fun=lambda p: S5.eval_kl_23(S5.werner_p(p), c_model) - nc),
            dict(type="eq", fun=lambda p: S5.chsh_S_sub(S5.werner_p(p)) - S_t)]
    try:
        r = minimize(lambda p: -S5.eval_kl_23(S5.werner_p(p), e_normal), x0, method="SLSQP",
                     bounds=bnds, constraints=cons, options=dict(maxiter=400, ftol=1e-12))
    except Exception:
        return (key, None)
    st = S5._shell_stats(S5.werner_p(r.x), e_normal, c_model, nc, S_t)
    resid = max(st["shell_dev"], st["s_dev"])
    if resid >= RESID_TOL:
        return (key, None)
    return (key, dict(kl_to_normal=float(st["kl_to_normal"]), residual=float(resid), p=r.x.tolist()))


def _sample_boxes(e, n, reps, rng):
    """n 라운드를 9 컨텍스트에 균등 배분 → reps 개 경험박스. 약 Laplace(0확률 방지)."""
    per = max(n // 9, 1)
    flat = e.reshape(9, 4)
    out = []
    for _ in range(reps):
        emp = np.empty((9, 2, 2))
        for c in range(9):
            emp[c] = rng.multinomial(per, flat[c]).reshape(2, 2)
        emp += 1.0 / (per * 4.0)
        out.append(emp / emp.sum(axis=(1, 2), keepdims=True))
    return out


def _auc_worker(t):
    """(V,N) 셀: 정상 vs 공격 → 탐지기별 AUC. 점수 = deviance KL(emp‖model), 높을수록 이상."""
    V, N, e_normal, e_atk, c_model, S_t, reps, seed = t
    _ensure_fast()
    e_normal = np.array(e_normal); e_atk = np.array(e_atk); c_model = np.array(c_model)
    rng = np.random.default_rng(seed)
    nb = _sample_boxes(e_normal, N, reps, rng)
    ab = _sample_boxes(e_atk, N, reps, rng)
    cb = _sample_boxes(e_normal, N, reps, rng)         # 대조군 → AUC ≈ 0.5

    def kl(b, m):
        return S5.eval_kl_23(b, m)

    return dict(V=float(V), N=int(N), reps=int(reps),
                quantum=float(D2.auc([kl(b, e_normal) for b in ab], [kl(b, e_normal) for b in nb])),
                classical=float(D2.auc([kl(b, c_model) for b in ab], [kl(b, c_model) for b in nb])),
                S_test_twosided=float(D2.auc([abs(S5.chsh_S_sub(b) - S_t) for b in ab],
                                             [abs(S5.chsh_S_sub(b) - S_t) for b in nb])),
                S_test_lower=float(D2.auc([-S5.chsh_S_sub(b) for b in ab],
                                          [-S5.chsh_S_sub(b) for b in nb])),
                control_quantum=float(D2.auc([kl(b, e_normal) for b in cb],
                                             [kl(b, e_normal) for b in nb])))


# ===========================================================================
# 분석
# ===========================================================================
def loglog_slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = (x > 0) & (y > 0)
    if m.sum() < 2:
        return None
    lx, ly = np.log(x[m]), np.log(y[m])
    A = np.vstack([lx, np.ones_like(lx)]).T
    coef, *_ = np.linalg.lstsq(A, ly, rcond=None)
    pred = A @ coef
    ss_res = float(np.sum((ly - pred) ** 2)); ss_tot = float(np.sum((ly - ly.mean()) ** 2))
    return dict(alpha=float(coef[0]), intercept=float(coef[1]), coef_c=float(np.exp(coef[1])),
                r2=float(1 - ss_res / ss_tot) if ss_tot > 0 else None, n_points=int(m.sum()))


def n_at_auc(ns, aucs, target=AUC_TARGET):
    for i in range(1, len(ns)):
        if aucs[i - 1] < target <= aucs[i]:
            f = (target - aucs[i - 1]) / (aucs[i] - aucs[i - 1])
            return float(np.exp(np.log(ns[i - 1]) + f * (np.log(ns[i]) - np.log(ns[i - 1]))))
    if aucs and aucs[0] >= target:
        return float(ns[0])
    return None


def run_margin(label, base13, V, floor, restarts, seed0):
    """한 (기하, V) 의 마진: restarts 개 SLSQP 를 병렬로. 반환 (M, best_p, n_ok, min_resid)."""
    tasks = [(label, base13, V, floor["e_normal"], floor["S_t"], floor["c_model"], floor["ncfloor"],
              (r > 0), seed0 + 7 * r) for r in range(restarts)]
    res = [r for (_, r) in _pmap(_margin_one, tasks) if r is not None]
    if not res:
        return 0.0, None, 0, None
    best = max(res, key=lambda d: d["kl_to_normal"])
    return best["kl_to_normal"], best["p"], len(res), min(d["residual"] for d in res)


# ===========================================================================
def main():
    global JOBS
    ap = argparse.ArgumentParser(description="STEP E: 마진 법칙 M(CF) — E1 앵커 / E2 스윕 / E2-c 정렬의존 / E3 붕괴.")
    ap.add_argument("--jobs", type=int, default=14)
    ap.add_argument("--restarts", type=int, default=13)
    ap.add_argument("--floor-restarts", type=int, default=8)
    ap.add_argument("--reps", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.restarts, args.floor_restarts, args.reps = 4, 3, 60
    JOBS = args.jobs
    os.makedirs(OUTDIR, exist_ok=True)

    # ---------------- E2-a ----------------
    log("=== E2-a: 정렬 기하(A₂=A₀,B₂=B₀) 정상점별 CF₂₃ · NCfloor₂₃ ===")
    floors = _pmap(_floor_worker, [(ALIGNED_23.tolist(), V, args.floor_restarts, SEED + 11 + 7 * i)
                                   for i, V in enumerate(V_GRID)])
    floors.sort(key=lambda d: d["V"])
    for f in floors:
        f["e_normal"] = S5.werner_23(ALIGNED_23, f["V"]).tolist()
        log("  V=%.3f CF₂₃=%.4f S_sub=%.4f NCfloor₂₃=%.6f  (NCfloor/CF²=%.5f)"
            % (f["V"], f["cf"], f["S_t"], f["ncfloor"], f["ncfloor"] / f["cf"] ** 2))
    fit_floor = loglog_slope([f["cf"] for f in floors], [f["ncfloor"] for f in floors])
    log("  >>> NCfloor₂₃ ≈ c₂₃·CF^α : α=%.3f (예측 2.0), R²=%.4f, c₂₃=%.5f"
        % (fit_floor["alpha"], fit_floor["r2"], fit_floor["coef_c"]))

    # ---------------- E2-b ----------------
    log("\n=== E2-b: 마진 M(V) — 등식제약 SLSQP (잔차<%.0e), restarts=%d ===" % (RESID_TOL, args.restarts))
    margins = []
    for vi, f in enumerate(floors):
        M, p_best, n_ok, resid = run_margin("aligned", ALIGNED_23.tolist(), f["V"], f, args.restarts,
                                            SEED + 500 + 101 * vi)
        kappa = M / f["ncfloor"] if f["ncfloor"] > 0 else None
        margins.append(dict(V=f["V"], cf=f["cf"], ncfloor=f["ncfloor"], M=float(M),
                            kappa=(float(kappa) if kappa is not None else None),
                            n_converged=n_ok, n_restarts=args.restarts,
                            min_residual=resid, p_best=p_best))
        log("  V=%.3f CF=%.4f | M=%.6f (수렴 %d/%d, 최소잔차=%s) | κ=M/NCfloor=%.3f"
            % (f["V"], f["cf"], M, n_ok, args.restarts,
               ("%.1e" % resid) if resid is not None else "—", kappa))
    fit_M = loglog_slope([m["cf"] for m in margins], [m["M"] for m in margins])
    kappas = [m["kappa"] for m in margins if m["kappa"]]
    if fit_M:
        log("  >>> M ≈ CF^α_M : α_M=%.3f (예측 2.0), R²=%.4f" % (fit_M["alpha"], fit_M["r2"]))

    # ★판정 분기: α 일치 → κ 상수 → M ≈ κ·c₂₃·CF²
    dalpha = abs(fit_M["alpha"] - fit_floor["alpha"]) if fit_M else None
    kappa_const = bool(dalpha is not None and dalpha < ALPHA_MATCH_TOL)
    if kappa_const:
        kbar = float(np.median(kappas))
        log("  >>> [전 구간] |α_M − α_NCfloor| = %.3f < %.2f → **κ 상수** (κ̄=%.3f, 범위 [%.3f,%.3f])"
            % (dalpha, ALPHA_MATCH_TOL, kbar, min(kappas), max(kappas)))
        log("      ⇒ M ≈ κ·c₂₃·CF² = %.4f·CF²" % (kbar * fit_floor["coef_c"]))
    else:
        log("  >>> [전 구간] |α_M − α_NCfloor| = %s ≥ %.2f → **상수 주장 금지**. κ(CF) 테이블:"
            % (("%.3f" % dalpha) if dalpha is not None else "—", ALPHA_MATCH_TOL))
        for m in margins:
            log("      CF=%.4f → κ=%.3f" % (m["cf"], m["kappa"]))

    # ★Tsirelson 경계(V=1.00) 제외 재적합 — 경계에서는 양자집합 자체가 공격을 제한해 κ 가 꺾인다.
    #   경계는 물리적 운용점이 아니므로(V=1 = 무잡음) 내부 구간의 법칙을 따로 보고한다.
    sub = [m for m in margins if m["V"] < 0.999]
    fit_M_in = loglog_slope([m["cf"] for m in sub], [m["M"] for m in sub])
    fit_F_in = loglog_slope([m["cf"] for m in sub], [m["ncfloor"] for m in sub])
    ks_in = [m["kappa"] for m in sub]
    dalpha_in = abs(fit_M_in["alpha"] - fit_F_in["alpha"]) if (fit_M_in and fit_F_in) else None
    kappa_const_in = bool(dalpha_in is not None and dalpha_in < ALPHA_MATCH_TOL
                          and max(ks_in) / min(ks_in) <= 1.5)
    kbar_in = float(np.median(ks_in))
    law_in = ("M ~= %.4f * CF^2" % (kbar_in * fit_F_in["coef_c"])) if kappa_const_in else "kappa(CF) table only"
    log("  >>> [경계 V=1.00 제외] α_M=%.3f α_NCfloor=%.3f Δα=%.3f | κ [%.3f,%.3f] (%.2f배, 중앙 %.3f) → κ 상수=%s"
        % (fit_M_in["alpha"], fit_F_in["alpha"], dalpha_in, min(ks_in), max(ks_in),
           max(ks_in) / min(ks_in), kbar_in, kappa_const_in))
    if kappa_const_in:
        log("      ⇒ M ≈ κ̄·c₂₃·CF² = %.4f·CF²  (경계 CF=%.3f 에서는 κ=%.3f 로 꺾임 — 양자집합 경계 효과)"
            % (kbar_in * fit_F_in["coef_c"], margins[-1]["cf"], margins[-1]["kappa"]))

    # ---------------- E2-c: 정렬 의존성 ----------------
    log("\n=== E2-c: 정렬 의존성 — 무엇이 마진을 만드는가 (V=0.856) ===")
    Vc = 0.856
    p_a2 = ALIGNED_23.copy(); p_a2[5], p_a2[6] = 0.7, 0.9
    p_b2 = ALIGNED_23.copy(); p_b2[11], p_b2[12] = 1.1, 2.0
    p_pl = ALIGNED_23.copy(); p_pl[5], p_pl[6], p_pl[11], p_pl[12] = 0.7, 0.0, 1.1, 0.0
    cfgs = [("aligned (A₂=A₀, B₂=B₀)", ALIGNED_23),
            ("A₂만 일반(θ.7,φ.9)", p_a2),
            ("B₂만 일반(θ1.1,φ2.0)", p_b2),
            ("둘 다 일반, φ=0 (평면내)", p_pl),
            ("GENERIC (둘 다 일반, φ≠0)", GENERIC_23)]
    cfloors = _pmap(_floor_worker, [(p.tolist(), Vc, args.floor_restarts, SEED + 61 + 3 * i)
                                    for i, (_, p) in enumerate(cfgs)])
    variants = []
    for (name, p), f in zip(cfgs, cfloors):
        f["e_normal"] = S5.werner_23(p, Vc).tolist()
        M, _, n_ok, resid = run_margin("var", p.tolist(), Vc, f, args.restarts, SEED + 700 + 17 * len(variants))
        variants.append(dict(name=name, cf=f["cf"], ncfloor=f["ncfloor"], M=float(M),
                             kappa=float(M / f["ncfloor"]), n_converged=n_ok))
        log("  %-28s CF=%.4f NCfloor=%.6f M=%.6f κ=%.3f" % (name, f["cf"], f["ncfloor"], M, M / f["ncfloor"]))
    log("  >>> 정렬을 풀수록 마진 감소 → 마진은 **키생성 설정의 정렬**에 기인(일반각·비평면에서 소멸)")

    # ---------------- E1 + E3 ----------------
    log("\n=== E1/E3: 탐지 AUC, N ∈ %s, reps=%d ===" % (N_GRID, args.reps))
    atasks = []
    for vi, (f, m) in enumerate(zip(floors, margins)):
        if m["p_best"] is None:
            continue
        e_atk = S5.werner_p(np.array(m["p_best"]))
        for ni, N in enumerate(N_GRID):
            atasks.append((f["V"], N, f["e_normal"], e_atk.tolist(), f["c_model"], f["S_t"],
                           args.reps, SEED + 900 + 31 * vi + 5 * ni))
    aucs = _pmap(_auc_worker, atasks)
    aucs.sort(key=lambda d: (d["V"], d["N"]))
    for a in aucs:
        log("  V=%.3f N=%6d | quantum=%.3f classical=%.3f S(양측)=%.3f S(단측)=%.3f | 대조=%.3f"
            % (a["V"], a["N"], a["quantum"], a["classical"], a["S_test_twosided"], a["S_test_lower"],
               a["control_quantum"]))
    anchor = [a for a in aucs if abs(a["V"] - 0.856) < 1e-9 and a["N"] in (245, 1000)]
    for a in anchor:
        log("  [E1 앵커] V=0.856 N=%d → quantum=%.3f vs classical=%.3f vs S(양측)=%.3f"
            % (a["N"], a["quantum"], a["classical"], a["S_test_twosided"]))

    collapse = []
    for m in margins:
        rows = [a for a in aucs if abs(a["V"] - m["V"]) < 1e-9]
        ns = [r["N"] for r in rows]; qs = [r["quantum"] for r in rows]
        nstar = n_at_auc(ns, qs)
        collapse.append(dict(V=m["V"], cf=m["cf"], M=m["M"], kappa=m["kappa"], n_star=nstar,
                             nM_star=(float(nstar * m["M"]) if nstar else None),
                             N0_cf2=(float(nstar * m["cf"] ** 2) if nstar else None),
                             auc_by_N={str(r["N"]): r["quantum"] for r in rows}))
        log("  [E3] V=%.3f CF=%.4f M=%.6f → N*(AUC≥0.9)=%s, N*·M=%s, N*·CF²=%s"
            % (m["V"], m["cf"], m["M"], ("%.0f" % nstar) if nstar else "미도달(N≤%d)" % max(N_GRID),
               ("%.3f" % (nstar * m["M"])) if nstar else "—",
               ("%.0f" % (nstar * m["cf"] ** 2)) if nstar else "—"))
    nMs = [c["nM_star"] for c in collapse if c["nM_star"]]
    nM_const = bool(nMs and len(nMs) >= 3 and max(nMs) / min(nMs) <= 1.5)
    N0 = None
    if nMs:
        N0 = float(np.median(nMs))
        log("  >>> N*·M 범위 [%.3f, %.3f] (중앙값 %.3f) — 상수성 %s"
            % (min(nMs), max(nMs), N0, "성립(≤1.5배)" if nM_const else "불충분 → 범위로 보고"))
        n0cf2 = [c["N0_cf2"] for c in collapse if c["N0_cf2"] and c["V"] < 0.999]
        if nM_const and kappa_const_in:
            log("      ⇒ required trials ≈ N₀/(κ·c₂₃·CF²), N₀=%.3f κ=%.3f c₂₃=%.5f → **N ≈ %.0f/CF²**"
                % (N0, kbar_in, fit_F_in["coef_c"], N0 / (kbar_in * fit_F_in["coef_c"])))
            log("      실측 N*·CF² (경계 제외) = %s → 중앙값 %.0f (예측 %.0f 과 대조)"
                % ([("%.0f" % x) for x in n0cf2], float(np.median(n0cf2)),
                   N0 / (kbar_in * fit_F_in["coef_c"])))

    # ---------------- self-check ----------------
    ctrl = [a["control_quantum"] for a in aucs]
    sc = dict(
        control_auc_half=dict(max_abs_dev=float(max(abs(np.array(ctrl) - 0.5))) if ctrl else None,
                              pass_=bool(ctrl and max(abs(np.array(ctrl) - 0.5)) < 0.08),
                              note="정상 vs 정상 → AUC≈0.5"),
        equality_residuals=dict(max_min_residual=float(max([m["min_residual"] or 0 for m in margins])),
                                tol=RESID_TOL,
                                pass_=bool(all((m["min_residual"] or 1) < RESID_TOL for m in margins))),
        cf_monotone=dict(cf=[m["cf"] for m in margins],
                         pass_=bool(all(margins[i]["cf"] < margins[i + 1]["cf"] for i in range(len(margins) - 1)))),
        margin_positive=dict(M=[m["M"] for m in margins], pass_=bool(all(m["M"] > 0 for m in margins))),
    )
    sc["all_pass"] = bool(all(v.get("pass_", True) for k, v in sc.items() if k != "all_pass"))
    log("\n=== self-check ===")
    for k, v in sc.items():
        if k != "all_pass":
            log("  %-22s pass=%s" % (k, v.get("pass_")))
    log("  >>> ALL PASS = %s" % sc["all_pass"])

    payload = dict(
        note=("STEP E 마진 법칙 M(CF) — (2,3,2) **정렬 기하**(A₂=A₀,B₂=B₀ = 실제 키생성 정렬)가 주. "
              "마진은 등식제약 SLSQP(잔차<1e-6)로 측정. E2-c 가 정렬 의존성(일반각에서 소멸)을 정량화한다. "
              "★R6 유지: (2,2,2) 탐지 커버리지 불가능성 정리는 그대로이며 M>0 은 기하가 다른 (2,3,2)의 결과다. "
              "AUC 고값은 **최악사례 공격점** 하나의 값이고 클래스 커버는 E3 붕괴곡선이 담당한다. "
              "변조의 다수가 미감시 컨텍스트에 살지만 전부는 아니므로 'A₂만 건드리는 공격' 서사는 금지. "
              "장비 무결성 맥락(키 보안 아님). 존재 증명 ≠ 실현 가능성. supremacy 아님."),
        params=dict(seed=SEED, aligned_23=ALIGNED_23.tolist(), generic_23=GENERIC_23.tolist(),
                    V_grid=V_GRID, N_grid=N_GRID, N_e3=N_E3, reps=args.reps, restarts=args.restarts,
                    floor_restarts=args.floor_restarts, resid_tol=RESID_TOL, auc_target=AUC_TARGET,
                    alpha_match_tol=ALPHA_MATCH_TOL, fastq=FASTQ23,
                    margin_method="equality-constrained SLSQP (max KL→normal s.t. KL(·‖c_model)=NCfloor, S=S_t)",
                    detectors="quantum=KL(emp‖e_normal), classical=KL(emp‖c_model), S=|S−S_t| 및 −S"),
        self_check=sc,
        e2a_floors=[{k: v for k, v in f.items() if k not in ("c_model", "e_normal")} for f in floors],
        e2b_margins=[{k: v for k, v in m.items() if k != "p_best"} for m in margins],
        e2c_alignment=variants,
        fits=dict(ncfloor_vs_cf=fit_floor, margin_vs_cf=fit_M, alpha_gap=dalpha, kappa_constant=kappa_const,
                  kappa=dict(values=kappas, min=(min(kappas) if kappas else None),
                             max=(max(kappas) if kappas else None),
                             median=(float(np.median(kappas)) if kappas else None)),
                  law=(("M ≈ %.4f·CF²" % (np.median(kappas) * fit_floor["coef_c"])) if kappa_const
                       else "전 구간: κ(CF) 테이블 참조 — 상수 주장 금지"),
                  boundary_excluded=dict(
                      note=("Tsirelson 경계 V=1.00(CF=√2−1, 무잡음)은 양자집합 경계라 공격 자유도가 잘려 κ 가 꺾인다. "
                            "물리적 운용점이 아니므로 내부 구간(V<1)의 법칙을 별도 보고한다."),
                      margin_vs_cf=fit_M_in, ncfloor_vs_cf=fit_F_in, alpha_gap=dalpha_in,
                      kappa_values=ks_in, kappa_median=kbar_in,
                      kappa_ratio=float(max(ks_in) / min(ks_in)), kappa_constant=kappa_const_in,
                      law=(("M ≈ %.4f·CF²" % (kbar_in * fit_F_in["coef_c"])) if kappa_const_in
                           else "κ(CF) 테이블 참조"),
                      boundary_point=dict(cf=margins[-1]["cf"], kappa=margins[-1]["kappa"]))),
        e1_anchor=anchor, e3_auc=aucs, e3_collapse=collapse,
        e3_summary=dict(nM_star_values=nMs, nM_constant=nM_const, N0=N0,
                        formula=("required trials ≈ N₀/(κ·c₂₃·CF²)" if (nM_const and kappa_const)
                                 else "붕괴 미확립 — N*(V) 테이블로만 보고")),
        lineage=dict(c2b_stationary_point_check=(
            "gate_s5 C-2b 의 δ-스윕(Powell)과 SLSQP 는 restarts 중 **첫 1개만** 비섭동 정상점 초기점이었고 "
            "나머지 11~12개는 ±π 섭동이었다(코드 확인). 보고된 해(KL→normal 0.0088~0.0102)는 섭동 초기점에서 "
            "나왔으므로 정류점 이슈는 restart 1개를 낭비했을 뿐 결론에 영향 없음. gate_s6 는 전 초기점을 섭동한다.")),
        limits=("(2,3,2) 도 최소 시나리오이며 일반 정리가 아니다. M 은 Werner 혼합 양자이미지(14 파라미터) 위의 "
                "**국소 최적 하한**이다(전역 최대 보장 없음; restarts 로 완화). 양자 템플릿은 모집단 참값 e_normal 을 "
                "그대로 쓴 이상화(훈련잡음 없음) → AUC 는 상한으로 읽어야 한다. N 은 9 컨텍스트 균등배분. "
                "E2-c 의 GENERIC M=0 은 restart·잔차 1e-13 조건에서의 측정이며 '마진 없음'의 증명은 아니다. "
                "★R6: (2,2,2) 커버리지 우위 주장 아님. 키 보안 아닌 장비 무결성. supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload)
        log("[저장] %s , %s" % (OUT_LAW, OUT_COL))
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(P):
    ms = P["e2b_margins"]; cf = np.array([m["cf"] for m in ms])
    nc = np.array([m["ncfloor"] for m in ms]); M = np.array([m["M"] for m in ms])
    ff, fm = P["fits"]["ncfloor_vs_cf"], P["fits"]["margin_vs_cf"]

    fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
    ax[0].loglog(cf, nc, "o-", color="#C1442E", label="NCfloor₂₃ (α=%.2f, R²=%.3f)" % (ff["alpha"], ff["r2"]))
    if fm:
        ax[0].loglog(cf, M, "s-", color="#7B2FBE", label="margin M (α=%.2f, R²=%.3f)" % (fm["alpha"], fm["r2"]))
    xs = np.linspace(cf.min(), cf.max(), 50)
    ax[0].loglog(xs, nc[0] * (xs / cf[0]) ** 2, "k:", lw=1, label="slope 2 reference")
    ax[0].set_xlabel("CF₂₃"); ax[0].set_ylabel("KL units")
    ax[0].set_title("E2 margin law (aligned geometry)\n%s" % P["fits"]["boundary_excluded"]["law"].replace("≈","~=").replace("·","*").replace("²","^2"))
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3, which="both")

    kap = [m["kappa"] for m in ms]
    ax[1].plot(cf, kap, "o-", color="#2E86C1", label="κ = M / NCfloor₂₃")
    ax[1].axhline(float(np.median([k for k in kap if k])), color="k", ls=":", lw=1,
                  label="median κ = %.2f" % np.median([k for k in kap if k]))
    ax[1].set_ylim(0, max(4.3, max(kap) * 1.2)); ax[1].axhline(4.0, color="r", ls="--", lw=1, label="κ = 4 bound")
    ax[1].set_xlabel("CF₂₃"); ax[1].set_ylabel("κ")
    ax[1].set_title("κ across sweep (constant claim: %s)" % P["fits"]["kappa_constant"])
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    va = P["e2c_alignment"]
    ax[2].bar(range(len(va)), [v["kappa"] for v in va], color="#7B2FBE")
    ax[2].set_xticks(range(len(va)))
    ax[2].set_xticklabels(["aligned", "A₂ gen", "B₂ gen", "both φ=0", "GENERIC"], rotation=20, fontsize=8)
    ax[2].set_ylabel("κ = M / NCfloor₂₃"); ax[2].set_title("E2-c: margin comes from key-setting alignment")
    ax[2].grid(alpha=0.3, axis="y")
    plt.suptitle("STEP E — margin law in (2,3,2) [device integrity; NOT key security; NOT supremacy]", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT_LAW, dpi=130); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(14, 5.6))
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(ms)))
    for i, m in enumerate(ms):
        rows = [a for a in P["e3_auc"] if abs(a["V"] - m["V"]) < 1e-9]
        ns = [r["N"] for r in rows]; q = [r["quantum"] for r in rows]
        ax[0].semilogx(ns, q, "o-", color=cols[i], label="CF=%.3f" % m["cf"])
        ax[1].semilogx([n * m["M"] for n in ns], q, "o-", color=cols[i], label="CF=%.3f" % m["cf"])
    for a in ax:
        a.axhline(0.9, color="r", ls="--", lw=1); a.axhline(0.5, color="k", lw=0.6)
        a.set_ylabel("AUC (quantum template)"); a.legend(fontsize=8); a.grid(alpha=0.3, which="both")
    ax[0].set_xlabel("N (rounds)"); ax[0].set_title("E3 raw: AUC vs N")
    ax[1].set_xlabel("N · M(V)"); ax[1].set_title("E3 collapse: AUC vs N·M  (worst-case attack point)")
    plt.suptitle("STEP E — detection collapse; template is population-ideal → AUC is an upper bound", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT_COL, dpi=130); plt.close()


if __name__ == "__main__":
    main()
