#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP A (gate_s4) — 파라미터 효율: 같은 커버리지를 **얼마나 싸게** 사는가.

★ A-0 주장 강도(정정): 지수 아님, **다항**이다.
   Carathéodory: 고전 사다리가 NCfloor 에 도달하는 데 필요한 컴포넌트 수 H ≤ d+1 (d = NC 폴리토프 차원).
   (2,m,2)에서 d = 2m + m² → H* = O(m²), 고전 파라미터 = H(2m+1) − 1 = **O(m³)**.
   양자는 상태 1 + 각도 4m(=party 2 × 설정 m × (θ,φ) 2) → 1+4m = **O(m)**.
   ∴ 삼차 대 선형 = **다항 분리**(Anschuetz & Gao 범주). 지수는 LP 꼭짓점 수 2^(2m) 에만 해당하고
   학습기 파라미터에는 해당 없음. "지수적"이라 쓰면 즉시 반박당한다.

구성:
  A-1  H*(ε) 실측 — CF ∈ {0.131,0.211,0.301,0.414}(양자 Werner 운용점), H ∈ {1,2,3,4,6,8,9,12,16},
       모집단(N=inf)에서 KL(H) − NCfloor < ε=1e-4 인 최소 H. ★검증점: (2,2,2)에서 H* ≈ 9 (= d+1 = 9).
  A-2  (2,3,2) 스케일링 — gate_s5 의 C 인프라 재사용. 예측 H* = d+1 = 16, 고전 111 vs 양자 13.
  A-3  표본복잡도 — 각 모델(H, quantum)을 n ∈ {500,2000,10000} 로 학습 → held-out(모집단 참값) KL.
       파라미터가 많으면 데이터가 더 든다 = A 와 B 를 잇는 실무 진술.

정합: R1′(2,2,2)는 G 트랙 무수정 사용. (2,3,2)만 신규 사다리(classical_behavior_m). R3 훈련·평가 분리.
      R5 결정성·JSON. R6 탐지 커버리지 우위 주장 없음(파라미터 카운팅 축). R7 동일 시드.

사용: python tools/gate_s4_params.py [--jobs 13] [--quick]
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
import gap_vs_cf_v0 as G
import gate_s2_detection as D2
import gate_s5_multifacet as S5
import fastq

FASTQ = fastq.install(G)                      # 성능 전용(수치동일 <1e-12). gap_vs_cf_v0.py 무수정.
FASTQ23 = fastq.install_23(S5)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt               # noqa: E402

OUTDIR = os.path.join("results2", "gate_s4")
OUT_JSON = os.path.join(OUTDIR, "s4.json")
OUT_PNG = os.path.join(OUTDIR, "s4_scaling.png")

SEED = 20260725
EPS_STAR = 1e-4                                # H* 판정 임계 (KL(H) − NCfloor < ε)
V_GRID = [0.80, 0.856, 0.92, 1.00]             # → CF ≈ 0.131, 0.211, 0.301, 0.414
H_GRID_22 = [1, 2, 3, 4, 6, 8, 9, 12, 16]      # 9(=d+1) 필수 포함
H_GRID_23 = [1, 2, 4, 8, 12, 14, 16, 18, 20]   # 16(=d+1) 필수 포함
N_GRID = [500, 2000, 10000]                    # A-3
H_SAMPLE = [1, 2, 4, 9, 16]                    # A-3 대상 사다리
TABLE1_H1_REF = 0.3681                         # 기존 Table I (CF=0.6, N=inf, restarts_c=2) 연속성 확인값

# ★A-2 는 **일반각** (2,3,2) 를 써야 한다. gate_s5 의 CANON_23 은 A₂=A₀·B₂=B₀ (키생성 복제) 라
#   9 컨텍스트가 중복되어 실질 (2,2,2) 로 퇴화 → H* 가 자라지 않는다(스케일링 측정 무효).
#   여기서는 세 번째 설정을 일반 방향으로 두어 폴리토프 차원 d=15 를 실제로 채운다.
GENERIC_23 = np.array([np.pi / 4,
                       0.0, 0.0, np.pi / 2, 0.0, 0.7, 0.9,              # A0=Z, A1=X, A2=일반
                       np.pi / 4, 0.0, -np.pi / 4, 0.0, 1.1, 2.0])      # B0, B1, B2=일반


def log(m):
    print(m, flush=True)


def dim_nc(m):
    """NC(=국소) 폴리토프의 아핀 차원: 주변확률 2m + 상관 m². (2,2,2)→8, (2,3,2)→15."""
    return 2 * m + m * m


def params_classical(H, m):
    """혼합가중 H−1 + 각 party 설정별 베르누이 mH ×2 → H(2m+1) − 1."""
    return H * (2 * m + 1) - 1


def params_quantum(m):
    """상태 1 + party 2 × 설정 m × (θ,φ) → 1 + 4m."""
    return 1 + 4 * m


# ===========================================================================
# (2,m,2) 고전 잠재모델 — G.classical_behavior 의 m-일반화 (m=2 에서 원본과 수치동일 검증)
# ===========================================================================
def classical_behavior_m(p, H, m, ctx):
    z = np.clip(p[:H], -30, 30)
    pi = np.exp(z - z.max()); pi = pi / pi.sum()
    u = np.clip(p[H:H + m * H], -30, 30).reshape(m, H)
    w = np.clip(p[H + m * H:H + 2 * m * H], -30, 30).reshape(m, H)
    pa1 = 1.0 / (1.0 + np.exp(-u))
    pb1 = 1.0 / (1.0 + np.exp(-w))
    pA = np.stack([1 - pa1, pa1])                       # (a, x, H)
    pB = np.stack([1 - pb1, pb1])
    xs = np.array([c[0] for c in ctx]); ys = np.array([c[1] for c in ctx])
    return np.einsum('act,bct,t->cab', pA[:, xs, :], pB[:, ys, :], pi)


def fit_classical_m(emp, e_true, H, m, ctx, rng, restarts, evalkl):
    ndim = H * (2 * m + 1)
    best = None
    for _ in range(restarts):
        x0 = rng.normal(0, 1.0, size=ndim)
        r = minimize(lambda p: G._ce(emp, classical_behavior_m(p, H, m, ctx)), x0, method="Powell",
                     options=dict(maxiter=2000, xtol=1e-6, ftol=1e-9))
        val = G._ce(emp, classical_behavior_m(r.x, H, m, ctx))
        if best is None or val < best[0]:
            best = (val, r.x)
    return evalkl(e_true, classical_behavior_m(best[1], H, m, ctx))


# ===========================================================================
# 워커
# ===========================================================================
def _w_h22(t):
    """A-1 셀: (V, H) → 모집단 KL. e_true 를 emp 로(N=inf)."""
    V, H, restarts, seed = t
    e = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
    kl = G.fit_classical(e, e, H, np.random.default_rng(seed), restarts)
    return dict(V=V, H=H, kl=float(kl))


def _w_h23(t):
    """A-2 셀: (V, H) → (2,3,2) 모집단 KL."""
    V, H, restarts, seed = t
    e = S5.werner_23(GENERIC_23, V)
    kl = fit_classical_m(e, e, H, 3, S5.CTX9, np.random.default_rng(seed), restarts, S5.eval_kl_23)
    return dict(V=V, H=H, kl=float(kl))


def _w_sample(t):
    """A-3 셀: (model, n, seed) → 훈련 emp(n) 학습 → 모집단 참값 held-out KL."""
    model, n, V, restarts, seed = t
    e = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
    rng = np.random.default_rng(seed)
    emp = G.sample_emp(e, n, rng)                        # 훈련
    if model == "quantum":
        kl = G.fit_quantum(emp, e, rng, restarts)        # 평가는 참값 e (R3 훈련·평가 분리)
    elif model == "ncfloor":
        kl = G.nc_floor(emp, e, rng, restarts)
    else:
        kl = G.fit_classical(emp, e, int(model[1:]), rng, restarts)
    return dict(model=model, n=n, seed=int(seed), kl=float(kl))


def run_many(fn, tasks, jobs):
    if jobs <= 1:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(fn, tasks))


def h_star(kls, floor, Hs, eps=EPS_STAR):
    for H in Hs:
        if kls[H] - floor < eps:
            return H
    return None


# ===========================================================================
def run_self_check(restarts_hi=8):
    """(1) Table I 연속성: CF=0.6·N=inf·원 프로토콜(restarts_c=2)에서 H=1 KL ≈ 0.3681.
       (2) restarts 를 늘리면 개선되는가(= Table I H=1 은 최적화-제한 값이었는가) — A 의 핵심 주의사항.
       (3) classical_behavior_m(m=2) ≡ G.classical_behavior (수치동일)."""
    checks = {}
    # ★ CF = 2λ−1 (PR·균등 등방혼합) → Table I 의 CF=0.6 은 **λ=0.8**. λ=0.6 은 CF=0.2 로 다른 셀이다.
    LAM_CF06 = 0.8
    e = G.mix(G.pr_box(), G.uniform_box(), LAM_CF06)
    cell = G.run_cell(G.pr_box(), G.uniform_box(), LAM_CF06, None, 5, 3, 2, [1])
    h1_orig = cell["per_model"]["H1"]["mean"]
    checks["table1_H1_continuity"] = dict(value=float(h1_orig), ref=TABLE1_H1_REF, cf=float(cell["cf"]), lam=LAM_CF06,
                                          diff=float(abs(h1_orig - TABLE1_H1_REF)),
                                          pass_=bool(abs(h1_orig - TABLE1_H1_REF) < 0.02))
    h1_hi = G.fit_classical(e, e, 1, np.random.default_rng(SEED), restarts_hi)
    checks["restarts_sensitivity_H1"] = dict(restarts2=float(h1_orig), restarts_hi=float(h1_hi),
                                             improved=bool(h1_hi < h1_orig - 1e-3),
                                             note="개선되면 Table I H=1 은 최적화-제한 값(이론 반증 아님)")
    # m-일반화가 m=2 에서 원본과 동일한가
    rng = np.random.default_rng(7)
    worst = 0.0
    for _ in range(200):
        H = int(rng.integers(1, 6)); p = rng.normal(0, 1.5, size=5 * H)
        worst = max(worst, float(np.abs(classical_behavior_m(p, H, 2, G.CTX) - G.classical_behavior(p, H)).max()))
    checks["classical_m_equiv_m2"] = dict(max_abs_diff=worst, pass_=bool(worst < 1e-12))
    checks["all_pass"] = bool(checks["table1_H1_continuity"]["pass_"] and checks["classical_m_equiv_m2"]["pass_"])
    return checks


def main():
    ap = argparse.ArgumentParser(description="STEP A 파라미터효율: H*(ε) 실측 + (2,3,2) 스케일링 + 표본복잡도.")
    ap.add_argument("--jobs", type=int, default=13)
    ap.add_argument("--restarts", type=int, default=8)
    ap.add_argument("--restarts23", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.restarts, args.restarts23, args.seeds = 2, 2, 4
    os.makedirs(OUTDIR, exist_ok=True)

    log("=== STEP A self-check ===")
    sc = run_self_check(args.restarts)
    log("  Table I 연속성 H1(CF=0.6,N=inf,restarts2)=%.4f (ref %.4f) pass=%s"
        % (sc["table1_H1_continuity"]["value"], TABLE1_H1_REF, sc["table1_H1_continuity"]["pass_"]))
    log("  restarts↑ 재확인: H1 %.4f → %.4f (개선=%s)"
        % (sc["restarts_sensitivity_H1"]["restarts2"], sc["restarts_sensitivity_H1"]["restarts_hi"],
           sc["restarts_sensitivity_H1"]["improved"]))
    log("  classical_behavior_m(m=2) ≡ 원본: max|diff|=%.2e pass=%s"
        % (sc["classical_m_equiv_m2"]["max_abs_diff"], sc["classical_m_equiv_m2"]["pass_"]))

    # ---------------- A-1: (2,2,2) H*(ε) ----------------
    log("\n=== A-1: (2,2,2) H*(ε=%.0e) — 예측 H* = d+1 = %d ===" % (EPS_STAR, dim_nc(2) + 1))
    floors22 = {}
    for V in V_GRID:
        e = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
        floors22[V] = dict(cf=float(G.contextual_fraction(e)),
                           ncfloor=float(G.nc_floor(e, e, np.random.default_rng(SEED + 11), args.restarts)))
    tasks = [(V, H, args.restarts, SEED + 100 * vi + H) for vi, V in enumerate(V_GRID) for H in H_GRID_22]
    res22 = run_many(_w_h22, tasks, args.jobs)
    a1 = []
    for V in V_GRID:
        kls = {r["H"]: r["kl"] for r in res22 if r["V"] == V}
        fl = floors22[V]["ncfloor"]
        hs = h_star(kls, fl, H_GRID_22)
        a1.append(dict(V=V, cf=floors22[V]["cf"], ncfloor=fl, kl_by_H={str(h): kls[h] for h in H_GRID_22},
                       h_star=hs, params_at_hstar=(params_classical(hs, 2) if hs else None),
                       floor_violation=bool(min(kls.values()) < fl - 1e-9)))
        log("  V=%.3f CF=%.3f NCfloor=%.5f | H*=%s (고전 %s params vs 양자 %d) | KL: %s"
            % (V, floors22[V]["cf"], fl, hs, params_classical(hs, 2) if hs else "—", params_quantum(2),
               " ".join("H%d=%.5f" % (h, kls[h]) for h in H_GRID_22)))
    hstars = [c["h_star"] for c in a1 if c["h_star"]]
    log("  >>> 검증점: H* 중앙값=%s vs Carathéodory 예측 d+1=%d"
        % (int(np.median(hstars)) if hstars else None, dim_nc(2) + 1))

    # ---------------- A-2: (2,3,2) 스케일링 ----------------
    log("\n=== A-2: (2,3,2) 스케일링 — 예측 H* = d+1 = %d (일반각: A₂·B₂ 비퇴화) ===" % (dim_nc(3) + 1))
    V23 = [0.856, 1.00]
    floors23 = {}
    for V in V23:
        e = S5.werner_23(GENERIC_23, V)
        nc, _ = S5.nc_floor_23(e, e, np.random.default_rng(SEED + 13), args.restarts)
        floors23[V] = dict(cf=float(S5.contextual_fraction_23(e)), ncfloor=float(nc))
    tasks = [(V, H, args.restarts23, SEED + 200 * vi + H) for vi, V in enumerate(V23) for H in H_GRID_23]
    res23 = run_many(_w_h23, tasks, args.jobs)
    a2 = []
    for V in V23:
        kls = {r["H"]: r["kl"] for r in res23 if r["V"] == V}
        fl = floors23[V]["ncfloor"]
        hs = h_star(kls, fl, H_GRID_23)
        a2.append(dict(V=V, cf=floors23[V]["cf"], ncfloor=fl, kl_by_H={str(h): kls[h] for h in H_GRID_23},
                       h_star=hs, params_at_hstar=(params_classical(hs, 3) if hs else None)))
        log("  V=%.3f CF23=%.3f NCfloor=%.5f | H*=%s (고전 %s params vs 양자 %d) | KL: %s"
            % (V, floors23[V]["cf"], fl, hs, params_classical(hs, 3) if hs else "—", params_quantum(3),
               " ".join("H%d=%.5f" % (h, kls[h]) for h in H_GRID_23)))

    scaling = [dict(m=m, d=dim_nc(m), h_star_pred=dim_nc(m) + 1,
                    classical_params_pred=params_classical(dim_nc(m) + 1, m),
                    quantum_params=params_quantum(m)) for m in (2, 3)]
    log("  이론표: " + " | ".join("m=%d d=%d H*pred=%d 고전=%d 양자=%d"
                                  % (s["m"], s["d"], s["h_star_pred"], s["classical_params_pred"],
                                     s["quantum_params"]) for s in scaling))

    # ---------------- A-3: 표본복잡도 ----------------
    log("\n=== A-3: 표본복잡도 (V=0.856 운용점, 훈련 emp(n) → 참값 held-out KL) ===")
    models = ["quantum"] + ["H%d" % h for h in H_SAMPLE]
    tasks = [(mdl, n, 0.856, max(3, args.restarts // 2), SEED + 300 * si + 17 * ni + 3 * mi)
             for mi, mdl in enumerate(models) for ni, n in enumerate(N_GRID) for si in range(args.seeds)]
    res3 = run_many(_w_sample, tasks, args.jobs)
    a3 = []
    for mdl in models:
        row = dict(model=mdl,
                   params=(params_quantum(2) if mdl == "quantum" else params_classical(int(mdl[1:]), 2)))
        for n in N_GRID:
            v = np.array([r["kl"] for r in res3 if r["model"] == mdl and r["n"] == n])
            row["n%d" % n] = dict(mean=float(v.mean()), std=float(v.std()))
        a3.append(row)
        log("  %-8s (%3d params) : " % (mdl, row["params"])
            + "  ".join("n=%5d KL=%.5f±%.5f" % (n, row["n%d" % n]["mean"], row["n%d" % n]["std"]) for n in N_GRID))

    payload = dict(
        note=("STEP A 파라미터효율. ★주장은 **다항 분리(삼차 대 선형)** — 지수 아님(지수는 LP 꼭짓점 2^(2m) 한정). "
              "A-1 H*(ε) 실측 vs Carathéodory d+1, A-2 (2,3,2) 교차확인(두 점 = 곡선 주장 불가), A-3 표본복잡도. "
              "R6: 탐지 커버리지 우위 주장 없음(불가능성 정리와 무모순). supremacy 아님."),
        params=dict(seed=SEED, eps_star=EPS_STAR, V_grid=V_GRID, H_grid_22=H_GRID_22, H_grid_23=H_GRID_23,
                    n_grid=N_GRID, H_sample=H_SAMPLE, generic_23=GENERIC_23.tolist(), restarts=args.restarts, restarts23=args.restarts23,
                    seeds=args.seeds, fastq=FASTQ, fastq23=FASTQ23),
        self_check=sc,
        a1_hstar_22=a1, a2_scaling_23=a2, scaling_theory=scaling, a3_sample_complexity=a3,
        verdicts=dict(
            hstar_median_22=(int(np.median(hstars)) if hstars else None),
            hstar_pred_22=dim_nc(2) + 1,
            hstar_matches_caratheodory=bool(hstars and abs(np.median(hstars) - (dim_nc(2) + 1)) <= 3),
            claim="polynomial separation: classical O(m^3) params vs quantum O(m) params (NOT exponential)"),
        limits=("H*(ε) 는 ε·restarts 의존 — H* ≫ d+1 이면 최적화 실패이지 이론 반증이 아님(restarts 늘려 재확인). "
                "m=2,3 두 점으로 곡선 주장 불가(이론 예측의 교차확인일 뿐). 모집단(N=inf)에서 H* 측정 → 표본잡음 배제. "
                "Carathéodory 상한은 **충분조건** — 실측 H* ≤ d+1 이 정상이며 H*<d+1 은 이론 반증이 아님. A-2 는 A₂·B₂ 를 일반각으로 둔 비퇴화 (2,3,2)(gate_s5 CANON_23 은 A₂=A₀ 복제라 (2,2,2)로 퇴화). R6: 탐지 커버리지 우위 주장 아님. supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload)
        log("[저장] %s" % OUT_PNG)
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(P):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    cols = plt.cm.viridis(np.linspace(0, 0.85, len(P["a1_hstar_22"])))
    for i, c in enumerate(P["a1_hstar_22"]):
        Hs = [int(h) for h in P["params"]["H_grid_22"]]
        ax[0].plot(Hs, [c["kl_by_H"][str(h)] for h in Hs], "o-", color=cols[i],
                   label="CF=%.3f (H*=%s)" % (c["cf"], c["h_star"]))
        ax[0].axhline(c["ncfloor"], color=cols[i], ls=":", lw=1)
    ax[0].axvline(9, color="r", ls="--", lw=1.5, label="Carathéodory d+1 = 9")
    ax[0].set_yscale("log"); ax[0].set_xlabel("classical ladder size H"); ax[0].set_ylabel("KL to truth (population)")
    ax[0].set_title("A-1 (2,2,2): KL(H) → NCfloor (dotted)\nH*(ε=1e-4) vs Carathéodory prediction")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    for i, c in enumerate(P["a2_scaling_23"]):
        Hs = [int(h) for h in P["params"]["H_grid_23"]]
        ax[1].plot(Hs, [c["kl_by_H"][str(h)] for h in Hs], "s-", label="(2,3,2) CF=%.3f (H*=%s)" % (c["cf"], c["h_star"]))
        ax[1].axhline(c["ncfloor"], ls=":", lw=1)
    ax[1].axvline(16, color="r", ls="--", lw=1.5, label="Carathéodory d+1 = 16")
    ax[1].set_yscale("log"); ax[1].set_xlabel("classical ladder size H"); ax[1].set_ylabel("KL to truth")
    ax[1].set_title("A-2 (2,3,2) cross-check"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)

    ms = np.arange(2, 9)
    ax[2].plot(ms, [params_classical(dim_nc(m) + 1, m) for m in ms], "o-", color="#C1442E",
               label="classical at H*=d+1  ~ O(m³)")
    ax[2].plot(ms, [params_quantum(m) for m in ms], "s-", color="#7B2FBE", label="quantum 1+4m ~ O(m)")
    for s in P["scaling_theory"]:
        ax[2].plot(s["m"], s["classical_params_pred"], "k*", ms=14)
    ax[2].set_yscale("log"); ax[2].set_xlabel("settings per party m"); ax[2].set_ylabel("model parameters")
    ax[2].set_title("Parameter scaling: polynomial (cubic vs linear)\n★ NOT exponential — 2^(2m) applies to LP vertices only")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=0.3, which="both")
    plt.suptitle("STEP A — parameter efficiency (polynomial separation; NOT detection coverage; NOT supremacy)",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=130); plt.close()


if __name__ == "__main__":
    main()
