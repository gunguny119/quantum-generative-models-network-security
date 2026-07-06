#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
양자 우위 스케일링 — n=6,8,10 깊이 곡선으로 "고전 대비 QCBM 효율 격차의 n-추세" 측정.

각 n 에서 저차(t=0) + 고랭크(random) 분포에 대해 깊이 L 스윕 → P=L·n·3 별 QCBM-KL(다seed best) vs 동일 P
고전(MLP/NMF). 교차점 P*(QCBM≤MLP 최소 P) 와 (고전−QCBM) KL 격차가 n 에 따라 어떻게 이동하는가.
합성만. 기존 코드 무수정(import). 단일 프로세스. 단정 금지(추세를 사실로).

★ n=10 statevector 학습은 매우 무거움 → --time-only 로 먼저 시간 게이트.

사용: python tools/exp_quantum_advantage_scaling.py --time-only --ns 8,10
      python tools/exp_quantum_advantage_scaling.py --smoke
      python tools/exp_quantum_advantage_scaling.py --ns 6,8 --Ls 2,3,4,6 --seeds 3
"""
import os
import sys
import json
import time
import argparse

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import experiment2 as E                   # noqa: E402  wshape, empirical_dist
import synth_highorder_distribution as SY  # noqa: E402  synth_dist, make_terms, ge3_share
import exp_synth_quantum_vs_classical as CL  # noqa: E402  nmf_reconstruct, mlp_energy_fit, fit_marginal, eff_rank
import qcbm_kl_variant as KLV             # noqa: E402  train_kl_one
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "qa_scaling")
EPS = 1e-12
N_PER_STATE = 500          # N = N_PER_STATE * 2^n → 샘플바닥 = 2^n/(2N) = 1/(2*N_PER_STATE) ≈ 1e-3 (n 무관)


def log(m):
    print(m, flush=True)


def make_point(nq, t, high_mode, seed=0):
    terms = SY.make_terms(nq, seed=seed, high_mode=high_mode)
    p_true, _ = SY.synth_dist(nq, t, terms=terms)
    N = N_PER_STATE * (1 << nq)
    rng = np.random.default_rng(12345)
    samples = rng.choice(1 << nq, size=N, p=p_true)
    q_emp = E.empirical_dist(samples, 1 << nq)
    return p_true, q_emp, samples, N


def qcbm_kl_multiseed(q_emp, p_true, nq, layers, epochs, seeds):
    kls = []
    for sd in range(seeds):
        p, ce, _ = KLV.train_kl_one(q_emp, nq, layers, epochs, seed=sd)
        kls.append((OD.kl_div(p_true, p, EPS), ce))
    best = min(kls, key=lambda x: x[1])[0]        # 최소 CE 선택
    arr = np.array([k for k, _ in kls])
    return best, float(arr.min()), float(arr.max()), float(arr.std())


def sweep(nq, p_true, q_emp, samples, Ls, epochs, seeds, mlp_iters):
    side = 1 << (nq // 2)
    rows = []
    for L in Ls:
        P = int(np.prod(E.wshape(nq, L)))
        r_nmf = max(1, round(P / (2 * side)))
        h_mlp = max(2, round(P / (nq + 2)))
        t0 = time.time()
        kl_q, qmin, qmax, qstd = qcbm_kl_multiseed(q_emp, p_true, nq, L, epochs, seeds)
        # 고전: 여러 seed best
        nmf_kls, mlp_kls = [], []
        for s in range(seeds):
            p_n, np_n = CL.nmf_reconstruct(q_emp, nq, r_nmf, seed=s)
            p_m, np_m = CL.mlp_energy_fit(q_emp, nq, h_mlp, iters=mlp_iters, seed=s)
            nmf_kls.append(OD.kl_div(p_true, p_n, EPS))
            mlp_kls.append(OD.kl_div(p_true, p_m, EPS))
        p_mg = CL.fit_marginal(samples, nq)
        secs = time.time() - t0
        row = dict(L=L, P=P, qcbm_KL=kl_q, qcbm_min=qmin, qcbm_max=qmax, qcbm_std=qstd,
                   mlp_KL=float(min(mlp_kls)), mlp_params=np_m,
                   nmf_KL=float(min(nmf_kls)), nmf_rank=r_nmf, nmf_params=np_n,
                   marginal_KL=OD.kl_div(p_true, p_mg, EPS), secs=secs)
        rows.append(row)
        log("    L=%d P=%3d | QCBM=%.4f(min%.4f max%.4f) | MLP=%.4f | NMF=%.4f | marg=%.3f (%.0fs)"
            % (L, P, kl_q, qmin, qmax, row["mlp_KL"], row["nmf_KL"], row["marginal_KL"], secs))
    return rows


def crossover_P(rows, floor):
    """QCBM_KL ≤ MLP_KL 되는 최소 P (없으면 None)."""
    for r in sorted(rows, key=lambda x: x["P"]):
        if r["qcbm_KL"] <= r["mlp_KL"] + 1e-6:
            return r["P"]
    return None


MODES = {"low": dict(t=0.0, high_mode="disjoint"),
         "highrank": dict(t=0.6, high_mode="random")}


def run(ns, Ls, seeds, epochs, mlp_iters, modes, tag="run"):
    os.makedirs(OUTDIR, exist_ok=True)
    out = {}
    for mode in modes:
        cfg = MODES[mode]
        out[mode] = {}
        log("\n##### MODE=%s (t=%.1f, %s) #####" % (mode, cfg["t"], cfg["high_mode"]))
        for nq in ns:
            p_true, q_emp, samples, N = make_point(nq, cfg["t"], cfg["high_mode"])
            share, _, _ = SY.ge3_share(p_true, nq)
            erank = CL.eff_rank(p_true, nq)
            floor = OD.kl_div(p_true, q_emp, EPS)
            log("  n=%d: N=%d 샘플바닥=%.4f ge3_share=%.3f rank99=%d" % (nq, N, floor, share, erank))
            rows = sweep(nq, p_true, q_emp, samples, Ls, epochs, seeds, mlp_iters)
            out[mode][nq] = dict(N=N, floor=floor, ge3_share=share, eff_rank99=erank,
                                 rows=rows, Pstar=crossover_P(rows, floor))
    trend = {mode: {nq: out[mode][nq]["Pstar"] for nq in out[mode]} for mode in out}
    sc = dict(n6_repro=({r["L"]: round(r["qcbm_KL"], 4) for r in out.get("low", {}).get(6, {}).get("rows", [])}
                        if 6 in ns else None))
    payload = dict(
        note=("n 스케일링: 각 n,mode 에서 L 스윕 P=L·n·3 별 QCBM-KL(다seed best) vs 동일 P MLP/NMF. "
              "교차점 P*=QCBM≤MLP 최소 P. N=500·2^n(바닥≈1e-3 일정). 합성만. 추세 기록, 단정 금지."),
        config=dict(ns=ns, Ls=Ls, seeds=seeds, epochs=epochs, mlp_iters=mlp_iters, modes=modes, N_per_state=N_PER_STATE),
        results=out, trend_Pstar=trend, self_check=sc)
    outjson = os.path.join(OUTDIR, "qa_scaling%s.json" % ("" if tag == "run" else "_" + tag))
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s | trend P*(by n): %s" % (outjson, trend))
    try:
        plot(out, modes, ns, tag)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(out, modes, ns, tag):
    nm = len(modes)
    fig, ax = plt.subplots(nm, 2, figsize=(15, 6 * nm), squeeze=False)
    cmap = plt.cm.viridis(np.linspace(0, 0.85, len(ns)))
    for mi, mode in enumerate(modes):
        for ci, c in zip(ns, cmap):
            if ci not in out[mode]:
                continue
            r = sorted(out[mode][ci]["rows"], key=lambda x: x["P"])
            Ps = [x["P"] for x in r]
            ax[mi][0].plot(Ps, [x["qcbm_KL"] for x in r], "*-", color=c, label="QCBM n=%d" % ci)
            ax[mi][0].plot(Ps, [x["mlp_KL"] for x in r], "^--", color=c, alpha=0.6)
            ax[mi][0].plot(Ps, [x["nmf_KL"] for x in r], "s:", color=c, alpha=0.4)
        ax[mi][0].axhline(out[mode][ns[0]]["floor"], color="gray", ls=":", alpha=.5, label="floor")
        ax[mi][0].set_xlabel("P"); ax[mi][0].set_ylabel("KL"); ax[mi][0].set_yscale("log")
        ax[mi][0].set_title("%s: P vs KL (solid=QCBM, dash=MLP, dot=NMF)" % mode); ax[mi][0].legend(fontsize=7); ax[mi][0].grid(alpha=.3)
        # n vs P*
        ns_ok = [c for c in ns if c in out[mode] and out[mode][c]["Pstar"] is not None]
        ax[mi][1].plot(ns_ok, [out[mode][c]["Pstar"] for c in ns_ok], "o-", color="#7B2FBE")
        ax[mi][1].set_xlabel("n (qubits)"); ax[mi][1].set_ylabel("P* (QCBM≤MLP 최소 P)")
        ax[mi][1].set_title("%s: crossover P* vs n" % mode); ax[mi][1].grid(alpha=.3)
    plt.suptitle("Quantum-advantage scaling: QCBM vs same-budget classical across n")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "qa_scaling%s.png" % ("" if tag == "run" else "_" + tag)), dpi=120); plt.close()
    log("[저장] qa_scaling png")


def time_only(ns, epochs, seeds):
    log("=== --time-only: 각 n L=4 단일점 QCBM 학습 시간 (게이트용) ===")
    for nq in ns:
        p_true, q_emp, _, N = make_point(nq, 0.0, "disjoint")
        t0 = time.time()
        KLV.train_kl_one(q_emp, nq, 4, epochs, seed=0)
        dt = time.time() - t0
        P = int(np.prod(E.wshape(nq, 4)))
        log("  n=%2d L=4 P=%3d epochs=%d 1seed: %.1fs  → %dseed 추정 %.0fs (%.1f분)"
            % (nq, P, epochs, dt, seeds, dt * seeds, dt * seeds / 60))


def main():
    ap = argparse.ArgumentParser(description="n 스케일링 깊이 곡선(QCBM vs 고전). 합성, 추세. 단정 금지.")
    ap.add_argument("--ns", type=str, default="6,8,10")
    ap.add_argument("--Ls", type=str, default="2,3,4,6")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--mlp-iters", type=int, default=2000)
    ap.add_argument("--modes", type=str, default="low,highrank")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--tag", type=str, default="run")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    if args.time_only:
        time_only(ns, epochs=200, seeds=args.seeds); return
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    if args.smoke:
        ns = ns or [6, 8]; ns = [n for n in ns if n <= 8]; Ls = [2, 4]; args.seeds = 2; args.epochs = 120; args.mlp_iters = 600
        if args.tag == "run":
            args.tag = "smoke"
    run(ns, Ls, args.seeds, args.epochs, args.mlp_iters, modes, tag=args.tag)


if __name__ == "__main__":
    main()
