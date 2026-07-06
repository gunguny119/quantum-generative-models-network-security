#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
깊이-표현력 곡선 (n=6, L 스윕) — QCBM 포화 KL vs 동일 파라미터 예산 고전(MLP/NMF).

직전: QCBM-KL 저차 0.197 은 L=3/P=54 표현력 포화(유한학습 아님), L=6/P=108 은 0.0002(바닥). 이 작업은
깊이 L∈[2,3,4,5,6,8] 을 스윕해 P=L·n·3 별 QCBM 포화 KL 과 같은 P 의 MLP/NMF KL 을 비교 →
"QCBM 이 같은/더 적은 P 로 고전(MLP) 동급/우위 되는 교차점" 정량화. n=6 기준선(n↑ 비교 토대). 합성만.

기존 코드 무수정(import): qcbm_kl_variant.train_kl_one, exp_synth_kl_vs_mmd(nmf/mlp/marginal/eval/eff_rank),
synth_highorder_distribution, experiment2(wshape/empirical_dist), diagnose_order_decomposition.kl_div.

사용: python tools/exp_depth_expressivity.py --smoke
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
import synth_highorder_distribution as SY  # noqa: E402  make_terms, synth_dist, ge3_share
import exp_synth_quantum_vs_classical as CL  # noqa: E402  nmf_reconstruct, mlp_energy_fit, fit_marginal, eval_model, eff_rank
import qcbm_kl_variant as KLV             # noqa: E402  train_kl_one
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "depth_expressivity")
EPS = 1e-12
NQ = 6
N = 80000
SAMPLE_SEED = 12345
SIDE = 1 << (NQ // 2)
MLP_FLOOR_REF = 0.001


def log(m):
    print(m, flush=True)


def make_point(t, high_mode):
    terms = SY.make_terms(NQ, seed=0, high_mode=high_mode)
    p_true, _ = SY.synth_dist(NQ, t, terms=terms)
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = rng.choice(1 << NQ, size=N, p=p_true)
    q_emp = E.empirical_dist(samples, 1 << NQ)
    return p_true, q_emp, samples


def qcbm_kl_best(q_emp, p_true, layers, epochs, restarts):
    best = None
    for sd in range(restarts):
        p, ce, _ = KLV.train_kl_one(q_emp, NQ, layers, epochs, seed=sd)
        kl = OD.kl_div(p_true, p, EPS)
        if best is None or ce < best[1]:
            best = (kl, ce)
    return best[0]


def sweep(p_true, q_emp, samples, Ls, epochs, restarts, mlp_iters):
    rows = []
    for L in Ls:
        P = int(np.prod(E.wshape(NQ, L)))
        r_nmf = max(1, round(P / (2 * SIDE)))
        h_mlp = max(2, round(P / (NQ + 2)))
        t0 = time.time()
        kl_q = qcbm_kl_best(q_emp, p_true, L, epochs, restarts)
        p_n, np_n = CL.nmf_reconstruct(q_emp, NQ, r_nmf, seed=0)
        p_m, np_m = CL.mlp_energy_fit(q_emp, NQ, h_mlp, iters=mlp_iters, seed=0)
        p_mg = CL.fit_marginal(samples, NQ)
        row = dict(L=L, P=P,
                   qcbm_KL=kl_q,
                   mlp_KL=OD.kl_div(p_true, p_m, EPS), mlp_params=np_m,
                   nmf_KL=OD.kl_div(p_true, p_n, EPS), nmf_rank=r_nmf, nmf_params=np_n,
                   marginal_KL=OD.kl_div(p_true, p_mg, EPS))
        rows.append(row)
        log("  L=%d P=%3d | QCBM=%.4f | MLP=%.4f(p%d) | NMF=%.4f(r%d,p%d) | marg=%.4f (%.0fs)"
            % (L, P, kl_q, row["mlp_KL"], np_m, row["nmf_KL"], r_nmf, np_n, row["marginal_KL"], time.time() - t0))
    return rows


def crossover_P(rows):
    """QCBM_KL <= MLP_KL 되는 최소 P (없으면 None)."""
    for r in sorted(rows, key=lambda x: x["P"]):
        if r["qcbm_KL"] <= r["mlp_KL"] + 1e-6:
            return r["P"]
    return None


def run(smoke=False, epochs=None, restarts=None, mlp_iters=None):
    os.makedirs(OUTDIR, exist_ok=True)
    if smoke:
        Ls = [2, 3, 6]; Ls_hr = [3, 6]
        epochs = epochs or 150; restarts = restarts or 2; mlp_iters = mlp_iters or 400
    else:
        Ls = [2, 3, 4, 5, 6, 8]; Ls_hr = [3, 6, 8]
        epochs = epochs or 800; restarts = restarts or 5; mlp_iters = mlp_iters or 1500

    samp_floor_low = None
    log("=== 저차 t=0 (n=6) ===")
    p_true, q_emp, samples = make_point(0.0, "disjoint")
    share, _, _ = SY.ge3_share(p_true, NQ)
    samp_floor_low = OD.kl_div(p_true, q_emp, EPS)
    log("share=%.3f | 샘플바닥=%.4f | epochs=%d restarts=%d" % (share, samp_floor_low, epochs, restarts))
    low = sweep(p_true, q_emp, samples, Ls, epochs, restarts, mlp_iters)
    cross_low = crossover_P(low)

    log("=== 고랭크 random t=0.6 (n=6) ===")
    p_hr, q_hr, s_hr = make_point(0.6, "random")
    sh_hr, _, _ = SY.ge3_share(p_hr, NQ)
    erank = CL.eff_rank(p_hr, NQ)
    log("share=%.3f rank99=%d" % (sh_hr, erank))
    high = sweep(p_hr, q_hr, s_hr, Ls_hr, epochs, restarts, mlp_iters)
    cross_high = crossover_P(high)

    sc = dict(
        low_qcbm_monotone=bool(all(low[i]["qcbm_KL"] >= low[i + 1]["qcbm_KL"] - 0.02 for i in range(len(low) - 1))),
        repro_L3=next((r["qcbm_KL"] for r in low if r["L"] == 3), None),
        repro_L6=next((r["qcbm_KL"] for r in low if r["L"] == 6), None),
    )
    payload = dict(
        note=("n=6, t=0 저차 + 고랭크 random. L 스윕으로 P=L·n·3 별 QCBM 포화 KL vs 동일 P MLP/NMF. "
              "교차점=QCBM_KL≤MLP_KL 최소 P. n=6 기준선(n↑ 비교 전). KL(p_true‖model). 양자우위 단정 금지."),
        config=dict(nq=NQ, N=N, sample_seed=SAMPLE_SEED, epochs=epochs, restarts=restarts,
                    mlp_iters=mlp_iters, sampling_floor_low=samp_floor_low, low_share=share, high_share=sh_hr, high_rank99=erank),
        low=low, crossover_P_low=cross_low, high=high, crossover_P_high=cross_high, self_check=sc)
    with open(os.path.join(OUTDIR, "depth_n6.json"), "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("[저장] depth_n6.json | 교차점 P(저차)=%s 고랭크=%s | self-check 단조=%s L3=%.4f L6=%.4f"
        % (cross_low, cross_high, sc["low_qcbm_monotone"],
           sc["repro_L3"] or -1, sc["repro_L6"] or -1))
    plot(low, high, samp_floor_low)
    return payload


def plot(low, high, floor):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for rows, a, title in [(low, ax[0], "low-order (t=0)"), (high, ax[1], "high-rank random (t=0.6)")]:
        rows = sorted(rows, key=lambda x: x["P"])
        Ps = [r["P"] for r in rows]
        a.plot(Ps, [r["qcbm_KL"] for r in rows], "*-", color="#7B2FBE", ms=11, label="QCBM-KL (entangling)")
        a.plot(Ps, [r["mlp_KL"] for r in rows], "^-", color="#4C9F70", label="MLP (same P)")
        a.plot(Ps, [r["nmf_KL"] for r in rows], "s-", color="#3B6EA5", label="NMF (same P)")
        a.axhline(floor, color="gray", ls=":", label="sampling floor")
        a.set_xlabel("parameters P (= L·n·3 for QCBM)"); a.set_ylabel("KL(p_true || model)")
        a.set_yscale("log"); a.set_title("Depth-expressivity: %s" % title); a.legend(fontsize=8); a.grid(alpha=.3)
    plt.suptitle("QCBM depth vs same-budget classical (n=6) — parameter-efficiency crossover?")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "depth_n6.png"), dpi=125); plt.close()
    log("[저장] depth_n6.png")


def main():
    ap = argparse.ArgumentParser(description="깊이-표현력 곡선(n=6, L 스윕). 합성, n↑ 기준선. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--restarts", type=int, default=None)
    ap.add_argument("--mlp-iters", type=int, default=None)
    args = ap.parse_args()
    run(smoke=args.smoke, epochs=args.epochs, restarts=args.restarts, mlp_iters=args.mlp_iters)


if __name__ == "__main__":
    main()
