#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCBM-MMD vs QCBM-KL 재비교 — 직전 합성 실험과 "완전히 동일한" 설정에서 QCBM loss만 교체.

직전(exp_synth_quantum_vs_classical) 의 합성 분포/예산/baseline(NMF/MLP/marginal)/평가(KL,TV)/t 스윕/N/seed 를
그대로 재사용하고, QCBM 만 두 버전(MMD²: EXP.qcbm_fit, KL: qcbm_kl_variant.qcbm_kl_fit)으로 학습해 비교한다.
합성만. 긴 학습 없음. 기존 코드 무수정(import 재사용).

사용: python tools/exp_synth_kl_vs_mmd.py --smoke
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

import experiment2 as E                   # noqa: E402  empirical_dist
import synth_highorder_distribution as SY  # noqa: E402  synth_dist, make_terms, ge3_share
import exp_synth_quantum_vs_classical as EXP  # noqa: E402  qcbm_fit(MMD), nmf_reconstruct, mlp_energy_fit, fit_marginal, eval_model, eff_rank
import qcbm_kl_variant as KLV             # noqa: E402  qcbm_kl_fit

OUTDIR = os.path.join("results2", "synth_highorder_kl")


def log(m):
    print(m, flush=True)


def run(nq, ts, N, layers, epochs, restarts, mlp_iters, seed, do_plot=True, tag="run", high_mode="disjoint"):
    os.makedirs(OUTDIR, exist_ok=True)
    terms = SY.make_terms(nq, seed=seed, high_mode=high_mode)
    P = int(np.prod(E.wshape(nq, layers)))
    side = 1 << (nq // 2)
    r_nmf = max(1, round(P / (2 * side)))
    h_mlp = max(2, round(P / (nq + 2)))
    log("[%s] 예산 P=%d | QCBM-MMD/KL params=%d | NMF rank=%d | MLP hidden=%d | n=%d N=%d epochs=%d restarts=%d"
        % (high_mode, P, P, r_nmf, h_mlp, nq, N, epochs, restarts))

    rng = np.random.default_rng(12345)      # 직전과 동일 샘플 seed
    rows = []
    for t in ts:
        p_true, _ = SY.synth_dist(nq, t, terms=terms)
        share, kl1, kl2 = SY.ge3_share(p_true, nq)
        erank = EXP.eff_rank(p_true, nq)
        samples = rng.choice(1 << nq, size=N, p=p_true)
        q_emp = E.empirical_dist(samples, 1 << nq)

        t0 = time.time()
        p_mmd, np_mmd, _mmd = EXP.qcbm_fit(q_emp, nq, layers, epochs, restarts)        # 기존 MMD²
        p_kl, np_kl, ce = KLV.qcbm_kl_fit(q_emp, nq, layers, epochs, restarts)         # 새 KL
        p_n, np_n = EXP.nmf_reconstruct(q_emp, nq, r_nmf, seed=seed)
        p_m, np_m = EXP.mlp_energy_fit(q_emp, nq, h_mlp, iters=mlp_iters, seed=seed)
        p_mg = EXP.fit_marginal(samples, nq)
        res = dict(t=t, ge3_share=share, eff_rank99=erank,
                   QCBM_MMD=EXP.eval_model(p_true, p_mmd), QCBM_KL=EXP.eval_model(p_true, p_kl),
                   NMF=EXP.eval_model(p_true, p_n), MLP=EXP.eval_model(p_true, p_m),
                   marginal=EXP.eval_model(p_true, p_mg),
                   nparams=dict(QCBM_MMD=np_mmd, QCBM_KL=np_kl, NMF=np_n, MLP=np_m, marginal=nq))
        rows.append(res)
        log("t=%.2f share=%.3f rank99=%d | KL  QCBM-MMD=%.4f QCBM-KL=%.4f NMF=%.4f MLP=%.4f marg=%.4f (%.0fs)"
            % (t, share, erank, res["QCBM_MMD"]["KL"], res["QCBM_KL"]["KL"], res["NMF"]["KL"],
               res["MLP"]["KL"], res["marginal"]["KL"], time.time() - t0))

    lo = min(rows, key=lambda x: x["ge3_share"])
    sc = dict(
        QCBM_MMD_KL_low=lo["QCBM_MMD"]["KL"], QCBM_KL_KL_low=lo["QCBM_KL"]["KL"],
        kl_le_mmd_all=bool(all(r["QCBM_KL"]["KL"] <= r["QCBM_MMD"]["KL"] + 1e-6 for r in rows)),
        kl_beats_nmf_any=bool(any(r["QCBM_KL"]["KL"] < r["NMF"]["KL"] for r in rows)),
        kl_beats_mlp_any=bool(any(r["QCBM_KL"]["KL"] < r["MLP"]["KL"] for r in rows)),
    )
    payload = dict(
        note=("QCBM loss만 MMD²→KL(cross-entropy) 교체, 나머지 직전과 동일. KL(p_true||model). 합성만. 단정 금지."),
        config=dict(nq=nq, ts=ts, N=N, layers=layers, epochs=epochs, restarts=restarts,
                    mlp_iters=mlp_iters, P=P, nmf_rank=r_nmf, mlp_hidden=h_mlp, seed=seed, high_mode=high_mode),
        rows=rows, self_check=sc)
    outjson = os.path.join(OUTDIR, "klmmd_%s.json" % tag)
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("[저장] %s" % outjson)
    log("self-check: QCBM-MMD KL(저차)=%.3f QCBM-KL KL(저차)=%.3f | KL≤MMD 전구간=%s | KL이 NMF넘는구간=%s MLP넘는구간=%s"
        % (sc["QCBM_MMD_KL_low"], sc["QCBM_KL_KL_low"], sc["kl_le_mmd_all"], sc["kl_beats_nmf_any"], sc["kl_beats_mlp_any"]))
    if do_plot:
        plot(rows, tag)
    return payload


def plot(rows, tag):
    rows = sorted(rows, key=lambda x: x["ge3_share"])
    x = [r["ge3_share"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    series = [("QCBM_MMD", "#C1442E", "o"), ("QCBM_KL", "#7B2FBE", "*"),
              ("NMF", "#3B6EA5", "s"), ("MLP", "#4C9F70", "^"), ("marginal", "#9AA0A6", "d")]
    for key, col, mk in series:
        ax[0].plot(x, [r[key]["KL"] for r in rows], mk + "-", color=col, label=key, ms=8)
        ax[1].plot(x, [r[key]["TV"] for r in rows], mk + "-", color=col, label=key, ms=8)
    ax[0].set_ylabel("KL(p_true || model)"); ax[1].set_ylabel("TV")
    for a in ax:
        a.set_xlabel("≥3rd-order share"); a.grid(alpha=.3); a.legend(fontsize=9)
    ax[0].set_title("QCBM loss MMD² vs KL — fit error vs high-order share")
    ax[1].set_title("TV")
    plt.suptitle("QCBM loss ablation (MMD² vs forward-KL) on synthetic high-order dists")
    plt.tight_layout()
    outpng = os.path.join(OUTDIR, "klmmd_%s.png" % tag)
    plt.savefig(outpng, dpi=125); plt.close()
    log("[저장] %s" % outpng)


def main():
    ap = argparse.ArgumentParser(description="QCBM MMD² vs KL 재비교(합성, 직전과 동일 설정).")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--betas", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--N", type=int, default=80000)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=250)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--mlp-iters", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--tag", type=str, default="run")
    ap.add_argument("--high-mode", type=str, default="disjoint", choices=["disjoint", "random"])
    args = ap.parse_args()
    if args.smoke:
        args.n, args.N, args.epochs, args.restarts, args.mlp_iters = 6, 40000, 100, 1, 500
        if args.tag == "run":
            args.tag = "smoke_" + args.high_mode
    ts = [float(x) for x in args.betas.split(",") if x.strip()]
    run(args.n, ts, args.N, args.layers, args.epochs, args.restarts, args.mlp_iters,
        args.seed, do_plot=not args.no_plot, tag=args.tag, high_mode=args.high_mode)


if __name__ == "__main__":
    main()
