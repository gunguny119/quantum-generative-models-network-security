#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
합성 고차-상관 분포에서 양자(QCBM) vs 고전(NMF/MLP-energy/marginal) 통제 비교.

고차 비중(t=0→1, ≥3차 share 사후측정)을 스윕하며, 동일 파라미터 예산에서 각 모델이 진짜 합성
분포를 얼마나 잘 적합하는지(KL(p_true||model), TV) 비교 → "고전이 무너지고 양자가 버티는 교차점" 탐색.

모델: QCBM(MMD², StronglyEntanglingLayers, experiment2 재사용) / NMF(저랭크, numpy) /
      MLP-energy(1-hidden, exact Z, numpy) / marginal(비트독립, 하한). torch 없음 → NMF/MLP 자체구현.
합성만(실데이터 미사용). 긴 학습 없음. 단일 프로세스.

사용: python tools/exp_synth_quantum_vs_classical.py --smoke
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

import experiment2 as E                   # noqa: E402  build_kernel, _init, train_one, wshape, empirical_dist, tv_dist
import train_qcbm_iat_b2 as TB2           # noqa: E402  marginal_dist
import diagnose_order_decomposition as OD  # noqa: E402  kl_div, bit_matrix
import synth_highorder_distribution as SY  # noqa: E402  synth_dist, ge3_share, make_terms

OUTDIR = os.path.join("results2", "synth_highorder")
EPS = 1e-12


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 고전 모델들
# ---------------------------------------------------------------------------
def fit_marginal(st_samples, nq):
    return TB2.marginal_dist(st_samples, nq)


def nmf_reconstruct(q_emp, nq, r, iters=400, seed=0):
    """q_emp(2^nq) -> side x side reshape -> rank-r NMF(multiplicative update) -> clip+정규화."""
    side = 1 << (nq // 2)
    M = q_emp.reshape(side, side).astype(np.float64)
    rng = np.random.default_rng(seed)
    W = rng.random((side, r)) + 1e-3
    H = rng.random((r, side)) + 1e-3
    for _ in range(iters):
        WH = W @ H + 1e-12
        H *= (W.T @ M) / (W.T @ WH + 1e-12)
        WH = W @ H + 1e-12
        W *= (M @ H.T) / (WH @ H.T + 1e-12)
    approx = np.clip((W @ H).reshape(-1), 0, None)
    s = approx.sum()
    approx = approx / s if s > 0 else np.full_like(approx, 1.0 / approx.size)
    nparams = 2 * side * r
    return approx, nparams


def mlp_energy_fit(q_emp, nq, h, iters=600, lr=0.05, seed=0):
    """E(s)=w2·tanh(W1 s+b1)+b2, p∝exp(-E), exact Z. NLL 경사하강(∇=E_data[∂E]-E_model[∂E]). numpy."""
    rng = np.random.default_rng(seed)
    N = 1 << nq
    s = (1 - 2 * OD.bit_matrix(nq)).astype(np.float64)   # (N, nq)
    W1 = rng.normal(0, 0.3, (h, nq)); b1 = np.zeros(h)
    w2 = rng.normal(0, 0.3, h); b2 = 0.0
    dataw = np.asarray(q_emp, float)
    params = [W1, b1, w2, np.array([b2])]
    m = [np.zeros_like(p) for p in params]; v = [np.zeros_like(p) for p in params]
    b1a, b2a, eps = 0.9, 0.999, 1e-8
    pm = np.full(N, 1.0 / N)
    for it in range(1, iters + 1):
        z = s @ W1.T + b1                      # (N,h)
        a = np.tanh(z)
        Evec = a @ w2 + params[3][0]           # (N,)
        logits = -(Evec - Evec.min())
        pm = np.exp(logits); pm /= pm.sum()
        diff = dataw - pm                      # (N,)
        da = (1 - a * a)                       # (N,h)
        # ∂E/∂params
        g_w2 = (diff[:, None] * a).sum(0)
        g_b2 = diff.sum()
        tmp = diff[:, None] * da * w2[None, :]  # (N,h)
        g_W1 = tmp.T @ s                        # (h,nq)
        g_b1 = tmp.sum(0)                        # (h,)
        grads = [g_W1, g_b1, g_w2, np.array([g_b2])]
        for k in range(4):
            m[k] = b1a * m[k] + (1 - b1a) * grads[k]
            v[k] = b2a * v[k] + (1 - b2a) * grads[k] ** 2
            mh = m[k] / (1 - b1a ** it); vh = v[k] / (1 - b2a ** it)
            params[k] -= lr * mh / (np.sqrt(vh) + eps)
        W1, b1, w2 = params[0], params[1], params[2]
    nparams = h * nq + h + h + 1
    return pm, nparams


def qcbm_fit(q_emp, nq, layers, epochs, restarts, lr=0.1, seed0=0):
    K = E.build_kernel(nq)
    E._init(q_emp, K, nq, layers, epochs, lr, max(epochs, 1), "synth")
    best = None
    for sd in range(restarts):
        r = E.train_one(seed0 + sd)
        if best is None or r["final_mmd2"] < best["final_mmd2"]:
            best = r
    nparams = int(np.prod(E.wshape(nq, layers)))
    return np.asarray(best["p_final"], float), nparams, best["final_mmd2"]


# ---------------------------------------------------------------------------
def eval_model(p_true, p_model):
    return dict(KL=OD.kl_div(p_true, p_model, EPS), TV=E.tv_dist(p_true, p_model))


def eff_rank(p, nq):
    """reshape side×side 의 특이값 기반 유효 랭크(에너지 99%) — 고차↔저랭크 confound 진단용."""
    side = 1 << (nq // 2)
    sv = np.linalg.svd(p.reshape(side, side), compute_uv=False)
    e = np.cumsum(sv ** 2) / max(np.sum(sv ** 2), 1e-30)
    return int(np.searchsorted(e, 0.99) + 1)


def run(nq, ts, N, layers, epochs, restarts, mlp_iters, seed, do_plot=True, tag="run", high_mode="disjoint"):
    os.makedirs(OUTDIR, exist_ok=True)
    terms = SY.make_terms(nq, seed=seed, high_mode=high_mode)
    P = int(np.prod(E.wshape(nq, layers)))           # QCBM 파라미터 예산
    side = 1 << (nq // 2)
    r_nmf = max(1, round(P / (2 * side)))
    h_mlp = max(2, round(P / (nq + 2)))
    log("예산 P(QCBM)=%d | NMF rank=%d | MLP hidden=%d | side=%d | n=%d N=%d epochs=%d restarts=%d"
        % (P, r_nmf, h_mlp, side, nq, N, epochs, restarts))

    rng = np.random.default_rng(12345)
    rows = []
    for t in ts:
        p_true, _ = SY.synth_dist(nq, t, terms=terms)
        share, kl1, kl2 = SY.ge3_share(p_true, nq)
        erank = eff_rank(p_true, nq)
        samples = rng.choice(1 << nq, size=N, p=p_true)
        q_emp = E.empirical_dist(samples, 1 << nq)

        t0 = time.time()
        p_q, np_q, mmd = qcbm_fit(q_emp, nq, layers, epochs, restarts)
        p_n, np_n = nmf_reconstruct(q_emp, nq, r_nmf, seed=seed)
        p_m, np_m = mlp_energy_fit(q_emp, nq, h_mlp, iters=mlp_iters, seed=seed)
        p_mg = fit_marginal(samples, nq)
        res = dict(t=t, ge3_share=share, kl1=kl1, kl2=kl2, eff_rank99=erank,
                   QCBM=eval_model(p_true, p_q), NMF=eval_model(p_true, p_n),
                   MLP=eval_model(p_true, p_m), marginal=eval_model(p_true, p_mg),
                   nparams=dict(QCBM=np_q, NMF=np_n, MLP=np_m, marginal=nq))
        rows.append(res)
        log("t=%.2f share=%.3f rank99=%d | KL  QCBM=%.4f NMF=%.4f MLP=%.4f marg=%.4f  (%.1fs)"
            % (t, share, erank, res["QCBM"]["KL"], res["NMF"]["KL"], res["MLP"]["KL"],
               res["marginal"]["KL"], time.time() - t0))

    # 교차점: QCBM_KL < NMF_KL 가 되는 가장 낮은 share
    cross = None
    for r in sorted(rows, key=lambda x: x["ge3_share"]):
        if r["QCBM"]["KL"] < r["NMF"]["KL"]:
            cross = r["ge3_share"]; break
    # self-check 관찰(완화): 저고차서 NMF 양호, 고고차서 marginal 붕괴
    lo = min(rows, key=lambda x: x["ge3_share"]); hi = max(rows, key=lambda x: x["ge3_share"])
    sc = dict(low_share=lo["ge3_share"], high_share=hi["ge3_share"],
              NMF_KL_low=lo["NMF"]["KL"], NMF_KL_high=hi["NMF"]["KL"],
              marg_KL_low=lo["marginal"]["KL"], marg_KL_high=hi["marginal"]["KL"],
              QCBM_KL_low=lo["QCBM"]["KL"], QCBM_KL_high=hi["QCBM"]["KL"],
              marginal_collapses=bool(hi["marginal"]["KL"] > lo["marginal"]["KL"] + 0.05),
              nmf_collapses=bool(hi["NMF"]["KL"] > lo["NMF"]["KL"] + 0.05))

    payload = dict(
        note=("합성 고차 비중(t→≥3차 share) 스윕. KL(p_true||model). torch 부재→NMF/MLP numpy 자체구현. "
              "QCBM=MMD²(StronglyEntanglingLayers). 교차점=QCBM_KL<NMF_KL 최소 share. 통제 합성, 단정 금지."),
        config=dict(nq=nq, ts=ts, N=N, layers=layers, epochs=epochs, restarts=restarts,
                    mlp_iters=mlp_iters, P=P, nmf_rank=r_nmf, mlp_hidden=h_mlp, seed=seed,
                    high_mode=high_mode),
        rows=rows, crossover_ge3_share=cross, self_check=sc,
    )
    outjson = os.path.join(OUTDIR, "synth_%s.json" % tag)
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s | 교차점(QCBM<NMF) ≥3차 share=%s" % (outjson, cross))
    log("self-check: NMF KL %.3f→%.3f, marginal KL %.3f→%.3f, QCBM KL %.3f→%.3f (저→고 share)"
        % (sc["NMF_KL_low"], sc["NMF_KL_high"], sc["marg_KL_low"], sc["marg_KL_high"],
           sc["QCBM_KL_low"], sc["QCBM_KL_high"]))
    if do_plot:
        plot(rows, cross, tag)
    return payload


def plot(rows, cross, tag):
    rows = sorted(rows, key=lambda x: x["ge3_share"])
    x = [r["ge3_share"] for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6))
    for key, col, mk in [("QCBM", "#C1442E", "o"), ("NMF", "#3B6EA5", "s"),
                         ("MLP", "#4C9F70", "^"), ("marginal", "#9AA0A6", "d")]:
        ax[0].plot(x, [r[key]["KL"] for r in rows], mk + "-", color=col, label=key)
        ax[1].plot(x, [r[key]["TV"] for r in rows], mk + "-", color=col, label=key)
    for a, yl in zip(ax, ["KL(p_true || model)", "TV(p_true, model)"]):
        a.set_xlabel("≥3rd-order share of synthetic dist"); a.set_ylabel(yl)
        a.grid(alpha=.3); a.legend(fontsize=9)
        if cross is not None:
            a.axvline(cross, color="purple", ls="--", alpha=.6)
    ax[0].set_title("Fit error vs high-order share (cross = QCBM beats NMF)")
    ax[1].set_title("TV vs high-order share")
    plt.suptitle("Synthetic high-order control: quantum (QCBM) vs classical (NMF/MLP/marginal)")
    plt.tight_layout()
    outpng = os.path.join(OUTDIR, "synth_%s.png" % tag)
    plt.savefig(outpng, dpi=125); plt.close()
    log("[저장] %s" % outpng)


def main():
    ap = argparse.ArgumentParser(description="합성 고차분포 양자 vs 고전 비교. 합성만, 긴 학습 없음.")
    ap.add_argument("--smoke", action="store_true", help="작은 설정(n=6, 5점, 작은 epoch)")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--betas", type=str, default="0,0.25,0.5,0.75,1.0", help="t 값 콤마구분")
    ap.add_argument("--N", type=int, default=60000)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--restarts", type=int, default=2)
    ap.add_argument("--mlp-iters", type=int, default=800)
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
