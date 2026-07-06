#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고차 상관 차수별 분해 — benign 경험적 분포의 ≥3차 KL 을 3차/4차/≥5차로 분해.

순수 측정(학습 없음). 직전 diagnose_maxent_kl.py 의 모멘트-보존 로그선형 max-entropy(L-BFGS) 인프라를
차수별로 일반화. import 재사용만. 단일 프로세스(병렬 없음).

차수 k 까지 보존 maxent: p_model(x) ∝ exp( Σ_{|S|≤k} θ_S * ∏_{i∈S} s_i ),  s_i = 1 - 2*bit_i.
NLL(θ) = logZ(θ) - θ·μ_emp 는 convex, ∇ = (model 모멘트 - empirical 모멘트). L-BFGS 로 적합.
p_k = 1..k 차 모멘트 보존. KL(p||p_k) 는 차수 k 까지로 설명 안 되는 잔여.
  c2 = KL(p||p1) - KL(p||p2)   (2차 기여)
  c3 = KL(p||p2) - KL(p||p3)   (3차 기여)
  c4 = KL(p||p3) - KL(p||p4)   (4차 기여)
  >=5차 잔여 = KL(p||p4)
이론상 KL1>=KL2>=KL3>=KL4>=0 (단조), 모든 기여 >=0. 음수 기여 = 수렴 실패 신호(진짜 값 아님).

사용: python tools/diagnose_order_decomposition.py [--dataset {unsw,bot,hoic,all}] [--kmax 4]
                                                  [--max-iter N] [--tol T] [--eps E] [--no-plot]
"""
import os
import sys
import json
import argparse
import itertools
from math import comb

import numpy as np
from scipy.optimize import minimize

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import train_qcbm_iat_b2 as TB2           # noqa: E402  build_b2()(HOIC)
import train_qcbm_bot_resumable as TBOT   # noqa: E402  build_bot()
import train_qcbm_unsw_resumable as TUNSW  # noqa: E402  build_unsw()

OUT = "results2"
OUT_JSON = os.path.join(OUT, "order_decomposition.json")
OUT_PNG = os.path.join(OUT, "order_decomposition.png")
MAXENT_JSON = os.path.join(OUT, "maxent_kl.json")   # 대조용(직전 KL2)

# UNSW(가장 작음) 먼저
DATASETS = {
    "unsw": dict(name="UNSW", build=TUNSW.build_unsw, nq=TUNSW.NQ),
    "bot":  dict(name="Bot",  build=TBOT.build_bot,   nq=TBOT.NQ),
    "hoic": dict(name="HOIC", build=TB2.build_b2,     nq=TB2.NQ),
}
ORDER = ["unsw", "bot", "hoic"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 기본 유틸
# ---------------------------------------------------------------------------
def bit_matrix(nq):
    xs = np.arange(1 << nq)
    return ((xs[:, None] >> np.arange(nq)[None, :]) & 1).astype(np.int64)


def kl_div(p, q, eps):
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    mask = p > 0
    qn = np.clip(q, eps, None)
    return float(np.sum(p[mask] * np.log(p[mask] / qn[mask])))


def entropy(p):
    p = np.asarray(p, dtype=np.float64)
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


# ---------------------------------------------------------------------------
# 차수 kmax 까지의 부분집합 특징 행렬 (spin 곱)
# ---------------------------------------------------------------------------
def subset_features(nq, kmax, bitmat):
    """반환 Phi (2^nq, D), D = sum_{k=1..kmax} C(nq,k). 열 = ∏_{i∈S} s_i."""
    s = (1 - 2 * bitmat).astype(np.float64)            # +-1 스핀 (N, nq)
    cols = []
    for k in range(1, kmax + 1):
        for S in itertools.combinations(range(nq), k):
            cols.append(np.prod(s[:, S], axis=1))
    return np.stack(cols, axis=1)


# ---------------------------------------------------------------------------
# 차수 kmax 까지 모멘트 보존 maxent (로그선형, L-BFGS) — diagnose_maxent_kl 일반화
# ---------------------------------------------------------------------------
def maxent_upto(p, nq, bitmat, kmax, max_iter=20000, tol=1e-8, eps=1e-12, verbose=False, Phi=None):
    if Phi is None:
        Phi = subset_features(nq, kmax, bitmat)
    mu_emp = Phi.T @ p

    def nll_grad(theta):
        e = Phi @ theta
        m = e.max()
        w = np.exp(e - m)
        Z = w.sum()
        q = w / Z
        nll = (m + np.log(Z)) - float(theta @ mu_emp)
        grad = (Phi.T @ q) - mu_emp
        return nll, grad

    theta0 = np.zeros(Phi.shape[1], dtype=np.float64)
    res = minimize(nll_grad, theta0, jac=True, method="L-BFGS-B",
                   options=dict(maxiter=max_iter, maxfun=max_iter * 10, gtol=tol, ftol=1e-18))
    e = Phi @ res.x
    e -= e.max()
    w = np.exp(e)
    q = w / w.sum()
    max_viol = float(np.max(np.abs(Phi.T @ q - mu_emp)))
    if verbose:
        log("    k≤%d: L-BFGS success=%s iters=%d  D=%d  max_moment_violation=%.3e"
            % (kmax, res.success, res.nit, Phi.shape[1], max_viol))
    info = dict(converged=bool(res.success and max_viol < 1e-6), iters=int(res.nit),
                max_violation=max_viol, nparams=int(Phi.shape[1]))
    return q, info


# ---------------------------------------------------------------------------
# 한 분포에 대한 차수별 분해
# ---------------------------------------------------------------------------
def decompose(p, nq, kmax=4, max_iter=20000, tol=1e-8, eps=1e-12, verbose=False):
    bm = bit_matrix(nq)
    Phi_full = subset_features(nq, kmax, bm)
    # 누적 열 수: 차수 k 까지의 부분집합 개수
    ncol = {}
    c = 0
    for k in range(1, kmax + 1):
        c += comb(nq, k)
        ncol[k] = c

    KL = {}
    per_k = {}
    for k in range(1, kmax + 1):
        Phi_k = Phi_full[:, :ncol[k]]
        qk, info = maxent_upto(p, nq, bm, k, max_iter=max_iter, tol=tol, eps=eps,
                               verbose=verbose, Phi=Phi_k)
        KL[k] = kl_div(p, qk, eps)
        per_k[k] = info

    contrib = {
        "c2": KL[1] - KL[2],
        "c3": KL[2] - KL[3],
        "c4": KL[3] - KL[4] if kmax >= 4 else None,
        "resid5": KL[kmax],
    }
    monotonic_ok = all(KL[k] >= KL[k + 1] - 1e-9 for k in range(1, kmax)) and KL[kmax] >= -1e-12
    neg_flags = {key: (v is not None and v < -1e-9) for key, v in contrib.items()}
    return dict(KL={str(k): KL[k] for k in KL}, per_k={str(k): per_k[k] for k in per_k},
                contrib=contrib, monotonic_ok=bool(monotonic_ok),
                negative_contrib=neg_flags, H=entropy(p))


# ---------------------------------------------------------------------------
# self-check (합성 분포)
# ---------------------------------------------------------------------------
def synth_independent(nq, rng):
    bm = bit_matrix(nq)
    pbit = rng.uniform(0.3, 0.7, size=nq)
    p = np.prod(np.where(bm == 1, pbit[None, :], 1 - pbit[None, :]), axis=1)
    return p / p.sum()


def synth_pairwise_ising(nq, rng):
    bm = bit_matrix(nq)
    s = 1 - 2 * bm
    h = rng.normal(0, 0.4, size=nq)
    J = np.triu(rng.normal(0, 0.4, size=(nq, nq)), 1)
    en = s @ h + np.einsum("xi,ij,xj->x", s, J, s)
    w = np.exp(en - en.max())
    return w / w.sum()


def synth_parity(nq, kbits, rng):
    """kbits-비트 parity 선호: 낮은 차수 상관 0, 순수 kbits 차 상관."""
    bm = bit_matrix(nq)
    par = np.zeros(1 << nq, dtype=np.int64)
    for b in range(kbits):
        par ^= bm[:, b]
    w = np.where(par == 0, 3.0, 1.0)
    return w / w.sum()


def self_check(eps):
    rng = np.random.default_rng(0)
    nq = 5
    out = {}

    # (a) 순수 3차 (3-bit parity)
    d3 = decompose(synth_parity(nq, 3, rng), nq, kmax=4, eps=eps)
    out["parity3"] = dict(c2=d3["contrib"]["c2"], c3=d3["contrib"]["c3"],
                          c4=d3["contrib"]["c4"], resid5=d3["contrib"]["resid5"],
                          monotonic=d3["monotonic_ok"])
    assert d3["contrib"]["c3"] > 1e-3, "parity3 c3 not >>0: %g" % d3["contrib"]["c3"]
    assert abs(d3["contrib"]["c4"]) < 1e-4, "parity3 c4 not ~0: %g" % d3["contrib"]["c4"]
    assert d3["contrib"]["resid5"] < 1e-4, "parity3 resid5 not ~0: %g" % d3["contrib"]["resid5"]

    # (b) 순수 4차 (4-bit parity)
    d4 = decompose(synth_parity(nq, 4, rng), nq, kmax=4, eps=eps)
    out["parity4"] = dict(c2=d4["contrib"]["c2"], c3=d4["contrib"]["c3"],
                          c4=d4["contrib"]["c4"], resid5=d4["contrib"]["resid5"],
                          monotonic=d4["monotonic_ok"])
    assert d4["contrib"]["c4"] > 1e-3, "parity4 c4 not >>0: %g" % d4["contrib"]["c4"]
    assert abs(d4["contrib"]["c3"]) < 1e-4, "parity4 c3 not ~0: %g" % d4["contrib"]["c3"]

    # (c) 2차 Ising → 3·4차 기여 ≈ 0
    di = decompose(synth_pairwise_ising(nq, rng), nq, kmax=4, eps=eps)
    out["ising2"] = dict(c3=di["contrib"]["c3"], c4=di["contrib"]["c4"], monotonic=di["monotonic_ok"])
    assert abs(di["contrib"]["c3"]) < 1e-4 and abs(di["contrib"]["c4"]) < 1e-4, \
        "ising2 c3/c4 not ~0: %g %g" % (di["contrib"]["c3"], di["contrib"]["c4"])

    # (d) 단조성 종합
    out["all_monotonic"] = bool(d3["monotonic_ok"] and d4["monotonic_ok"] and di["monotonic_ok"])
    assert out["all_monotonic"], "monotonicity violated in self-check"
    return out


# ---------------------------------------------------------------------------
def load_prior_kl2():
    try:
        d = json.load(open(MAXENT_JSON))
        return {k: v.get("KL_2nd_nats") for k, v in d.get("datasets", {}).items()}
    except Exception:
        return {}


def run(keys, kmax=4, do_plot=True, max_iter=20000, tol=1e-8, eps=1e-12):
    log("=== self-check (합성 분포, 차수 분해) ===")
    sc = self_check(eps)
    log("  parity3: c3=%.4f c4=%.2e resid5=%.2e (PASS)" % (sc["parity3"]["c3"], sc["parity3"]["c4"], sc["parity3"]["resid5"]))
    log("  parity4: c3=%.2e c4=%.4f (PASS)" % (sc["parity4"]["c3"], sc["parity4"]["c4"]))
    log("  ising2 : c3=%.2e c4=%.2e (PASS) | 단조성 all=%s" % (sc["ising2"]["c3"], sc["ising2"]["c4"], sc["all_monotonic"]))

    prior_kl2 = load_prior_kl2()
    results = {}
    for key in keys:
        cfg = DATASETS[key]
        nm, nq = cfg["name"], cfg["nq"]
        log("\n=== %s (%d bit / %d states) ===" % (nm, nq, 1 << nq))
        d = cfg["build"]()
        p = np.asarray(d["q_train"], dtype=np.float64); p = p / p.sum()
        occ = int(d.get("occ", int((p > 0).sum())))
        log("  점유 %d/%d | 차수별 maxent 적합(k=1..%d)..." % (occ, 1 << nq, kmax))
        dec = decompose(p, nq, kmax=kmax, max_iter=max_iter, tol=tol, eps=eps, verbose=True)

        KL = {int(k): v for k, v in dec["KL"].items()}
        ct = dec["contrib"]
        ge3_sum = ct["c3"] + (ct["c4"] or 0.0) + ct["resid5"]   # = KL2 (≥3차 합)
        pk2 = prior_kl2.get(nm)
        rec = dict(
            bits=nq, nstates=1 << nq, occ=occ, H=dec["H"],
            KL_1=KL[1], KL_2=KL[2], KL_3=KL[3], KL_4=KL.get(4),
            c2=ct["c2"], c3=ct["c3"], c4=ct["c4"], resid5=ct["resid5"],
            ge3_sum=ge3_sum, prior_KL2=pk2,
            ge3_sum_vs_priorKL2=(None if pk2 is None else ge3_sum - pk2),
            ratio_of_ge3=dict(
                c3=(ct["c3"] / ge3_sum if ge3_sum > 0 else None),
                c4=((ct["c4"] or 0.0) / ge3_sum if ge3_sum > 0 else None),
                resid5=(ct["resid5"] / ge3_sum if ge3_sum > 0 else None)),
            per_k=dec["per_k"], monotonic_ok=dec["monotonic_ok"],
            negative_contrib=dec["negative_contrib"],
        )
        results[nm] = rec
        log("  KL: k1=%.4f k2=%.4f k3=%.4f k4=%.4f" % (KL[1], KL[2], KL[3], KL.get(4, float("nan"))))
        log("  기여(nats): c2=%.4f c3=%.4f c4=%.4f ≥5차잔여=%.4f | ≥3차합=%.4f (직전KL2=%s, Δ=%s)"
            % (ct["c2"], ct["c3"], ct["c4"], ct["resid5"], ge3_sum, pk2,
               None if pk2 is None else round(ge3_sum - pk2, 5)))
        log("  단조성=%s | 음수기여=%s | per_k 수렴=%s"
            % (dec["monotonic_ok"], dec["negative_contrib"],
               {k: v["converged"] for k, v in dec["per_k"].items()}))

    payload = dict(
        note=("차수별 모멘트-보존 로그선형 maxent(L-BFGS)로 p1..p_kmax 적합, 연속 KL 차이로 차수별 순수 기여 분해. "
              "c2=KL1-KL2, c3=KL2-KL3, c4=KL3-KL4, ≥5차잔여=KL_kmax. 이론상 모두 ≥0(단조); 음수=미수렴 신호. "
              "≥3차합=c3+c4+잔여 는 직전 maxent_kl 의 KL2 와 일치해야. 학습 없음. 단정 금지."),
        params=dict(kmax=kmax, max_iter=max_iter, tol=tol, eps=eps, units="nats"),
        self_check=sc, datasets=results,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if do_plot and results:
        plot(results)
    return payload


def plot(results):
    names = list(results.keys())
    x = np.arange(len(names))
    c2 = [results[n]["c2"] for n in names]
    c3 = [results[n]["c3"] for n in names]
    c4 = [results[n]["c4"] or 0.0 for n in names]
    r5 = [results[n]["resid5"] for n in names]
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.0))
    b = np.zeros(len(names))
    for vals, lab, col in [(c2, "2nd", "#4C9F70"), (c3, "3rd", "#3B6EA5"),
                           (c4, "4th", "#E2A13B"), (r5, ">=5th residual", "#C1442E")]:
        ax[0].bar(x, vals, bottom=b, label=lab, color=col)
        b = b + np.array(vals)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("KL contribution (nats)")
    ax[0].set_title("Per-order correlation contribution\n(total = KL(p || 1st-maxent))")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")
    # 오른쪽: ≥3차 내부 비율
    for xi, n in enumerate(names):
        ge3 = results[n]["c3"] + (results[n]["c4"] or 0.0) + results[n]["resid5"]
        if ge3 <= 0:
            continue
        parts = [results[n]["c3"] / ge3, (results[n]["c4"] or 0.0) / ge3, results[n]["resid5"] / ge3]
        bb = 0.0
        for v, col in zip(parts, ["#3B6EA5", "#E2A13B", "#C1442E"]):
            ax[1].bar(xi, v, bottom=bb, color=col)
            bb += v
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("share of >=3rd-order (KL2)")
    ax[1].set_title("Within >=3rd-order: 3rd / 4th / >=5th split")
    ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Higher-order decomposition: how far does >=3rd-order correlation spread?")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="고차 상관 차수별 분해(2/3/4/≥5차). 학습 없음.")
    ap.add_argument("--dataset", default="all", choices=["all"] + ORDER, help="기본 all (UNSW 먼저)")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--eps", type=float, default=1e-12)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    keys = ORDER if args.dataset == "all" else [args.dataset]
    run(keys, kmax=args.kmax, do_plot=not args.no_plot,
        max_iter=args.max_iter, tol=args.tol, eps=args.eps)


if __name__ == "__main__":
    main()
