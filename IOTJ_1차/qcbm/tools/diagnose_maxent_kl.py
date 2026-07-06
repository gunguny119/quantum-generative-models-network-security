#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
저차(1·2차) max-entropy 대비 KL 진단 — "2차로 설명 안 되는 기약 고차(≥3) 상관" 측정.

순수 측정(학습 없음). 기존 인코딩/분포 코드 무수정, import 재사용만. 단일 프로세스 순차(병렬화 없음).

배경: 직전 raw k-body Pauli-Z 스펙트럼은 분포 희소성/단일비트 bias 에 confound(독립분포인데 raw high≠0).
이번엔 1차(단일비트)·2차(쌍) 통계만 보존하는 max-entropy 분포 p2 를 만들고 KL(p||p2) 를 잰다. 이는
정보기하상 "2차로 설명 안 되는 잔여 = 순수 ≥3차 고차 상관" 이다.
  KL(p||p1) = KL(p||p2) + KL(p2||p1)   (p1=1차 maxent=독립, nested exponential family)
- KL(p||p2) 작으면 저차로 충분(고전 surrogate 쌈, 양자 무망), 크면 고차 본질(양자 여지).
- p1 = marginal_dist(단일비트 곱) 재사용. p2 = 1·2차 모멘트 보존 pairwise 로그선형 maxent(L-BFGS).
  (평면 IPF 는 선형 수렴이 너무 느려 tol 미달 → convex 로그선형 적합으로 대체. 사유는 코드 주석/.md 참조.)

사용: python tools/diagnose_maxent_kl.py [--dataset {unsw,bot,hoic,all}] [--no-plot]
                                        [--max-iter N] [--tol T] [--eps E]
"""
import os
import sys
import json
import argparse
import itertools

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

import train_qcbm_iat_b2 as TB2           # noqa: E402  build_b2()(HOIC), marginal_dist (=1차 maxent)
import train_qcbm_bot_resumable as TBOT   # noqa: E402  build_bot()
import train_qcbm_unsw_resumable as TUNSW  # noqa: E402  build_unsw()

OUT = "results2"
OUT_JSON = os.path.join(OUT, "maxent_kl.json")
OUT_PNG = os.path.join(OUT, "maxent_kl.png")
HOC_JSON = os.path.join(OUT, "higher_order_correlation.json")   # 대조용(raw high_order_ratio)

# UNSW(가장 작음) 먼저 처리
DATASETS = {
    "unsw": dict(name="UNSW", build=TUNSW.build_unsw, nq=TUNSW.NQ),
    "bot":  dict(name="Bot",  build=TBOT.build_bot,   nq=TBOT.NQ),
    "hoic": dict(name="HOIC", build=TB2.build_b2,     nq=TB2.NQ),
}
ORDER = ["unsw", "bot", "hoic"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# 비트 행렬 / KL
# ---------------------------------------------------------------------------
def bit_matrix(nq):
    """bitmat[x,i] = (x>>i)&1,  shape (2^nq, nq)."""
    xs = np.arange(1 << nq)
    return ((xs[:, None] >> np.arange(nq)[None, :]) & 1).astype(np.int64)


def kl_div(p, q, eps):
    """KL(p||q) in nats. p=0 항은 0. q는 eps 하한."""
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
# 2차 max-entropy 분포 = pairwise 로그선형(Ising) 적합, convex, L-BFGS
# ---------------------------------------------------------------------------
# 참고: 평면 IPF 는 marginal 위반이 선형(매우 느리게) 수렴해 tol(1e-8) 도달에 수백만 sweep 이 필요했다
# (UNSW smoke: 2000 sweep 후 위반 3.8e-5 에서 정체). 동일한 1·2차 maxent 를 모멘트(E[s_i], E[s_i s_j]) 를
# 보존하는 pairwise 로그선형 모델 p∝exp(Σ h_i s_i + Σ J_ij s_i s_j) 로 두면 음의 로그가능도가 convex 이고
# 그 gradient = (model moment − empirical moment) 이므로 L-BFGS 로 위반을 tol 이하로 빠르게 내릴 수 있다.
def maxent_2nd(p, nq, bitmat, max_iter=20000, tol=1e-8, eps=1e-12, verbose=False):
    """
    1·2차 모멘트(E[s_i], E[s_i s_j])를 보존하는 maxent 분포(= pairwise 로그선형)를 L-BFGS 로 적합.
    q_train 벡터 p 만 사용. 반환 (q, info), info=dict(converged, iters, max_violation).
    """
    N = 1 << nq
    s = (1 - 2 * bitmat).astype(np.float64)            # +-1 스핀 (N, nq)
    pairs = list(itertools.combinations(range(nq), 2))
    cols = [s[:, i] for i in range(nq)] + [s[:, i] * s[:, j] for (i, j) in pairs]
    Phi = np.stack(cols, axis=1)                        # (N, D), D = nq + C(nq,2)
    mu_emp = Phi.T @ p                                  # 보존할 1·2차 모멘트

    def nll_grad(theta):
        e = Phi @ theta
        m = e.max()
        w = np.exp(e - m)
        Z = w.sum()
        q = w / Z
        logZ = m + np.log(Z)
        nll = logZ - float(theta @ mu_emp)
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
        log("    L-BFGS: success=%s iters=%d  max_moment_violation=%.3e"
            % (res.success, res.nit, max_viol))
    return q, dict(converged=bool(res.success and max_viol < 1e-6),
                   iters=int(res.nit), max_violation=max_viol)


# ---------------------------------------------------------------------------
# self-check (합성 분포)
# ---------------------------------------------------------------------------
def synth_independent(nq, rng):
    bm = bit_matrix(nq)
    pbit = rng.uniform(0.25, 0.75, size=nq)
    p = np.prod(np.where(bm == 1, pbit[None, :], 1 - pbit[None, :]), axis=1)
    return p / p.sum()


def synth_pairwise_ising(nq, rng):
    bm = bit_matrix(nq)
    s = 1 - 2 * bm                          # +-1 spins, (N, nq)
    h = rng.normal(0, 0.5, size=nq)
    J = rng.normal(0, 0.5, size=(nq, nq)); J = np.triu(J, 1)
    E_lin = s @ h
    E_pair = np.einsum("xi,ij,xj->x", s, J, s)
    en = E_lin + E_pair
    w = np.exp(en - en.max())
    return w / w.sum()


def synth_parity3(nq, rng):
    """3-bit parity 선호: 쌍상관 ~0, 순수 3차 상관. (bit0^bit1^bit2)=0 인 상태에 가중."""
    bm = bit_matrix(nq)
    par = (bm[:, 0] ^ bm[:, 1] ^ bm[:, 2])
    base = synth_independent(nq, rng)       # 약한 1차 배경
    w = base * np.where(par == 0, 3.0, 1.0)
    return w / w.sum()


def self_check(eps):
    rng = np.random.default_rng(0)
    nq = 4
    bm = bit_matrix(nq)
    out = {}

    p_ind = synth_independent(nq, rng)
    q2, info = maxent_2nd(p_ind, nq, bm, eps=eps)
    kl_ind = kl_div(p_ind, q2, eps)
    out["independent_KL2_nats"] = kl_ind
    assert kl_ind < 1e-6, "independent KL(p||p2) not ~0: %g" % kl_ind

    p_is = synth_pairwise_ising(nq, rng)
    q2b, info_b = maxent_2nd(p_is, nq, bm, eps=eps)
    kl_is = kl_div(p_is, q2b, eps)
    out["pairwise_ising_KL2_nats"] = kl_is
    assert kl_is < 1e-6, "pairwise-Ising KL(p||p2) not ~0: %g" % kl_is

    p_par = synth_parity3(nq, rng)
    q2c, info_c = maxent_2nd(p_par, nq, bm, eps=eps)
    kl_par = kl_div(p_par, q2c, eps)
    out["parity3_KL2_nats"] = kl_par
    assert kl_par > 1e-3, "parity3 KL(p||p2) not >> 0: %g" % kl_par

    out["lbfgs_independent_iters"] = info["iters"]
    out["lbfgs_parity3_iters"] = info_c["iters"]
    return out


# ---------------------------------------------------------------------------
# 1차 maxent (p1) — q_train 벡터에서 직접(원자료 없이도): 단일비트 marginal 곱
# ---------------------------------------------------------------------------
def maxent_1st_from_p(p, nq, bitmat):
    pbit = np.array([float(np.sum(p[bitmat[:, i] == 1])) for i in range(nq)])
    q = np.prod(np.where(bitmat == 1, pbit[None, :], 1 - pbit[None, :]), axis=1)
    return q / q.sum()


# ---------------------------------------------------------------------------
# raw high_order_ratio 대조 로드
# ---------------------------------------------------------------------------
def load_raw_high():
    try:
        d = json.load(open(HOC_JSON))
        return {k: v.get("high_order_ratio") for k, v in d.get("datasets", {}).items()}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(keys, do_plot=True, max_iter=2000, tol=1e-8, eps=1e-12):
    log("=== self-check (n=4 합성 분포) ===")
    sc = self_check(eps)
    log("  독립 KL2=%.3e (≈0 PASS) | 2차Ising KL2=%.3e (≈0 PASS) | 3차parity KL2=%.4f (>>0 PASS)"
        % (sc["independent_KL2_nats"], sc["pairwise_ising_KL2_nats"], sc["parity3_KL2_nats"]))

    raw_high = load_raw_high()
    results = {}
    for key in keys:
        cfg = DATASETS[key]
        nm, nq = cfg["name"], cfg["nq"]
        N = 1 << nq
        log("\n=== %s (%d bit / %d states) ===" % (nm, nq, N))
        d = cfg["build"]()
        p = np.asarray(d["q_train"], dtype=np.float64)
        p = p / p.sum()
        occ = int(d.get("occ", int((p > 0).sum())))
        bm = bit_matrix(nq)

        p1 = maxent_1st_from_p(p, nq, bm)
        log("  maxent(2차) L-BFGS 적합...")
        p2, info = maxent_2nd(p, nq, bm, max_iter=max_iter, tol=tol, eps=eps, verbose=True)

        Hp = entropy(p)
        kl2 = kl_div(p, p2, eps)
        kl1 = kl_div(p, p1, eps)
        kl21 = kl_div(p2, p1, eps)          # 2차가 설명하는 양
        pyth = kl1 - (kl2 + kl21)           # ≈0 이어야(정보기하 일관성)

        rec = dict(
            bits=nq, nstates=N, occ=occ,
            KL_2nd_nats=kl2, KL_2nd_bits=kl2 / np.log(2),
            KL_1st_nats=kl1, KL_1st_bits=kl1 / np.log(2),
            KL_2nd_over_1st=(kl2 / kl1 if kl1 > 0 else float("nan")),  # ≥3차가 차지하는 비율(전체 상관 중)
            KL_p2_p1_nats=kl21,
            H_p_nats=Hp, KL_2nd_over_H=(kl2 / Hp if Hp > 0 else float("nan")),
            pythagoras_residual=pyth,
            ipf_converged=info["converged"], ipf_iters=info["iters"],
            ipf_max_violation=info["max_violation"],
            raw_high_order_ratio=raw_high.get(nm),
        )
        results[nm] = rec
        log("  점유 %d/%d | H(p)=%.4f nats" % (occ, N, Hp))
        log("  KL(p||p1=1차)=%.4f | KL(p||p2=2차)=%.4f nats (=%.4f bits) | KL(p2||p1)=%.4f"
            % (kl1, kl2, kl2 / np.log(2), kl21))
        log("  ≥3차 비율 KL2/KL1=%.3f | KL2/H(p)=%.4f | maxent conv=%s iters=%d viol=%.2e | Pythagoras잔차=%.2e"
            % (rec["KL_2nd_over_1st"], rec["KL_2nd_over_H"], info["converged"], info["iters"],
               info["max_violation"], pyth))
        log("  (대조) raw high_order_ratio=%s" % raw_high.get(nm))

    payload = dict(
        note=("benign-train empirical dist p 의 KL(p||p2): p2=1·2차 marginal 보존 max-entropy(IPF). "
              "KL(p||p2)=순수 ≥3차 고차 상관(confound 제거). KL(p||p1)=독립 초과 전체 상관(p1=marginal_dist). "
              "KL2/KL1=전체 상관 중 ≥3차 비율. 학습 없음. 단정 금지."),
        params=dict(max_iter=max_iter, tol=tol, eps=eps, units="nats(주)/bits"),
        self_check=sc,
        datasets=results,
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
    kl1 = [results[n]["KL_1st_nats"] for n in names]
    kl2 = [results[n]["KL_2nd_nats"] for n in names]
    kl2_over_h = [results[n]["KL_2nd_over_H"] for n in names]

    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6.0))
    # 왼쪽: KL 분해 (총 KL1 = 2차기여 + ≥3차). 막대: ≥3차(KL2) + 2차기여(KL1-KL2)
    second_contrib = [a - b for a, b in zip(kl1, kl2)]
    ax[0].bar(x, second_contrib, label="2nd-order contribution  KL(p2||p1)", color="#4C9F70")
    ax[0].bar(x, kl2, bottom=second_contrib, label="higher-order (>=3)  KL(p||p2)", color="#C1442E")
    for xi, (s, h) in enumerate(zip(second_contrib, kl2)):
        ax[0].text(xi, s + h, "%.3f" % h, ha="center", va="bottom", fontsize=10)
    ax[0].set_xticks(x); ax[0].set_xticklabels(names)
    ax[0].set_ylabel("KL from independent (nats)")
    ax[0].set_title("Correlation decomposition: 2nd-order vs higher-order(>=3)\n(total bar = KL(p||1st-maxent))")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")
    # 오른쪽: KL2/H(p) 정규화 + ≥3차 비율
    ratio3 = [results[n]["KL_2nd_over_1st"] for n in names]
    w = 0.38
    ax[1].bar(x - w / 2, kl2_over_h, w, label="KL(p||p2) / H(p)", color="#C1442E")
    ax[1].bar(x + w / 2, ratio3, w, label="KL2/KL1 (>=3 share of all corr.)", color="#3B6EA5")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names)
    ax[1].set_ylabel("ratio"); ax[1].set_title("Normalized higher-order indicators")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Max-entropy KL: irreducible higher-order(>=3) correlation (benign-train empirical)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125)
    plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(
        description="1·2차 max-entropy 대비 KL 진단(=순수 ≥3차 고차 상관). 학습 없음.")
    ap.add_argument("--dataset", default="all", choices=["all"] + ORDER,
                    help="처리할 데이터셋 (기본 all; UNSW 먼저)")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--tol", type=float, default=1e-8)
    ap.add_argument("--eps", type=float, default=1e-12)
    args = ap.parse_args()
    keys = ORDER if args.dataset == "all" else [args.dataset]
    run(keys, do_plot=not args.no_plot, max_iter=args.max_iter, tol=args.tol, eps=args.eps)


if __name__ == "__main__":
    main()
