#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고차 상관 진단 — benign 경험적 분포의 k-body Pauli-Z correlator 스펙트럼 (HOIC/Bot/UNSW).

순수 측정(학습 없음). 기존 인코딩/평가 코드 무수정, import 재사용만.

이론: spin s_i = 1 - 2*bit_i 일 때 k-body Pauli-Z correlator
  p_hat(S) = sum_x p(x) * (-1)^popcount(S & x) = E[ prod_{i in S} s_i ]
는 (자연순서) Walsh-Hadamard 변환 계수와 정확히 같다. 차수 k = popcount(S).
  W_k = sum_{popcount(S)=k} p_hat(S)^2   (raw 스펙트럼; k=0 상수항 p_hat(∅)=1 은 비율에서 제외)
QCBM 은 Fourier(=Pauli-Z) 모델이므로, 이 스펙트럼이 저차에 몰리면 고전 저랭크가 싸게 대체 가능,
고차로 퍼지면 고전이 비싸진다(양자 여지).

해석 주의: raw 스펙트럼은 분포의 peakiness/단일비트 bias 에 민감하다(델타함수는 전 차수 평탄).
독립(곱) 분포에서도 raw 고차항은 0 이 아니라 prod E[s_i] 로 기하 감쇠한다. "독립 대비 초과 구조"는
connected(cumulant) 또는 marginal(독립) baseline 과의 대조로 본다 -> 둘 다 산출/병기한다.

사용: python tools/diagnose_higher_order_correlation.py [--datasets hoic,bot,unsw] [--no-plot]
"""
import os
import sys
import json
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

import experiment2 as E                   # noqa: E402  empirical_dist, marginal_dist via TB2
import train_qcbm_iat_b2 as TB2           # noqa: E402  build_b2() (HOIC), marginal_dist
import train_qcbm_bot_resumable as TBOT   # noqa: E402  build_bot()
import train_qcbm_unsw_resumable as TUNSW  # noqa: E402  build_unsw()

OUT = "results2"
OUT_JSON = os.path.join(OUT, "higher_order_correlation.json")
OUT_PNG = os.path.join(OUT, "higher_order_correlation.png")

# 데이터셋 정의: native 인코딩 빌드 + 동일 인코딩 SVD reach 출처
DATASETS = {
    "hoic": dict(name="HOIC", build=TB2.build_b2,   nq=TB2.NQ,
                 svd_json="attack_complexity.json",        svd_tag="HOIC"),
    "bot":  dict(name="Bot",  build=TBOT.build_bot, nq=TBOT.NQ,
                 svd_json="attack_complexity.json",        svd_tag="Bot"),
    "unsw": dict(name="UNSW", build=TUNSW.build_unsw, nq=TUNSW.NQ,
                 svd_json="cross_dataset_diagnosis.json",  svd_tag="UNSW"),
}


# ---------------------------------------------------------------------------
# Walsh-Hadamard 변환 (자연/Sylvester 순서, 정규화 없음): y[S]=sum_x (-1)^popcount(S&x) x_j
# ---------------------------------------------------------------------------
def fwht(a):
    a = np.asarray(a, dtype=np.float64).copy()
    n = a.shape[0]
    assert (n & (n - 1)) == 0, "length must be power of 2"
    h = 1
    while h < n:
        for i in range(0, n, h * 2):
            x = a[i:i + h].copy()
            y = a[i + h:i + 2 * h].copy()
            a[i:i + h] = x + y
            a[i + h:i + 2 * h] = x - y
        h *= 2
    return a


def popcounts(nstates):
    """index 0..nstates-1 의 set-bit 수."""
    pc = np.zeros(nstates, dtype=np.int64)
    i = 1
    while i < nstates:
        pc[i:2 * i] = pc[:i] + 1
        i *= 2
    return pc


def order_spectrum_raw(p, nq):
    """raw W_k (k=0..nq) = sum_{|S|=k} p_hat(S)^2,  p_hat = FWHT(p)."""
    p = np.asarray(p, dtype=np.float64)
    nstates = 1 << nq
    assert p.shape[0] == nstates, (p.shape[0], nstates)
    phat = fwht(p)
    pc = popcounts(nstates)
    W = np.zeros(nq + 1, dtype=np.float64)
    sq = phat * phat
    for k in range(nq + 1):
        W[k] = sq[pc == k].sum()
    return W, phat, pc


def ratios_from_W(W):
    """W_0 제외 정규화 + low/high ratio + 누적곡선."""
    Wpos = W[1:].copy()
    tot = Wpos.sum()
    if tot <= 0:
        return None
    norm = Wpos / tot                          # k=1..nq 에 대한 정규화 가중치
    low = float((W[1] + (W[2] if len(W) > 2 else 0.0)) / tot)
    high = float(W[3:].sum() / tot) if len(W) > 3 else 0.0
    cum = np.cumsum(norm).tolist()             # k<=1, k<=2, ... 누적
    return dict(W_norm=norm.tolist(), low_order_ratio=low,
                high_order_ratio=high, cumulative=cum)


def connected_2body(phat, pc):
    """2-body connected mass: sum_{i<j}(E[s_i s_j]-E[s_i]E[s_j])^2 = sum_{pairs}(phat_pair - phat_i*phat_j)^2.
    독립(곱) 분포에서 정확히 0 이어야 함(검증용/대조용)."""
    nstates = phat.shape[0]
    nbits = int(np.log2(nstates))
    single = {b: phat[1 << b] for b in range(nbits)}     # E[s_b]
    total = 0.0
    for i in range(nbits):
        for j in range(i + 1, nbits):
            idx = (1 << i) | (1 << j)
            c = phat[idx] - single[i] * single[j]
            total += c * c
    return float(total)


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
def self_check():
    rng = np.random.default_rng(0)
    n = 4
    N = 1 << n
    out = {}

    # (a) FWHT 기반 W_k == 직접 곱-평균 W_k (임의 분포)
    p = rng.random(N)
    p = p / p.sum()
    W_fwht, phat, pc = order_spectrum_raw(p, n)
    # 직접: 각 부분집합 S 에 대해 sum_x p(x) prod_{i in S} (1-2*bit_i(x))
    bits = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1)      # (N, n)
    spins = 1 - 2 * bits                                                # +-1
    W_direct = np.zeros(n + 1)
    for S in range(N):
        sel = [i for i in range(n) if (S >> i) & 1]
        prod = np.ones(N) if not sel else np.prod(spins[:, sel], axis=1)
        corr = float((p * prod).sum())
        W_direct[bin(S).count("1")] += corr * corr
    max_abs_diff = float(np.max(np.abs(W_fwht - W_direct)))
    assert max_abs_diff < 1e-10, "FWHT vs direct mismatch: %g" % max_abs_diff
    out["wht_vs_direct_max_abs_diff"] = max_abs_diff

    # (b) 독립(곱) 분포: connected 2-body ≈ 0, raw 고차는 0 아님(기하 감쇠)
    pbit = np.array([0.30, 0.60, 0.45, 0.70])
    bits_b = ((np.arange(N)[:, None] >> np.arange(n)[None, :]) & 1)
    pind = np.prod(np.where(bits_b == 1, pbit[None, :], 1 - pbit[None, :]), axis=1)
    pind = pind / pind.sum()
    Wi, phati, pci = order_spectrum_raw(pind, n)
    c2 = connected_2body(phati, pci)
    ri = ratios_from_W(Wi)
    out["independent_connected_2body"] = c2
    out["independent_raw_high_order_ratio"] = ri["high_order_ratio"]
    assert c2 < 1e-12, "independent connected-2body not ~0: %g" % c2

    return out


# ---------------------------------------------------------------------------
# SVD reach 읽기 (데이터셋 native 인코딩 출처)
# ---------------------------------------------------------------------------
def read_svd_reach(svd_json, tag):
    path = os.path.join(OUT, svd_json)
    try:
        d = json.load(open(path))
    except Exception as e:
        return dict(error="%r" % e)
    entries = d.get("results") or d.get("datasets") or []
    for x in entries:
        nm = x.get("tag") or x.get("name") or x.get("dataset")
        if nm == tag:
            svd = x.get("svd") or {}
            return dict(source="%s:%s" % (svd_json, tag),
                        reach=svd.get("reach_rank"),
                        cum_energy=svd.get("cum_energy"))
    return dict(error="tag %r not found in %s" % (tag, svd_json))


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def run(keys, do_plot=True):
    print("=== self-check ===")
    sc = self_check()
    print("  WHT vs direct max|diff| = %.2e (PASS)" % sc["wht_vs_direct_max_abs_diff"])
    print("  독립분포 connected-2body = %.2e (≈0 PASS) | 독립분포 raw high_order_ratio = %.4f (0 아님: raw 한계)"
          % (sc["independent_connected_2body"], sc["independent_raw_high_order_ratio"]))

    results = {}
    for key in keys:
        cfg = DATASETS[key]
        nm, nq = cfg["name"], cfg["nq"]
        print("\n=== %s (%d bit / %d states) ===" % (nm, nq, 1 << nq))
        d = cfg["build"]()
        q = np.asarray(d["q_train"], dtype=np.float64)
        occ = int(d.get("occ", int((q > 0).sum())))

        # 경험적(joint) 스펙트럼
        W, phat, pc = order_spectrum_raw(q, nq)
        r = ratios_from_W(W)
        c2 = connected_2body(phat, pc)

        # marginal(독립) baseline 스펙트럼 (동일 states 의 비트독립 곱)
        qm = np.asarray(TB2.marginal_dist(d["st_tr"], nq), dtype=np.float64)
        Wm, phatm, pcm = order_spectrum_raw(qm, nq)
        rm = ratios_from_W(Wm)

        svd = read_svd_reach(cfg["svd_json"], cfg["svd_tag"])

        print("  점유 %d/%d (%.2f%%)" % (occ, 1 << nq, 100 * occ / (1 << nq)))
        print("  W_k(raw, k=0..%d) = %s" % (nq, np.array2string(W, precision=4, suppress_small=True)))
        print("  low_order_ratio(k<=2)=%.4f  high_order_ratio(k>=3)=%.4f  | marginal high=%.4f"
              % (r["low_order_ratio"], r["high_order_ratio"], rm["high_order_ratio"]))
        print("  connected-2body(emp)=%.4e | SVD reach=%s" % (c2, svd.get("reach")))

        results[nm] = dict(
            bits=nq, nstates=1 << nq, occ=occ,
            W_raw=W.tolist(),
            W_norm=r["W_norm"], low_order_ratio=r["low_order_ratio"],
            high_order_ratio=r["high_order_ratio"], cumulative=r["cumulative"],
            connected_2body=c2,
            marginal_W_norm=rm["W_norm"], marginal_high_order_ratio=rm["high_order_ratio"],
            svd_reach=svd,
        )

    payload = dict(
        note=("benign-train empirical dist 의 raw k-body Pauli-Z correlator 스펙트럼(=FWHT). "
              "raw 는 peakiness/bias 에 민감 -> marginal baseline 및 connected-2body 병기. "
              "각 데이터셋 SVD reach 는 동일 native 인코딩 출처에서 읽음. 학습 없음. 판정 금지."),
        self_check=sc,
        method=dict(spin="s_i=1-2*bit_i", correlator="p_hat(S)=FWHT(p)[S]=E[prod s_i]",
                    W_k="sum_{|S|=k} p_hat(S)^2 (k=0 제외 정규화)",
                    low="(W1+W2)/sum_{k>=1}", high="sum_{k>=3}/sum_{k>=1}",
                    dist="benign-train empirical (q_train), 전수, no sampling"),
        datasets=results,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    print("\n[저장] %s" % OUT_JSON)

    if do_plot and results:
        plot(results)
    return payload


def plot(results):
    names = list(results.keys())
    fig, ax = plt.subplots(1, 2, figsize=(15.5, 6.2))
    colors = plt.cm.tab10(np.linspace(0, 1, max(3, len(names))))
    for c, nm in zip(colors, names):
        r = results[nm]
        nq = r["bits"]
        ks = np.arange(1, nq + 1)
        wn = np.asarray(r["W_norm"])
        lbl = "%s (%db, SVDreach=%s, high=%.2f)" % (
            nm, nq, (r["svd_reach"] or {}).get("reach", {}).get("tv<=0.05") if isinstance(r["svd_reach"], dict) else "?",
            r["high_order_ratio"])
        ax[0].plot(ks, wn, "o-", color=c, label=lbl)
        ax[0].plot(ks, r["marginal_W_norm"], "x--", color=c, alpha=0.4)
        ax[1].plot(ks, np.cumsum(wn), "o-", color=c, label="%s" % nm)
    ax[0].set_xlabel("correlator order k"); ax[0].set_ylabel("normalized weight W_k / sum_{k>=1}")
    ax[0].set_title("k-body Pauli-Z spectrum (solid=empirical, dashed=marginal baseline)")
    ax[0].set_yscale("log"); ax[0].grid(alpha=.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("correlator order k"); ax[1].set_ylabel("cumulative weight (k'<=k)")
    ax[1].set_title("Cumulative order spectrum (how fast low orders saturate)")
    ax[1].axhline(1.0, color="gray", lw=0.8); ax[1].grid(alpha=.3); ax[1].legend(fontsize=9)
    plt.suptitle("Higher-order correlation spectrum: HOIC / Bot / UNSW (benign-train empirical)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125)
    plt.close()
    print("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="benign 경험적 분포의 k-body Pauli-Z correlator 스펙트럼 진단(학습 없음).")
    ap.add_argument("--datasets", default="hoic,bot,unsw",
                    help="콤마구분: hoic,bot,unsw (기본 전체)")
    ap.add_argument("--no-plot", action="store_true", help="PNG 생략")
    args = ap.parse_args()
    keys = [k.strip().lower() for k in args.datasets.split(",") if k.strip()]
    bad = [k for k in keys if k not in DATASETS]
    if bad:
        ap.error("알 수 없는 데이터셋: %s (가능: %s)" % (bad, list(DATASETS)))
    run(keys, do_plot=not args.no_plot)


if __name__ == "__main__":
    main()
