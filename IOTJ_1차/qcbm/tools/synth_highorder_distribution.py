#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고차 비중 조절 가능한 합성 분포 생성기.

log p(s) = alpha * (랜덤 1차+2차 항) + beta * (선택 k>=3 parity 항),  s_i = 1 - 2*bit_i
softmax 정규화(전체 2^n, 정확). beta(고차 가중)로 고차 비중 조절.
각 분포의 "실제 ≥3차 비중"은 maxent 차수분해로 사후측정: ge3 = KL(p||p2)/KL(p||p1).

학습 없음. diagnose_order_decomposition(maxent_upto/kl_div/bit_matrix) 재사용.

사용: python tools/synth_highorder_distribution.py --selftest
"""
import os
import sys
import argparse
import itertools

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import diagnose_order_decomposition as OD  # noqa: E402  maxent_upto, kl_div, bit_matrix

EPS = 1e-12


def _spin(nq):
    bm = OD.bit_matrix(nq)                 # (2^nq, nq) 0/1
    return (1 - 2 * bm).astype(np.float64)  # +-1


LOW_SCALE = 1.0
HIGH_SCALE = 3.0   # 고차 블록이 t=1 에서 분포를 지배하도록


def make_terms(nq, seed, n_high=None, high_mode="disjoint"):
    """저차(1·2차) 항 + 고차(k≥3 parity) 항(랜덤, 고정).
    high_mode:
      'disjoint' = 겹치지 않는 3-bit parity 블록 → ≤2차 모멘트=0, share→1, 단 reshape분할에 대해 저랭크(분리가능).
      'random'   = 겹치는 3·4-bit parity 다수(랜덤부호) → 고차+고랭크(비분리). 교차점 테스트용. share는 1 미만."""
    rng = np.random.default_rng(seed)
    low = []   # (subset, coef)
    for i in range(nq):
        low.append(((i,), rng.normal(0, 1.0)))
    for i, j in itertools.combinations(range(nq), 2):
        low.append(((i, j), rng.normal(0, 1.0)))
    high = []
    if high_mode == "disjoint":
        i = 0
        while i + 3 <= nq:
            high.append((tuple(range(i, i + 3)), float(rng.choice([-1.0, 1.0]))))
            i += 3
    elif high_mode == "random":
        pool = list(itertools.combinations(range(nq), 3)) + list(itertools.combinations(range(nq), 4))
        rng.shuffle(pool)
        n_take = n_high if n_high else max(4, 2 * nq)
        for S in pool[:n_take]:
            high.append((S, float(rng.choice([-1.0, 1.0]))))
    else:
        raise ValueError(high_mode)
    return low, high


def synth_dist(nq, t, seed=0, n_high=None, terms=None, high_mode="disjoint"):
    """합성 분포 p (길이 2^nq) 반환.
    t in [0,1]: 저차↔고차 혼합. t=0 순수 저차(≥3차 share≈0), t=1 순수 고차."""
    s = _spin(nq)
    if terms is None:
        terms = make_terms(nq, seed, n_high, high_mode=high_mode)
    low, high = terms
    a = (1.0 - t) * LOW_SCALE
    b = t * HIGH_SCALE
    energy = np.zeros(s.shape[0], dtype=np.float64)
    for S, c in low:
        energy += a * c * np.prod(s[:, list(S)], axis=1)
    for S, c in high:
        energy += b * c * np.prod(s[:, list(S)], axis=1)
    energy -= energy.max()
    p = np.exp(energy)
    return p / p.sum(), terms


def ge3_share(p, nq, max_iter=20000):
    """≥3차 비중 = KL(p||p2)/KL(p||p1). maxent_upto 재사용(p1=1차, p2=1·2차)."""
    bm = OD.bit_matrix(nq)
    q1, _ = OD.maxent_upto(p, nq, bm, kmax=1, max_iter=max_iter)
    q2, _ = OD.maxent_upto(p, nq, bm, kmax=2, max_iter=max_iter)
    kl1 = OD.kl_div(p, q1, EPS)
    kl2 = OD.kl_div(p, q2, EPS)
    return (kl2 / kl1 if kl1 > 0 else 0.0), kl1, kl2


def selftest():
    nq = 6
    terms = make_terms(nq, seed=1)
    print("=== synth self-test (n=%d) ===" % nq)
    shares = []
    for t in [0.0, 0.25, 0.5, 0.75, 1.0]:
        p, _ = synth_dist(nq, t, terms=terms)
        sh, kl1, kl2 = ge3_share(p, nq)
        shares.append(sh)
        print("  t=%.2f -> ≥3차 share=%.4f (KL1=%.4f KL2=%.4f)" % (t, sh, kl1, kl2))
    assert shares[0] < 0.02, "t=0 인데 ≥3차 share 큼: %.4f" % shares[0]
    assert shares[-1] > 0.8, "t=1(순수고차)인데 ≥3차 share 작음: %.4f" % shares[-1]
    mono = all(shares[i] <= shares[i + 1] + 0.08 for i in range(len(shares) - 1))
    print("  단조(허용오차)=%s | t0 share=%.4f, t1 share=%.4f" % (mono, shares[0], shares[-1]))
    print("  PASS")
    return True


def main():
    ap = argparse.ArgumentParser(description="고차 비중 조절 합성 분포 생성기. 학습 없음.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    p, _ = synth_dist(args.n, args.beta, seed=args.seed)
    sh, kl1, kl2 = ge3_share(p, args.n)
    print("n=%d beta=%.2f: ≥3차 share=%.4f (KL1=%.4f KL2=%.4f) sum=%.6f"
          % (args.n, args.beta, sh, kl1, kl2, p.sum()))


if __name__ == "__main__":
    main()
