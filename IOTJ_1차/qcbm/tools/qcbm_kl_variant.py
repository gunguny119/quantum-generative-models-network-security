#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCBM loss 변형: MMD² 대신 forward KL(cross-entropy) 로 학습.

기존 experiment2.train_one 의 cost(MMD²)는 함수 내부 클로저라 "cost만 교체"가 불가 →
여기서 Adam 학습 루프를 재구현하되 ansatz(experiment2.make_circuit/wshape)·파라미터수(L·nq·3)·
Adam·qml.probs(exact)·타깃 분포 직접 입력은 그대로 두고 **cost만 KL** 로 바꾼다. experiment2 무수정.

cost(w) = - sum_s q_target[s] * log(circuit(w)[s] + eps)
        = forward KL(q_target || p_model) - H(q_target)   (상수차)
→ 최소화하면 forward KL 최소화(=MLE/cross-entropy).

사용: python tools/qcbm_kl_variant.py --selftest
"""
import os
import sys
import time
import argparse

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import pennylane as qml                    # noqa: E402
from pennylane import numpy as pnp         # noqa: E402
import experiment2 as E                    # noqa: E402  make_circuit, wshape (ansatz 재사용, 무수정)

EPS_LOG = 1e-10


def train_kl_one(q_target, nq, layers, epochs, lr=0.1, seed=0, log_every=0, eps=EPS_LOG):
    """KL(cross-entropy) loss 로 QCBM 1회 학습. train_one(experiment2) 의 Adam 구조 복제, cost만 KL."""
    circuit = E.make_circuit(nq, layers)
    shp = E.wshape(nq, layers)
    rng = np.random.default_rng(2000 + seed)        # train_one 과 동일 seed 규약
    w = pnp.array(rng.uniform(0, 2 * np.pi, size=shp), requires_grad=True)
    qt = pnp.array(np.asarray(q_target, dtype=float))

    def cost(ww):
        p = circuit(ww)
        return -pnp.sum(qt * pnp.log(p + eps))      # forward KL(q||p) + const

    gfun = qml.grad(cost)
    b1, b2, adam_eps = 0.9, 0.999, 1e-8
    m = np.zeros(shp); v = np.zeros(shp)
    loss_hist = []
    for ep in range(epochs):
        g = np.array(gfun(w), dtype=float)
        loss = float(cost(w))
        loss_hist.append(loss)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** (ep + 1))
        vhat = v / (1 - b2 ** (ep + 1))
        w = pnp.array(np.array(w) - lr * mhat / (np.sqrt(vhat) + adam_eps), requires_grad=True)
        if log_every and (ep % log_every == 0 or ep == epochs - 1):
            print("    [KL seed %d] ep %d/%d  CE=%.5f" % (seed, ep, epochs, loss), flush=True)

    p_final = np.array(circuit(w), dtype=float)
    p_final = np.clip(p_final, 0, None); p_final /= p_final.sum()
    return p_final, float(loss_hist[-1]), loss_hist


def qcbm_kl_fit(q_emp, nq, layers, epochs, restarts, lr=0.1, seed0=0):
    """여러 restart 중 최종 cross-entropy(=KL) 최소 모델 선택. params=L·nq·3."""
    best = None
    for sd in range(restarts):
        p, ce, _ = train_kl_one(q_emp, nq, layers, epochs, lr=lr, seed=seed0 + sd)
        if best is None or ce < best[1]:
            best = (p, ce)
    nparams = int(np.prod(E.wshape(nq, layers)))
    return best[0], nparams, best[1]


def _kl(q, p, eps=1e-12):
    q = np.asarray(q, float); p = np.asarray(p, float)
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / np.clip(p, eps, None)[m])))


def selftest():
    print("=== qcbm_kl self-test (KL 감소 확인) ===")
    nq, layers, epochs = 4, 3, 80
    rng = np.random.default_rng(0)
    q = rng.random(1 << nq); q /= q.sum()
    p, ce, hist = train_kl_one(q, nq, layers, epochs, seed=0)
    kl0 = None
    # 초기 가중치 분포의 KL 재계산용: 첫 epoch CE → KL = CE - H(q)
    Hq = -np.sum(q * np.log(q + 1e-12))
    kl_start = hist[0] - Hq
    kl_end = hist[-1] - Hq
    klf = _kl(q, p)
    print("  CE: %.4f -> %.4f | KL(q||p): start≈%.4f end≈%.4f (최종모델 KL=%.4f)"
          % (hist[0], hist[-1], kl_start, kl_end, klf))
    assert hist[-1] < hist[0] - 1e-3, "CE(=KL) 가 감소하지 않음"
    assert kl_end < kl_start, "KL 감소 안함"
    print("  PASS (KL 단조 감소)")
    return True


def main():
    ap = argparse.ArgumentParser(description="QCBM KL(forward) loss 변형. experiment2 무수정 재사용.")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=80)
    args = ap.parse_args()
    if args.selftest:
        selftest(); return
    rng = np.random.default_rng(0)
    q = rng.random(1 << args.n); q /= q.sum()
    t0 = time.time()
    p, npar, ce = qcbm_kl_fit(q, args.n, args.layers, args.epochs, restarts=1)
    print("n=%d params=%d final CE=%.4f KL=%.4f (%.1fs)"
          % (args.n, npar, ce, _kl(q, p), time.time() - t0))


if __name__ == "__main__":
    main()
