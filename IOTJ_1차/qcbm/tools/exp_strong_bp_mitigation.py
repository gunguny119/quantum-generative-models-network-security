#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
강한 BP 완화 — identity-block(Grant V†V) + layerwise vs uniform/near-identity. loss=KL 유지.

직전: near-identity(소각) 부분완화(n10 KL 0.643→0.391, 바닥 미도달). 이번: 더 강한 loss-독립 완화로 n10 바닥 도달 여부.
방법(같은 P=L·n·3, 같은 총 epoch, 같은 lr, seeds best):
  M1 uniform      : full SEL, uniform[0,2π]        (baseline, ~0.643 재현)
  M2 near-identity: full SEL, N(0,0.1)             (~0.391 재현)
  M3 identity-block: Grant V†V = SEL(wA)+adjoint(SEL)(wB), 각 L/2층, wB=wA+N(0,pert=0.1)
                     ※ exact(wB=wA)는 U=I=확률의 임계점→grad=0(함정). pert 필수. pert0.1서 grad가 uniform의 ~6배(측정).
  M4 layerwise    : full SEL, L=2→4→6 warm-start(새 층 소각 append), 총 epoch 분할.
+ gradient 재측정: uniform vs identity-block(pert) 의 ⟨Z₀⟩ grad 분산 → 완화가 grad 보존/증대하는지.

★ 이 ansatz·완화법 한정, 우위와 별개, loss=KL 유지, lr 동일(loss-종속 미세튜닝 배제). 합성만. 기존 코드 무수정(import).
사용: python tools/exp_strong_bp_mitigation.py --smoke
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

import pennylane as qml                   # noqa: E402
from pennylane import numpy as pnp        # noqa: E402
import experiment2 as E                   # noqa: E402  make_circuit, wshape
import synth_highorder_distribution as SY  # noqa: E402  make_terms, synth_dist
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "strong_bp_mitigation")
EPS = 1e-10
EPS_KL = 1e-12
PERT = 0.1
LR = 0.1
MLP_REF = 0.001


def log(m):
    print(m, flush=True)


def synth_low(nq):
    terms = SY.make_terms(nq, seed=0, high_mode="disjoint")
    p, _ = SY.synth_dist(nq, 0.0, terms=terms)
    return np.asarray(p, float)


def adam_step(w, g, m, v, ep, lr=LR):
    b1, b2, ae = 0.9, 0.999, 1e-8
    m = b1 * m + (1 - b1) * g
    v = b2 * v + (1 - b2) * (g * g)
    mh = m / (1 - b1 ** (ep + 1)); vh = v / (1 - b2 ** (ep + 1))
    w = w - lr * mh / (np.sqrt(vh) + ae)
    return w, m, v


# ---------------------------------------------------------------------------
# 표준 SEL 학습 (M1/M2/M4) — 초기 가중치 주입, 최종 가중치 반환
# ---------------------------------------------------------------------------
def train_sel(q, nq, layers, epochs, w0, lr=LR):
    circuit = E.make_circuit(nq, layers)
    shp = E.wshape(nq, layers)
    qt = pnp.array(np.asarray(q, float))
    w = pnp.array(np.asarray(w0, float), requires_grad=True)

    def cost(ww):
        p = circuit(ww)
        return -pnp.sum(qt * pnp.log(p + EPS))
    gf = qml.grad(cost)
    m = np.zeros(shp); v = np.zeros(shp); last = None
    for ep in range(epochs):
        g = np.array(gf(w), float); last = float(cost(w))
        wn, m, v = adam_step(np.array(w), g, m, v, ep, lr)
        w = pnp.array(wn, requires_grad=True)
    p = np.array(circuit(w), float); p = np.clip(p, 0, None); p /= p.sum()
    return p, last, np.array(w)


# ---------------------------------------------------------------------------
# identity-block (Grant V†V) 학습 (M3)
# ---------------------------------------------------------------------------
def make_idb(nq):
    dev = qml.device("lightning.qubit", wires=nq)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circ(wA, wB):
        qml.StronglyEntanglingLayers(wA, wires=range(nq))
        qml.adjoint(qml.StronglyEntanglingLayers)(wB, wires=range(nq))
        return qml.probs(wires=range(nq))
    return circ


def train_idblock(q, nq, layers, epochs, seed, lr=LR, pert=PERT):
    half = layers // 2                         # 총 params = 2·half·n·3 = layers·n·3 (동일 예산)
    shp = (half, nq, 3)
    circ = make_idb(nq)
    rng = np.random.default_rng(2000 + seed)
    wA0 = rng.uniform(0, 2 * np.pi, shp)
    wB0 = wA0 + rng.normal(0, pert, shp)       # wB≈wA → U≈I(near-δ₀). pert=0 이면 임계점(grad0).
    wA = pnp.array(wA0, requires_grad=True); wB = pnp.array(wB0, requires_grad=True)
    qt = pnp.array(np.asarray(q, float))

    def cost(a, b):
        p = circ(a, b)
        return -pnp.sum(qt * pnp.log(p + EPS))
    gf = qml.grad(cost)
    mA = np.zeros(shp); vA = np.zeros(shp); mB = np.zeros(shp); vB = np.zeros(shp); last = None
    for ep in range(epochs):
        gA, gB = gf(wA, wB); gA = np.array(gA, float); gB = np.array(gB, float); last = float(cost(wA, wB))
        wAn, mA, vA = adam_step(np.array(wA), gA, mA, vA, ep, lr)
        wBn, mB, vB = adam_step(np.array(wB), gB, mB, vB, ep, lr)
        wA = pnp.array(wAn, requires_grad=True); wB = pnp.array(wBn, requires_grad=True)
    p = np.array(circ(wA, wB), float); p = np.clip(p, 0, None); p /= p.sum()
    return p, last


# ---------------------------------------------------------------------------
# layerwise warm-start (M4)
# ---------------------------------------------------------------------------
def train_layerwise(q, nq, L_final, epochs_total, seed, lr=LR, stages=(2, 4, 6)):
    stages = [s for s in stages if s <= L_final]
    if stages[-1] != L_final:
        stages.append(L_final)
    ep_each = max(1, epochs_total // len(stages))
    rng = np.random.default_rng(2000 + seed)
    w = None; last = None
    for Ls in stages:
        shp = (Ls, nq, 3)
        if w is None:
            w0 = rng.normal(0, 0.1, shp)                    # near-identity 시작
        else:
            w0 = np.zeros(shp); prevL = w.shape[0]
            w0[:prevL] = w                                  # warm-start(학습된 층 복사)
            w0[prevL:] = rng.normal(0, 0.05, (Ls - prevL, nq, 3))  # 새 층 소각 append
        p, last, w = train_sel(q, nq, Ls, ep_each, w0, lr)
    return p, last


# ---------------------------------------------------------------------------
# gradient 재측정 (⟨Z₀⟩, 중간층 RY) — uniform vs identity-block(pert)
# ---------------------------------------------------------------------------
def z0_sel(nq, L):
    dev = qml.device("lightning.qubit", wires=nq)

    @qml.qnode(dev)
    def c(w):
        qml.StronglyEntanglingLayers(w, wires=range(nq)); return qml.expval(qml.PauliZ(0))
    return c


def z0_idb(nq, half):
    dev = qml.device("lightning.qubit", wires=nq)

    @qml.qnode(dev)
    def c(wA, wB):
        qml.StronglyEntanglingLayers(wA, wires=range(nq))
        qml.adjoint(qml.StronglyEntanglingLayers)(wB, wires=range(nq))
        return qml.expval(qml.PauliZ(0))
    return c


def grad_measure(nq, L, R, pert=PERT):
    gu = qml.grad(z0_sel(nq, L)); pm = (L // 2, 0, 1); vu = []
    for r in range(R):
        rng = np.random.default_rng(500 + r); w = pnp.array(rng.uniform(0, 2 * np.pi, (L, nq, 3)), requires_grad=True)
        vu.append(float(np.array(gu(w))[pm]))
    half = L // 2; gi = qml.grad(z0_idb(nq, half)); pmh = (half // 2, 0, 1); vi = []
    for r in range(R):
        rng = np.random.default_rng(500 + r); wA = rng.uniform(0, 2 * np.pi, (half, nq, 3)); wB = wA + rng.normal(0, pert, (half, nq, 3))
        gA, _ = gi(pnp.array(wA, requires_grad=True), pnp.array(wB, requires_grad=True)); vi.append(float(np.array(gA)[pmh]))
    return dict(uniform_var=float(np.var(vu)), uniform_meanabs=float(np.mean(np.abs(vu))),
                idblock_var=float(np.var(vi)), idblock_meanabs=float(np.mean(np.abs(vi))), R=R)


# ---------------------------------------------------------------------------
def run_methods(nq, L, epochs, seeds):
    p_true = synth_low(nq)
    shp = E.wshape(nq, L)
    res = {}
    for name in ["uniform", "near_identity", "identity_block", "layerwise"]:
        kls = []; t0 = time.time()
        for s in range(seeds):
            rng = np.random.default_rng(2000 + s)
            if name == "uniform":
                p, _ = train_sel(p_true, nq, L, epochs, rng.uniform(0, 2 * np.pi, shp))[:2]
            elif name == "near_identity":
                p, _ = train_sel(p_true, nq, L, epochs, rng.normal(0, 0.1, shp))[:2]
            elif name == "identity_block":
                p, _ = train_idblock(p_true, nq, L, epochs, seed=s)
            else:
                p, _ = train_layerwise(p_true, nq, L, epochs, seed=s)
            kls.append(OD.kl_div(p_true, p, EPS_KL))
        a = np.array(kls)
        res[name] = dict(best=float(a.min()), mean=float(a.mean()), spread=float(a.max() - a.min()), all=a.tolist())
        log("  n=%d %-15s best=%.4f mean=%.4f spread=%.4f (%.0fs)"
            % (nq, name, res[name]["best"], res[name]["mean"], res[name]["spread"], time.time() - t0))
    return res


def run(ns, L, epochs, seeds, R, tag="run"):
    os.makedirs(OUTDIR, exist_ok=True)
    results = {}; grads = {}
    for nq in ns:
        log("\n=== n=%d (L=%d, epochs=%d, seeds=%d, lr=%.2f) 4 방법 ===" % (nq, L, epochs, seeds, LR))
        results[str(nq)] = run_methods(nq, L, epochs, seeds)
        g = grad_measure(nq, L, R)
        grads[str(nq)] = g
        log("  [grad] uniform Var=%.3e | identity-block Var=%.3e (%.1fx)"
            % (g["uniform_var"], g["idblock_var"], g["idblock_var"] / max(g["uniform_var"], 1e-30)))
    sc = dict(
        repro_uniform_n10=results.get("10", {}).get("uniform", {}).get("best"),
        repro_nearid_n10=results.get("10", {}).get("near_identity", {}).get("best"),
        floor_ref=MLP_REF,
    )
    payload = dict(
        note=("identity-block=Grant V†V(qml.adjoint), wB=wA+N(0,%.2f)(exact는 확률 임계점→grad0). layerwise=L2→4→6 warm-start. "
              "loss=KL, lr=%.2f 동일(loss-종속 미세튜닝 배제). 바닥=MLP~%.3f. 이 ansatz·완화법 한정, 우위와 별개. 단정 금지." % (PERT, LR, MLP_REF)),
        config=dict(ns=ns, L=L, epochs=epochs, seeds=seeds, lr=LR, pert=PERT, R=R),
        results=results, grads=grads, self_check=sc)
    outjson = os.path.join(OUTDIR, "strong_bp%s.json" % ("" if tag == "run" else "_" + tag))
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % outjson)
    try:
        plot(results, grads, ns, tag)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(results, grads, ns, tag):
    methods = ["uniform", "near_identity", "identity_block", "layerwise"]
    cols = ["#9AA0A6", "#4C9F70", "#7B2FBE", "#E2A13B"]
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    x = np.arange(len([n for n in ns if str(n) in results]))
    nsr = [n for n in ns if str(n) in results]
    w = 0.2
    for i, (mth, c) in enumerate(zip(methods, cols)):
        ax[0].bar(x + (i - 1.5) * w, [results[str(n)][mth]["best"] for n in nsr], w, color=c, label=mth)
    ax[0].axhline(MLP_REF, color="red", ls="--", label="floor(MLP~%.3f)" % MLP_REF)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["n=%d" % n for n in nsr]); ax[0].set_yscale("log")
    ax[0].set_ylabel("best KL(p_true||model)"); ax[0].set_title("Strong BP mitigation (loss=KL)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    gn = [n for n in ns if str(n) in grads]
    ax[1].semilogy(gn, [grads[str(n)]["uniform_var"] for n in gn], "o-", color="#9AA0A6", label="uniform")
    ax[1].semilogy(gn, [grads[str(n)]["idblock_var"] for n in gn], "*-", color="#7B2FBE", label="identity-block(pert)")
    ax[1].set_xlabel("n"); ax[1].set_ylabel("Var(∂⟨Z₀⟩/∂θ)"); ax[1].set_title("gradient 보존 (완화 vs uniform)"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    plt.suptitle("Strong barren-plateau mitigation: identity-block(Grant)+layerwise — reach floor? (this ansatz only)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "strong_bp%s.png" % ("" if tag == "run" else "_" + tag)), dpi=120); plt.close()
    log("[저장] strong_bp png")


def main():
    ap = argparse.ArgumentParser(description="강한 BP 완화(identity-block+layerwise). KL 유지. 합성. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ns", type=str, default="10,8")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--R", type=int, default=40)
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    if args.smoke:
        run([8], 6, 120, 2, 12, tag="smoke"); return
    run(ns, args.L, args.epochs, args.seeds, args.R, tag="run")


if __name__ == "__main__":
    main()
