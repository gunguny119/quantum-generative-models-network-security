#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
바렌 플래토(BP) 직접 진단(A) + near-identity 초기화 완화 시험(B).

직전 n↑ 스케일링서 QCBM 격차가 n↑일수록 확대 → BP류 trainability 정황. 이 작업:
(A) n별 gradient 분산을 랜덤 init 다수에서 직접 재 BP(지수 감소) 확정.
(B) near-identity(소각) 초기화로 큰 n QCBM 학습이 개선되는지 시험.

ansatz=StronglyEntanglingLayers, cost=forward KL(−Σ q·log p), params=L·n·3. 기존 qcbm_kl_variant 와 동일 cost/Adam,
init 만 바꾼 변형을 여기서 작성(experiment2/qcbm_kl_variant 무수정 import). 합성만.
★ 해석은 "이 ansatz·옵티마이저·초기화 한정", 우위와 별개, 단정 금지.

사용: python tools/exp_barren_plateau.py --smoke
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

OUTDIR = os.path.join("results2", "barren_plateau")
EPS = 1e-10
EPS_KL = 1e-12


def log(m):
    print(m, flush=True)


def synth_low(nq):
    terms = SY.make_terms(nq, seed=0, high_mode="disjoint")
    p_true, _ = SY.synth_dist(nq, 0.0, terms=terms)
    return np.asarray(p_true, float)


# ---------------------------------------------------------------------------
# (A) gradient 진단
# ---------------------------------------------------------------------------
def z0_qnode(nq, layers):
    dev = qml.device("lightning.qubit", wires=nq)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circ(weights):
        qml.StronglyEntanglingLayers(weights, wires=range(nq))
        return qml.expval(qml.PauliZ(0))
    return circ


def grad_diag(nq, layers, R, seed0=0):
    """랜덤 init R개에서 전체 gradient 측정 → BP 지표.
    probe 파라미터 = 중간 레이어 qubit0 의 RY 각도 (인덱스 (L//2,0,1)).
      ※ (0,0,0)=|0⟩에 처음 걸리는 RZ(위상)각이라 확률 gradient가 항등적 0 → 부적합. 중간층 RY 사용.
    var_*  = 그 probe 성분의 init간 분산(교과서 단일파라미터 BP 척도).
    meanvar_* = 모든 파라미터의 (init간 분산)을 평균(robust 집계)."""
    p_true = synth_low(nq)
    qt = pnp.array(p_true)
    circuit = E.make_circuit(nq, layers)
    shp = E.wshape(nq, layers)
    pm = (layers // 2, 0, 1)                       # 중간층, qubit0, RY 각

    def cost_kl(ww):
        p = circuit(ww)
        return -pnp.sum(qt * pnp.log(p + EPS))

    z0 = z0_qnode(nq, layers)
    gk = qml.grad(cost_kl)
    gz = qml.grad(z0)

    GK, GZ = [], []
    for r in range(R):
        rng = np.random.default_rng(10000 + seed0 + r)
        w = pnp.array(rng.uniform(0, 2 * np.pi, size=shp), requires_grad=True)
        GK.append(np.array(gk(w), dtype=float))
        GZ.append(np.array(gz(w), dtype=float))
    GK = np.stack(GK); GZ = np.stack(GZ)           # (R, *shp)
    pk = GK[:, pm[0], pm[1], pm[2]]; pz = GZ[:, pm[0], pm[1], pm[2]]
    return dict(L=layers, P=int(np.prod(shp)), R=R, probe=str(pm),
                var_kl=float(pk.var()), mean_abs_kl=float(np.abs(pk).mean()), mean_kl=float(pk.mean()),
                meanvar_kl=float(GK.var(axis=0).mean()),
                var_z0=float(pz.var()), mean_abs_z0=float(np.abs(pz).mean()), mean_z0=float(pz.mean()),
                meanvar_z0=float(GZ.var(axis=0).mean()))


# ---------------------------------------------------------------------------
# (B) 초기화 변형 학습 (uniform vs near-identity small-angle)
# ---------------------------------------------------------------------------
def train_kl_init(q_target, nq, layers, epochs, init_mode, seed, lr=0.1, sigma=0.1):
    circuit = E.make_circuit(nq, layers)
    shp = E.wshape(nq, layers)
    rng = np.random.default_rng(2000 + seed)
    if init_mode == "uniform":
        w0 = rng.uniform(0, 2 * np.pi, size=shp)          # = train_kl_one baseline
    elif init_mode == "small":
        w0 = rng.normal(0, sigma, size=shp)               # near-identity(소각)
    else:
        raise ValueError(init_mode)
    w = pnp.array(w0, requires_grad=True)
    qt = pnp.array(np.asarray(q_target, float))

    def cost(ww):
        p = circuit(ww)
        return -pnp.sum(qt * pnp.log(p + EPS))

    gfun = qml.grad(cost)
    b1, b2, ae = 0.9, 0.999, 1e-8
    m = np.zeros(shp); v = np.zeros(shp)
    last = None
    for ep in range(epochs):
        g = np.array(gfun(w), dtype=float)
        last = float(cost(w))
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** (ep + 1)); vhat = v / (1 - b2 ** (ep + 1))
        w = pnp.array(np.array(w) - lr * mhat / (np.sqrt(vhat) + ae), requires_grad=True)
    p = np.array(circuit(w), float); p = np.clip(p, 0, None); p /= p.sum()
    return p, last


def mitig(nq, layers, epochs, seeds):
    p_true = synth_low(nq)
    out = {}
    for mode in ["uniform", "small"]:
        kls = []
        for s in range(seeds):
            p, ce = train_kl_init(p_true, nq, layers, epochs, mode, seed=s)
            kls.append(OD.kl_div(p_true, p, EPS_KL))
        a = np.array(kls)
        out[mode] = dict(best=float(a.min()), mean=float(a.mean()), spread=float(a.max() - a.min()), all=a.tolist())
    return out


# ---------------------------------------------------------------------------
def run(ns_A, ns_B, L, R, epochs, seeds, do_B=True, tag="run"):
    os.makedirs(OUTDIR, exist_ok=True)
    A = {}
    log("=== (A) gradient 진단 (L=%d, R=%d, θ₀ 분산) ===" % (L, R))
    for nq in ns_A:
        t0 = time.time()
        d = grad_diag(nq, L, R)
        A[str(nq)] = d
        log("  n=%2d P=%3d | Var(KL grad)=%.3e mean|KL|=%.3e | Var(Z0 grad)=%.3e mean|Z0|=%.3e (%.0fs)"
            % (nq, d["P"], d["var_kl"], d["mean_abs_kl"], d["var_z0"], d["mean_abs_z0"], time.time() - t0))
    # 깊이 효과(1점): n=8 L=3 비교(있으면)
    depth = None
    if 8 in ns_A:
        d3 = grad_diag(8, 3, R)
        depth = dict(n=8, L3=d3["var_z0"], L_main=A["8"]["var_z0"])
        log("  [깊이] n=8 Var(Z0): L=3 %.3e vs L=%d %.3e" % (d3["var_z0"], L, A["8"]["var_z0"]))

    B = {}
    if do_B:
        log("\n=== (B) near-identity 완화 (L=%d, epochs=%d, seeds=%d) ===" % (L, epochs, seeds))
        for nq in ns_B:
            t0 = time.time()
            r = mitig(nq, L, epochs, seeds)
            B[str(nq)] = r
            log("  n=%2d | uniform best=%.4f(spread %.4f) | near-identity best=%.4f(spread %.4f) (%.0fs)"
                % (nq, r["uniform"]["best"], r["uniform"]["spread"], r["small"]["best"], r["small"]["spread"], time.time() - t0))

    # self-check
    sc = dict(
        randinit_mean_kl_small=bool(all(abs(A[n]["mean_kl"]) < 5 * A[n]["mean_abs_kl"] + 1e-9 for n in A)),
        baseline_n10_repro=(B.get("10", {}).get("uniform", {}).get("best") if do_B else None),
        var_z0_decreasing=bool(all(A[str(ns_A[i])]["var_z0"] >= A[str(ns_A[i + 1])]["var_z0"]
                                   for i in range(len(ns_A) - 1)) if len(ns_A) > 1 else True),
    )
    payload = dict(
        note=("(A) 랜덤 init R개에서 ∂C/∂θ₀ 분산 — KL-cost & canonical ⟨Z₀⟩. n↑서 지수감소면 BP. "
              "(B) uniform vs near-identity(소각 N(0,0.1)) 초기화 학습 KL. ansatz·옵티마이저·초기화 한정. 우위와 별개. 단정 금지."),
        config=dict(L=L, R=R, epochs=epochs, seeds=seeds, ns_A=ns_A, ns_B=ns_B),
        A=A, depth=depth, B=B, self_check=sc)
    with open(os.path.join(OUTDIR, "barren_plateau%s.json" % ("" if tag == "run" else "_" + tag)), "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] barren_plateau.json | self-check: randMean작음=%s, Var(Z0)감소=%s, baseline_n10=%s"
        % (sc["randinit_mean_kl_small"], sc["var_z0_decreasing"], sc["baseline_n10_repro"]))
    try:
        plot(A, B, ns_A, ns_B, tag)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(A, B, ns_A, ns_B, tag):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ns = [int(n) for n in A]
    ax[0].semilogy(ns, [A[str(n)]["var_kl"] for n in ns], "o-", color="#C1442E", label="Var(∂C_KL/∂θ₀)")
    ax[0].semilogy(ns, [A[str(n)]["var_z0"] for n in ns], "s-", color="#3B6EA5", label="Var(∂⟨Z₀⟩/∂θ₀) (canonical)")
    ax[0].set_xlabel("n (qubits)"); ax[0].set_ylabel("Var(gradient) [log]")
    ax[0].set_title("(A) Barren plateau: gradient variance vs n\n(직선 하락=지수감소=BP)"); ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    if B:
        nsB = [int(n) for n in B]; x = np.arange(len(nsB)); w = 0.35
        ax[1].bar(x - w / 2, [B[str(n)]["uniform"]["best"] for n in nsB], w, color="#9AA0A6", label="uniform init")
        ax[1].bar(x + w / 2, [B[str(n)]["small"]["best"] for n in nsB], w, color="#4C9F70", label="near-identity init")
        ax[1].set_xticks(x); ax[1].set_xticklabels(["n=%d" % n for n in nsB]); ax[1].set_yscale("log")
        ax[1].set_ylabel("best KL(p_true||model)"); ax[1].set_title("(B) near-identity 완화"); ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("QCBM barren-plateau diagnosis (A) + near-identity mitigation (B) — this ansatz/optimizer/init only")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "barren_plateau%s.png" % ("" if tag == "run" else "_" + tag)), dpi=120); plt.close()
    log("[저장] barren_plateau png")


def main():
    ap = argparse.ArgumentParser(description="바렌 플래토 진단+완화. 합성. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--ns-A", type=str, default="6,8,10")
    ap.add_argument("--ns-B", type=str, default="8,10")
    ap.add_argument("--L", type=int, default=6)
    ap.add_argument("--R", type=int, default=50)
    ap.add_argument("--epochs", type=int, default=500)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--no-B", action="store_true")
    args = ap.parse_args()
    ns_A = [int(x) for x in args.ns_A.split(",") if x.strip()]
    ns_B = [int(x) for x in args.ns_B.split(",") if x.strip()]
    if args.smoke:
        ns_A = [6, 8]; ns_B = [8]; args.R = 8; args.epochs = 120; args.seeds = 2
        run(ns_A, ns_B, args.L, args.R, args.epochs, args.seeds, do_B=not args.no_B, tag="smoke"); return
    run(ns_A, ns_B, args.L, args.R, args.epochs, args.seeds, do_B=not args.no_B, tag="run")


if __name__ == "__main__":
    main()
