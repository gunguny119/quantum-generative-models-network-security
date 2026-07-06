#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
잔여 병목 분리 (ii) — 국소최적 진단. n10 identity-block 잔여 정체(0.087, 소수 모드 미스매치)가 국소최적인가.

P(=180, L6)·loss(KL)·pert(0.5) 고정, restart·epoch(찾는 능력)만 키운다.
  축1 restart↑: seed 다수 @ 400ep → best/min/median/max. 직전 0.087 대비 내려가나.
  축2 epoch↑  : 대표 seed @ 1200ep, ckpt{400,800,1200} KL 곡선 → 평평(수렴)인가 계속↓인가.
  축3 지배모드 : seed별 최종 p 상위5 KL기여 상태 → seed 간 겹침율(Jaccard). 매번 다르면 국소최적, 겹치면 구조적.
국소최적 지지 = restart↑로 best↓ + 모드 seed마다 다름. 기각 = 정체 + 모드 일관 → (iii)표현력.
이 ansatz·완화법·KL·P=180 한정. 단정 금지. 합성만. 기존 코드 무수정(import).

사용: python tools/exp_local_minima.py --smoke | --time-only
"""
import os
import sys
import json
import time
import argparse
import itertools

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

from pennylane import numpy as pnp        # noqa: E402
import exp_strong_bp_mitigation as SBP    # noqa: E402  make_idb, adam_step, synth_low, EPS
import exp_kl_sensitivity as KLS          # noqa: E402  kl_contributions
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "local_minima")
EPS = SBP.EPS
NQ, L, PERT, LR = 10, 6, 0.5, 0.1


def log(m):
    print(m, flush=True)


def train_idblock_ckpt(p_true, nq, layers, epochs, seed, pert=PERT, ckpts=None):
    """identity-block(Grant V†V) 학습. ckpt epoch 에서 KL(p_true||p) 기록. (SBP.make_idb/adam_step 재사용, cost 동일)"""
    half = layers // 2; shp = (half, nq, 3)
    circ = SBP.make_idb(nq)
    rng = np.random.default_rng(2000 + seed)               # SBP.train_idblock 과 동일 seed 규약
    wA = pnp.array(rng.uniform(0, 2 * np.pi, shp), requires_grad=True)
    wB = pnp.array(np.array(wA) + rng.normal(0, pert, shp), requires_grad=True)
    qt = pnp.array(np.asarray(p_true, float))
    import pennylane as qml

    def cost(a, b):
        p = circ(a, b); return -pnp.sum(qt * pnp.log(p + EPS))
    gf = qml.grad(cost)
    mA = np.zeros(shp); vA = np.zeros(shp); mB = np.zeros(shp); vB = np.zeros(shp)
    ckpts = set(ckpts or [])
    curve = {}
    for ep in range(epochs):
        gA, gB = gf(wA, wB)
        wAn, mA, vA = SBP.adam_step(np.array(wA), np.array(gA, float), mA, vA, ep, LR)
        wBn, mB, vB = SBP.adam_step(np.array(wB), np.array(gB, float), mB, vB, ep, LR)
        wA = pnp.array(wAn, requires_grad=True); wB = pnp.array(wBn, requires_grad=True)
        if (ep + 1) in ckpts:
            pp = np.array(circ(wA, wB), float); pp = np.clip(pp, 0, None); pp /= pp.sum()
            curve[ep + 1] = OD.kl_div(p_true, pp, 1e-12)
    p = np.array(circ(wA, wB), float); p = np.clip(p, 0, None); p /= p.sum()
    if epochs in (ckpts or {epochs}):
        curve[epochs] = OD.kl_div(p_true, p, 1e-12)
    return p, OD.kl_div(p_true, p, 1e-12), curve


def top_states(p_true, p, k=5):
    c = KLS.kl_contributions(p_true, p, EPS)
    return [t["state"] for t in c["top_states"][:k]]


def run(S, ep_axis1, ep_axis2_max, seeds_axis2, tag="run"):
    os.makedirs(OUTDIR, exist_ok=True)
    p_true = SBP.synth_low(NQ)
    log("n=%d L=%d P=180 pert=%.1f loss=KL(EPS=%.0e) | 직전 pert0.5 best=0.087" % (NQ, L, PERT, EPS))

    # ── 축1 restart↑ ──
    log("\n=== 축1 restart↑ (%d seed @ %dep) ===" % (S, ep_axis1))
    kls = []; pfin = {}
    for s in range(S):
        t0 = time.time()
        p, k, _ = train_idblock_ckpt(p_true, NQ, L, ep_axis1, seed=s)
        kls.append(k); pfin[s] = p
        log("  seed %2d: KL=%.4f (%.0fs)" % (s, k, time.time() - t0))
    a = np.array(kls)
    axis1 = dict(S=S, epochs=ep_axis1, best=float(a.min()), min=float(a.min()), median=float(np.median(a)),
                 max=float(a.max()), all=a.tolist(), best_seed=int(np.argmin(a)))
    log("  → best=%.4f median=%.4f max=%.4f (직전 0.087 대비 %s)"
        % (axis1["best"], axis1["median"], axis1["max"], "내려감" if axis1["best"] < 0.087 - 1e-4 else "정체/미개선"))

    # ── 축2 epoch↑ 수렴곡선 ──
    log("\n=== 축2 epoch↑ 곡선 (%d seed @ %dep, ckpt) ===" % (seeds_axis2, ep_axis2_max))
    ck = [c for c in [400, 800, ep_axis2_max] if c <= ep_axis2_max]
    axis2 = {}
    for s in range(seeds_axis2):
        _, _, curve = train_idblock_ckpt(p_true, NQ, L, ep_axis2_max, seed=s, ckpts=ck)
        axis2["seed%d" % s] = {str(c): curve.get(c) for c in ck}
        log("  seed %d curve: %s" % (s, {c: round(curve.get(c, float('nan')), 4) for c in ck}))

    # ── 축3 지배모드 seed 일관성 ──
    log("\n=== 축3 지배모드 seed 일관성 (상위5) ===")
    tops = {s: set(top_states(p_true, pfin[s], 5)) for s in pfin}
    pairs = list(itertools.combinations(pfin.keys(), 2))
    jac = [len(tops[a] & tops[b]) / len(tops[a] | tops[b]) for a, b in pairs] if pairs else [0]
    from collections import Counter
    freq = Counter()
    for s in tops:
        freq.update(tops[s])
    recur = [(int(st), c) for st, c in freq.most_common(8)]
    axis3 = dict(mean_jaccard=float(np.mean(jac)), per_seed_top5={str(s): sorted(int(x) for x in tops[s]) for s in tops},
                 recurring=recur)
    log("  평균 Jaccard(상위5 겹침율)=%.3f | 반복 상태(빈도): %s" % (axis3["mean_jaccard"], recur[:5]))

    sc = dict(repro_pert05=float(min(kls[0], kls[1])) if S >= 2 else float(kls[0]))
    payload = dict(
        note=("(ii) 국소최적 진단. P=180·loss=KL·pert=0.5 고정, restart/epoch만 키움. 직전 pert0.5 best=0.087. "
              "국소최적 지지=restart로 best↓+모드 seed 다름 / 기각=정체+모드 일관→(iii). 이 ansatz·완화법·KL·P=180 한정. 단정 금지."),
        config=dict(nq=NQ, L=L, P=180, pert=PERT, lr=LR, EPS=EPS, S=S, ep_axis1=ep_axis1, ep_axis2_max=ep_axis2_max),
        axis1=axis1, axis2=axis2, axis3=axis3, self_check=sc)
    outjson = os.path.join(OUTDIR, "local_minima%s.json" % ("" if tag == "run" else "_" + tag))
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s | self-check pert0.5(2seed)=%.4f (직전 0.087)" % (outjson, sc["repro_pert05"]))
    try:
        plot(axis1, axis2, axis3, tag)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(axis1, axis2, axis3, tag):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ax[0].plot(range(len(axis1["all"])), sorted(axis1["all"]), "o-", color="#7B2FBE")
    ax[0].axhline(0.087, color="red", ls="--", label="직전 best 0.087")
    ax[0].set_xlabel("seed rank"); ax[0].set_ylabel("KL"); ax[0].set_title("축1 restart 분포 (%d seed)" % axis1["S"]); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    for k, cv in axis2.items():
        xs = sorted(int(e) for e in cv if cv[e] is not None); ys = [cv[str(e)] for e in xs]
        ax[1].plot(xs, ys, "o-", label=k)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("KL"); ax[1].set_title("축2 epoch↑ 수렴곡선"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[2].bar([str(s) for s, _ in axis3["recurring"]], [c for _, c in axis3["recurring"]], color="#3B6EA5")
    ax[2].set_xlabel("state"); ax[2].set_ylabel("seed 등장 빈도"); ax[2].set_title("축3 반복 지배모드 (Jaccard=%.2f)" % axis3["mean_jaccard"]); ax[2].grid(alpha=.3, axis="y")
    plt.suptitle("Residual bottleneck (ii): local-minima diagnosis (P·loss·pert fixed; restart/epoch only)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "local_minima%s.png" % ("" if tag == "run" else "_" + tag)), dpi=120); plt.close()
    log("[저장] local_minima png")


def time_only():
    p_true = SBP.synth_low(NQ)
    for ep in [400, 800]:
        t0 = time.time(); train_idblock_ckpt(p_true, NQ, L, ep, seed=0)
        dt = time.time() - t0
        log("  n=%d pert%.1f %dep 1seed: %.0fs (%.1f분)" % (NQ, PERT, ep, dt, dt / 60))


def main():
    ap = argparse.ArgumentParser(description="국소최적 진단(P·loss 고정, restart/epoch). 합성. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--S", type=int, default=12)
    ap.add_argument("--ep-axis1", type=int, default=400)
    ap.add_argument("--ep-axis2", type=int, default=1200)
    ap.add_argument("--seeds-axis2", type=int, default=2)
    args = ap.parse_args()
    if args.time_only:
        time_only(); return
    if args.smoke:
        run(2, 100, 200, 1, tag="smoke"); return
    run(args.S, args.ep_axis1, args.ep_axis2, args.seeds_axis2, tag="run")


if __name__ == "__main__":
    main()
