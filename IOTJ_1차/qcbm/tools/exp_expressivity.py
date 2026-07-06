#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
잔여 병목 분리 (iii) — 표현력 진단. P↑(L=6/8/10/12)로 n10 반복 미스매치 모드가 잡히고 바닥 근접하는가.

idblock(Grant V†V, pert0.5)·loss(KL, −Σq·log(p+ε))·n=10·t=0 저차 고정, 오직 L(=P=L·n·3)만 키움.
  축1   L 스윕: 각 L best KL(seeds) → 바닥(1e-3) 근접?  (L6=직전 0.087 재현)
  축1-b gradient: 각 L ⟨Z₀⟩ 분산(idblock) → L↑로 유지되나(BP 재발 통제; 작아지면 표현력/BP 혼재).
  축2   반복모드 추적: 각 L best p 에서 s958/808/1018/517/1 의 |q_s−p_s| (L↑에 잡히나) + 비트 decode.
★ t=0 target은 순수 1·2차 저차분포인데도 미스매치 → 회로 표현 기하 단서. 이 ansatz·완화법·KL 한정. 단정 금지. 합성만. 기존 코드 무수정(import).

사용: python tools/exp_expressivity.py --smoke | --time-only
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

import experiment2 as E                   # noqa: E402  wshape
import exp_strong_bp_mitigation as SBP    # noqa: E402  train_idblock, grad_measure, synth_low, EPS
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "expressivity")
NQ, PERT = 10, 0.5
MODES = [958, 808, 1018, 517, 1]          # 국소최적 진단서 seed 무관 반복된 지배 미스매치 모드
FLOOR = 1e-3


def log(m):
    print(m, flush=True)


def run(Ls, seeds, epochs, R, tag="run"):
    os.makedirs(OUTDIR, exist_ok=True)
    p_true = SBP.synth_low(NQ)
    log("n=%d pert=%.1f loss=KL(EPS=%.0e) t=0 저차 | 바닥≈%.0e | 반복모드 %s" % (NQ, PERT, SBP.EPS, FLOOR, MODES))
    log("반복모드 비트(10b): " + " ".join("s%d=%s(q%.4f,ham%d)" % (s, format(s, "010b"), p_true[s], bin(s).count("1")) for s in MODES))

    axis1, grad, axis2 = {}, {}, {}
    for L in Ls:
        P = int(np.prod(E.wshape(NQ, L)))
        kls = []; best_p = None; t0 = time.time()
        for s in range(seeds):
            p, _ = SBP.train_idblock(p_true, NQ, L, epochs, seed=s, pert=PERT)
            k = OD.kl_div(p_true, p, 1e-12); kls.append(k)
            if best_p is None or k < best_p[0]:
                best_p = (k, p)
        a = np.array(kls)
        axis1["%d" % L] = dict(P=P, best=float(a.min()), median=float(np.median(a)), max=float(a.max()), all=a.tolist())
        # gradient (BP 재발 통제)
        g = SBP.grad_measure(NQ, L, R, pert=PERT)
        grad["%d" % L] = dict(idblock_var=g["idblock_var"], uniform_var=g["uniform_var"])
        # 반복모드 |q-p|
        bp = best_p[1]
        axis2["%d" % L] = {str(s): dict(q=float(p_true[s]), p=float(bp[s]), absdiff=float(abs(p_true[s] - bp[s]))) for s in MODES}
        log("  L=%2d P=%3d | best KL=%.4f (median %.4f) | grad idblock Var=%.3e | modes|q-p|: %s (%.0fs)"
            % (L, P, a.min(), np.median(a), g["idblock_var"],
               " ".join("s%d=%.3f" % (s, axis2["%d" % L][str(s)]["absdiff"]) for s in MODES), time.time() - t0))

    Lsi = [int(x) for x in axis1]
    sc = dict(repro_L6=axis1.get("6", {}).get("best"),
              P_formula_ok=all(axis1["%d" % L]["P"] == L * NQ * 3 for L in Lsi),
              best_decreasing=bool(all(axis1["%d" % Lsi[i]]["best"] >= axis1["%d" % Lsi[i + 1]]["best"] - 1e-4
                                       for i in range(len(Lsi) - 1)) if len(Lsi) > 1 else True))
    payload = dict(
        note=("(iii) 표현력 진단. idblock pert0.5·KL·n10·t=0 고정, L(=P=L·n·3)만 키움. 바닥≈1e-3. "
              "L↑ best↓+반복모드 잡힘+grad 유지→표현력부족; grad 저하→BP재발 혼재. 이 ansatz·완화법·KL 한정. 단정 금지."),
        config=dict(nq=NQ, pert=PERT, EPS=SBP.EPS, Ls=Ls, seeds=seeds, epochs=epochs, R=R, modes=MODES, floor=FLOOR),
        axis1=axis1, grad=grad, axis2=axis2, self_check=sc)
    outjson = os.path.join(OUTDIR, "expressivity%s.json" % ("" if tag == "run" else "_" + tag))
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s | self-check L6=%.4f(직전0.087) P공식=%s best단조=%s"
        % (outjson, sc["repro_L6"] or -1, sc["P_formula_ok"], sc["best_decreasing"]))
    try:
        plot(axis1, grad, axis2, tag)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(axis1, grad, axis2, tag):
    Ls = sorted(int(x) for x in axis1)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    Ps = [axis1["%d" % L]["P"] for L in Ls]
    ax[0].plot(Ps, [axis1["%d" % L]["best"] for L in Ls], "o-", color="#7B2FBE", label="QCBM best KL")
    ax[0].axhline(FLOOR, color="red", ls="--", label="floor ~1e-3")
    ax[0].set_xlabel("P (=L·n·3)"); ax[0].set_ylabel("best KL"); ax[0].set_yscale("log")
    ax[0].set_title("축1 P↑ vs best KL (바닥 근접?)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[1].semilogy(Ls, [grad["%d" % L]["idblock_var"] for L in Ls], "*-", color="#3B6EA5")
    ax[1].set_xlabel("L"); ax[1].set_ylabel("⟨Z₀⟩ grad Var (idblock)"); ax[1].set_title("축1-b gradient (BP 재발?)"); ax[1].grid(alpha=.3)
    for s in MODES:
        ax[2].plot(Ls, [axis2["%d" % L][str(s)]["absdiff"] for L in Ls], "o-", label="s%d" % s)
    ax[2].set_xlabel("L"); ax[2].set_ylabel("|q_s - p_s|"); ax[2].set_title("축2 반복모드 미스매치 (L↑에 잡히나)"); ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.suptitle("Residual bottleneck (iii): expressivity vs P (idblock·pert0.5·KL fixed, n=10)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "expressivity%s.png" % ("" if tag == "run" else "_" + tag)), dpi=120); plt.close()
    log("[저장] expressivity png")


def time_only(Ls):
    p_true = SBP.synth_low(NQ)
    for L in Ls:
        t0 = time.time(); SBP.train_idblock(p_true, NQ, L, 400, seed=0, pert=PERT)
        dt = time.time() - t0
        log("  L=%d P=%d 400ep 1seed: %.0fs (%.1f분)" % (L, L * NQ * 3, dt, dt / 60))


def main():
    ap = argparse.ArgumentParser(description="표현력 진단(L=P만 키움, idblock·KL 고정). 합성. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--time-only", action="store_true")
    ap.add_argument("--Ls", type=str, default="6,8,10,12")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--R", type=int, default=30)
    args = ap.parse_args()
    Ls = [int(x) for x in args.Ls.split(",") if x.strip()]
    if args.time_only:
        time_only([12]); return
    if args.smoke:
        run([6, 8], seeds=1, epochs=100, R=8, tag="smoke"); return
    run(Ls, args.seeds, args.epochs, args.R, tag="run")


if __name__ == "__main__":
    main()
