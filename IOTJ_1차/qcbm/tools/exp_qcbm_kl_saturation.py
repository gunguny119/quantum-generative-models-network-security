#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCBM-KL 학습 포화 실험 — epoch/restart(+참고 depth)를 키워 저차 KL 바닥을 찾는다.

직전 QCBM-KL 저차 KL=0.197 이 (a)유한 학습 탓인지 (b)이 ansatz(StronglyEntangling)·예산(P=54)·옵티마이저
표현력 포화인지 판별. 한 점(저차 t=0) 집중 + 참고 L=6 + 고랭크 random 1점. 합성만. 기존 코드 무수정(import).

평가 = KL(p_true || model) (직전 0.197 과 동일 지표). 학습 곡선 = KL(q_emp || model) = CE - H(q_emp).
샘플링 KL 바닥 ≈ #states/(2N) (이 이하론 안 내려감). 단정 금지: "이 ansatz·예산·옵티마이저 포화"로 한정.

사용: python tools/exp_qcbm_kl_saturation.py --smoke
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

import experiment2 as E                   # noqa: E402  empirical_dist, wshape
import synth_highorder_distribution as SY  # noqa: E402  make_terms, synth_dist, ge3_share
import qcbm_kl_variant as KLV             # noqa: E402  train_kl_one
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "qcbm_kl_saturation")
EPS = 1e-12
NQ = 6
N = 80000
SAMPLE_SEED = 12345
MLP_REF, NMF_REF = 0.001, 0.005   # 직전 저차 기준선


def log(m):
    print(m, flush=True)


def entropy(q):
    q = np.asarray(q, float); m = q > 0
    return float(-np.sum(q[m] * np.log(q[m])))


def make_point(t, high_mode):
    terms = SY.make_terms(NQ, seed=0, high_mode=high_mode)
    p_true, _ = SY.synth_dist(NQ, t, terms=terms)
    rng = np.random.default_rng(SAMPLE_SEED)
    samples = rng.choice(1 << NQ, size=N, p=p_true)
    q_emp = E.empirical_dist(samples, 1 << NQ)
    return p_true, q_emp


def train_combo(p_true, q_emp, layers, epochs, restarts):
    """restart 중 best(최소 final CE) 선택. 반환 dict(best_KL_ptrue, restart_KLs, curve, params)."""
    Hq = entropy(q_emp)
    best = None
    restart_KLs = []
    for sd in range(restarts):
        p, ce, hist = KLV.train_kl_one(q_emp, NQ, layers, epochs, seed=sd)
        kl_ptrue = OD.kl_div(p_true, p, EPS)
        restart_KLs.append(kl_ptrue)
        if best is None or ce < best["ce"]:
            best = dict(ce=ce, kl_ptrue=kl_ptrue, hist=hist)
    # 곡선: KL(q_emp||model) per epoch = CE - H(q_emp), 다운샘플
    curve = [max(c - Hq, 0.0) for c in best["hist"]]
    step = max(1, len(curve) // 60)
    curve_ds = [(i, curve[i]) for i in range(0, len(curve), step)]
    params = int(np.prod(E.wshape(NQ, layers)))
    return dict(layers=layers, epochs=epochs, restarts=restarts, params=params,
                best_KL_ptrue=best["kl_ptrue"], restart_KLs=restart_KLs,
                train_KL_final=curve[-1], curve=curve_ds)


def run(smoke=False):
    os.makedirs(OUTDIR, exist_ok=True)
    if smoke:
        ladder = [(50, 1), (150, 2)]; lref = (150, 2); hr = (150, 2)
    else:
        ladder = [(250, 3), (750, 5), (1500, 10)]; lref = (1000, 4); hr = (1200, 8)

    # ---- 저차 t=0 ladder (P=54, L=3), early-stop ----
    p_true, q_emp = make_point(0.0, "disjoint")
    share, _, _ = SY.ge3_share(p_true, NQ)
    samp_floor = OD.kl_div(p_true, q_emp, EPS)
    log("저차 t=0: share=%.3f | 샘플링 KL바닥(p_true‖q_emp)=%.4f | 기준 MLP=%.3f NMF=%.3f" % (share, samp_floor, MLP_REF, NMF_REF))
    ladder_rows = []
    prev = None; saturated_at = None
    for (ep, rs) in ladder:
        t0 = time.time()
        r = train_combo(p_true, q_emp, 3, ep, rs)
        ladder_rows.append(r)
        impr = None if prev is None else (prev - r["best_KL_ptrue"]) / max(prev, 1e-9)
        log("  L=3 ep=%4d rs=%2d params=%d -> best KL(p_true)=%.4f (restart 분포 min=%.4f max=%.4f) train_KL=%.4f %s (%.0fs)"
            % (ep, rs, r["params"], r["best_KL_ptrue"], min(r["restart_KLs"]), max(r["restart_KLs"]),
               r["train_KL_final"], "" if impr is None else "개선=%.1f%%" % (100 * impr), time.time() - t0))
        if impr is not None and impr < 0.05:
            saturated_at = (ep, rs); log("  -> 포화(개선<5%%) at ep=%d rs=%d. ladder 중단." % (ep, rs)); break
        prev = r["best_KL_ptrue"]
    sat_KL = min(r["best_KL_ptrue"] for r in ladder_rows)

    # ---- 참고: L=6 (P=108, 예산↑) ----
    log("참고 L=6(P=108, 예산↑):")
    t0 = time.time()
    lref_row = train_combo(p_true, q_emp, 6, lref[0], lref[1])
    log("  L=6 ep=%d rs=%d params=%d -> best KL(p_true)=%.4f (%.0fs)"
        % (lref[0], lref[1], lref_row["params"], lref_row["best_KL_ptrue"], time.time() - t0))

    # ---- 고랭크 random t=0.6 ----
    log("고랭크 random t=0.6:")
    p_hr, q_hr = make_point(0.6, "random")
    sh_hr, _, _ = SY.ge3_share(p_hr, NQ)
    t0 = time.time()
    hr_row = train_combo(p_hr, q_hr, 3, hr[0], hr[1])
    log("  random t=0.6 share=%.3f L=3 ep=%d rs=%d -> best KL(p_true)=%.4f (%.0fs)"
        % (sh_hr, hr[0], hr[1], hr_row["best_KL_ptrue"], time.time() - t0))

    sc = dict(
        ladder_nonincreasing=bool(all(ladder_rows[i]["best_KL_ptrue"] >= ladder_rows[i + 1]["best_KL_ptrue"] - 1e-6
                                      for i in range(len(ladder_rows) - 1))),
        curves_decreasing=bool(all(r["curve"][0][1] >= r["curve"][-1][1] - 1e-6 for r in ladder_rows)),
    )
    payload = dict(
        note=("QCBM-KL 저차 t=0 학습 포화. 평가 KL(p_true‖model). 샘플링 바닥 %.4f. "
              "포화 KL vs MLP/NMF로 '유한학습 vs 이 ansatz·예산·옵티마이저 포화' 판단. 단정 금지." % samp_floor),
        config=dict(nq=NQ, N=N, sample_seed=SAMPLE_SEED, P_base=54, low_share=share,
                    sampling_floor=samp_floor, MLP_ref=MLP_REF, NMF_ref=NMF_REF),
        low_ladder=ladder_rows, saturated_at=saturated_at, saturation_KL=sat_KL,
        Lref=lref_row, highrank=dict(share=sh_hr, **hr_row),
        saturation=dict(sat_KL=sat_KL, vs_MLP=sat_KL - MLP_REF, vs_NMF=sat_KL - NMF_REF, sampling_floor=samp_floor),
        self_check=sc)
    with open(os.path.join(OUTDIR, "saturation.json"), "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("[저장] %s/saturation.json | 포화 KL=%.4f (MLP %.3f, NMF %.3f, 샘플바닥 %.4f)"
        % (OUTDIR, sat_KL, MLP_REF, NMF_REF, samp_floor))
    log("self-check: ladder 비증가=%s | 곡선 감소=%s" % (sc["ladder_nonincreasing"], sc["curves_decreasing"]))
    plot(ladder_rows, lref_row, hr_row, samp_floor)
    return payload


def plot(ladder_rows, lref_row, hr_row, samp_floor):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # 좌: 학습량(epoch*restart) vs best KL
    budg = [r["epochs"] * r["restarts"] for r in ladder_rows]
    ax[0].plot(budg, [r["best_KL_ptrue"] for r in ladder_rows], "o-", color="#7B2FBE", label="QCBM-KL L=3 (P=54)")
    ax[0].scatter([lref_row["epochs"] * lref_row["restarts"]], [lref_row["best_KL_ptrue"]],
                  color="#E2A13B", marker="D", s=70, label="L=6 (P=108, ref)", zorder=5)
    ax[0].axhline(MLP_REF, color="#4C9F70", ls="--", label="MLP ref 0.001")
    ax[0].axhline(NMF_REF, color="#3B6EA5", ls="--", label="NMF ref 0.005")
    ax[0].axhline(samp_floor, color="gray", ls=":", label="sampling floor")
    ax[0].set_xlabel("training budget (epochs x restarts)"); ax[0].set_ylabel("best KL(p_true || model)")
    ax[0].set_yscale("log"); ax[0].set_title("Low-order KL saturation vs training"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # 우: 대표 학습 곡선
    for r in ladder_rows:
        xs = [c[0] for c in r["curve"]]; ys = [c[1] for c in r["curve"]]
        ax[1].plot(xs, ys, label="ep=%d rs=%d" % (r["epochs"], r["restarts"]))
    ax[1].axhline(MLP_REF, color="#4C9F70", ls="--"); ax[1].axhline(samp_floor, color="gray", ls=":")
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("train KL(q_emp || model)")
    ax[1].set_yscale("log"); ax[1].set_title("Best-run learning curves (low-order)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.suptitle("QCBM-KL training saturation (StronglyEntangling, P=54, n=6) — finite-training vs expressivity")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "saturation.png"), dpi=125); plt.close()
    log("[저장] %s/saturation.png" % OUTDIR)


def main():
    ap = argparse.ArgumentParser(description="QCBM-KL 학습 포화 실험(합성, 저차 집중). 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
