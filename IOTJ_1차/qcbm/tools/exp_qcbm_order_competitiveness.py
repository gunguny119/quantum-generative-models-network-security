#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCBM 상대 경쟁력 통제 실험 — 저차 vs 고차 부분집합에서 QCBM vs 차수사다리(M1/M2/full)의 KL 대결.

순수 분포 표현 대결(탐지 아님, y_test 안 씀). 기존 코드 재사용(import만, 무수정):
  qcbm_kl_variant(QK): train_kl_one(q_target, nq, layers, epochs, lr, seed) — 임의 목표로 forward-KL 학습(StronglyEntangling, exact probs).
  diagnose_order_decomposition(OD): maxent_upto(order별 maxent), kl_div, bit_matrix.
  diagnose_drebin_order(DR): emp_dist, load_split. 부분집합 인덱스는 results2/drebin_targeted/drebin_targeted.json.

가설: 데이터 ≥3차 비중↑ → QCBM의 (pairwise M2 대비) 상대 위치↑. pairwise는 2차까지만이라 고차 subset서 M2가 ≥3차를 놓침.
저차(highvar)/고차(targeted) 부분집합에서 같은 목표 q 에 대해 KL(q‖M1/M2/full/QCBM) 비교 → gap_QCBM_M2 의 저차→고차 변화.

목표 2버전: A=order-4 maxent(매끈, 표본잡음 제거) / B=Laplace 평활 경험분포(현실). 모델 full=q(참조 KL=0).
★ 이기지 않아도 됨 — 상대 위치 변화가 결과. SOTA·양자우위·탐지 주장 없음. QCBM seed std 그대로. 단정 금지.

사용: python tools/exp_qcbm_order_competitiveness.py [--smoke] [--splits 0,1,2] [--k 10]
                                                    [--layers 4] [--epochs 120] [--seeds 3] [--targets A,B]
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

import qcbm_kl_variant as QK              # noqa: E402  train_kl_one (experiment2 ansatz)
import diagnose_order_decomposition as OD  # noqa: E402  maxent_upto, kl_div, bit_matrix
import diagnose_drebin_order as DR         # noqa: E402  emp_dist, load_split
import diagnose_drift_auc as DA            # noqa: E402  laplace_emp, _load_tr_te (평활·로드 재사용)

DDIR = DR.DDIR
OUTDIR = os.path.join("results2", "qcbm_order_comp")
OUT_JSON = os.path.join(OUTDIR, "qcbm_order_comp.json")
OUT_PNG = os.path.join(OUTDIR, "qcbm_order_comp.png")
TGT_JSON = os.path.join("results2", "drebin_targeted", "drebin_targeted.json")
EPS = 1e-12
TARGET_MAXENT_ORDER = 4     # 목표A: order-4 maxent (k=10서 수렴 확인)


def log(m):
    print(m, flush=True)


def load_subsets():
    d = json.load(open(TGT_JSON))
    tgt, hv = {}, {}
    for k, recs in d["targeted"].items():
        for r in recs:
            tgt[(int(r["split"]), int(k))] = r["subset"]
    for k, recs in d["per_family"].items():
        for r in recs:
            hv[(int(r["split"]), int(k))] = r["subset"]
    return tgt, hv


def make_target(p_emp, N, k, bm, version):
    """목표 분포 생성. A=order-4 maxent(매끈), B=Laplace 평활 경험분포."""
    if version == "A":
        q, info = OD.maxent_upto(p_emp, k, bm, TARGET_MAXENT_ORDER, max_iter=20000, tol=1e-8)
        return np.asarray(q, float), bool(info["converged"])
    else:  # B
        return DA.laplace_emp(p_emp, N, k), True


def run_cell(split, k, subtype, idx, version, layers, epochs, seeds, Xtr, save_cb):
    p_emp, occ = DR.emp_dist(Xtr[:, idx])
    N = len(Xtr)
    bm = OD.bit_matrix(k)
    q, tconv = make_target(p_emp, N, k, bm, version)
    # 차수사다리 모델 (목표 q 에 대한 KL)
    M1, _ = OD.maxent_upto(p_emp, k, bm, 1)
    M2, _ = OD.maxent_upto(p_emp, k, bm, 2)
    kl_M1 = OD.kl_div(q, M1, EPS)
    kl_M2 = OD.kl_div(q, M2, EPS)
    kl_full = OD.kl_div(q, q, EPS)                      # 참조 = 0
    share_ge3 = (kl_M2 / kl_M1) if kl_M1 > 0 else None  # 목표의 ≥3차 비중
    # QCBM seed별 학습
    qcbm = []
    for sd in range(seeds):
        t0 = time.time()
        p_q, ce, hist = QK.train_kl_one(q, k, layers, epochs, lr=0.1, seed=sd)
        klq = OD.kl_div(q, p_q, EPS)
        qcbm.append(dict(seed=sd, kl=float(klq), ce_first=float(hist[0]), ce_last=float(hist[-1]),
                         decreasing=bool(hist[-1] < hist[0] - 1e-4), sec=round(time.time() - t0, 1)))
        log("      QCBM seed%d KL=%.4f (CE %.3f→%.3f, %s, %.0fs)"
            % (sd, klq, hist[0], hist[-1], "↓" if qcbm[-1]["decreasing"] else "×", qcbm[-1]["sec"]))
    kls = np.array([c["kl"] for c in qcbm])
    rec = dict(
        split=split, k=k, subtype=subtype, subset=[int(i) for i in idx], target_version=version,
        occ=int(occ), target_converged=tconv, params=int(layers * k * 3),
        kl_M1=float(kl_M1), kl_M2=float(kl_M2), kl_full=float(kl_full),
        target_share_ge3=(float(share_ge3) if share_ge3 is not None else None),
        qcbm_seeds=qcbm, kl_QCBM_mean=float(kls.mean()), kl_QCBM_std=float(kls.std()),
        kl_QCBM_min=float(kls.min()),
        gap_QCBM_M2=float(kls.mean() - kl_M2),        # >0 = QCBM 이 M2 보다 나쁨
        gap_QCBM_M1=float(kls.mean() - kl_M1),        # <0 = QCBM 이 M1(독립) 보다 좋음(상관 포착)
        rel_pos_QCBM=(float((kl_M1 - kls.mean()) / kl_M1) if kl_M1 > 0 else None),  # M1 격차 중 QCBM이 닫은 비율
        rel_pos_M2=(float((kl_M1 - kl_M2) / kl_M1) if kl_M1 > 0 else None),
    )
    log("    [%s/%s tgt%s] ≥3차=%.2f | KL: M1=%.3f M2=%.3f QCBM=%.3f±%.3f | gapQCBM-M2=%+.3f gapQCBM-M1=%+.3f"
        % (subtype, "k%d" % k, version, rec["target_share_ge3"] or -1, kl_M1, kl_M2,
           rec["kl_QCBM_mean"], rec["kl_QCBM_std"], rec["gap_QCBM_M2"], rec["gap_QCBM_M1"]))
    save_cb(rec)
    return rec


def run(splits, k, layers, epochs, seeds, targets, do_plot=True, smoke=False):
    os.makedirs(OUTDIR, exist_ok=True)
    tgt, hv = load_subsets()
    subsets = {"loword_highvar": hv, "highord_targeted": tgt}

    log("=== self-check (합성 QCBM KL 감소) ===")
    ok = QK.selftest()
    log("  QK.selftest PASS=%s" % ok)

    results = []
    payload = dict(
        note=("QCBM 상대 경쟁력 통제. 저차(highvar)/고차(targeted) 부분집합·같은 목표 q에 대해 KL(q‖M1/M2/full/QCBM). "
              "목표A=order-4 maxent(매끈), B=Laplace 평활 경험. full=q(참조 KL=0). gap_QCBM_M2 의 저차→고차 변화가 핵심. "
              "표현 대결(탐지 아님). SOTA·양자우위 주장 없음. QCBM seed std 그대로. 단정 금지."),
        params=dict(splits=splits, k=k, layers=layers, epochs=epochs, seeds=seeds, targets=targets,
                    lr=0.1, qcbm_params=layers * k * 3, target_maxent_order=TARGET_MAXENT_ORDER, eps=EPS),
        selftest_pass=bool(ok), results=results)

    def save_cb(rec):
        results.append(rec)
        with open(OUT_JSON, "w") as f:
            json.dump(payload, f, indent=1, ensure_ascii=False)

    for split in splits:
        Xtr = DA._load_tr_te(split)[0]
        log("\n=== split %d (k=%d, L=%d, ep=%d, seeds=%d) ===" % (split, k, layers, epochs, seeds))
        for subtype, sdict in subsets.items():
            idx = sdict[(split, k)]
            for version in targets:
                run_cell(split, k, subtype, idx, version, layers, epochs, seeds, Xtr, save_cb)

    log("\n[저장] %s (%d cells)" % (OUT_JSON, len(results)))
    if do_plot and not smoke and results:
        try:
            plot(results, targets)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def plot(results, targets):
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 6))
    colors = {"A": "#C1442E", "B": "#3B6EA5"}
    # 왼쪽: gap_QCBM_M2 저차 vs 고차 (목표버전별)
    for version in targets:
        lows = [r["gap_QCBM_M2"] for r in results if r["subtype"] == "loword_highvar" and r["target_version"] == version]
        his = [r["gap_QCBM_M2"] for r in results if r["subtype"] == "highord_targeted" and r["target_version"] == version]
        xs = [0, 1]
        ax[0].plot(xs, [np.mean(lows), np.mean(his)], "o-", color=colors.get(version, "gray"),
                   label="target %s" % version)
        # 개별 점
        for r in results:
            if r["target_version"] == version:
                xx = 0 if r["subtype"] == "loword_highvar" else 1
                ax[0].scatter([xx], [r["gap_QCBM_M2"]], color=colors.get(version, "gray"), alpha=0.4, s=20)
    ax[0].axhline(0, color="black", lw=1, ls="--")
    ax[0].set_xticks([0, 1]); ax[0].set_xticklabels(["저차\n(highvar)", "고차\n(targeted)"])
    ax[0].set_ylabel("gap = KL(QCBM) − KL(M2)   (>0: QCBM worse than pairwise)")
    ax[0].set_title("QCBM−M2 gap: low- vs high-order subset")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")
    # 오른쪽: x=목표 ≥3차 비중, y=gap_QCBM_M2
    for version in targets:
        xs = [r["target_share_ge3"] for r in results if r["target_version"] == version]
        ys = [r["gap_QCBM_M2"] for r in results if r["target_version"] == version]
        ax[1].scatter(xs, ys, color=colors.get(version, "gray"), label="target %s" % version, s=40)
    ax[1].axhline(0, color="black", lw=1, ls="--")
    ax[1].set_xlabel("target >=3rd-order share  (KL_M2/KL_M1)")
    ax[1].set_ylabel("gap = KL(QCBM) − KL(M2)")
    ax[1].set_title("higher-order share vs QCBM relative position")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)
    plt.suptitle("QCBM competitiveness vs data higher-order content (classical order-ladder, KL contest)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="QCBM 저차/고차 상대 경쟁력(KL 대결). 탐지 아님. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--targets", type=str, default="A,B")
    args = ap.parse_args()
    if args.smoke:
        run([0], args.k, layers=2, epochs=30, seeds=1, targets=["A"], do_plot=False, smoke=True)
        return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    targets = [t.strip() for t in args.targets.split(",") if t.strip()]
    run(splits, args.k, args.layers, args.epochs, args.seeds, targets, do_plot=True)


if __name__ == "__main__":
    main()
