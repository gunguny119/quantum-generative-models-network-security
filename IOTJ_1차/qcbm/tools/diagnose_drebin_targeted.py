#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drebin 고차성 "최강 버전" 시험 — 표적 greedy 부분집합(≥3차 최대화)+holdout, per-family 분해.

순수 측정(학습 없음, QCBM 없음). 기존 도구 재사용(import만, 무수정):
  diagnose_order_decomposition(OD): maxent_upto, kl_div, bit_matrix, decompose, self_check, entropy
  diagnose_drebin_order(DR): emp_dist, select_subset, load_split, top_triples

배경: 일반 선정에서 Drebin ≥3차 최대 21.8%(highvar k12) < flow UNSW 27%. 두 미검 사항:
 (b) ≥3차를 *일부러 최대화*하는 부분집합이 있는가 — greedy 탐색. 단 표본 희소(train 1912 vs 2^12)로 과적합 위험.
     ★holdout 필수: X_train 내부 selA(선정)/selB(iid 검증) 분할이 1차, X_test(CADE drift, 다른 가족)는 2차 검증.
 (a) 8가족 pooling이 mixture로 겉보기 고차를 부풀렸는지 per-family 분해로 분리.

지표: ≥3차 share = KL2/KL1 (KL1=1차 maxent, KL2=2차 maxent). flow와 동일 잣대.
★ 측정값만. holdout 미검증(selA만 높음)=과적합으로 명시. "고차다/양자 유리" 단정 금지. 우위·QCBM 언급 없음.

사용: python tools/diagnose_drebin_targeted.py [--smoke] [--splits 0,1,2] [--ks 10,12] [--pool-M 40] [--no-plot]
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

import diagnose_order_decomposition as OD  # noqa: E402
import diagnose_drebin_order as DR         # noqa: E402

DDIR = DR.DDIR
OUTDIR = os.path.join("results2", "drebin_targeted")
OUT_JSON = os.path.join(OUTDIR, "drebin_targeted.json")
OUT_PNG = os.path.join(OUTDIR, "drebin_targeted.png")
EPS = 1e-12
FAMILY_MIN = 150          # per-family 최소 표본(미만은 불가 명시)
SEED = 12345


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# ≥3차 share = KL2/KL1  (임의 데이터 X 와 feature 인덱스 idx)
# ---------------------------------------------------------------------------
def ge3_share(X, idx, max_iter=20000, tol=1e-8):
    p, occ = DR.emp_dist(X[:, idx])
    k = len(idx)
    bm = OD.bit_matrix(k)
    q1, i1 = OD.maxent_upto(p, k, bm, 1, max_iter=max_iter, tol=tol)
    q2, i2 = OD.maxent_upto(p, k, bm, 2, max_iter=max_iter, tol=tol)
    KL1 = OD.kl_div(p, q1, EPS)
    KL2 = OD.kl_div(p, q2, EPS)
    share = (KL2 / KL1) if KL1 > 0 else 0.0
    return dict(share_ge3=float(share), share_2nd=float((KL1 - KL2) / KL1) if KL1 > 0 else None,
                KL1=float(KL1), KL2=float(KL2), occ=int(occ), n=int(len(X)),
                converged=bool(i1["converged"] and i2["converged"]))


def load_tr_te(split):
    npz = os.path.join(DDIR, "drebin_new_%d.npz" % split)
    with np.load(npz) as z:
        Xtr = z["X_train"].astype(np.int8); ytr = np.asarray(z["y_train"])
        Xte = z["X_test"].astype(np.int8);  yte = np.asarray(z["y_test"])
    return Xtr, ytr, Xte, yte


# ---------------------------------------------------------------------------
# (b) 표적 greedy: selA 에서 ≥3차 최대화하며 feature 추가 → selA/selB/test 재측정
# ---------------------------------------------------------------------------
def greedy_subset(Xsel, cats, k, pool_M, seed_mode="triple", max_iter=6000):
    """Xsel(선정셋)에서 ≥3차 share 최대화 greedy. 후보풀=고분산 top-M. 씨앗=connected-3body 최대 triple.
    반환 선정 인덱스(원본 열 기준) + greedy 경로."""
    mean = Xsel.mean(0); var = mean * (1 - mean)
    nonc = np.where((mean > 0) & (mean < 1))[0]
    pool = nonc[np.argsort(-var[nonc])][:pool_M].tolist()     # 후보 원본 인덱스
    # 씨앗: pool 안에서 connected-3body 최대 triple (없으면 top-var 2개)
    chosen = []
    if seed_mode == "triple" and len(pool) >= 3:
        names_pool = ["c%d" % i for i in pool]; cats_pool = [cats[i] for i in pool]
        trs = DR.top_triples(Xsel[:, pool], names_pool, cats_pool, topn=1)
        if trs:
            chosen = [pool[j] for j in trs[0]["triple"]]
    if len(chosen) < 2:
        chosen = pool[:2]
    path = [(len(chosen), ge3_share(Xsel, chosen, max_iter=max_iter)["share_ge3"])]
    # greedy 추가
    while len(chosen) < k:
        best = None
        for c in pool:
            if c in chosen:
                continue
            g = ge3_share(Xsel, chosen + [c], max_iter=max_iter)["share_ge3"]
            if best is None or g > best[0]:
                best = (g, c)
        if best is None:
            break
        chosen.append(best[1]); path.append((len(chosen), best[0]))
    return chosen, path


def run_targeted(splits, ks, cats_by_split, data_by_split, pool_M):
    res = {}   # res[k] = list over splits
    for split in splits:
        Xtr, ytr, Xte, yte = data_by_split[split]
        cats = cats_by_split[split]
        rng = np.random.default_rng(SEED + split)
        idxp = rng.permutation(len(Xtr))
        half = len(Xtr) // 2
        selA, selB = Xtr[idxp[:half]], Xtr[idxp[half:]]
        for k in ks:
            t0 = time.time()
            chosen, path = greedy_subset(selA, cats, k, pool_M)
            gA = ge3_share(selA, chosen)
            gB = ge3_share(selB, chosen)
            gT = ge3_share(Xte, chosen)
            rec = dict(split=split, k=k, subset=[int(c) for c in chosen],
                       cats=[cats[c] for c in chosen],
                       train_selA=gA, holdout_selB=gB, test_drift=gT,
                       overfit_gap=float(gA["share_ge3"] - gB["share_ge3"]),
                       greedy_path=[[int(a), float(b)] for a, b in path])
            res.setdefault(str(k), []).append(rec)
            log("  [표적] split%d k%d | selA ≥3차=%.1f%% → selB(holdout)=%.1f%% (gap %.1f%%) | test(drift)=%.1f%% | occA=%d occB=%d (%.0fs)"
                % (split, k, 100 * gA["share_ge3"], 100 * gB["share_ge3"], 100 * rec["overfit_gap"],
                   100 * gT["share_ge3"], gA["occ"], gB["occ"], time.time() - t0))
    return res


# ---------------------------------------------------------------------------
# (a) per-family: 고정 부분집합에서 가족별 ≥3차 vs pooling
# ---------------------------------------------------------------------------
def run_perfamily(splits, ks, cats_by_split, data_by_split):
    res = {}
    for split in splits:
        Xtr, ytr, Xte, yte = data_by_split[split]
        cats = cats_by_split[split]
        # pooled(train+test) 로 highvar 부분집합 선정 (직전 21.8% 재현 잣대와 동일)
        Xpool = np.vstack([Xtr, Xte])
        for k in ks:
            idx = DR.select_subset(Xpool, cats, "highvar", k)
            pooled = ge3_share(Xpool, idx)          # 전가족 pooling
            fams = {}
            for f in sorted(set(ytr.tolist())):
                Xf = Xtr[ytr == f]
                if len(Xf) < FAMILY_MIN:
                    fams[str(int(f))] = dict(n=int(len(Xf)), measurable=False)
                    continue
                g = ge3_share(Xf, idx)
                g["measurable"] = True; g["n"] = int(len(Xf))
                fams[str(int(f))] = g
            res.setdefault(str(k), []).append(dict(
                split=split, k=k, subset=[int(i) for i in idx],
                pooled=pooled, per_family=fams))
            meas = {f: v for f, v in fams.items() if v.get("measurable")}
            log("  [per-family] split%d k%d | pooling ≥3차=%.1f%% | 가족(n≥%d)=%d개: %s"
                % (split, k, 100 * pooled["share_ge3"], FAMILY_MIN, len(meas),
                   " ".join("f%s=%.0f%%(n%d,occ%d)" % (f, 100 * v["share_ge3"], v["n"], v["occ"]) for f, v in meas.items())))
    return res


# ---------------------------------------------------------------------------
def self_check(data_by_split, cats_by_split):
    out = {}
    # (i) pooling highvar k12 ≥3차 = 21.8% 재현 (split0, pooled)
    Xtr, ytr, Xte, yte = data_by_split[0]
    Xpool = np.vstack([Xtr, Xte]); cats = cats_by_split[0]
    idx = DR.select_subset(Xpool, cats, "highvar", 12)
    g = ge3_share(Xpool, idx)
    out["repro_pooling_highvar_k12"] = dict(share_ge3=g["share_ge3"], expected=0.218)
    # (ii) OD 합성 self-check
    try:
        sc = OD.self_check(1e-12)
        out["synthetic_all_monotonic"] = bool(sc["all_monotonic"])
    except Exception as e:
        out["synthetic_all_monotonic"] = "ERR:%r" % e
    # (iii) flow UNSW 27% 재현
    try:
        import train_qcbm_unsw_resumable as TUNSW
        d = TUNSW.build_unsw()
        p = np.asarray(d["q_train"], float); p /= p.sum()
        dec = OD.decompose(p, TUNSW.NQ, kmax=4, eps=1e-12)
        ct = dec["contrib"]; KL1 = dec["KL"]["1"]
        ge3 = ct["c3"] + (ct["c4"] or 0) + ct["resid5"]
        out["repro_flow_unsw"] = dict(share_ge3=ge3 / KL1, expected=0.270)
    except Exception as e:
        out["repro_flow_unsw"] = dict(error=repr(e))
    return out


def run(splits, ks, pool_M, do_plot=True):
    os.makedirs(OUTDIR, exist_ok=True)
    data_by_split, cats_by_split = {}, {}
    for s in splits:
        Xtr, ytr, Xte, yte = load_tr_te(s)
        _, _, cats = DR.load_split(s)   # feature명/카테고리(열 1:1)
        data_by_split[s] = (Xtr, ytr, Xte, yte); cats_by_split[s] = cats

    log("=== self-check (잣대 일관성) ===")
    sc = self_check(data_by_split, cats_by_split)
    log("  pooling highvar k12 재현=%.3f (기대 0.218) | 합성 mono=%s | flow UNSW=%.3f (기대 0.270)"
        % (sc["repro_pooling_highvar_k12"]["share_ge3"], sc.get("synthetic_all_monotonic"),
           sc.get("repro_flow_unsw", {}).get("share_ge3", float("nan"))))

    log("\n=== (b) 표적 greedy 탐색 (selA 선정 → selB holdout, X_test drift) ===")
    targeted = run_targeted(splits, ks, cats_by_split, data_by_split, pool_M)

    log("\n=== (a) per-family 분해 (고정 highvar 부분집합, 가족별 vs pooling) ===")
    perfam = run_perfamily(splits, ks, cats_by_split, data_by_split)

    # flow 기준
    flow = {"UNSW": 0.270, "Bot": 0.188, "HOIC": 0.012}
    payload = dict(
        note=("Drebin 고차성 최강버전. (b)표적 greedy: selA(train50%)서 ≥3차 최대화→selB(iid holdout)·X_test(drift) 재측정. "
              "(a)per-family: 고정 highvar 부분집합서 가족별 ≥3차 vs pooling. 지표=KL2/KL1. flow와 동일 잣대. "
              "holdout(selB) 미검증(selA만 높음)=과적합. 측정값만, 단정 금지, 우위·QCBM 언급 없음."),
        params=dict(splits=splits, ks=ks, pool_M=pool_M, seed=SEED, family_min=FAMILY_MIN, eps=EPS,
                    holdout_note="X_test=CADE drift(train과 다른 가족) → selB(train 내부 random half)가 1차 holdout"),
        self_check=sc, targeted=targeted, per_family=perfam, flow=flow)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if do_plot:
        try:
            plot(targeted, perfam, flow, ks)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def plot(targeted, perfam, flow, ks):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # 왼쪽: 표적 selA vs selB vs test (k별 평균)
    kk = [str(k) for k in ks]
    x = np.arange(len(kk)); w = 0.25
    def avg(recs, field):
        return float(np.mean([r[field]["share_ge3"] for r in recs]))
    A = [100 * avg(targeted[k], "train_selA") for k in kk]
    B = [100 * avg(targeted[k], "holdout_selB") for k in kk]
    T = [100 * avg(targeted[k], "test_drift") for k in kk]
    ax[0].bar(x - w, A, w, label="selA (train, 선정)", color="#E2A13B")
    ax[0].bar(x, B, w, label="selB (iid holdout)", color="#3B6EA5")
    ax[0].bar(x + w, T, w, label="X_test (drift)", color="#9AA0A6")
    ax[0].axhline(100 * flow["UNSW"], color="red", ls="--", lw=1, label="flow UNSW 27%")
    ax[0].axhline(21.8, color="green", ls=":", lw=1, label="일반선정 21.8%")
    ax[0].set_xticks(x); ax[0].set_xticklabels(["k=%s" % k for k in kk])
    ax[0].set_ylabel(">=3rd-order share [%]")
    ax[0].set_title("(b) Targeted greedy: train vs holdout vs drift\n(selA-only high = overfit)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    # 오른쪽: per-family vs pooling (k=max, split0)
    kmax = str(max(ks))
    rec = perfam[kmax][0]
    fams = {f: v for f, v in rec["per_family"].items() if v.get("measurable")}
    labels = ["f%s\n(n%d)" % (f, v["n"]) for f, v in fams.items()]
    vals = [100 * v["share_ge3"] for v in fams.values()]
    xf = np.arange(len(labels))
    ax[1].bar(xf, vals, color="#4C9F70", label="per-family")
    ax[1].axhline(100 * rec["pooled"]["share_ge3"], color="purple", ls="--", label="pooling %.0f%%" % (100 * rec["pooled"]["share_ge3"]))
    ax[1].axhline(100 * flow["UNSW"], color="red", ls="--", lw=1, label="flow UNSW 27%")
    ax[1].set_xticks(xf); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set_ylabel(">=3rd-order share [%]")
    ax[1].set_title("(a) per-family vs pooling (k=%s, split0)" % kmax)
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Drebin strongest-case: targeted subset (holdout) + per-family >=3rd-order")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="Drebin 표적 greedy(holdout)+per-family 고차 측정. 학습 없음. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--ks", type=str, default="10,12")
    ap.add_argument("--pool-M", type=int, default=40)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run([0], [10], 20, do_plot=False)
        return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    run(splits, ks, args.pool_M, do_plot=not args.no_plot)


if __name__ == "__main__":
    main()
