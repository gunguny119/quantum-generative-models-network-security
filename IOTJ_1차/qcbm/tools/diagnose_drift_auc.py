#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
차수별 drift 탐지 AUC 진단 — 고차(≥3차) 밀도 모델링이 pairwise(2차)보다 unseen-가족 탐지 AUC를 올리는가.

고전 전용(QCBM 없음). maxent 적합·MI 계산 외 학습 없음. 기존 도구 재사용(import만, 무수정):
  diagnose_order_decomposition(OD): maxent_upto(2^k full-support 벡터+수렴), bit_matrix, kl_div
  diagnose_drebin_order(DR): emp_dist(정수코드=상태매핑), load_split
  diagnose_drebin_targeted(TG): 표적/highvar 부분집합 인덱스는 results2/drebin_targeted/drebin_targeted.json에서 로드.

질문: 직전 표적 부분집합(≥3차 34~64%)의 고차 구조가 unseen 가족(concept drift) 탐지에 쓸모 있는가.
같은 부분집합·train/test·평활화로 차수만 바꾼 모델(M1 1차 / M2 2차 / M3 3차 / M4 4차 / M_full 평활 경험분포)의
anomaly score=−log p[state] 로 AUC(unseen=양성)를 비교. 격차(고차−M2)·split 일관성만 본다.

★ 상대 비교 한정. 절대 탐지 성능·CADE 비교 아님. 우위·QCBM 언급 없음. 단정 금지(판정은 웹 AI).
※ k=12는 3·4차 maxent 비수렴이라 제외(M_full 프록시). M_full=memorization 성격·평활 c 민감 → maxent 사다리(M1→M4,k10)가 1차 근거.

사용: python tools/diagnose_drift_auc.py [--smoke] [--splits 0,1,2] [--ks 10,12] [--no-plot]
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
OUTDIR = os.path.join("results2", "drift_auc")
OUT_JSON = os.path.join(OUTDIR, "drift_auc.json")
OUT_PNG = os.path.join(OUTDIR, "drift_auc.png")
TGT_JSON = os.path.join("results2", "drebin_targeted", "drebin_targeted.json")
EPS = 1e-12


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def state_codes(Xsub):
    """이진 부분집합 → 정수 상태코드 (DR.emp_dist 와 동일 인코딩: bit j << j)."""
    Xsub = np.asarray(Xsub, dtype=np.int64)
    k = Xsub.shape[1]
    code = np.zeros(len(Xsub), dtype=np.int64)
    for j in range(k):
        code += (Xsub[:, j] << j)
    return code


def auc_rank(scores, pos_mask):
    """Mann-Whitney rank AUC (numpy). pos_mask=양성(unseen). 동점=평균순위."""
    scores = np.asarray(scores, dtype=np.float64)
    pos_mask = np.asarray(pos_mask, dtype=bool)
    n_pos = int(pos_mask.sum()); n_neg = int((~pos_mask).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # 동점 평균순위 처리
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            avg = (ranks[order[i]] + ranks[order[j]]) / 2.0
            ranks[order[i:j + 1]] = avg
        i = j + 1
    sum_pos = ranks[pos_mask].sum()
    auc = (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def laplace_emp(p_emp, N, k, c=None):
    """평활 경험분포. 총 pseudo-mass=1 → p_s=(n_s + c)/(N + c*2^k), c=1/2^k. full support."""
    K = 1 << k
    if c is None:
        c = 1.0 / K
    counts = p_emp * N                      # p_emp = counts/N (정확)
    ps = (counts + c) / (N + c * K)
    return ps / ps.sum()


# ---------------------------------------------------------------------------
# 모델 적합 (모두 같은 Xtr_sub) + test 점수 + AUC
# ---------------------------------------------------------------------------
def fit_models(Xtr_sub, k, do_high=True, max_iter=20000, tol=1e-8):
    p_emp, occ = DR.emp_dist(Xtr_sub)       # counts/N over 2^k
    N = len(Xtr_sub)
    bm = OD.bit_matrix(k)
    models = {}; conv = {}
    orders = [1, 2] + ([3, 4] if (do_high and k <= 10) else [])
    for o in orders:
        q, info = OD.maxent_upto(p_emp, k, bm, o, max_iter=max_iter, tol=tol)
        models["M%d" % o] = np.asarray(q, float)
        conv["M%d" % o] = bool(info["converged"])
    models["M_full"] = laplace_emp(p_emp, N, k)
    conv["M_full"] = True
    return models, conv, occ, N


def score_auc(models, Xte_sub, pos_mask):
    codes = state_codes(Xte_sub)
    aucs = {}
    for name, p in models.items():
        s = -np.log(np.clip(p[codes], EPS, None))
        aucs[name] = auc_rank(s, pos_mask)
    return aucs


def run_axis(subset, split, k, Xtr, Xte, pos_mask):
    Xtr_sub = Xtr[:, subset]; Xte_sub = Xte[:, subset]
    models, conv, occ, N = fit_models(Xtr_sub, k)
    aucs = score_auc(models, Xte_sub, pos_mask)
    # 고차 대표: k<=10 이면 최고차 수렴 모델(M4>M3), 아니면 M_full
    if k <= 10 and conv.get("M4"):
        hi_name, hi = "M4", aucs["M4"]
    elif k <= 10 and conv.get("M3"):
        hi_name, hi = "M3", aucs["M3"]
    else:
        hi_name, hi = "M_full", aucs["M_full"]
    m2 = aucs["M2"]
    # 참고: train ≥3차 share (KL2/KL1) — full-train
    KL1 = OD.kl_div(*(_kl_args(Xtr_sub, k, 1)))
    KL2 = OD.kl_div(*(_kl_args(Xtr_sub, k, 2)))
    share_ge3 = (KL2 / KL1) if KL1 > 0 else None
    return dict(split=split, k=k, subset=[int(i) for i in subset], occ=int(occ), N=int(N),
                auc=aucs, converged=conv,
                hi_name=hi_name, gap_hi_minus_M2=float(hi - m2),
                gap_Mfull_minus_M2=float(aucs["M_full"] - m2),
                train_share_ge3=(float(share_ge3) if share_ge3 is not None else None))


_KLCACHE = {}
def _kl_args(Xtr_sub, k, order):
    key = (id(Xtr_sub), k, order)
    if key not in _KLCACHE:
        p_emp, _ = DR.emp_dist(Xtr_sub)
        bm = OD.bit_matrix(k)
        q, _i = OD.maxent_upto(p_emp, k, bm, order)
        _KLCACHE[key] = (p_emp, q)
    return (_KLCACHE[key][0], _KLCACHE[key][1], EPS)


# ---------------------------------------------------------------------------
# 축3: feature-가족 MI (train known 가족)  — 측정만
# ---------------------------------------------------------------------------
def feature_family_mi(Xtr, ytr):
    """각 feature(binary)와 가족 레이블 간 MI(nats). 반환 길이 = feature 수."""
    N = len(ytr)
    fams = np.unique(ytr)
    pf = np.array([(ytr == f).mean() for f in fams])       # p(family)
    mi = np.zeros(Xtr.shape[1], dtype=np.float64)
    x1 = Xtr.mean(0)                                        # p(x=1)
    for jf, f in enumerate(fams):
        mask = (ytr == f)
        # p(x=1, f) and p(x=0, f)
        p_x1_f = Xtr[mask].sum(0) / N                       # count(x=1,f)/N
        p_x0_f = (mask.sum() - Xtr[mask].sum(0)) / N
        for pxf, px in [(p_x1_f, x1), (p_x0_f, 1 - x1)]:
            with np.errstate(divide="ignore", invalid="ignore"):
                term = pxf * np.log(pxf / (px * pf[jf]))
            term[~np.isfinite(term)] = 0.0
            mi += term
    return mi


# ---------------------------------------------------------------------------
def self_check():
    out = {}
    rng = np.random.default_rng(0)
    # 무작위 ≈ 0.5
    sc = rng.random(2000); pm = rng.random(2000) < 0.4
    out["auc_random"] = auc_rank(sc, pm)
    # 완전분리 = 1.0 (양성 점수 큼)
    n = 1000; pm2 = np.array([True] * 500 + [False] * 500)
    sc2 = np.concatenate([rng.random(500) + 10, rng.random(500)])
    out["auc_separated"] = auc_rank(sc2, pm2)
    # 완전 역분리 = 0.0
    out["auc_reversed"] = auc_rank(-sc2, pm2)
    return out


def load_subsets():
    d = json.load(open(TGT_JSON))
    tgt = {}; hv = {}
    for k, recs in d["targeted"].items():
        for r in recs:
            tgt[(int(r["split"]), int(k))] = r["subset"]
    for k, recs in d["per_family"].items():
        for r in recs:
            hv[(int(r["split"]), int(k))] = r["subset"]
    return tgt, hv


def run(splits, ks, do_plot=True):
    os.makedirs(OUTDIR, exist_ok=True)
    log("=== self-check (AUC 검증) ===")
    sc = self_check()
    log("  AUC 무작위=%.3f (≈0.5) | 완전분리=%.3f (=1.0) | 역분리=%.3f (=0.0)"
        % (sc["auc_random"], sc["auc_separated"], sc["auc_reversed"]))

    tgt, hv = load_subsets()
    axis1, axis2, axis3 = {}, {}, {}
    for split in splits:
        Xtr, ytr, Xte, yte = _load_tr_te(split)
        unseen = sorted(set(yte.tolist()) - set(ytr.tolist()))
        pos_mask = np.isin(yte, unseen)
        log("\n=== split %d | unseen 가족=%s | test unseen=%d known=%d ==="
            % (split, unseen, int(pos_mask.sum()), int((~pos_mask).sum())))
        # 축3 MI (split 공통 feature 공간)
        mi = feature_family_mi(Xtr, ytr)
        for k in ks:
            # 축1 표적
            r1 = run_axis(tgt[(split, k)], split, k, Xtr, Xte, pos_mask)
            axis1.setdefault(str(k), []).append(r1)
            a = r1["auc"]
            log("  [표적 k%d] M1=%.3f M2=%.3f%s M_full=%.3f | 고차(%s)−M2=%+.3f Mfull−M2=%+.3f | trainge3=%.2f"
                % (k, a["M1"], a["M2"],
                   (" M3=%.3f M4=%.3f" % (a.get("M3", float("nan")), a.get("M4", float("nan")))) if k <= 10 else "",
                   a["M_full"], r1["hi_name"], r1["gap_hi_minus_M2"], r1["gap_Mfull_minus_M2"],
                   r1["train_share_ge3"]))
            # 축2 highvar
            r2 = run_axis(hv[(split, k)], split, k, Xtr, Xte, pos_mask)
            axis2.setdefault(str(k), []).append(r2)
            b = r2["auc"]
            log("  [highvar k%d] M2=%.3f M_full=%.3f | 고차(%s)−M2=%+.3f"
                % (k, b["M2"], b["M_full"], r2["hi_name"], r2["gap_hi_minus_M2"]))
        # 축3 정리: 표적/highvar/전체 평균 MI (k=max 기준 subset)
        kmax = max(ks)
        tsub = tgt[(split, kmax)]; hsub = hv[(split, kmax)]
        axis3[str(split)] = dict(
            unseen=unseen, k=kmax,
            mi_targeted_mean=float(mi[tsub].mean()),
            mi_highvar_mean=float(mi[hsub].mean()),
            mi_all_mean=float(mi.mean()),
            mi_targeted=[float(mi[i]) for i in tsub],
        )
        log("  [MI] 표적 평균=%.4f | highvar 평균=%.4f | 전체 평균=%.4f (nats)"
            % (axis3[str(split)]["mi_targeted_mean"], axis3[str(split)]["mi_highvar_mean"],
               axis3[str(split)]["mi_all_mean"]))

    payload = dict(
        note=("차수별 unseen-가족 drift 탐지 AUC. 같은 부분집합·train·test·평활화(Laplace c=1/2^k)로 차수만 변경. "
              "score=−log p[state], unseen=양성, AUC=rank(Mann-Whitney). k12는 3·4차 maxent 비수렴→제외(M_full 프록시). "
              "상대 비교(고차−M2)만. 절대 성능·CADE 비교 아님. 우위·QCBM 언급 없음. 단정 금지."),
        params=dict(splits=splits, ks=ks, eps=EPS, laplace_c="1/2^k", auc="rank Mann-Whitney"),
        self_check=sc, axis1_targeted=axis1, axis2_highvar=axis2, axis3_mi=axis3)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if do_plot:
        try:
            plot(axis1, axis2, axis3, ks)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def _load_tr_te(split):
    npz = os.path.join(DDIR, "drebin_new_%d.npz" % split)
    with np.load(npz) as z:
        Xtr = z["X_train"].astype(np.int8); ytr = np.asarray(z["y_train"])
        Xte = z["X_test"].astype(np.int8);  yte = np.asarray(z["y_test"])
    return Xtr, ytr, Xte, yte


def plot(axis1, axis2, axis3, ks):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    # 축1: 표적 차수별 AUC (k10, split 평균) + M_full
    def avg(recs, name):
        vs = [r["auc"].get(name) for r in recs if r["auc"].get(name) is not None and np.isfinite(r["auc"].get(name))]
        return float(np.mean(vs)) if vs else float("nan")
    k10 = "10"
    names = ["M1", "M2", "M3", "M4", "M_full"]
    if k10 in axis1:
        vals = [avg(axis1[k10], n) for n in names]
        ax[0].bar(range(len(names)), vals, color=["#9AA0A6", "#4C9F70", "#3B6EA5", "#E2A13B", "#C1442E"])
        ax[0].set_xticks(range(len(names))); ax[0].set_xticklabels(names)
        ax[0].axhline(0.5, color="gray", ls="--", lw=1)
        for i, v in enumerate(vals):
            ax[0].text(i, v, "%.3f" % v, ha="center", va="bottom", fontsize=8)
        ax[0].set_ylabel("drift AUC (unseen vs known)")
        ax[0].set_title("(axis1) Targeted k=10: AUC by order")
        ax[0].grid(alpha=.3, axis="y")
    # 축2: 표적 vs highvar 고차−M2 격차 (k별)
    x = np.arange(len(ks)); w = 0.35
    gt = [float(np.mean([r["gap_hi_minus_M2"] for r in axis1[str(k)]])) for k in ks]
    gh = [float(np.mean([r["gap_hi_minus_M2"] for r in axis2[str(k)]])) for k in ks]
    ax[1].bar(x - w / 2, gt, w, label="targeted", color="#C1442E")
    ax[1].bar(x + w / 2, gh, w, label="highvar", color="#3B6EA5")
    ax[1].axhline(0, color="black", lw=1)
    ax[1].set_xticks(x); ax[1].set_xticklabels(["k=%d" % k for k in ks])
    ax[1].set_ylabel("AUC(고차) − AUC(M2)")
    ax[1].set_title("(axis1/2) higher-order − pairwise gap")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis="y")
    # 축3: MI 표적/highvar/전체
    sp = sorted(axis3.keys())
    tm = [axis3[s]["mi_targeted_mean"] for s in sp]
    hm = [axis3[s]["mi_highvar_mean"] for s in sp]
    am = [axis3[s]["mi_all_mean"] for s in sp]
    xs = np.arange(len(sp))
    ax[2].bar(xs - w, tm, w, label="targeted", color="#C1442E")
    ax[2].bar(xs, hm, w, label="highvar", color="#3B6EA5")
    ax[2].bar(xs + w, am, w, label="all features", color="#9AA0A6")
    ax[2].set_xticks(xs); ax[2].set_xticklabels(["split%s" % s for s in sp])
    ax[2].set_ylabel("mean MI(feature; family) [nats]")
    ax[2].set_title("(axis3) family discriminability (MI)")
    ax[2].legend(fontsize=9); ax[2].grid(alpha=.3, axis="y")
    plt.suptitle("Order-wise drift-detection AUC: does higher-order help over pairwise? (classical)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="차수별 drift 탐지 AUC(고전). maxent/MI 외 학습 없음. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--ks", type=str, default="10,12")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run([0], [10], do_plot=False)
        return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    run(splits, ks, do_plot=not args.no_plot)


if __name__ == "__main__":
    main()
