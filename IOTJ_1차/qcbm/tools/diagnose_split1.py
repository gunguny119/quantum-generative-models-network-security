#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
split1 이상 신호 진단 — drift(unseen family) 탐지에서 split1만 고차−pairwise AUC 이득이 일관 +0.15인 원인 규명.

측정 전용(새 학습 없음). 기존 도구 재사용(import, 무수정):
  diagnose_order_decomposition(OD): maxent_upto, kl_div, bit_matrix, entropy
  diagnose_drebin_order(DR): emp_dist, load_split
  diagnose_drift_auc(DA): _load_tr_te, laplace_emp  (AUC 격차는 results2/drift_auc/drift_auc.json 인용)
  부분집합: results2/drebin_targeted/drebin_targeted.json 의 targeted[k][split]['subset'].

배경(AUC 고차−M2): split0 −0.069 / split1 +0.152 / split2 −0.094 (k10). split1만 양(+). split1은 M2 AUC도 유독 낮음(0.605).

4종 측정(모든 split 동일):
 1) 구조 상속: KL(unseen ∥ train의 ≥3차 maxent 투영) + train 각 가족 top-k 3차 연결상관 vs unseen 의 자카드/코사인.
 2) 이탈의 차수 분해: KL(unseen ∥ base) 를 1차/2차/≥3차로 근사 분해(base=known-test / train 둘 다). 각 차수 절대·비중.
 3) 아티팩트: split별 unseen/known 표본수·비율, subset feature 활성률 평균/분산(train vs unseen).
 4) 선택 정합: 표적 subset feature vs unseen 상위 활성 feature 겹침률.

★ 측정값만. 재현 위해 seed/표본수 병기. 판정(H1~H4)은 지표-AUC 정렬로. 단정 금지, 혼재면 정직 보고.

사용: python tools/diagnose_split1.py [--splits 0,1,2] [--k 10] [--topk 10]
"""
import os
import sys
import json
import itertools
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
import diagnose_drift_auc as DA            # noqa: E402

OUTDIR = os.path.join("results2", "split1_diag")
OUT_JSON = os.path.join(OUTDIR, "split1_diag.json")
OUT_PNG = os.path.join(OUTDIR, "split1_diag.png")
TGT_JSON = os.path.join("results2", "drebin_targeted", "drebin_targeted.json")
AUC_JSON = os.path.join("results2", "drift_auc", "drift_auc.json")
EPS = 1e-12
GE3_ORDER = 4          # train ≥3차 투영 = order-4 maxent (k=10 수렴 확인됨)


def log(m):
    print(m, flush=True)


def smoothed(X, k):
    p, occ = DR.emp_dist(X)
    return DA.laplace_emp(p, len(X), k), p, occ


def maxent(p_emp, k, bm, order):
    q, info = OD.maxent_upto(p_emp, k, bm, order, max_iter=20000, tol=1e-8)
    return np.asarray(q, float), bool(info["converged"])


# ---------------------------------------------------------------------------
# 차수별 연결(connected) 상관 통계 — 지지/로그 문제 없는 robust moment 기반
# ---------------------------------------------------------------------------
def moment_stats(X):
    """1차(활성), 2차(연결 2점=공분산 off-diag), 3차(연결 3점) 상관 벡터. ±1 스핀."""
    s = (1 - 2 * np.asarray(X, float))       # ±1  (N,k)
    k = s.shape[1]
    m1 = s.mean(0)                           # ⟨s_i⟩  (1차)
    m2 = (s.T @ s) / len(s)                  # ⟨s_i s_j⟩
    conn2 = m2 - np.outer(m1, m1)            # 연결 2점
    iu = np.triu_indices(k, 1)
    v2 = conn2[iu]                           # 2차 벡터 (off-diag)
    triples = list(itertools.combinations(range(k), 3))
    v3 = np.zeros(len(triples))
    for t, (i, j, l) in enumerate(triples):
        m3 = float(np.mean(s[:, i] * s[:, j] * s[:, l]))
        v3[t] = (m3 - (m1[i] * m2[j, l] + m1[j] * m2[i, l] + m1[l] * m2[i, j])
                 + 2 * m1[i] * m1[j] * m1[l])
    return dict(m1=m1, v2=v2, v3=v3), triples


def conn3_vector(X):
    st, tr = moment_stats(X)
    return st["v3"], tr


def topk_set(vec, topk):
    return set(np.argsort(-np.abs(vec))[:topk].tolist())


def cosine(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else 0.0


def jaccard(a, b):
    return len(a & b) / len(a | b) if (a | b) else 0.0


# ---------------------------------------------------------------------------
def diagnose_split(split, k, topk, bm):
    Xtr, ytr, Xte, yte = DA._load_tr_te(split)
    unseen = sorted(set(yte.tolist()) - set(ytr.tolist()))
    tgt = json.load(open(TGT_JSON))
    idx = {(int(r["split"]), int(kk)): r["subset"]
           for kk, recs in tgt["targeted"].items() for r in recs}[(split, k)]

    Xtr_s = Xtr[:, idx].astype(np.int64)
    Xun = Xte[np.isin(yte, unseen)][:, idx].astype(np.int64)     # unseen family (test 양성)
    Xkn = Xte[~np.isin(yte, unseen)][:, idx].astype(np.int64)    # known families (test 음성)

    q_train, pe_train, occ_tr = smoothed(Xtr_s, k)
    p_unseen, pe_un, occ_un = smoothed(Xun, k)
    q_known, pe_kn, occ_kn = smoothed(Xkn, k)

    M1_tr, _ = maxent(pe_train, k, bm, 1)
    M2_tr, _ = maxent(pe_train, k, bm, 2)

    st_un, triples = moment_stats(Xun)
    st_tr, _ = moment_stats(Xtr_s)
    st_kn, _ = moment_stats(Xkn)

    # ── 측정1: 구조 상속 (smooth ladder KL; order-4 maxent는 train 과적합으로 unseen서 KL 폭발→제외) ──
    kl_un_M1 = OD.kl_div(p_unseen, M1_tr, EPS)
    kl_un_M2 = OD.kl_div(p_unseen, M2_tr, EPS)
    kl_un_full = OD.kl_div(p_unseen, q_train, EPS)         # full = Laplace 평활(지지 안전)
    ladder_r2 = kl_un_M1 - kl_un_M2                        # 2차 train 구조가 unseen 설명 개선분
    ladder_rge3 = kl_un_M2 - kl_un_full                   # ≥3차 train 구조가 unseen 설명 개선분(상속)
    # per-family 3차 연결상관 겹침
    un_vec = st_un["v3"]
    un_top = topk_set(un_vec, topk)
    fam_jac, fam_cos, known_vecs = [], [], []
    for f in sorted(set(ytr.tolist())):
        Xf = Xtr[ytr == f][:, idx].astype(np.int64)
        if len(Xf) < 50:
            continue
        fv, _ = conn3_vector(Xf)
        known_vecs.append(fv)
        fam_jac.append(jaccard(un_top, topk_set(fv, topk)))
        fam_cos.append(cosine(un_vec, fv))

    # ── 측정2: 이탈의 차수 분해 (robust moment 기반; base=known-test, train 둘 다) ──
    # 차수별 dissimilarity = 1 − cos(unseen 통계, base 통계). 지지/로그 문제 없음, scale-free.
    def order_decomp(st_base, tag):
        d1 = 1.0 - cosine(st_un["m1"], st_base["m1"])
        d2 = 1.0 - cosine(st_un["v2"], st_base["v2"])
        d3 = 1.0 - cosine(st_un["v3"], st_base["v3"])
        # L2 이탈(절대) 도 병기
        l1 = float(np.linalg.norm(st_un["m1"] - st_base["m1"]))
        l2 = float(np.linalg.norm(st_un["v2"] - st_base["v2"]))
        l3 = float(np.linalg.norm(st_un["v3"] - st_base["v3"]))
        tot = d1 + d2 + d3
        return dict(base=tag, diss1=float(d1), diss2=float(d2), diss3=float(d3),
                    share_1st=float(d1 / tot) if tot > 0 else None,
                    share_2nd=float(d2 / tot) if tot > 0 else None,
                    share_ge3=float(d3 / tot) if tot > 0 else None,
                    l2dev_1st=l1, l2dev_2nd=l2, l2dev_3rd=l3)
    dec_known = order_decomp(st_kn, "known_test")
    dec_train = order_decomp(st_tr, "train")

    # ── 측정3: 아티팩트 ──
    act_tr = Xtr_s.mean(0); act_un = Xun.mean(0)
    artifact = dict(
        n_unseen=int(len(Xun)), n_known=int(len(Xkn)),
        ratio_known_over_unseen=float(len(Xkn) / max(1, len(Xun))),
        occ_unseen=int(occ_un), occ_known=int(occ_kn), occ_train=int(occ_tr),
        feat_act_train_mean=float(act_tr.mean()), feat_act_train_var=float(act_tr.var()),
        feat_act_unseen_mean=float(act_un.mean()), feat_act_unseen_var=float(act_un.var()),
        feat_act_shift_L1=float(np.abs(act_tr - act_un).sum()))    # 주변활성 총이동

    # ── 측정4: 선택 정합 ──
    # unseen 상위 활성 feature(전체 1376 중) vs 표적 subset(idx)
    un_full_act = Xte[np.isin(yte, unseen)].mean(0)
    un_top_feats = set(np.argsort(-un_full_act)[:k].tolist())
    tr_full_act = Xtr.mean(0)
    tr_top_feats = set(np.argsort(-tr_full_act)[:k].tolist())
    sel = dict(
        overlap_subset_vs_unseen_topk=len(set(idx) & un_top_feats),
        overlap_subset_vs_train_topk=len(set(idx) & tr_top_feats),
        subset_mean_act_in_unseen=float(un_full_act[idx].mean()),
        subset_mean_act_in_train=float(tr_full_act[idx].mean()))

    return dict(
        split=split, k=k, unseen=unseen, subset=[int(i) for i in idx],
        m1_structure=dict(kl_unseen_M1train=float(kl_un_M1), kl_unseen_M2train=float(kl_un_M2),
                          kl_unseen_fulltrain=float(kl_un_full),
                          ladder_r2=float(ladder_r2), ladder_rge3=float(ladder_rge3),
                          conn3_jaccard_mean=float(np.mean(fam_jac)) if fam_jac else None,
                          conn3_cosine_perfam_mean=float(np.mean(fam_cos)) if fam_cos else None,
                          conn3_cosine_knownpooled=cosine(un_vec, st_kn["v3"])),
        m2_deviation=dict(known_base=dec_known, train_base=dec_train),
        m3_artifact=artifact, m4_selection=sel)


def load_auc_gaps():
    d = json.load(open(AUC_JSON))
    g = {}
    for k, recs in d["axis1_targeted"].items():
        for r in recs:
            g[(int(r["split"]), int(k))] = dict(gap=r["gap_hi_minus_M2"], M1=r["auc"]["M1"],
                                                M2=r["auc"]["M2"], Mfull=r["auc"]["M_full"])
    return g


def run(splits, k, topk):
    os.makedirs(OUTDIR, exist_ok=True)
    bm = OD.bit_matrix(k)
    gaps = load_auc_gaps()
    res = {}
    for s in splits:
        log("\n=== split %d 진단 (k=%d) ===" % (s, k))
        r = diagnose_split(s, k, topk, bm)
        r["auc"] = gaps.get((s, k))
        res[str(s)] = r
        m1, m2k, m2t, m3, m4 = (r["m1_structure"], r["m2_deviation"]["known_base"],
                                r["m2_deviation"]["train_base"], r["m3_artifact"], r["m4_selection"])
        log("  AUC 고차−M2=%+.3f (M2=%.3f) | unseen=%s n_un=%d n_kn=%d" %
            (r["auc"]["gap"], r["auc"]["M2"], r["unseen"], m3["n_unseen"], m3["n_known"]))
        log("  [1 구조상속] KL(un‖M1tr)=%.3f KL(un‖M2tr)=%.3f KL(un‖fulltr)=%.3f rge3=%+.3f | conn3 cos(un,known)=%.3f jac=%.3f" %
            (m1["kl_unseen_M1train"], m1["kl_unseen_M2train"], m1["kl_unseen_fulltrain"],
             m1["ladder_rge3"], m1["conn3_cosine_knownpooled"], m1["conn3_jaccard_mean"]))
        log("  [2 이탈차수/known] diss 1/2/≥3=%.3f/%.3f/%.3f (비중 %.2f/%.2f/%.2f) | train base ≥3차비중=%.2f" %
            (m2k["diss1"], m2k["diss2"], m2k["diss3"], m2k["share_1st"], m2k["share_2nd"], m2k["share_ge3"],
             m2t["share_ge3"]))
        log("  [3 아티팩트] known/unseen 비=%.2f | feat활성 train_var=%.4f unseen_var=%.4f L1이동=%.3f" %
            (m3["ratio_known_over_unseen"], m3["feat_act_train_var"], m3["feat_act_unseen_var"], m3["feat_act_shift_L1"]))
        log("  [4 선택정합] subset∩unseen_top%d=%d subset∩train_top%d=%d | subset활성 un=%.3f tr=%.3f" %
            (k, m4["overlap_subset_vs_unseen_topk"], k, m4["overlap_subset_vs_train_topk"],
             m4["subset_mean_act_in_unseen"], m4["subset_mean_act_in_train"]))

    payload = dict(
        note=("split1 이상 신호(고차−M2 AUC +0.15) 원인 진단. 측정 전용(새 학습 없음). "
              "1구조상속 2이탈차수분해 3아티팩트 4선택정합 을 split별 AUC 이득과 정렬. "
              "정의: unseen=y_test에만 있는 가족(양성), known=train 가족(음성). base=known-test/train. "
              "KL Laplace 평활(c=1/2^k). conn3=연결 3점상관. 판정 H1~H4는 정렬로. 단정 금지."),
        params=dict(splits=splits, k=k, topk=topk, ge3_order=GE3_ORDER, eps=EPS,
                    laplace_c="1/2^k", auc_source=AUC_JSON), results=res)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(res, splits, k)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(res, splits, k):
    sp = [str(s) for s in splits]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    gap = [res[s]["auc"]["gap"] for s in sp]
    # 왼쪽: AUC gap + M2 AUC
    x = np.arange(len(sp))
    ax[0].bar(x - 0.2, gap, 0.4, label="AUC(고차−M2)", color="#C1442E")
    ax[0].bar(x + 0.2, [res[s]["auc"]["M2"] for s in sp], 0.4, label="M2 AUC", color="#4C9F70")
    ax[0].axhline(0, color="black", lw=1)
    ax[0].set_xticks(x); ax[0].set_xticklabels(["split%s" % s for s in sp]); ax[0].legend(fontsize=8)
    ax[0].set_title("AUC 이득 vs M2 절대 AUC"); ax[0].grid(alpha=.3, axis="y")
    # 가운데: 측정2 차수별 이탈비중(known base) + 측정1 low-order KL
    ax[1].plot(x, [res[s]["m2_deviation"]["known_base"]["share_1st"] for s in sp], "o-", label="1차 이탈비중")
    ax[1].plot(x, [res[s]["m2_deviation"]["known_base"]["share_2nd"] for s in sp], "s-", label="2차 이탈비중")
    ax[1].plot(x, [res[s]["m2_deviation"]["known_base"]["share_ge3"] for s in sp], "D-", label="≥3차 이탈비중")
    ax[1].plot(x, [res[s]["auc"]["gap"] for s in sp], "^:", color="#C1442E", label="AUC 이득")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["split%s" % s for s in sp]); ax[1].legend(fontsize=8)
    ax[1].set_title("측정2 차수별 이탈비중(known) vs AUC 이득"); ax[1].grid(alpha=.3)
    # 오른쪽: 측정3 feat 활성 이동 + 측정1 cosine
    ax[2].plot(x, [res[s]["m3_artifact"]["feat_act_shift_L1"] for s in sp], "o-", label="feat활성 L1이동")
    ax[2].plot(x, [res[s]["m1_structure"]["conn3_cosine_knownpooled"] for s in sp], "s--", label="conn3 cos(un,known)")
    ax[2].plot(x, [res[s]["auc"]["gap"] for s in sp], "^:", color="#C1442E", label="AUC 이득")
    ax[2].set_xticks(x); ax[2].set_xticklabels(["split%s" % s for s in sp]); ax[2].legend(fontsize=8)
    ax[2].set_title("측정1/3 vs AUC 이득"); ax[2].grid(alpha=.3)
    plt.suptitle("split1 anomaly diagnosis: which metric aligns with higher-order AUC gain?")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="split1 이상 신호 진단(측정 전용). 단정 금지.")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--topk", type=int, default=10)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run([1], args.k, args.topk); return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    run(splits, args.k, args.topk)


if __name__ == "__main__":
    main()
