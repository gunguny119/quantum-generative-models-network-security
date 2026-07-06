#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
인코딩 정보 손실 진단 (학습 없음, 순수 측정)
=============================================
배경: B1/B2 모두 고전이 QCBM보다 효율적 -> 분포가 희소·저랭크. B2 비트그룹에서 단서:
      3bit×4 TV 0.506 -> 4bit×3 TV 0.020 (25배 폭락). size가 정확히 4비트 -> 정보가 size에
      집중되고 나머지(proto/syn/psh/Fwd IAT/Flow IAT)는 거의 죽었을 가능성.

세 가지를 측정(판정은 웹 AI):
  (1) feature별 엔트로피(bits) + 활용률(엔트로피/비트수). benign/attack 각각. + 단일feature AUC.
  (2) IAT 해상도 곡선: Fwd/Flow IAT를 4/8/16/32/64버킷으로 -> 엔트로피·점유·단일feature AUC.
      (버킷 늘릴 때 포화하면 원본에 정보 없음 / 계속 늘면 4버킷이 정보를 죽인 것.)
  (3) 원본 연속값(size=Pkt Len Mean, Fwd/Flow IAT Mean) 고해상도 히스토그램·통계. benign vs attack.

재사용: experiment2(proto_code, size_edges_from_benign, discretize, empirical_dist, auc_score),
        diagnose_iat_joint(log1p_quantile_edges). IAT를 4버킷 외 해상도로 자르는 부분만 동일
        recipe(log1p+분위수+digitize)를 nbuckets로 일반화. 기존 소스 무수정. QCBM 학습 없음.
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E                      # noqa: E402
import diagnose_iat_joint as DIAG            # noqa: E402  log1p_quantile_edges, iat_to_bits, append_bits

DATA = "data/cic_0221_ddos.csv"
OUT = "results2"
ATTACK = "ddos attack-hoic"
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
EXPECT_OCC_B2 = 94
EPS = 1e-12
IAT_BUCKET_SWEEP = [4, 8, 16, 32, 64]
# 현재 인코딩의 feature 정의: (이름, 비트수)
FEATURES = [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4),
            ("fwd_iat", 2), ("flow_iat", 2)]


def entropy_bits(levels, nlevels):
    c = np.bincount(np.asarray(levels, int), minlength=nlevels).astype(float)
    p = c / c.sum()
    p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def single_feature_auc(tr_levels, te_levels, atk_levels, nlevels):
    """그 feature 하나의 marginal p_benign(level)로 s=-log p, attack=positive AUC."""
    p = E.empirical_dist(np.asarray(tr_levels, int), nlevels)
    s_te = -np.log(p[np.asarray(te_levels, int)] + EPS)
    s_atk = -np.log(p[np.asarray(atk_levels, int)] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def iat_quantize(values, train_values, nbuckets):
    """동일 recipe(log1p+benign-train 분위수+digitize)를 nbuckets로 일반화.
       반환: (levels, n_effective_buckets)."""
    edges = DIAG.log1p_quantile_edges(train_values, nbuckets)   # inner edges (np.unique 적용됨)
    v = np.log1p(np.clip(np.asarray(values, float), 0, None))
    lv = np.clip(np.digitize(v, edges), 0, nbuckets - 1).astype(int)
    n_eff = len(edges) + 1                                       # 중복 제거 후 실제 버킷 수
    return lv, n_eff


def feature_levels(df, name, size_edges, fwd_edges, flow_edges):
    """현재 인코딩 기준 feature별 양자화 레벨 (experiment2 로직 재사용)."""
    if name == "proto":
        return E.discretize(df, {"features": [("proto", 2)]}, size_edges)
    if name == "syn":
        return E.discretize(df, {"features": [("syn", 1)]}, size_edges)
    if name == "psh":
        return E.discretize(df, {"features": [("psh", 1)]}, size_edges)
    if name == "size":
        return E.discretize(df, {"features": [("size", 4)]}, size_edges)
    if name == "fwd_iat":
        return DIAG.iat_to_bits(df["Fwd IAT Mean"], fwd_edges)    # 4버킷(현재)
    if name == "flow_iat":
        return DIAG.iat_to_bits(df["Flow IAT Mean"], flow_edges)
    raise ValueError(name)


def load_split():
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean",
            "Fwd IAT Mean", "Flow IAT Mean", "Label"]
    E.log("CSV 로드: %s" % DATA)
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    for c in cols[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    lab = df["Label"].astype(str).str.strip().str.lower()
    ben = df[lab == "benign"].reset_index(drop=True)
    atk = df[lab == ATTACK].reset_index(drop=True)
    rng = np.random.default_rng(7)                 # build_b2와 동일
    idx = rng.permutation(len(ben)); cut = int(0.7 * len(ben))
    ben_tr = ben.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = ben.iloc[idx[cut:]].reset_index(drop=True)
    E.log("benign=%d (train=%d/test=%d)  attack=%d"
          % (len(ben), len(ben_tr), len(ben_te), len(atk)))
    return ben_tr, ben_te, atk


def cont_stats(series_tr, series_atk, name, logspace=False):
    a = np.asarray(series_tr, float); b = np.asarray(series_atk, float)
    if logspace:
        a = np.log1p(np.clip(a, 0, None)); b = np.log1p(np.clip(b, 0, None))
    qs = [0, 25, 50, 75, 90, 99, 100]
    lo = min(a.min(), b.min()); hi = max(a.max(), b.max())
    bins = np.linspace(lo, hi, 120)
    h_ben, _ = np.histogram(a, bins=bins, density=True)
    h_atk, _ = np.histogram(b, bins=bins, density=True)
    # 봉우리 추정(거친 50-bin 스무딩 위 국소최대)
    coarse, _ = np.histogram(a, bins=50)
    peaks = int(np.sum((coarse[1:-1] > coarse[:-2]) & (coarse[1:-1] > coarse[2:])
                       & (coarse[1:-1] > 0.05 * coarse.max())))
    return {
        "name": name, "logspace": logspace,
        "nunique_benign": int(pd.Series(series_tr).nunique()),
        "nunique_attack": int(pd.Series(series_atk).nunique()),
        "benign_quantiles": {str(q): float(np.percentile(a, q)) for q in qs},
        "attack_quantiles": {str(q): float(np.percentile(b, q)) for q in qs},
        "approx_peaks_benign_coarse50": peaks,
        "bins": bins.tolist(), "hist_benign": h_ben.tolist(), "hist_attack": h_atk.tolist(),
    }


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 78)
    E.log("인코딩 정보 손실 진단 (feature 엔트로피 / IAT 해상도 / 원본 구조)")
    E.log("=" * 78)

    ben_tr, ben_te, atk = load_split()
    size_edges = E.size_edges_from_benign(ben_tr, 16)
    fwd_edges = DIAG.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], 4)
    flow_edges = DIAG.log1p_quantile_edges(ben_tr["Flow IAT Mean"], 4)

    # --- 재현 검증: B2 점유 94 ---
    s = E.discretize(ben_tr, SPEC8, size_edges)
    s = DIAG.append_bits(s, DIAG.iat_to_bits(ben_tr["Fwd IAT Mean"], fwd_edges), 2)
    s = DIAG.append_bits(s, DIAG.iat_to_bits(ben_tr["Flow IAT Mean"], flow_edges), 2)
    occ = int(np.bincount(s, minlength=4096).astype(bool).sum())
    E.log("재현 검증: B2 점유=%d (기대 %d)" % (occ, EXPECT_OCC_B2))
    if occ != EXPECT_OCC_B2:
        E.log("!!! 점유 불일치 -> split/인코딩 재현 어긋남. 중단."); sys.exit(2)
    E.log("검증 OK.")

    # =========================================================
    # (1) feature별 엔트로피 + 활용률 + 단일feature AUC
    # =========================================================
    E.log("-" * 78)
    E.log("[(1) feature 엔트로피] benign/attack, 활용률=엔트로피/비트수, 단일feature AUC")
    feat_rows = []
    for name, bits in FEATURES:
        nlev = 1 << bits
        lv_tr = np.asarray(feature_levels(ben_tr, name, size_edges, fwd_edges, flow_edges), int)
        lv_te = np.asarray(feature_levels(ben_te, name, size_edges, fwd_edges, flow_edges), int)
        lv_atk = np.asarray(feature_levels(atk, name, size_edges, fwd_edges, flow_edges), int)
        H_ben = entropy_bits(lv_tr, nlev)
        H_atk = entropy_bits(lv_atk, nlev)
        auc = single_feature_auc(lv_tr, lv_te, lv_atk, nlev)
        rec = {"feature": name, "bits": bits, "max_entropy": bits,
               "entropy_benign": H_ben, "utilization": H_ben / bits,
               "entropy_attack": H_atk, "single_feature_auc": auc,
               "occupied_levels_benign": int(np.bincount(lv_tr, minlength=nlev).astype(bool).sum())}
        feat_rows.append(rec)
        E.log("  %-9s bits=%d H_ben=%.4f (활용 %.1f%%) H_atk=%.4f AUC=%.4f 점유레벨=%d/%d"
              % (name, bits, H_ben, 100 * H_ben / bits, H_atk, auc,
                 rec["occupied_levels_benign"], nlev))

    # =========================================================
    # (2) IAT 해상도 곡선
    # =========================================================
    E.log("-" * 78)
    E.log("[(2) IAT 해상도] 버킷 4->64: 엔트로피 / 점유버킷 / 단일feature AUC")
    iat_curves = {}
    for col, tag in [("Fwd IAT Mean", "fwd_iat"), ("Flow IAT Mean", "flow_iat")]:
        rows = []
        for nb in IAT_BUCKET_SWEEP:
            lv_tr, n_eff = iat_quantize(ben_tr[col], ben_tr[col], nb)
            lv_te, _ = iat_quantize(ben_te[col], ben_tr[col], nb)
            lv_atk, _ = iat_quantize(atk[col], ben_tr[col], nb)
            H = entropy_bits(lv_tr, nb)
            occ_b = int(np.bincount(lv_tr, minlength=nb).astype(bool).sum())
            auc = single_feature_auc(lv_tr, lv_te, lv_atk, nb)
            rows.append({"requested_buckets": nb, "effective_buckets": n_eff,
                         "entropy_bits": H, "occupied_buckets": occ_b,
                         "single_feature_auc": auc})
            E.log("  %-9s nb=%-3d eff=%-3d H=%.4f 점유=%-3d AUC=%.4f"
                  % (tag, nb, n_eff, H, occ_b, auc))
        iat_curves[tag] = rows

    # =========================================================
    # (3) 원본 연속값 구조
    # =========================================================
    E.log("-" * 78)
    E.log("[(3) 원본 연속값] nunique / 분위수 / 봉우리 (benign vs attack)")
    cont = {
        "size_PktLenMean": cont_stats(ben_tr["Pkt Len Mean"], atk["Pkt Len Mean"],
                                      "Pkt Len Mean", logspace=False),
        "fwd_iat_log1p": cont_stats(ben_tr["Fwd IAT Mean"], atk["Fwd IAT Mean"],
                                    "Fwd IAT Mean (log1p)", logspace=True),
        "flow_iat_log1p": cont_stats(ben_tr["Flow IAT Mean"], atk["Flow IAT Mean"],
                                     "Flow IAT Mean (log1p)", logspace=True),
    }
    for k, c in cont.items():
        E.log("  %-16s nunique ben=%d atk=%d | 봉우리(거친)=%d | ben med=%.4g atk med=%.4g"
              % (k, c["nunique_benign"], c["nunique_attack"],
                 c["approx_peaks_benign_coarse50"],
                 c["benign_quantiles"]["50"], c["attack_quantiles"]["50"]))

    # --- 저장 ---
    payload = {
        "encoding": "B2 12bit: proto2 syn1 psh1 size4 + FwdIAT2 + FlowIAT2",
        "reproduce_occupied_b2": occ,
        "feature_entropy": feat_rows,
        "iat_resolution_curve": iat_curves,
        "continuous_structure": cont,
        "note": "사실 측정만. 판정(인코딩 vs 데이터 책임)은 웹 AI. QCBM 학습 없음.",
    }
    with open(os.path.join(OUT, "encoding_info_loss.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/encoding_info_loss.json")

    # --- 그림 (3패널) ---
    try:
        fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
        # 패널1: feature 엔트로피 활용률
        names = [r["feature"] for r in feat_rows]
        util = [100 * r["utilization"] for r in feat_rows]
        aucs = [r["single_feature_auc"] for r in feat_rows]
        x = np.arange(len(names))
        ax[0].bar(x, util, color="steelblue")
        for xi, r in zip(x, feat_rows):
            ax[0].text(xi, util[x.tolist().index(xi)] + 1,
                       "%.0f%%\nAUC%.2f" % (100 * r["utilization"], r["single_feature_auc"]),
                       ha="center", va="bottom", fontsize=8)
        ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=30)
        ax[0].set_ylabel("entropy utilization (%)"); ax[0].set_ylim(0, 110)
        ax[0].set_title("(1) feature entropy utilization (+single-feat AUC)")
        ax[0].grid(alpha=.3, axis="y")
        # 패널2: IAT 해상도 곡선 (엔트로피 + AUC)
        for tag, c, col in [("fwd_iat", iat_curves["fwd_iat"], "tab:blue"),
                            ("flow_iat", iat_curves["flow_iat"], "tab:green")]:
            nb = [r["requested_buckets"] for r in c]
            ax[1].plot(nb, [r["entropy_bits"] for r in c], "o-", color=col, label="%s H" % tag)
            ax[1].plot(nb, [r["single_feature_auc"] for r in c], "s--", color=col, alpha=.6,
                       label="%s AUC" % tag)
        ax[1].set_xscale("log", base=2); ax[1].set_xlabel("buckets (log2)")
        ax[1].set_title("(2) IAT resolution: entropy(bits) & single-feat AUC")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, which="both")
        # 패널3: 원본 연속값 히스토그램 (size)
        c = cont["size_PktLenMean"]; b = np.array(c["bins"]); ctr = 0.5 * (b[1:] + b[:-1])
        ax[2].plot(ctr, c["hist_benign"], color="steelblue", label="benign")
        ax[2].plot(ctr, c["hist_attack"], color="crimson", label="attack")
        ax[2].set_title("(3) raw Pkt Len Mean (size) density"); ax[2].set_xlabel("Pkt Len Mean")
        ax[2].legend(); ax[2].grid(alpha=.3)
        plt.tight_layout()
        fp = os.path.join(OUT, "encoding_info_loss.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
