#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
교차 데이터셋 재현 진단 — joint>marginal·복잡도·적합도탐지 해리가 UNSW에서도 성립하는가
================================================================================
배경: 모든 결론이 CIC-IDS2018 한 데이터셋에서만 나왔다. 독립 UNSW-NB15에서 같은 방법론·같은
feature 세트로 재현되는지 본다. UNSW엔 TCP 플래그(SYN/PSH)가 없으므로(확인됨), 양쪽 공통
feature(proto+size+fwd_iat+flow_iat=10bit)만으로 인코딩하고, 공정 비교를 위해 CIC도 동일
세트로 재계산한다(플래그 제거가 교란변수 안 되도록). QCBM 학습 없음(빈도/SVD). 판정은 웹 AI.

공통 feature 매핑 (억지 매핑 없음; 불확실은 명시):
  proto    : CIC Protocol(숫자)        / UNSW proto(문자열 tcp/udp..)  -> tcp=6,udp=17,기타=-1 후 proto_code  [정확(의미)]
  size(4)  : CIC Pkt Len Mean          / UNSW smean(src 평균 패킷크기)                                      [근사]
  fwd_iat  : CIC Fwd IAT Mean          / UNSW sinpkt(Source interpacket arrival, mSec)                      [근사~정확]
  flow_iat : CIC Flow IAT Mean         / UNSW dinpkt(Dest interpacket arrival, mSec)                        [불확실 — 전체흐름 단일대응 없음]
  ※ syn/psh : UNSW 부재 -> 미사용. 따라서 CIC도 10bit로 재계산(기존 12bit 결과 직접 비교 불가).

재사용(기존 소스 무수정, import): experiment2(proto_code, discretize, size_edges_from_benign,
  empirical_dist, tv_dist, auc_score), train_qcbm_iat_b2(log1p_quantile_edges, iat_to_bits,
  append_bits, marginal_dist). 분위수 경계는 각 데이터 benign train(seed=7,0.7)에서 생성(구조 동일).
  ★ 전수(benign 전체/attack 전체), 샘플링 없음. raw MWU 유지(0.5 미만도 그대로).
"""
import os
import sys
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E          # noqa: E402  proto_code/discretize/size_edges/empirical_dist/tv_dist/auc_score
import train_qcbm_iat_b2 as TB2  # noqa: E402  log1p_quantile_edges/iat_to_bits/append_bits/marginal_dist

OUT = "results2"
NQ = 10
NSTATES = 1 << NQ                # 1024
SIDE = 32                        # 32x32 reshape (SVD)
IAT_BITS = 2
IAT_BUCKETS = 1 << IAT_BITS      # 4
SIZE_BUCKETS = 16
EPS = 1e-12
SPEC_PS = {"features": [("proto", 2), ("size", 4)]}     # 6bit (MSB) — 플래그 제외
FEATURES = [("proto", 2), ("size", 4), ("fwd_iat", 2), ("flow_iat", 2)]  # 10bit
CUM_R = [1, 2, 3, 5, 10]
TV_TARGETS = [0.05, 0.01]
RMAX = 20

CANON = ["Protocol", "Pkt Len Mean", "Fwd IAT Mean", "Flow IAT Mean"]


# ---------------------------------------------------------------------------
# 로더 (전수, 캐노니컬 컬럼으로 변환 -> 기존 discretize 재사용)
# ---------------------------------------------------------------------------
def load_cic(csv, attack_label, tag):
    cols = CANON + ["Label"]
    E.log("[%s] CSV 로드: %s" % (tag, csv))
    df = pd.read_csv(csv, usecols=cols, low_memory=False)
    for c in CANON:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=cols)
    lab = df["Label"].astype(str).str.strip().str.lower()
    ben = df[lab == "benign"][CANON].reset_index(drop=True)
    atk = df[lab == attack_label][CANON].reset_index(drop=True)
    E.log("[%s] dropna=%d | benign=%d  %s=%d" % (tag, n0 - len(df), len(ben), attack_label, len(atk)))
    if len(ben) == 0 or len(atk) == 0:
        E.log("[%s] !!! benign/attack 0 -> 중단." % tag); sys.exit(2)
    return ben, {attack_label: atk}, (n0 - len(df))


def load_unsw(tag):
    src = ["proto", "smean", "sinpkt", "dinpkt", "attack_cat", "label"]
    paths = ["data/unsw_nb15_training-set.csv", "data/unsw_nb15_testing-set.csv"]
    E.log("[%s] CSV 로드: %s" % (tag, paths))
    parts = [pd.read_csv(p, usecols=src, low_memory=False) for p in paths]
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    proto_num = df["proto"].astype(str).str.lower().map({"tcp": 6, "udp": 17}).fillna(-1).astype(int)
    can = pd.DataFrame({
        "Protocol": proto_num,
        "Pkt Len Mean": pd.to_numeric(df["smean"], errors="coerce"),
        "Fwd IAT Mean": pd.to_numeric(df["sinpkt"], errors="coerce"),
        "Flow IAT Mean": pd.to_numeric(df["dinpkt"], errors="coerce"),
        "attack_cat": df["attack_cat"].astype(str).str.strip(),
        "label": pd.to_numeric(df["label"], errors="coerce"),
    })
    can = can.dropna(subset=CANON + ["label"])
    ben = can[can["attack_cat"].str.lower() == "normal"][CANON].reset_index(drop=True)
    attacks = {}
    for cat in sorted(can.loc[can["attack_cat"].str.lower() != "normal", "attack_cat"].unique()):
        attacks[cat] = can[can["attack_cat"] == cat][CANON].reset_index(drop=True)
    attacks["ALL"] = can[can["attack_cat"].str.lower() != "normal"][CANON].reset_index(drop=True)
    E.log("[%s] benign(Normal)=%d | 공격종류=%d | 공격 전체=%d"
          % (tag, len(ben), len(attacks) - 1, len(attacks["ALL"])))
    return ben, attacks, (n0 - len(df))


def split_benign(ben):
    rng = np.random.default_rng(7)                 # 기존과 동일 seed/정책
    idx = rng.permutation(len(ben)); cut = int(0.7 * len(ben))
    return ben.iloc[idx[:cut]].reset_index(drop=True), ben.iloc[idx[cut:]].reset_index(drop=True)


# ---------------------------------------------------------------------------
# 인코딩/지표 (기존 함수 재사용)
# ---------------------------------------------------------------------------
def make_edges(ben_tr):
    size_edges = E.size_edges_from_benign(ben_tr, SIZE_BUCKETS)
    fwd_edges = TB2.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], IAT_BUCKETS)
    flow_edges = TB2.log1p_quantile_edges(ben_tr["Flow IAT Mean"], IAT_BUCKETS)
    return size_edges, fwd_edges, flow_edges


def encode(df, size_edges, fwd_edges, flow_edges):
    s = E.discretize(df, SPEC_PS, size_edges)                                  # proto2+size4 (6bit MSB)
    s = TB2.append_bits(s, TB2.iat_to_bits(df["Fwd IAT Mean"], fwd_edges), IAT_BITS)
    s = TB2.append_bits(s, TB2.iat_to_bits(df["Flow IAT Mean"], flow_edges), IAT_BITS)
    return s


def feature_levels(df, name, size_edges, fwd_edges, flow_edges):
    if name in ("proto", "size"):
        bits = dict(FEATURES)[name]
        return np.asarray(E.discretize(df, {"features": [(name, bits)]}, size_edges), int)
    if name == "fwd_iat":
        return np.asarray(TB2.iat_to_bits(df["Fwd IAT Mean"], fwd_edges), int)
    if name == "flow_iat":
        return np.asarray(TB2.iat_to_bits(df["Flow IAT Mean"], flow_edges), int)
    raise ValueError(name)


def entropy_bits(levels, nlevels):
    c = np.bincount(np.asarray(levels, int), minlength=nlevels).astype(float)
    p = c / c.sum(); p = p[p > 0]
    return float(-np.sum(p * np.log2(p)))


def auc_of_dist(p, st_te, st_atk):
    """benign_test(0) vs attack(1), s=-log p, raw MWU."""
    p = np.asarray(p, float)
    s_te = -np.log(p[st_te] + EPS); s_atk = -np.log(p[st_atk] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def single_feature_auc(tr_lv, te_lv, atk_lv, nlevels):
    p = E.empirical_dist(tr_lv, nlevels)
    s_te = -np.log(p[te_lv] + EPS); s_atk = -np.log(p[atk_lv] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def normalize_clip(p):
    p = np.clip(np.asarray(p, float), 0, None); s = p.sum()
    return p / s if s > 0 else p


def svd_benign(q):
    """benign q_train 복잡도: 32x32 SVD 누적에너지 + 저랭크 TV reach."""
    M = np.asarray(q, float).reshape(SIDE, SIDE)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    energy = S ** 2; tot = float(energy.sum()) or 1.0
    cum_energy = {str(r): float(energy[:r].sum() / tot) for r in CUM_R}
    tv_curve = []
    recon = {}
    for r in range(1, RMAX + 1):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        qa = normalize_clip(approx.reshape(-1))
        recon[r] = qa
        tv_curve.append({"rank": r, "params": 2 * SIDE * r, "tv": float(E.tv_dist(qa, q))})
    reach = {}
    for thr in TV_TARGETS:
        ok = [row["rank"] for row in tv_curve if row["tv"] <= thr]
        reach["tv<=%.2f" % thr] = (min(ok) if ok else None)
    return {"singular_values_norm": (S[:SIDE] / (S[0] if S[0] > 0 else 1.0)).tolist(),
            "cum_energy": cum_energy, "tv_curve": tv_curve, "reach_rank": reach}, recon


def dissociation(recon, st_te, st_atk, tv_curve):
    """저랭크 근사 r별 (TV, AUC). 적합 좋아질수록(TV↓) AUC 따라오는가?"""
    rows = []
    for row in tv_curve:
        r = row["rank"]
        rows.append({"rank": r, "tv": row["tv"], "auc": auc_of_dist(recon[r], st_te, st_atk)})
    tvs = np.array([x["tv"] for x in rows]); aucs = np.array([x["auc"] for x in rows])
    ranks = np.array([x["rank"] for x in rows])
    sp_tv_auc = float(spearmanr(tvs, aucs).correlation)      # +면 적합 나쁠수록 AUC↑ (해리/역)
    sp_rank_auc = float(spearmanr(ranks, aucs).correlation)  # +면 랭크(적합)↑ 시 AUC↑
    return {"curve": rows, "spearman_tv_auc": sp_tv_auc, "spearman_rank_auc": sp_rank_auc,
            "auc_lowrank_max": float(aucs.max()), "auc_lowrank_min": float(aucs.min())}


# ---------------------------------------------------------------------------
def diagnose_dataset(tag, ben, attacks, dropped):
    ben_tr, ben_te = split_benign(ben)
    size_edges, fwd_edges, flow_edges = make_edges(ben_tr)
    st_tr = encode(ben_tr, size_edges, fwd_edges, flow_edges)
    st_te = encode(ben_te, size_edges, fwd_edges, flow_edges)
    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    svd, recon = svd_benign(q_train)
    marg = TB2.marginal_dist(st_tr, NQ)

    # benign feature 엔트로피/활용률 (분포 진단)
    feat_info = []
    for name, bits in FEATURES:
        nlev = 1 << bits
        lv = feature_levels(ben_tr, name, size_edges, fwd_edges, flow_edges)
        feat_info.append({"feature": name, "bits": bits,
                          "entropy_benign": entropy_bits(lv, nlev),
                          "utilization": entropy_bits(lv, nlev) / bits})

    E.log("-" * 78)
    E.log("[%s] benign 점유=%d/%d (%.2f%%) | SVD 누적E(r3)=%.3f reach(TV<=0.05)=%s"
          % (tag, occ, NSTATES, 100 * occ / NSTATES, svd["cum_energy"]["3"],
             svd["reach_rank"]["tv<=0.05"]))

    rows = []
    for aname, atk in attacks.items():
        st_atk = encode(atk, size_edges, fwd_edges, flow_edges)
        auc_emp = auc_of_dist(q_train, st_te, st_atk)
        auc_marg = auc_of_dist(marg, st_te, st_atk)
        feats = []
        for name, bits in FEATURES:
            nlev = 1 << bits
            feats.append({"feature": name,
                          "auc": single_feature_auc(
                              feature_levels(ben_tr, name, size_edges, fwd_edges, flow_edges),
                              feature_levels(ben_te, name, size_edges, fwd_edges, flow_edges),
                              feature_levels(atk, name, size_edges, fwd_edges, flow_edges), nlev)})
        max_single = max(feats, key=lambda r: r["auc"])
        diss = dissociation(recon, st_te, st_atk, svd["tv_curve"])
        row = {
            "dataset": tag, "attack": aname, "n_attack": int(len(atk)),
            "auc_emp_joint": auc_emp, "auc_marginal": auc_marg,
            "joint_gain": auc_emp - auc_marg,
            "max_single_feature": max_single["feature"], "max_single_auc": max_single["auc"],
            "single_feature_auc": {f["feature"]: f["auc"] for f in feats},
            "dissociation": diss,
        }
        rows.append(row)
        E.log("  [%s/%-14s] n=%-7d emp=%.4f marg=%.4f gain=%+.4f | max단일 %s=%.4f | "
              "저랭크AUCmax=%.4f sp(tv,auc)=%+.2f"
              % (tag, aname, len(atk), auc_emp, auc_marg, auc_emp - auc_marg,
                 max_single["feature"], max_single["auc"],
                 diss["auc_lowrank_max"], diss["spearman_tv_auc"]))
    return {"tag": tag, "n_benign_train": len(ben_tr), "n_benign_test": len(ben_te),
            "rows_dropped_na": int(dropped), "occupied": occ, "occ_frac": occ / NSTATES,
            "svd": svd, "feature_entropy": feat_info, "attacks": rows}


def plot_all(datasets, fp):
    # 비교용 평탄화
    flat = []
    for ds in datasets:
        for r in ds["attacks"]:
            flat.append((r["dataset"], r["attack"], r["joint_gain"], r["auc_emp_joint"],
                         r["auc_marginal"], r["max_single_auc"], r["dissociation"]["spearman_tv_auc"]))
    fig, ax = plt.subplots(1, 3, figsize=(20, 6.2))
    # 패널1: SVD 특이값 감소 (benign 복잡도)
    colors = {"CIC_HOIC": "steelblue", "CIC_Bot": "crimson", "UNSW": "darkgreen"}
    for ds in datasets:
        sv = np.array(ds["svd"]["singular_values_norm"])
        ax[0].plot(np.arange(1, len(sv) + 1), sv, "o-", ms=3,
                   color=colors.get(ds["tag"], "gray"), label=ds["tag"])
    ax[0].set_yscale("log"); ax[0].set_xlabel("singular value index")
    ax[0].set_ylabel("normalized singular value (log)")
    ax[0].set_title("benign joint complexity: SVD decay (10-feature set)")
    ax[0].legend(); ax[0].grid(alpha=.3, which="both")
    # 패널2: joint gain (emp-marg) per (dataset, attack)
    labels = ["%s/%s" % (f[0].replace("CIC_", ""), f[1]) for f in flat]
    gains = [f[2] for f in flat]
    cols = [colors.get(f[0], "gray") for f in flat]
    y = np.arange(len(labels))
    ax[1].barh(y, gains, color=cols, alpha=.8, edgecolor="k")
    ax[1].axvline(0, color="k", lw=1)
    ax[1].set_yticks(y); ax[1].set_yticklabels(labels, fontsize=7); ax[1].invert_yaxis()
    ax[1].set_xlabel("joint gain = emp_joint AUC - marginal AUC")
    ax[1].set_title("joint>marginal reproduction (>0 = joint helps)")
    ax[1].grid(alpha=.3, axis="x")
    # 패널3: 해리 — max단일AUC vs emp_joint AUC
    for f in flat:
        ax[2].scatter(f[5], f[3], color=colors.get(f[0], "gray"), s=45, alpha=.8, edgecolor="k")
        ax[2].annotate(f[1][:4], (f[5], f[3]), fontsize=6, alpha=.7)
    lim = [0.4, 1.0]
    ax[2].plot(lim, lim, "k--", alpha=.4, label="emp=maxSingle")
    ax[2].set_xlim(lim); ax[2].set_ylim(lim)
    ax[2].set_xlabel("max single-feature AUC"); ax[2].set_ylabel("emp_joint AUC")
    ax[2].set_title("joint vs best single feature (above diag = joint adds)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
    plt.suptitle("Cross-dataset reproduction: CIC(HOIC/Bot) vs UNSW-NB15, common 10-feature set (no QCBM)")
    plt.tight_layout()
    plt.savefig(fp, dpi=120); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("#" * 78)
    E.log("교차 데이터셋 재현 진단 — CIC(HOIC/Bot) vs UNSW, 공통 10-feature 세트 (학습 없음)")
    E.log("#" * 78)

    datasets = []
    # CIC (동일 10-feature 세트로 재계산: 플래그 제외)
    ben, atks, drp = load_cic("data/cic_0221_ddos.csv", "ddos attack-hoic", "CIC_HOIC")
    datasets.append(diagnose_dataset("CIC_HOIC", ben, atks, drp))
    ben, atks, drp = load_cic("data/cic_bot_0203.csv", "bot", "CIC_Bot")
    datasets.append(diagnose_dataset("CIC_Bot", ben, atks, drp))
    # UNSW (공격별 + 전체)
    ben, atks, drp = load_unsw("UNSW")
    datasets.append(diagnose_dataset("UNSW", ben, atks, drp))

    # 비교표 로그
    E.log("=" * 78); E.log("[교차 비교표] (공통 10-feature, 전수)")
    E.log("  %-22s %-8s %-8s %-9s %-12s %-10s" %
          ("dataset/attack", "empAUC", "margAUC", "jointGain", "maxSingle", "spTVAUC"))
    for ds in datasets:
        for r in ds["attacks"]:
            E.log("  %-22s %-8.4f %-8.4f %-+9.4f %-12s %-+10.2f" % (
                "%s/%s" % (ds["tag"], r["attack"]), r["auc_emp_joint"], r["auc_marginal"],
                r["joint_gain"], "%s%.3f" % (r["max_single_feature"][:4], r["max_single_auc"]),
                r["dissociation"]["spearman_tv_auc"]))

    payload = {
        "feature_set": {
            "bits": 10, "nstates": NSTATES,
            "features": [{"feature": "proto", "bits": 2, "cic": "Protocol", "unsw": "proto(tcp=6,udp=17,else=-1)",
                          "status": "정확(의미)"},
                         {"feature": "size", "bits": 4, "cic": "Pkt Len Mean", "unsw": "smean", "status": "근사"},
                         {"feature": "fwd_iat", "bits": 2, "cic": "Fwd IAT Mean", "unsw": "sinpkt", "status": "근사~정확"},
                         {"feature": "flow_iat", "bits": 2, "cic": "Flow IAT Mean", "unsw": "dinpkt",
                          "status": "불확실(전체흐름 단일대응 없음; 대안 rate/sinpkt+dinpkt)"}],
            "excluded": ["syn", "psh (UNSW 부재 -> 양쪽 제외; CIC도 10bit 재계산)"],
        },
        "encoding": "proto2+size4(linear 16버킷)+fwd_iat2(log1p 4버킷)+flow_iat2(log1p 4버킷)=10bit(1024)",
        "split": {"seed": 7, "benign_train_frac": 0.7, "note": "각 데이터 benign train에서 경계 생성(재사용 안 함)"},
        "eval": "s(x)=-log p(x), attack=positive, raw MWU (experiment2.auc_score)",
        "sampling": "전수(benign 전체/attack 전체). 샘플링 없음.",
        "datasets": datasets,
        "note": "사실 측정만. 판정은 웹 AI. QCBM 학습 없음. 기존 소스/데이터/결과 무수정.",
    }
    with open(os.path.join(OUT, "cross_dataset_diagnosis.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/cross_dataset_diagnosis.json")

    try:
        fp = os.path.join(OUT, "cross_dataset_diagnosis.png")
        plot_all(datasets, fp)
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)
    E.log("DONE")


if __name__ == "__main__":
    main()
