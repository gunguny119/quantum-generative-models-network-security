#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
공격 분포 복잡도 진단 — HOIC vs Bot vs Infiltration (학습 없음, 빈도/SVD 기반)
================================================================================
목적:
  HOIC(기존, 단순: 점유 2.3%, psh 단독 AUC 0.878 지배, 저랭크)와 비교해
  Bot(필수)·Infiltration 분포가 더 "고랭크·조밀·단일feature 지배 약함"인지
  (= 양자가 이길 가능성 있는 복잡도인지) 학습 없이 측정한다. 판정은 웹 AI.

방법(각 공격마다, attack=positive, benign=정상):
  - benign seed=7로 0.7/0.3 분할. size(16버킷)/IAT(log1p 4버킷) 경계는 각 데이터 benign train에서 생성.
  - B2 인코딩(12bit/4096상태): proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2.
  - 지표1 joint이득 : emp_joint AUC vs marginal AUC (s=-log p, raw MWU).
  - 지표2 분포 랭크 : benign q_train 64x64 reshape -> SVD. 특이값 감소/누적에너지(r=1,2,3,5,10),
                      저랭크 근사 TV(r) 곡선, emp_joint 근접(TV<=0.05/0.01) 도달 r (=고전 저랭크 비용).
  - 지표3 점유율   : benign train 점유 상태수 / 4096.
  - 지표4 단일지배 : feature(proto/syn/psh/size/fwd_iat/flow_iat) 단독 AUC, 최대값.
  - 지표5 엔트로피 : feature별 엔트로피(bits) + 활용률.

재사용(기존 소스 무수정, import만):
  experiment2: proto_code, size_edges_from_benign, discretize, empirical_dist, tv_dist,
               auc_score, anomaly_eval.
  train_qcbm_iat_b2: log1p_quantile_edges, iat_to_bits, append_bits, marginal_dist, eval_dist.
HOIC 재계산이 기존값(점유 2.3%, psh 0.878, emp_joint 0.9258, marginal 0.8741)과 불일치하면 중단.
QCBM 학습 없음. 새 dependency 없음.
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

import experiment2 as E          # noqa: E402  discretize/size_edges/empirical_dist/tv_dist/auc_score/anomaly_eval
import train_qcbm_iat_b2 as TB2  # noqa: E402  log1p_quantile_edges/iat_to_bits/append_bits/marginal_dist/eval_dist

OUT = "results2"
NQ = 12
NSTATES = 1 << NQ                # 4096
SIDE = 64                        # 64x64 reshape
IAT_BITS = 2
IAT_BUCKETS = 1 << IAT_BITS      # 4
SIZE_BUCKETS = 16
EPS = 1e-12
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
FEATURES = [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4),
            ("fwd_iat", 2), ("flow_iat", 2)]
CUM_R = [1, 2, 3, 5, 10]
TV_TARGETS = [0.05, 0.01]
RMAX_TV = 20

# (tag, csv, benign_label, attack_label)
DATASETS = [
    ("HOIC",         "data/cic_0221_ddos.csv", "benign", "ddos attack-hoic"),
    ("Bot",          "data/cic_bot_0203.csv",  "benign", "bot"),
    ("Infiltration", "data/cic_thursday.csv",  "benign", "infilteration"),
]
# HOIC 재계산 정합 기준(기존 results2/qcbm_iat_b2.json, encoding_info_loss.json)
HOIC_REF = {"occupied": 94, "marginal_auc": 0.8741, "emp_joint_auc": 0.9258,
            "psh_single_auc": 0.8779}
TOL = 0.0015


# ---------------------------------------------------------------------------
def load_split(csv, benign_label, attack_label, tag):
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean",
            "Fwd IAT Mean", "Flow IAT Mean", "Label"]
    E.log("[%s] CSV 로드: %s" % (tag, csv))
    df = pd.read_csv(csv, usecols=cols, low_memory=False)
    for c in cols[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=cols)
    dropped = n0 - len(df)
    lab = df["Label"].astype(str).str.strip().str.lower()
    ben = df[lab == benign_label].reset_index(drop=True)
    atk = df[lab == attack_label].reset_index(drop=True)
    E.log("[%s] dropna 제거=%d | benign=%d  %s=%d" % (tag, dropped, len(ben), attack_label, len(atk)))
    if len(ben) == 0 or len(atk) == 0:
        E.log("[%s] !!! benign/attack 표본 0 -> 중단." % tag); sys.exit(2)
    rng = np.random.default_rng(7)               # 기존과 동일 seed
    idx = rng.permutation(len(ben)); cut = int(0.7 * len(ben))
    ben_tr = ben.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = ben.iloc[idx[cut:]].reset_index(drop=True)
    return ben_tr, ben_te, atk, dropped


def make_edges(ben_tr):
    size_edges = E.size_edges_from_benign(ben_tr, SIZE_BUCKETS)
    fwd_edges = TB2.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], IAT_BUCKETS)
    flow_edges = TB2.log1p_quantile_edges(ben_tr["Flow IAT Mean"], IAT_BUCKETS)
    return size_edges, fwd_edges, flow_edges


def encode_b2(df, size_edges, fwd_edges, flow_edges):
    s = E.discretize(df, SPEC8, size_edges)                                  # 8bit(MSB)
    s = TB2.append_bits(s, TB2.iat_to_bits(df["Fwd IAT Mean"], fwd_edges), IAT_BITS)
    s = TB2.append_bits(s, TB2.iat_to_bits(df["Flow IAT Mean"], flow_edges), IAT_BITS)
    return s


def feature_levels(df, name, size_edges, fwd_edges, flow_edges):
    if name in ("proto", "syn", "psh", "size"):
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


def single_feature_auc(tr_lv, te_lv, atk_lv, nlevels):
    """그 feature 하나의 marginal p_benign(level)로 s=-log p, attack=positive AUC."""
    p = E.empirical_dist(tr_lv, nlevels)
    s_te = -np.log(p[te_lv] + EPS)
    s_atk = -np.log(p[atk_lv] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def normalize_clip(p):
    p = np.clip(np.asarray(p, float), 0, None)
    s = p.sum()
    return p / s if s > 0 else p


def svd_analysis(q):
    """q_train(4096) -> 64x64 -> SVD. 특이값/누적에너지/저랭크 TV(r) 곡선."""
    M = np.asarray(q, float).reshape(SIDE, SIDE)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    energy = S ** 2
    tot = float(energy.sum()) if energy.sum() > 0 else 1.0
    cum_energy = {str(r): float(energy[:r].sum() / tot) for r in CUM_R}
    tv_curve = []
    for r in range(1, RMAX_TV + 1):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        qa = normalize_clip(approx.reshape(-1))
        tv_curve.append({"rank": r, "params": 128 * r, "tv": float(E.tv_dist(qa, q))})
    reach = {}
    for thr in TV_TARGETS:
        ok = [row["rank"] for row in tv_curve if row["tv"] <= thr]
        reach["tv<=%.2f" % thr] = (min(ok) if ok else None)
    return {"singular_values": S[:32].tolist(),
            "singular_values_norm": (S[:32] / (S[0] if S[0] > 0 else 1.0)).tolist(),
            "cum_energy": cum_energy, "tv_curve": tv_curve, "reach_rank": reach}


# ---------------------------------------------------------------------------
def diagnose_one(tag, csv, benign_label, attack_label):
    E.log("=" * 78)
    E.log("[%s] 복잡도 진단 시작" % tag)
    ben_tr, ben_te, atk, dropped = load_split(csv, benign_label, attack_label, tag)
    size_edges, fwd_edges, flow_edges = make_edges(ben_tr)

    st_tr = encode_b2(ben_tr, size_edges, fwd_edges, flow_edges)
    st_te = encode_b2(ben_te, size_edges, fwd_edges, flow_edges)
    st_atk = encode_b2(atk, size_edges, fwd_edges, flow_edges)

    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    occ_frac = occ / NSTATES

    # 지표1: joint 이득 (기존 b2 baseline과 동일 경로: eval_dist/marginal_dist)
    auc_emp = TB2.eval_dist(q_train, st_te, st_atk, "%s/emp_joint" % tag)["auc"]
    auc_marg = TB2.eval_dist(TB2.marginal_dist(st_tr, NQ), st_te, st_atk, "%s/marginal" % tag)["auc"]
    joint_gain = auc_emp - auc_marg

    # 지표4/5: 단일 feature AUC + 엔트로피
    feat_rows = []
    for name, bits in FEATURES:
        nlev = 1 << bits
        lv_tr = feature_levels(ben_tr, name, size_edges, fwd_edges, flow_edges)
        lv_te = feature_levels(ben_te, name, size_edges, fwd_edges, flow_edges)
        lv_atk = feature_levels(atk, name, size_edges, fwd_edges, flow_edges)
        H = entropy_bits(lv_tr, nlev)
        auc = single_feature_auc(lv_tr, lv_te, lv_atk, nlev)
        feat_rows.append({"feature": name, "bits": bits, "entropy_benign": H,
                          "utilization": H / bits, "single_feature_auc": auc})
    max_single = max(feat_rows, key=lambda r: r["single_feature_auc"])

    # 지표2: SVD 랭크
    svd = svd_analysis(q_train)

    E.log("[%s] 점유=%d/%d (%.2f%%) | emp_joint=%.4f marginal=%.4f (이득 %+.4f) | "
          "최대단일 %s=%.4f | SVD 누적E(r3)=%.3f reach(TV<=0.05)=%s"
          % (tag, occ, NSTATES, 100 * occ_frac, auc_emp, auc_marg, joint_gain,
             max_single["feature"], max_single["single_feature_auc"],
             svd["cum_energy"]["3"], svd["reach_rank"]["tv<=0.05"]))

    return {
        "tag": tag, "csv": csv, "attack_label": attack_label,
        "n_benign_train": len(ben_tr), "n_benign_test": len(ben_te),
        "n_attack": len(atk), "rows_dropped_na": int(dropped),
        "occupied": occ, "occ_frac": occ_frac,
        "auc_emp_joint": auc_emp, "auc_marginal": auc_marg, "joint_gain": joint_gain,
        "feature_entropy": feat_rows,
        "max_single_feature": {"feature": max_single["feature"],
                               "auc": max_single["single_feature_auc"]},
        "svd": svd,
    }


def check_hoic(res):
    """HOIC 재계산 정합 검증. 불일치 시 중단."""
    psh = next(r for r in res["feature_entropy"] if r["feature"] == "psh")["single_feature_auc"]
    checks = [
        ("occupied", res["occupied"], HOIC_REF["occupied"], 0),
        ("marginal_auc", res["auc_marginal"], HOIC_REF["marginal_auc"], TOL),
        ("emp_joint_auc", res["auc_emp_joint"], HOIC_REF["emp_joint_auc"], TOL),
        ("psh_single_auc", psh, HOIC_REF["psh_single_auc"], TOL),
    ]
    ok = True
    E.log("-" * 78); E.log("[HOIC 재계산 정합 검증]")
    detail = []
    for name, got, ref, tol in checks:
        passed = abs(got - ref) <= tol
        ok = ok and passed
        E.log("  %-16s 재계산=%.4f 기존=%.4f %s"
              % (name, got, ref, "OK" if passed else "!!! 불일치"))
        detail.append({"metric": name, "recomputed": got, "reference": ref, "pass": bool(passed)})
    return ok, detail


def plot_all(results, fp):
    colors = {"HOIC": "steelblue", "Bot": "crimson", "Infiltration": "darkgreen"}
    fig, ax = plt.subplots(1, 3, figsize=(19, 5.5))
    # 패널1: SVD 특이값 감소(정규화, log)
    for r in results:
        sv = np.array(r["svd"]["singular_values_norm"])
        ax[0].plot(np.arange(1, len(sv) + 1), sv, "o-", ms=3,
                   color=colors.get(r["tag"], "gray"), label=r["tag"])
    ax[0].set_yscale("log"); ax[0].set_xlabel("singular value index")
    ax[0].set_ylabel("normalized singular value (log)")
    ax[0].set_title("(2) SVD singular-value decay (low-rank=fast / high-rank=slow)")
    ax[0].legend(); ax[0].grid(alpha=.3, which="both")
    # 패널2: 저랭크 TV(r)
    for r in results:
        tc = r["svd"]["tv_curve"]
        ax[1].plot([x["rank"] for x in tc], [x["tv"] for x in tc], "o-", ms=3,
                   color=colors.get(r["tag"], "gray"), label=r["tag"])
    for thr in TV_TARGETS:
        ax[1].axhline(thr, color="gray", ls="--", alpha=.4)
    ax[1].set_yscale("log"); ax[1].set_xlabel("SVD rank r (params=128r)")
    ax[1].set_ylabel("low-rank TV vs q_train (log)")
    ax[1].set_title("(2) classical low-rank cost: TV vs rank")
    ax[1].legend(); ax[1].grid(alpha=.3, which="both")
    # 패널3: 스칼라 지표 그룹 막대 (점유%/최대단일AUC/joint이득)
    tags = [r["tag"] for r in results]
    occ = [100 * r["occ_frac"] for r in results]
    maxauc = [r["max_single_feature"]["auc"] for r in results]
    jgain = [r["joint_gain"] for r in results]
    x = np.arange(len(tags)); w = 0.6
    a2 = ax[2]; a3 = a2.twinx()
    a2.bar(x - w / 3, occ, w / 3, color="slategray", label="occupancy %")
    a3.bar(x, maxauc, w / 3, color="orange", label="max single-feat AUC")
    a3.bar(x + w / 3, jgain, w / 3, color="purple", label="joint gain (emp-marg)")
    a2.set_xticks(x); a2.set_xticklabels(tags)
    a2.set_ylabel("occupancy % (gray)")
    a3.set_ylabel("AUC / joint-gain (orange/purple)")
    a2.set_title("occupancy / max single-feat AUC / joint gain")
    h2, l2 = a2.get_legend_handles_labels(); h3, l3 = a3.get_legend_handles_labels()
    a2.legend(h2 + h3, l2 + l3, fontsize=8, loc="upper center")
    a2.grid(alpha=.3, axis="y")
    plt.suptitle("Attack distribution complexity: HOIC vs Bot vs Infiltration (no training)")
    plt.tight_layout()
    plt.savefig(fp, dpi=130); plt.close()


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("#" * 78)
    E.log("공격 분포 복잡도 진단 (HOIC vs Bot vs Infiltration) — 학습 없음")
    E.log("#" * 78)

    results = []
    hoic_check = None
    for tag, csv, bl, al in DATASETS:
        if not os.path.exists(csv):
            E.log("[%s] 데이터 없음: %s -> 스킵." % (tag, csv))
            continue
        res = diagnose_one(tag, csv, bl, al)
        results.append(res)
        if tag == "HOIC":
            ok, detail = check_hoic(res)
            hoic_check = {"pass": bool(ok), "detail": detail}
            if not ok:
                E.log("!!! HOIC 재계산이 기존값과 불일치 -> 분석 신뢰성 전제 깨짐. 중단.")
                with open(os.path.join(OUT, "attack_complexity.json"), "w") as f:
                    json.dump({"hoic_consistency": hoic_check, "results": results,
                               "aborted": "HOIC 재계산 불일치"}, f, indent=2, ensure_ascii=False)
                sys.exit(3)

    # 비교표 로그
    E.log("=" * 78)
    E.log("[복잡도 비교표]")
    hdr = "  %-13s %-9s %-9s %-9s %-11s %-9s %-9s" % (
        "attack", "occ%", "empAUC", "margAUC", "jointGain", "maxSingl", "reachTV05")
    E.log(hdr)
    for r in results:
        E.log("  %-13s %-9.2f %-9.4f %-9.4f %-+11.4f %-9s %-9s" % (
            r["tag"], 100 * r["occ_frac"], r["auc_emp_joint"], r["auc_marginal"],
            r["joint_gain"],
            "%s%.3f" % (r["max_single_feature"]["feature"][:3], r["max_single_feature"]["auc"]),
            str(r["svd"]["reach_rank"]["tv<=0.05"])))

    payload = {
        "encoding": "B2 12bit: proto2 syn1 psh1 size4 + FwdIAT2 + FlowIAT2 (4096)",
        "split": {"seed": 7, "benign_train_frac": 0.7},
        "edges_note": "size 16버킷 + IAT log1p 4버킷, 각 데이터 benign train에서 생성(HOIC 경계 재사용 안 함)",
        "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (experiment2.auc_score)",
        "metrics_def": {
            "joint_gain": "emp_joint AUC - marginal AUC",
            "occ_frac": "benign train 점유 상태수 / 4096",
            "max_single_feature": "feature 6개 단독 AUC 중 최대 (단일지배 강도)",
            "svd_cum_energy": "64x64 SVD 상위 r 특이값^2 누적비율 (r=1,2,3,5,10)",
            "svd_reach_rank": "저랭크 근사 TV가 임계 이하 되는 최소 r (params=128r; 고전 저랭크 비용)",
        },
        "hoic_consistency": hoic_check,
        "results": results,
        "note": "사실 측정만. 양자 베팅 판정은 웹 AI. QCBM 학습 없음. 새 dependency 없음.",
    }
    with open(os.path.join(OUT, "attack_complexity.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/attack_complexity.json")

    try:
        fp = os.path.join(OUT, "attack_complexity.png")
        plot_all(results, fp)
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
