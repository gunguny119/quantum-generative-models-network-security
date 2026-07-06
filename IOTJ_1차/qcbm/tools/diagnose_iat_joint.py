#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
시간축(IAT) feature 추가 후 HOIC의 joint vs marginal 재측정 (학습 없음)
=====================================================================
가설 마지막 점검:
  핵심 가설 "joint > marginal"이 지금까지 미지지(Infiltration·DDoS 모두 joint 이득 ~0).
  기존 4 feature(proto/syn/psh/size)는 모두 "패킷 자체" 속성이라 상호작용이 약했을 수 있다.
  시간축(IAT)을 추가하면 "패킷 속성 × 시간 간격"의 다른 축 조합이 생겨 joint 이득이 나타나는지 본다.
  대상: HOIC (emp_joint AUC 0.88, 천장 아님). LOIC-UDP(0.997)는 천장이라 제외.

구성:
  A  : proto2+syn1+psh1+size4          = 8bit  (256)   [기존 baseline, 비교 기준]
  B1 : A + Fwd IAT Mean(2bit)          = 10bit (1024)
  B2 : A + Fwd IAT Mean(2bit) + Flow IAT Mean(2bit) = 12bit (4096)

IAT 이산화: benign train에서 log1p(value) 후 분위수 4버킷(2bit) 경계 생성, train/test/attack 공유.
비트 패킹: 기존 8bit(MSB) 뒤(LSB)에 IAT 비트 추가 -> state = (state8 << iat_bits) | iat_bits_value.
평가: experiment2.anomaly_eval 동일 (s=-log p, attack=positive, raw MWU AUC, 방향반전 없음).
QCBM/학습 없음. 기존 소스(experiment2/diagnose_*)는 import/재사용만, 수정 없음.
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
if QCBM_DIR not in sys.path:
    sys.path.insert(0, QCBM_DIR)
os.chdir(QCBM_DIR)

import experiment2 as E  # noqa: E402

DATA = "data/cic_0221_ddos.csv"
OUT = "results2"
ATTACK = "ddos attack-hoic"
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
IAT_BITS = 2                     # feature당 비트 (4버킷)
IAT_BUCKETS = 1 << IAT_BITS      # 4
SUBSAMPLE_SEED = 12345
EPS = 1e-9


def log1p_quantile_edges(values, nbuckets):
    """benign train의 IAT에 log1p 후 분위수 inner edges (size_edges_from_benign와 동일 철학)."""
    v = np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
    qs = np.linspace(0, 1, nbuckets + 1)[1:-1]
    return np.unique(np.quantile(v, qs))         # log 공간 inner edges


def iat_to_bits(values, edges):
    """log1p 후 digitize -> 0..(IAT_BUCKETS-1)."""
    v = np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
    b = np.digitize(v, edges)
    return np.clip(b, 0, IAT_BUCKETS - 1).astype(int)


def append_bits(state, add_val, add_bits):
    """기존 state(MSB) 뒤에 add_bits 폭의 add_val(LSB)을 이어붙임."""
    return (np.asarray(state, dtype=int) << add_bits) | np.asarray(add_val, dtype=int)


def marginal_dist(states, nq):
    states = np.asarray(states, dtype=int)
    bit_idx = np.arange(nq)
    p_bit = (((states[:, None] >> bit_idx) & 1).mean(axis=0))
    all_states = np.arange(1 << nq)
    sb = (all_states[:, None] >> bit_idx) & 1
    q = np.where(sb == 1, p_bit, 1.0 - p_bit).prod(axis=1)
    return q / q.sum(), p_bit


def eval_dist(q, st_te, st_atk, tag):
    fake_cfg = {"best": {"p_final": np.asarray(q, dtype=float).tolist(),
                         "final_mmd2": float("nan"), "tv": float("nan")}}
    return E.anomaly_eval(fake_cfg, st_te, st_atk, tag)


def measure(nq, st_tr, st_te, st_atk, tag):
    nstates = 1 << nq
    occ = int(np.bincount(st_tr, minlength=nstates).astype(bool).sum())
    q_emp = E.empirical_dist(st_tr, nstates)
    q_marg, _ = marginal_dist(st_tr, nq)
    an_e = eval_dist(q_emp, st_te, st_atk, "%s/emp" % tag)
    an_m = eval_dist(q_marg, st_te, st_atk, "%s/marg" % tag)
    rec = {
        "config": tag, "nbits": nq, "nstates": nstates,
        "occupied": occ, "occ_frac": occ / nstates,
        "n_attack": int(len(st_atk)),
        "emp_joint_auc": an_e["auc"], "marginal_auc": an_m["auc"],
        "joint_gain": an_e["auc"] - an_m["auc"],
        "emp_joint_auc_flipped": 1.0 - an_e["auc"],
        "youden_j_emp": an_e["youden_j"], "youden_j_marg": an_m["youden_j"],
        "logp_gap_emp": an_e["mean_logp_ben"] - an_e["mean_logp_atk"],
        "logp_gap_marg": an_m["mean_logp_ben"] - an_m["mean_logp_atk"],
        "_an_e": an_e, "_an_m": an_m,
    }
    E.log("  [%-12s] nbits=%2d states=%4d occ=%4d(%.1f%%) | emp=%.4f marg=%.4f gain=%+.4f"
          % (tag, nq, nstates, occ, 100 * occ / nstates,
             rec["emp_joint_auc"], rec["marginal_auc"], rec["joint_gain"]))
    return rec


def load():
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
    E.log("benign=%d  %s=%d" % (len(ben), ATTACK, len(atk)))
    return ben, atk


def main():
    if not os.path.exists(DATA):
        E.log("데이터 없음: %s -> 실행 불가." % DATA)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 78)
    E.log("IAT joint 점검: HOIC, 구성 A(8bit) vs B1(+FwdIAT 10bit) vs B2(+FlowIAT 12bit)")
    E.log("=" * 78)

    ben, atk = load()
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(ben))
    cut = int(0.7 * len(ben))
    ben_tr = ben.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = ben.iloc[idx[cut:]].reset_index(drop=True)
    E.log("benign train=%d / test=%d | attack=%d" % (len(ben_tr), len(ben_te), len(atk)))

    # 8bit base 상태 (구성 A)
    size_edges = E.size_edges_from_benign(ben_tr, 16)
    s8_tr = E.discretize(ben_tr, SPEC8, size_edges)
    s8_te = E.discretize(ben_te, SPEC8, size_edges)
    s8_atk = E.discretize(atk, SPEC8, size_edges)

    # IAT 경계 (benign train, log1p 후 분위수 4버킷)
    fwd_edges = log1p_quantile_edges(ben_tr["Fwd IAT Mean"], IAT_BUCKETS)
    flow_edges = log1p_quantile_edges(ben_tr["Flow IAT Mean"], IAT_BUCKETS)
    E.log("Fwd IAT Mean log1p inner edges = %s" % np.round(fwd_edges, 4).tolist())
    E.log("Flow IAT Mean log1p inner edges = %s" % np.round(flow_edges, 4).tolist())

    def fwd_bits(df):  return iat_to_bits(df["Fwd IAT Mean"], fwd_edges)
    def flow_bits(df): return iat_to_bits(df["Flow IAT Mean"], flow_edges)

    # 구성 B1: 8bit + Fwd IAT (2bit) -> 10bit
    b1_tr = append_bits(s8_tr, fwd_bits(ben_tr), IAT_BITS)
    b1_te = append_bits(s8_te, fwd_bits(ben_te), IAT_BITS)
    b1_atk = append_bits(s8_atk, fwd_bits(atk), IAT_BITS)
    # 구성 B2: B1 + Flow IAT (2bit) -> 12bit
    b2_tr = append_bits(b1_tr, flow_bits(ben_tr), IAT_BITS)
    b2_te = append_bits(b1_te, flow_bits(ben_te), IAT_BITS)
    b2_atk = append_bits(b1_atk, flow_bits(atk), IAT_BITS)

    E.log("-" * 78)
    E.log("[full] HOIC 대상 AUC (s=-log p, attack=positive, raw MWU)")
    res = {}
    res["A"] = measure(8, s8_tr, s8_te, s8_atk, "A:8bit")
    res["B1"] = measure(10, b1_tr, b1_te, b1_atk, "B1:+FwdIAT")
    res["B2"] = measure(12, b2_tr, b2_te, b2_atk, "B2:+FlowIAT")

    # 서브샘플 (attack을 benign_test 크기로, seed 고정) — 결론 안정성
    E.log("-" * 78)
    E.log("[subsample] attack -> benign_test(n=%d) (seed=%d)" % (len(ben_te), SUBSAMPLE_SEED))
    srng = np.random.default_rng(SUBSAMPLE_SEED)
    sel = srng.choice(len(atk), size=min(len(ben_te), len(atk)), replace=False)
    res_sub = {}
    res_sub["A"] = measure(8, s8_tr, s8_te, s8_atk[sel], "A:8bit(sub)")
    res_sub["B1"] = measure(10, b1_tr, b1_te, b1_atk[sel], "B1:+FwdIAT(sub)")
    res_sub["B2"] = measure(12, b2_tr, b2_te, b2_atk[sel], "B2:+FlowIAT(sub)")

    # 표
    E.log("=" * 78)
    E.log("[비교표 full] joint 이득이 IAT 추가로 커지는가?")
    E.log("  %-12s %-6s %-6s %-7s %-9s %-9s %-9s" %
          ("config", "bits", "occ", "occ%", "emp_AUC", "marg_AUC", "gain"))
    for k in ["A", "B1", "B2"]:
        r = res[k]
        E.log("  %-12s %-6d %-6d %-7.1f %-9.4f %-9.4f %+9.4f"
              % (r["config"], r["nbits"], r["occupied"], 100 * r["occ_frac"],
                 r["emp_joint_auc"], r["marginal_auc"], r["joint_gain"]))

    def clean(d):
        return {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
                for k, r in d.items()}
    payload = {
        "data": DATA, "attack": ATTACK,
        "split": {"seed": 7, "benign_train_frac": 0.7},
        "subsample_seed": SUBSAMPLE_SEED,
        "iat_bits_per_feature": IAT_BITS, "iat_buckets": IAT_BUCKETS,
        "iat_discretization": "log1p then benign-train quantile (4 buckets)",
        "fwd_iat_log1p_inner_edges": fwd_edges.tolist(),
        "flow_iat_log1p_inner_edges": flow_edges.tolist(),
        "packing": "state = (8bit_base << iat_bits) | iat ; base=proto2 syn1 psh1 size4 (MSB->LSB)",
        "n_benign_train": int(len(ben_tr)), "n_benign_test": int(len(ben_te)),
        "n_attack": int(len(atk)),
        "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (no direction selection)",
        "results_full": clean(res),
        "results_subsample": clean(res_sub),
    }
    out_json = os.path.join(OUT, "iat_joint.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: %s" % out_json)

    # ROC 그림 (full)
    try:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
        for ax, k in zip(axes, ["A", "B1", "B2"]):
            r = res[k]
            for rk, c, lbl in [("_an_e", "darkgreen", "emp_joint"),
                               ("_an_m", "darkorange", "marginal")]:
                an = r[rk]
                ax.plot(an["fpr"], an["tpr"], lw=2, color=c,
                        label="%s (AUC=%.3f)" % (lbl, an["auc"]))
            ax.plot([0, 1], [0, 1], "k--", alpha=.5)
            ax.set_title("%s  gain=%+.4f" % (r["config"], r["joint_gain"]))
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.legend(loc="lower right"); ax.grid(alpha=.3)
        plt.suptitle("HOIC: emp_joint vs marginal as IAT features added")
        plt.tight_layout()
        fp = os.path.join(OUT, "iat_joint.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("ROC 그림 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
