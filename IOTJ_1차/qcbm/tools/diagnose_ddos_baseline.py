#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DDoS 파일의 emp_joint vs marginal AUC 사전 측정 (학습 없음)
============================================================
목적:
  Infiltration에서는 joint < marginal(0.428<0.438), 분포기반 상한 0.43으로 약분리였다.
  가설: DDoS처럼 feature 조합에 시그니처가 있는 공격은 joint > marginal일 수 있다.
  data/cic_0221_ddos.csv 에서 기존 8bit 인코딩(proto2+syn1+psh1+size4=256상태)으로
  emp_joint와 marginal 분포의 이상탐지 AUC를 공격(HOIC, LOIC-UDP)별로 측정하고,
  joint 이득(emp_joint - marginal)과 단일 feature 분리력(천장 여부)을 함께 본다.

평가는 experiment2.anomaly_eval 과 동일: s(x)=-log p(x), attack=positive, raw MWU AUC.
QCBM/학습은 하지 않는다. emp_joint/marginal은 순수 계산이다.

기존 소스(experiment2.py, diagnose_classical_baseline.py)는 import/재사용만 하고 수정하지 않는다.
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

import experiment2 as E  # noqa: E402  (가드 있음 -> import 시 main() 자동실행 안 됨)

DATA = "data/cic_0221_ddos.csv"
OUT = "results2"
NQ = 8
NSTATES = 1 << NQ
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
# 비트 위치(bit0=LSB..bit7=MSB): state = proto*64 + syn*32 + psh*16 + size
BIT_FEATURE = {7: "proto[hi]", 6: "proto[lo]", 5: "syn", 4: "psh",
               3: "size[b3]", 2: "size[b2]", 1: "size[b1]", 0: "size[b0]"}
ATTACKS = ["ddos attack-hoic", "ddos attack-loic-udp"]
SUBSAMPLE_SEED = 12345
EPS = 1e-9


def build_marginal_dist(states, nq=NQ):
    """비트 독립(곱) 분포. diagnose_classical_baseline.py 와 동일 방식."""
    states = np.asarray(states, dtype=int)
    bit_idx = np.arange(nq)
    sample_bits = (states[:, None] >> bit_idx) & 1
    p_bit = sample_bits.mean(axis=0)
    all_states = np.arange(1 << nq)
    state_bits = (all_states[:, None] >> bit_idx) & 1
    probs = np.where(state_bits == 1, p_bit, 1.0 - p_bit)
    q = probs.prod(axis=1)
    return q / q.sum(), p_bit


def eval_dist(q, st_te, st_atk, tag):
    """experiment2.anomaly_eval 재사용 (cfg_out['best']['p_final'] 만 사용)."""
    fake_cfg = {"best": {"p_final": np.asarray(q, dtype=float).tolist(),
                         "final_mmd2": float("nan"), "tv": float("nan")}}
    return E.anomaly_eval(fake_cfg, st_te, st_atk, tag)


def bit_separation(st_ben, st_atk, nq=NQ):
    """각 비트의 표준화 평균차 |p_ben-p_atk|/(pooled_std+eps). p는 P(bit=1)."""
    bi = np.arange(nq)
    pb = ((np.asarray(st_ben)[:, None] >> bi) & 1).mean(axis=0)
    pa = ((np.asarray(st_atk)[:, None] >> bi) & 1).mean(axis=0)
    sb = np.sqrt(pb * (1 - pb)); sa = np.sqrt(pa * (1 - pa))
    pooled = np.sqrt((sb * sb + sa * sa) / 2.0)
    sep = np.abs(pb - pa) / (pooled + EPS)
    return pb, pa, sep


def load_ddos():
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean", "Label"]
    E.log("CSV 로드: %s | %s" % (DATA, cols))
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    for c in ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    lab = df["Label"].astype(str).str.strip().str.lower()
    benign = df[lab == "benign"].reset_index(drop=True)
    atks = {a: df[lab == a].reset_index(drop=True) for a in ATTACKS}
    E.log("benign=%d | %s" % (len(benign),
          " | ".join("%s=%d" % (a, len(atks[a])) for a in ATTACKS)))
    return benign, atks


def main():
    if not os.path.exists(DATA):
        E.log("데이터 없음: %s -> 실행 불가." % DATA)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 70)
    E.log("DDoS baseline: emp_joint vs marginal AUC (학습 없음)")
    E.log("=" * 70)

    benign, atks = load_ddos()

    # benign seed=7 0.7/0.3 분할 (experiment2/anomaly_redo 와 동일 방식)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(benign))
    cut = int(0.7 * len(benign))
    ben_tr = benign.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = benign.iloc[idx[cut:]].reset_index(drop=True)
    E.log("benign train=%d / test=%d" % (len(ben_tr), len(ben_te)))

    edges = E.size_edges_from_benign(ben_tr, 16)
    st_tr = E.discretize(ben_tr, SPEC8, edges)
    st_te = E.discretize(ben_te, SPEC8, edges)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    E.log("benign train 점유 상태=%d/256" % occ)

    # baseline 분포 (benign train 기반)
    q_emp = E.empirical_dist(st_tr, NSTATES)
    q_marg, p_bit = build_marginal_dist(st_tr, NQ)

    st_atk = {a: E.discretize(atks[a], SPEC8, edges) for a in ATTACKS}

    # ---- 단일 feature(비트) 분리력 ----
    E.log("=" * 70)
    E.log("[단일 feature 분리력] |P(bit=1)_ben - P(bit=1)_atk| / (pooled_std+eps)")
    E.log("  %-4s %-10s %-10s %-10s" % ("bit", "feature", "HOIC_sep", "LOIC_sep"))
    bit_sep = {}
    seps = {}
    for a in ATTACKS:
        pb, pa, sep = bit_separation(st_tr, st_atk[a], NQ)
        seps[a] = (pb, pa, sep)
    for b in range(NQ - 1, -1, -1):
        E.log("  %-4d %-10s %-10.4f %-10.4f" %
              (b, BIT_FEATURE[b], seps[ATTACKS[0]][2][b], seps[ATTACKS[1]][2][b]))
        bit_sep[b] = {"feature": BIT_FEATURE[b],
                      ATTACKS[0]: float(seps[ATTACKS[0]][2][b]),
                      ATTACKS[1]: float(seps[ATTACKS[1]][2][b])}

    # ---- AUC 측정 (full + subsample) ----
    def measure(a_states, tag):
        an_emp = eval_dist(q_emp, st_te, a_states, "%s/emp" % tag)
        an_marg = eval_dist(q_marg, st_te, a_states, "%s/marg" % tag)
        return {
            "tag": tag, "n_attack": int(len(a_states)),
            "emp_joint_auc": an_emp["auc"], "marginal_auc": an_marg["auc"],
            "joint_gain": an_emp["auc"] - an_marg["auc"],
            "emp_joint_auc_flipped": 1.0 - an_emp["auc"],
            "marginal_auc_flipped": 1.0 - an_marg["auc"],
            "youden_j_emp": an_emp["youden_j"], "youden_j_marg": an_marg["youden_j"],
            "logp_gap_emp": an_emp["mean_logp_ben"] - an_emp["mean_logp_atk"],
            "logp_gap_marg": an_marg["mean_logp_ben"] - an_marg["mean_logp_atk"],
            "_an_emp": an_emp, "_an_marg": an_marg,
        }

    results = {"full": {}, "subsample": {}}
    E.log("=" * 70)
    E.log("[AUC] full (s=-log p, attack=positive, raw MWU)")
    for a in ATTACKS:
        r = measure(st_atk[a], a)
        results["full"][a] = r
        E.log("  %-22s n=%-7d emp_joint=%.4f marginal=%.4f gain=%+.4f (1-emp=%.4f)"
              % (a, r["n_attack"], r["emp_joint_auc"], r["marginal_auc"],
                 r["joint_gain"], r["emp_joint_auc_flipped"]))

    # 불균형 보정: attack을 benign_test 크기로 서브샘플 (seed 고정)
    E.log("-" * 70)
    E.log("[AUC] subsample: attack을 benign_test(n=%d) 크기로 무작위 축소 (seed=%d)"
          % (len(st_te), SUBSAMPLE_SEED))
    srng = np.random.default_rng(SUBSAMPLE_SEED)
    for a in ATTACKS:
        sa = st_atk[a]
        if len(sa) > len(st_te):
            sub = sa[srng.choice(len(sa), size=len(st_te), replace=False)]
        else:
            sub = sa  # 이미 작으면 그대로
        r = measure(sub, "%s(sub)" % a)
        results["subsample"][a] = r
        E.log("  %-22s n=%-7d emp_joint=%.4f marginal=%.4f gain=%+.4f"
              % (a, r["n_attack"], r["emp_joint_auc"], r["marginal_auc"], r["joint_gain"]))

    # ---- 저장 (내부 _an_* 제거) ----
    def clean(d):
        return {k: {kk: vv for kk, vv in r.items() if not kk.startswith("_")}
                for k, r in d.items()}
    payload = {
        "data": DATA, "nq": NQ, "nstates": NSTATES, "spec": SPEC8["features"],
        "split": {"seed": 7, "benign_train_frac": 0.7},
        "subsample_seed": SUBSAMPLE_SEED,
        "n_benign_train": int(len(ben_tr)), "n_benign_test": int(len(ben_te)),
        "n_attack": {a: int(len(st_atk[a])) for a in ATTACKS},
        "benign_train_occupied_states": occ,
        "marginal_p_bit_lsb_to_msb": p_bit.tolist(),
        "bit_feature_map": BIT_FEATURE,
        "bit_separation": bit_sep,
        "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (no direction selection)",
        "results_full": clean(results["full"]),
        "results_subsample": clean(results["subsample"]),
    }
    out_json = os.path.join(OUT, "ddos_baseline.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: %s" % out_json)

    # ---- (선택) ROC 그림 (full) ----
    try:
        fig, axes = plt.subplots(1, len(ATTACKS), figsize=(6 * len(ATTACKS), 5.5))
        for ax, a in zip(np.atleast_1d(axes), ATTACKS):
            for r_key, c, lbl in [("_an_emp", "darkgreen", "emp_joint"),
                                   ("_an_marg", "darkorange", "marginal")]:
                an = results["full"][a][r_key]
                ax.plot(an["fpr"], an["tpr"], lw=2, color=c,
                        label="%s (AUC=%.3f)" % (lbl, an["auc"]))
            ax.plot([0, 1], [0, 1], "k--", alpha=.5, label="chance")
            ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
            ax.set_title("%s ROC (8bit baseline)" % a)
            ax.legend(loc="lower right"); ax.grid(alpha=.3)
        plt.tight_layout()
        fp = os.path.join(OUT, "ddos_baseline.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("ROC 그림 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
