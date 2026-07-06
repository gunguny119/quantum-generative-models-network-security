#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psh 강신호 통제 + 약신호 joint 가치 검증 (학습 없음, 빈도 기반)
==============================================================
질문: "joint>marginal"(B2 emp_joint 0.9258 > marginal 0.8741, +0.0518)이 진짜 joint(상호작용)
      가치인가, 아니면 psh 단독 AUC 0.878 때문인 artifact인가? 논문 핵심 라인의 생존 판정.

세 가지(판정은 웹 AI):
  (1) psh 제거 통제: psh 뺀 11비트(proto2+syn1+size4+Fwd2+Flow2)에서 emp_joint vs marginal.
      psh 없이도 joint가 이기는가.
  (2) joint 순수 기여: 12비트 marginal / emp_joint / psh단독 AUC 한 표.
      emp_joint - marginal (상호작용분), emp_joint - psh단독.
  (3) 상호작용 위치: feature 쌍별 joint(결합 경험분포) vs marginal(두 feature 독립곱) AUC 이득.

정의:
  - emp_joint: 선택 feature 조립 상태의 경험적 결합분포 q(x).
  - marginal((1)(2)): 비트독립 곱(train_qcbm_iat_b2.marginal_dist) — 프로젝트 기준선 정의(0.8741 재현).
  - 쌍 marginal((3)): 두 feature를 각자 marginal로 두고 곱(feature-level 독립) = 두 feature 상호작용 분리.
  - 평가: s(x)=-log p_benign(x)+eps, attack=positive, experiment2.auc_score (raw MWU, eps=1e-12).
재사용: experiment2(empirical_dist, auc_score), diagnose_iat_joint(IAT 인코딩),
        diagnose_encoding_info_loss(load_split, feature_levels), train_qcbm_iat_b2(marginal_dist).
QCBM 학습 없음. 기존 소스 무수정.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E                          # noqa: E402  empirical_dist, auc_score
import diagnose_iat_joint as DIAG                # noqa: E402  log1p_quantile_edges
import diagnose_encoding_info_loss as INFO       # noqa: E402  load_split, feature_levels
from train_qcbm_iat_b2 import marginal_dist      # noqa: E402  비트독립 곱(기준선 정의)

OUT = "results2"
EPS = 1e-12
FEAT_BITS = {"proto": 2, "syn": 1, "psh": 1, "size": 4, "fwd_iat": 2, "flow_iat": 2}
FULL_ORDER = ["proto", "syn", "psh", "size", "fwd_iat", "flow_iat"]      # MSB->LSB (build_b2와 동일)
NOPSH_ORDER = ["proto", "syn", "size", "fwd_iat", "flow_iat"]            # psh 제거 11bit
# 12비트 기준선(검증 대상)
EXPECT_EMP, EXPECT_MARG = 0.9258, 0.8741
PAIRS = [("psh", "size"), ("psh", "fwd_iat"), ("psh", "flow_iat"),
         ("size", "fwd_iat"), ("size", "flow_iat"), ("fwd_iat", "flow_iat"),
         ("proto", "size")]                                              # proto×size=대조


def eval_auc(q, st_te, st_atk):
    s_te = -np.log(q[np.asarray(st_te, int)] + EPS)
    s_atk = -np.log(q[np.asarray(st_atk, int)] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def assemble(levels, order):
    """선택 feature를 MSB->LSB로 비트팩한 결합 상태 + 상태수."""
    state = None; nbits = 0
    for name in order:
        v = np.asarray(levels[name], int)
        b = FEAT_BITS[name]
        state = v if state is None else ((state << b) | v)
        nbits += b
    return state, (1 << nbits), nbits


def emp_and_marg(levels_tr, levels_te, levels_atk, order):
    """조립 상태에서 emp_joint(경험결합) / marginal(비트독립 곱) AUC."""
    s_tr, nstates, nbits = assemble(levels_tr, order)
    s_te, _, _ = assemble(levels_te, order)
    s_atk, _, _ = assemble(levels_atk, order)
    q_emp = E.empirical_dist(s_tr, nstates)
    q_marg = marginal_dist(s_tr, nbits)
    occ = int(np.bincount(s_tr, minlength=nstates).astype(bool).sum())
    return {"nbits": nbits, "nstates": nstates, "occupied": occ,
            "emp_joint_auc": eval_auc(q_emp, s_te, s_atk),
            "marginal_auc": eval_auc(q_marg, s_te, s_atk)}


def feature_single_auc(levels_tr, levels_te, levels_atk, name):
    nlev = 1 << FEAT_BITS[name]
    q = E.empirical_dist(np.asarray(levels_tr[name], int), nlev)
    return eval_auc(q, levels_te[name], levels_atk[name])


def pair_joint_marg(levels_tr, levels_te, levels_atk, f1, f2):
    """쌍 (f1,f2): joint=결합 경험분포 / marginal=두 feature 독립 곱(feature-level)."""
    n1, n2 = 1 << FEAT_BITS[f1], 1 << FEAT_BITS[f2]
    def comb(lv):  # 결합 인덱스
        return np.asarray(lv[f1], int) * n2 + np.asarray(lv[f2], int)
    c_tr, c_te, c_atk = comb(levels_tr), comb(levels_te), comb(levels_atk)
    q_joint = E.empirical_dist(c_tr, n1 * n2)
    p1 = E.empirical_dist(np.asarray(levels_tr[f1], int), n1)
    p2 = E.empirical_dist(np.asarray(levels_tr[f2], int), n2)
    q_marg = (p1[:, None] * p2[None, :]).reshape(-1)            # feature-level 독립 곱
    q_marg = q_marg / q_marg.sum()
    return {"joint_auc": eval_auc(q_joint, c_te, c_atk),
            "marginal_auc": eval_auc(q_marg, c_te, c_atk)}


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 78)
    E.log("psh 통제 + 약신호 joint 가치 검증 (빈도 기반, 학습 없음)")
    E.log("=" * 78)

    ben_tr, ben_te, atk = INFO.load_split()
    size_edges = E.size_edges_from_benign(ben_tr, 16)
    fwd_edges = DIAG.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], 4)
    flow_edges = DIAG.log1p_quantile_edges(ben_tr["Flow IAT Mean"], 4)

    def all_levels(df):
        return {n: np.asarray(INFO.feature_levels(df, n, size_edges, fwd_edges, flow_edges), int)
                for n in FEAT_BITS}
    L_tr, L_te, L_atk = all_levels(ben_tr), all_levels(ben_te), all_levels(atk)

    # --- 정합 검증: 12비트 ---
    full = emp_and_marg(L_tr, L_te, L_atk, FULL_ORDER)
    E.log("12비트 재계산: emp_joint=%.4f marginal=%.4f 점유=%d (기대 %.4f/%.4f, 94)"
          % (full["emp_joint_auc"], full["marginal_auc"], full["occupied"],
             EXPECT_EMP, EXPECT_MARG))
    if abs(full["emp_joint_auc"] - EXPECT_EMP) > 0.01 or abs(full["marginal_auc"] - EXPECT_MARG) > 0.01:
        E.log("!!! 12비트 정합 실패 -> 평가/인코딩 재현 어긋남. 중단."); sys.exit(2)
    E.log("정합 OK.")

    # --- (1) psh 제거 통제 ---
    nopsh = emp_and_marg(L_tr, L_te, L_atk, NOPSH_ORDER)
    E.log("-" * 78)
    E.log("[(1) psh 제거] 11비트: emp_joint=%.4f marginal=%.4f gain=%+.4f (점유 %d/%d)"
          % (nopsh["emp_joint_auc"], nopsh["marginal_auc"],
             nopsh["emp_joint_auc"] - nopsh["marginal_auc"], nopsh["occupied"], nopsh["nstates"]))
    E.log("       (12비트 gain=%+.4f 와 비교)" % (full["emp_joint_auc"] - full["marginal_auc"]))

    # --- (2) joint 순수 기여 ---
    psh_solo = feature_single_auc(L_tr, L_te, L_atk, "psh")
    contrib = {"marginal": full["marginal_auc"], "emp_joint": full["emp_joint_auc"],
               "psh_solo": psh_solo,
               "emp_minus_marginal": full["emp_joint_auc"] - full["marginal_auc"],
               "emp_minus_psh_solo": full["emp_joint_auc"] - psh_solo,
               "marginal_minus_psh_solo": full["marginal_auc"] - psh_solo}
    E.log("-" * 78)
    E.log("[(2) joint 순수 기여] marginal=%.4f emp_joint=%.4f psh단독=%.4f"
          % (full["marginal_auc"], full["emp_joint_auc"], psh_solo))
    E.log("    emp-marginal=%+.4f | emp-psh단독=%+.4f | marginal-psh단독=%+.4f"
          % (contrib["emp_minus_marginal"], contrib["emp_minus_psh_solo"],
             contrib["marginal_minus_psh_solo"]))

    # --- (3) 쌍별 상호작용 ---
    E.log("-" * 78)
    E.log("[(3) 쌍별 상호작용] joint vs marginal(두 feature 독립곱)")
    pairs = []
    for f1, f2 in PAIRS:
        r = pair_joint_marg(L_tr, L_te, L_atk, f1, f2)
        g = r["joint_auc"] - r["marginal_auc"]
        pairs.append({"pair": "%sx%s" % (f1, f2), "joint_auc": r["joint_auc"],
                      "marginal_auc": r["marginal_auc"], "gain": g})
        E.log("  %-16s joint=%.4f marg=%.4f gain=%+.4f" % ("%sx%s" % (f1, f2),
              r["joint_auc"], r["marginal_auc"], g))
    pairs_sorted = sorted(pairs, key=lambda d: d["gain"], reverse=True)
    E.log("  >> 최대 상호작용 쌍: %s (gain %+.4f)"
          % (pairs_sorted[0]["pair"], pairs_sorted[0]["gain"]))

    # --- 저장 ---
    payload = {
        "eval": "s=-log p_benign(x)+1e-12, attack=positive, experiment2.auc_score (raw MWU)",
        "marginal_def_12_11bit": "비트독립 곱 (train_qcbm_iat_b2.marginal_dist)",
        "marginal_def_pairs": "두 feature 독립 곱 (feature-level)",
        "verify_12bit": {"emp_joint": full["emp_joint_auc"], "marginal": full["marginal_auc"],
                         "occupied": full["occupied"],
                         "expected": {"emp_joint": EXPECT_EMP, "marginal": EXPECT_MARG, "occ": 94}},
        "control_remove_psh": {
            "full_12bit": {"emp_joint": full["emp_joint_auc"], "marginal": full["marginal_auc"],
                           "gain": full["emp_joint_auc"] - full["marginal_auc"],
                           "occupied": full["occupied"], "nstates": full["nstates"]},
            "nopsh_11bit": {"emp_joint": nopsh["emp_joint_auc"], "marginal": nopsh["marginal_auc"],
                            "gain": nopsh["emp_joint_auc"] - nopsh["marginal_auc"],
                            "occupied": nopsh["occupied"], "nstates": nopsh["nstates"]}},
        "joint_contribution": contrib,
        "pairwise_interaction": pairs,
        "max_interaction_pair": pairs_sorted[0],
        "note": "사실 측정만. 판정(joint 진짜 가치 vs psh artifact)은 웹 AI. QCBM 학습 없음.",
    }
    with open(os.path.join(OUT, "psh_control_joint.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/psh_control_joint.json")

    # --- 그림 ---
    try:
        fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
        # 패널1: psh有/無 joint이득 + 기준선들
        labels = ["12bit\n(psh포함)", "11bit\n(psh제거)"]
        marg = [full["marginal_auc"], nopsh["marginal_auc"]]
        emp = [full["emp_joint_auc"], nopsh["emp_joint_auc"]]
        x = np.arange(2); w = 0.36
        ax[0].bar(x - w / 2, marg, w, label="marginal", color="darkorange")
        ax[0].bar(x + w / 2, emp, w, label="emp_joint", color="steelblue")
        ax[0].axhline(psh_solo, color="crimson", ls="--", label="psh 단독 (%.3f)" % psh_solo)
        for xi, m, e in zip(x, marg, emp):
            ax[0].text(xi - w / 2, m + .003, "%.3f" % m, ha="center", fontsize=8)
            ax[0].text(xi + w / 2, e + .003, "%.3f" % e, ha="center", fontsize=8)
        ax[0].set_xticks(x); ax[0].set_xticklabels(labels)
        ax[0].set_ylim(0.5, 1.0); ax[0].set_ylabel("AUC")
        ax[0].set_title("(1)(2) psh control: marginal vs emp_joint vs psh-solo")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
        # 패널2: 쌍별 이득
        pn = [p["pair"] for p in pairs]; pg = [p["gain"] for p in pairs]
        xi = np.arange(len(pn))
        ax[1].bar(xi, pg, color=["crimson" if g == max(pg) else "slateblue" for g in pg])
        for i, g in enumerate(pg):
            ax[1].text(i, g + (0.002 if g >= 0 else -0.004), "%+.3f" % g,
                       ha="center", va="bottom" if g >= 0 else "top", fontsize=8)
        ax[1].set_xticks(xi); ax[1].set_xticklabels(pn, rotation=35, fontsize=8)
        ax[1].set_ylabel("joint - marginal AUC"); ax[1].axhline(0, color="k", lw=.8)
        ax[1].set_title("(3) pairwise interaction gain")
        ax[1].grid(alpha=.3, axis="y")
        plt.tight_layout()
        fp = os.path.join(OUT, "psh_control_joint.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
