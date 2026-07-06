#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
flow_iat 매핑 민감도 점검 — 교차 데이터 재현이 flow_iat 컬럼 선택에 흔들리는가 (재학습 없음)
================================================================================
교차 진단(diagnose_cross_dataset.py)에서 joint>marginal이 12/12 재현됐으나, UNSW flow_iat을
dinpkt로 매핑한 것이 "불확실"(flow_iat=전체흐름 IAT vs dinpkt=목적지방향 패킷간격)했다.
본 작업은 UNSW flow_iat을 대안 컬럼으로 바꿔 재계산해 핵심 결과의 강건성을 본다.

flow_iat 대안 (모두 UNSW 실제 nonneg 컬럼; 억지 매핑 없음. CIC는 Flow IAT Mean 고정):
  M0 = dinpkt              (기존 기준선; Dest interpacket arrival)
  M1 = rate                (흐름 속도 pkts/s; 시간역수 의미 — 흐름수준 timing)
  M2 = (sinpkt+dinpkt)/2   (양방향 평균 interpacket; "전체 흐름"에 의미상 근접)

재사용: diagnose_cross_dataset(diagnose_dataset/load_cic/split_benign/encode 등) import.
  -> 인코딩/평가/SVD/해리 로직 100% 동일, flow_iat 소스 컬럼만 파라미터화. ★전수, raw MWU.
QCBM 학습 없음. 새 dependency 없음. 판정은 웹 AI.
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

import experiment2 as E                 # noqa: E402
import diagnose_cross_dataset as DCD     # noqa: E402  diagnose_dataset/load_cic/CANON 재사용

OUT = "results2"
UNSW_PATHS = ["data/unsw_nb15_training-set.csv", "data/unsw_nb15_testing-set.csv"]
MAPPINGS = ["M0_dinpkt", "M1_rate", "M2_sinpkt_dinpkt_mean"]


def load_unsw_mapped(flow_mode, tag):
    """UNSW 캐노니컬 로더 — flow_iat 소스만 flow_mode로 교체. 나머지는 기존과 동일."""
    src = ["proto", "smean", "sinpkt", "dinpkt", "rate", "attack_cat", "label"]
    parts = [pd.read_csv(p, usecols=src, low_memory=False) for p in UNSW_PATHS]
    df = pd.concat(parts, ignore_index=True)
    n0 = len(df)
    proto_num = df["proto"].astype(str).str.lower().map({"tcp": 6, "udp": 17}).fillna(-1).astype(int)
    sinpkt = pd.to_numeric(df["sinpkt"], errors="coerce")
    dinpkt = pd.to_numeric(df["dinpkt"], errors="coerce")
    rate = pd.to_numeric(df["rate"], errors="coerce")
    if flow_mode == "M0_dinpkt":
        flow = dinpkt
    elif flow_mode == "M1_rate":
        flow = rate
    elif flow_mode == "M2_sinpkt_dinpkt_mean":
        flow = 0.5 * (sinpkt + dinpkt)
    else:
        raise ValueError(flow_mode)
    can = pd.DataFrame({
        "Protocol": proto_num,
        "Pkt Len Mean": pd.to_numeric(df["smean"], errors="coerce"),
        "Fwd IAT Mean": sinpkt,
        "Flow IAT Mean": flow,
        "attack_cat": df["attack_cat"].astype(str).str.strip(),
        "label": pd.to_numeric(df["label"], errors="coerce"),
    })
    can = can.dropna(subset=DCD.CANON + ["label"])
    ben = can[can["attack_cat"].str.lower() == "normal"][DCD.CANON].reset_index(drop=True)
    attacks = {}
    for cat in sorted(can.loc[can["attack_cat"].str.lower() != "normal", "attack_cat"].unique()):
        attacks[cat] = can[can["attack_cat"] == cat][DCD.CANON].reset_index(drop=True)
    attacks["ALL"] = can[can["attack_cat"].str.lower() != "normal"][DCD.CANON].reset_index(drop=True)
    E.log("[%s/%s] benign=%d 공격종류=%d ALL=%d" % (tag, flow_mode, len(ben), len(attacks) - 1, len(attacks["ALL"])))
    return ben, attacks, (n0 - len(df))


def classify(emp, gain):
    """고정 규칙(매핑 간 일관 비교용; 임계는 사실 기술용)."""
    if emp < 0.75:
        return "feature_inadequate"
    if gain >= 0.15:
        return "complex_joint_dependent"
    return "simple_easy"


def slim(ds, mapping):
    """diagnose_dataset 결과에서 비교 핵심만 추출."""
    benign = {"occupied": ds["occupied"], "occ_frac": ds["occ_frac"],
              "reach_tv05": ds["svd"]["reach_rank"]["tv<=0.05"],
              "reach_tv01": ds["svd"]["reach_rank"]["tv<=0.01"],
              "cum_energy_r3": ds["svd"]["cum_energy"]["3"]}
    rows = []
    for r in ds["attacks"]:
        rows.append({"attack": r["attack"], "n_attack": r["n_attack"],
                     "emp_joint": r["auc_emp_joint"], "marginal": r["auc_marginal"],
                     "joint_gain": r["joint_gain"],
                     "max_single_feature": r["max_single_feature"], "max_single_auc": r["max_single_auc"],
                     "flow_iat_single_auc": r["single_feature_auc"]["flow_iat"],
                     "class": classify(r["auc_emp_joint"], r["joint_gain"])})
    return {"mapping": mapping, "tag": ds["tag"], "benign_complexity": benign, "rows": rows}


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("#" * 78)
    E.log("flow_iat 매핑 민감도 점검 — UNSW flow_iat: dinpkt vs rate vs (sinpkt+dinpkt)/2")
    E.log("#" * 78)

    # --- CIC 고정 (flow_iat = Flow IAT Mean, 매핑 무관) ---
    cic = []
    for csv, al, tag in [("data/cic_0221_ddos.csv", "ddos attack-hoic", "CIC_HOIC"),
                         ("data/cic_bot_0203.csv", "bot", "CIC_Bot")]:
        ben, atks, drp = DCD.load_cic(csv, al, tag)
        cic.append(slim(DCD.diagnose_dataset(tag, ben, atks, drp), "fixed"))

    # --- UNSW: 매핑별 ---
    unsw_by_map = {}
    for m in MAPPINGS:
        ben, atks, drp = load_unsw_mapped(m, "UNSW")
        unsw_by_map[m] = slim(DCD.diagnose_dataset("UNSW", ben, atks, drp), m)

    # --- joint>marginal 카운트 (CIC 2 + UNSW 10 = 12), 매핑별 ---
    cic_pos = sum(1 for ds in cic for r in ds["rows"] if r["joint_gain"] > 0)   # 2 (고정)
    summary = {}
    for m in MAPPINGS:
        u = unsw_by_map[m]
        u_pos = sum(1 for r in u["rows"] if r["joint_gain"] > 0)
        total = cic_pos + u_pos
        allrow = next(r for r in u["rows"] if r["attack"] == "ALL")
        summary[m] = {
            "joint_gt_marginal": "%d/%d" % (total, cic_pos + len(u["rows"])),
            "joint_gt_marginal_count": total, "denom": cic_pos + len(u["rows"]),
            "unsw_all_joint_gain": allrow["joint_gain"],
            "unsw_all_emp": allrow["emp_joint"], "unsw_all_marginal": allrow["marginal"],
            "benign_reach_tv05": u["benign_complexity"]["reach_tv05"],
            "benign_occ_frac": u["benign_complexity"]["occ_frac"],
        }

    # --- 매핑 간 변화: 각 UNSW 공격 joint_gain의 매핑별 값 + 변동폭, 분류 안정성 ---
    attacks = [r["attack"] for r in unsw_by_map["M0_dinpkt"]["rows"]]
    per_attack = []
    flips = []
    for a in attacks:
        gains = {m: next(r["joint_gain"] for r in unsw_by_map[m]["rows"] if r["attack"] == a) for m in MAPPINGS}
        emps = {m: next(r["emp_joint"] for r in unsw_by_map[m]["rows"] if r["attack"] == a) for m in MAPPINGS}
        cls = {m: next(r["class"] for r in unsw_by_map[m]["rows"] if r["attack"] == a) for m in MAPPINGS}
        gvals = list(gains.values())
        rec = {"attack": a, "joint_gain": gains, "emp_joint": emps, "class": cls,
               "gain_range": float(max(gvals) - min(gvals)),
               "all_positive": all(g > 0 for g in gvals),
               "class_stable": len(set(cls.values())) == 1}
        per_attack.append(rec)
        if not rec["class_stable"]:
            flips.append({"attack": a, "class_by_mapping": cls})

    max_gain_shift = max(p["gain_range"] for p in per_attack)
    any_joint_broken = any(not p["all_positive"] for p in per_attack)

    E.log("=" * 78)
    E.log("[매핑별 요약]")
    for m in MAPPINGS:
        s = summary[m]
        E.log("  %-22s joint>marg %s | UNSW/ALL gain=%+.4f emp=%.4f marg=%.4f | benign reach(TV05)=%s occ=%.1f%%"
              % (m, s["joint_gt_marginal"], s["unsw_all_joint_gain"], s["unsw_all_emp"],
                 s["unsw_all_marginal"], s["benign_reach_tv05"], 100 * s["benign_occ_frac"]))
    E.log("[매핑 간] 공격별 joint이득 최대 변동폭=%.4f | joint>marginal 깨진 공격 있음=%s | 분류 뒤집힘=%d개"
          % (max_gain_shift, any_joint_broken, len(flips)))
    for f in flips:
        E.log("  분류변화: %s -> %s" % (f["attack"], f["class_by_mapping"]))

    payload = {
        "purpose": "UNSW flow_iat 매핑(dinpkt/rate/(sinpkt+dinpkt)/2) 민감도. CIC는 Flow IAT Mean 고정.",
        "mappings": {"M0_dinpkt": "dinpkt(Dest interpacket)", "M1_rate": "rate(pkts/s, 흐름속도)",
                     "M2_sinpkt_dinpkt_mean": "(sinpkt+dinpkt)/2 양방향 평균 interpacket"},
        "encoding": "proto2+size4+fwd_iat2+flow_iat2=10bit(1024), seed7 0.7 split, raw MWU. 전수.",
        "joint_gt_marginal_counting": "CIC_HOIC+CIC_Bot(고정 2) + UNSW(9공격+ALL=10) = 12",
        "cic_fixed": cic,
        "unsw_by_mapping": unsw_by_map,
        "summary_by_mapping": summary,
        "per_attack_across_mappings": per_attack,
        "robustness": {
            "max_joint_gain_shift_across_mappings": max_gain_shift,
            "any_joint_gt_marginal_broken": any_joint_broken,
            "n_classification_flips": len(flips), "flips": flips,
        },
        "note": "사실 측정만. 판정은 웹 AI. QCBM 학습 없음. 기존 소스/데이터/결과 무수정.",
    }
    with open(os.path.join(OUT, "flow_iat_sensitivity.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/flow_iat_sensitivity.json")

    # --- 그림: (좌) UNSW 공격별 joint_gain 매핑 3종 grouped bar, (우) emp_joint 매핑 비교 ---
    try:
        order = [a for a in attacks if a != "ALL"] + ["ALL"]
        x = np.arange(len(order)); w = 0.26
        colors = {"M0_dinpkt": "darkgreen", "M1_rate": "teal", "M2_sinpkt_dinpkt_mean": "olive"}
        fig, ax = plt.subplots(1, 2, figsize=(18, 6.5))
        for i, m in enumerate(MAPPINGS):
            g = [next(r["joint_gain"] for r in unsw_by_map[m]["rows"] if r["attack"] == a) for a in order]
            ax[0].bar(x + (i - 1) * w, g, w, color=colors[m], alpha=.85, label=m, edgecolor="k")
        ax[0].axhline(0, color="k", lw=1)
        ax[0].set_xticks(x); ax[0].set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax[0].set_ylabel("joint gain (emp_joint - marginal)")
        ax[0].set_title("UNSW joint gain by flow_iat mapping (all >0 = joint>marginal robust)")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
        for i, m in enumerate(MAPPINGS):
            ev = [next(r["emp_joint"] for r in unsw_by_map[m]["rows"] if r["attack"] == a) for a in order]
            ax[1].bar(x + (i - 1) * w, ev, w, color=colors[m], alpha=.85, label=m, edgecolor="k")
        ax[1].axhline(0.75, color="red", ls="--", alpha=.6, label="feature-inadequate <0.75")
        ax[1].set_xticks(x); ax[1].set_xticklabels(order, rotation=40, ha="right", fontsize=8)
        ax[1].set_ylabel("emp_joint AUC"); ax[1].set_ylim(0.5, 1.02)
        ax[1].set_title("UNSW emp_joint AUC by flow_iat mapping")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
        plt.suptitle("flow_iat mapping sensitivity (UNSW): dinpkt vs rate vs (sinpkt+dinpkt)/2 — CIC fixed")
        plt.tight_layout()
        fp = os.path.join(OUT, "flow_iat_sensitivity.png")
        plt.savefig(fp, dpi=120); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)
    E.log("DONE")


if __name__ == "__main__":
    main()
