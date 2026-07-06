#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNSW 파라미터 효율 + 견고성 + 비지도 선택 — QCBM(180p) vs 고전, Bot 패턴 재현 여부 (재학습 없음)
====================================================================================================
학습된 results2/qcbm_unsw_restarts.npz(16 p_final, 10bit/1024)를 사용. 판정은 웹 AI.
분석2(효율): QCBM best AUC 도달 고전 params, 180p 동일예산 고전 최대 AUC vs QCBM.
분석3(견고성+선택): 고전 180p-이하 최고 초과 restart 수/16, median vs 고전, corr(MMD²/TV, AUC).
  -> Bot 패턴(best운/median미달/선택불가 corr0.077)이 UNSW에서 재현되는가를 사실로.

평가 raw MWU(s=-log p, 방향반전 없음). 재사용(기존 소스 무수정, import):
  experiment2: tv_dist, auc_score, empirical_dist.
  diagnose_param_efficiency_b2: normalize_clip, nmf_factorize, group_product_dist, group_index.
  train_qcbm_unsw_resumable: build_unsw() (q_train/st_tr/st_te/st_atk 재현, 학습 split과 동일).
QCBM 재학습 없음. 새 dependency 없음(scipy.stats 허용).
"""
import os
import sys
import json

import numpy as np
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

import experiment2 as E                     # noqa: E402  tv_dist, auc_score
import diagnose_param_efficiency_b2 as PB2  # noqa: E402  normalize_clip, nmf_factorize, group_product_dist
import train_qcbm_unsw_resumable as TUNSW   # noqa: E402  build_unsw()

OUT = "results2"
NQ = 10
NSTATES = 1 << NQ
SIDE = 32                                   # 32x32 reshape, low-rank params = 2*SIDE*r = 64r
EPS = 1e-12
NPZ = os.path.join(OUT, "qcbm_unsw_restarts.npz")
UNSW_JSON = os.path.join(OUT, "qcbm_unsw.json")
QCBM_PARAMS = 6 * NQ * 3                     # 180
SVD_RMAX = 24
NMF_RMAX = 16
REF_EMP, REF_MARG = 0.8705, 0.7027
TOL = 0.003

BITGROUPS = {
    "marginal(1x10)": [[i] for i in range(NQ)],
    "2bitx5":         [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
    "5+5":            [list(range(5)), list(range(5, 10))],
    "6+4":            [list(range(6)), list(range(6, 10))],
    "8+2":            [list(range(8)), [8, 9]],
    "full(1x10)":     [list(range(NQ))],
}


def auc_of_dist(p, st_te, st_atk):
    p = np.asarray(p, dtype=float)
    s_te = -np.log(p[st_te] + EPS); s_atk = -np.log(p[st_atk] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def marginal_dist(st_tr, nq):
    states = np.asarray(st_tr, dtype=int)
    bit_idx = np.arange(nq)
    p_bit = (((states[:, None] >> bit_idx) & 1).mean(axis=0))
    all_states = np.arange(1 << nq)
    sb = (all_states[:, None] >> bit_idx) & 1
    q = np.where(sb == 1, p_bit, 1.0 - p_bit).prod(axis=1)
    return q / q.sum()


def svd_curve(q, st_te, st_atk):
    M = np.asarray(q, float).reshape(SIDE, SIDE)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rows = []
    for r in range(1, SVD_RMAX + 1):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        qa = PB2.normalize_clip(approx.reshape(-1))
        rows.append({"rank": r, "params": 2 * SIDE * r, "tv": float(E.tv_dist(qa, q)),
                     "auc": auc_of_dist(qa, st_te, st_atk)})
    return rows


def nmf_curve(q, st_te, st_atk):
    M = np.asarray(q, float).reshape(SIDE, SIDE)
    rows = []
    for r in range(1, NMF_RMAX + 1):
        approx, ferr = PB2.nmf_factorize(M, r)
        qa = PB2.normalize_clip(approx.reshape(-1))
        rows.append({"rank": r, "params": 2 * SIDE * r, "tv": float(E.tv_dist(qa, q)),
                     "auc": auc_of_dist(qa, st_te, st_atk), "frob_err": float(ferr)})
    return rows


def bitgroup_curve(st_tr, q, st_te, st_atk):
    rows = []
    for name, gr in BITGROUPS.items():
        qd, params = PB2.group_product_dist(np.asarray(st_tr, int), gr, NSTATES)
        rows.append({"config": name, "params": int(params), "tv": float(E.tv_dist(qd, q)),
                     "auc": auc_of_dist(qd, st_te, st_atk)})
    return rows


def reach_params(rows, target_auc):
    hit = [r for r in rows if r["auc"] >= target_auc]
    aucs = [r["auc"] for r in rows]
    nonmono = any(aucs[i + 1] < aucs[i] - 1e-9 for i in range(len(aucs) - 1))
    if not hit:
        return {"reached": False, "min_params": None, "nonmonotonic": nonmono,
                "max_auc": float(max(aucs)), "max_auc_params": int(rows[int(np.argmax(aucs))]["params"])}
    best = min(hit, key=lambda r: r["params"])
    return {"reached": True, "min_params": int(best["params"]),
            "at": best.get("config") or ("r=%d" % best["rank"]),
            "auc_there": best["auc"], "nonmonotonic": nonmono}


def best_within_budget(rows, budget):
    elig = [r for r in rows if r["params"] <= budget]
    if not elig:
        return None
    b = max(elig, key=lambda r: r["auc"])
    return {"config": b.get("config") or ("r=%d" % b["rank"]), "params": int(b["params"]),
            "auc": b["auc"], "tv": b["tv"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 80)
    E.log("UNSW 효율+견고성: QCBM(180p) vs 고전(SVD/NMF/비트그룹) — 재학습 없음")
    E.log("=" * 80)

    if not os.path.exists(NPZ):
        E.log("!!! %s 없음 — 학습 미완료. 중단." % NPZ); sys.exit(1)
    z = np.load(NPZ, allow_pickle=False)
    d = TUNSW.build_unsw()
    q = np.asarray(d["q_train"], float)
    st_tr, st_te, st_atk = d["st_tr"], d["st_te"], d["st_atk"]
    occ = int(d["occ"])

    # --- 정합 게이트 ---
    if not np.allclose(q, z["q_train"]):
        E.log("!!! q_train 재현 불일치 -> 중단."); sys.exit(2)
    p_finals = z["p_finals"]; seeds = z["seeds"]; mmd2 = z["mmd2"].astype(float); tv_arr = z["tv"].astype(float)
    rec_auc = np.array([auc_of_dist(p_finals[i], st_te, st_atk) for i in range(len(p_finals))])
    if float(np.max(np.abs(rec_auc - z["aucs"]))) > 1e-9:
        E.log("!!! QCBM AUC 재계산 불일치 -> 중단."); sys.exit(2)
    auc_emp = auc_of_dist(q, st_te, st_atk)
    auc_marg = auc_of_dist(marginal_dist(st_tr, NQ), st_te, st_atk)
    if abs(auc_emp - REF_EMP) > TOL or abs(auc_marg - REF_MARG) > TOL:
        E.log("!!! emp/marg 정합 불일치 emp=%.4f marg=%.4f -> 중단." % (auc_emp, auc_marg)); sys.exit(3)
    best_idx = int(np.argmin(mmd2))            # 실제 선택규칙: 최소 MMD²
    best_seed = int(seeds[best_idx]); best_auc = float(rec_auc[best_idx]); best_tv = float(tv_arr[best_idx])
    E.log("[정합] occ=%d | emp=%.4f marg=%.4f | QCBM best(min-MMD² seed%d) AUC=%.4f TV=%.4f"
          % (occ, auc_emp, auc_marg, best_seed, best_auc, best_tv))

    # --- 고전 곡선 ---
    E.log("고전 곡선 계산: SVD/NMF/비트그룹 (10bit, low-rank params=64r)...")
    svd = svd_curve(q, st_te, st_atk); nmf = nmf_curve(q, st_te, st_atk)
    bg = bitgroup_curve(st_tr, q, st_te, st_atk)
    allrows = ([{"family": "SVD", "name": "r=%d" % r["rank"], **r} for r in svd]
               + [{"family": "NMF", "name": "r=%d" % r["rank"], **r} for r in nmf]
               + [{"family": "bitgroup", "name": r["config"], **r} for r in bg])

    # === 분석2: 효율 ===
    reach = {"SVD": reach_params(svd, best_auc), "NMF": reach_params(nmf, best_auc),
             "bitgroup": reach_params(bg, best_auc)}
    budget = {"SVD": best_within_budget(svd, QCBM_PARAMS),
              "NMF": best_within_budget(nmf, QCBM_PARAMS),
              "bitgroup": best_within_budget(bg, QCBM_PARAMS)}
    le180 = [r for r in allrows if r["params"] <= QCBM_PARAMS]
    cbest = max(le180, key=lambda r: r["auc"])     # 고전 180p-이하 최고
    cbest_auc = float(cbest["auc"])

    # === 분석3: 견고성 + 선택 ===
    aucs = rec_auc
    stat = {"min": float(aucs.min()), "p25": float(np.percentile(aucs, 25)),
            "median": float(np.median(aucs)), "p75": float(np.percentile(aucs, 75)),
            "p90": float(np.percentile(aucs, 90)), "max": float(aucs.max()), "std": float(aucs.std())}
    n_beat = int((aucs > cbest_auc).sum())
    corr_mmd_auc = float(np.corrcoef(mmd2, aucs)[0, 1])
    corr_tv_auc = float(np.corrcoef(tv_arr, aucs)[0, 1])
    sp_mmd_auc = float(spearmanr(mmd2, aucs).correlation)
    rank_of_pick = int((aucs >= best_auc).sum())   # min-MMD² 모델 AUC 순위(상위 몇 번째)
    reach_median = {fam: reach_params([r for r in allrows if r["family"] == fam], stat["median"])
                    for fam in ("SVD", "NMF", "bitgroup")}

    E.log("-" * 80)
    E.log("[고전 180p-이하 최고] %s (%dp, AUC=%.4f)" % (cbest["name"], cbest["params"], cbest_auc))
    E.log("[효율 기준1] QCBM best AUC=%.4f 도달 최소 고전 params:" % best_auc)
    for nm in ("SVD", "NMF", "bitgroup"):
        rr = reach[nm]
        if rr["reached"]:
            E.log("  %-9s %d params (%s, %.4f) %s" % (nm, rr["min_params"], rr["at"], rr["auc_there"],
                  ">180" if rr["min_params"] > QCBM_PARAMS else "<=180"))
        else:
            E.log("  %-9s 미도달 (max %.4f @%dp)" % (nm, rr["max_auc"], rr["max_auc_params"]))
    E.log("[효율 기준2] params<=180 동일예산 고전 최대 AUC (vs QCBM best %.4f):" % best_auc)
    for nm in ("SVD", "NMF", "bitgroup"):
        b = budget[nm]
        if b:
            E.log("  %-9s %.4f (%s,%dp) %s" % (nm, b["auc"], b["config"], b["params"],
                  "고전우위" if b["auc"] >= best_auc else "QCBM우위"))
    E.log("[견고성] 고전최고(%.4f) 초과 QCBM restart = %d/16" % (cbest_auc, n_beat))
    E.log("[분포] min %.4f p25 %.4f median %.4f p75 %.4f p90 %.4f max %.4f"
          % (stat["min"], stat["p25"], stat["median"], stat["p75"], stat["p90"], stat["max"]))
    E.log("[위치] 고전최고 대비: best %s p90 %s p75 %s median %s"
          % ("위" if best_auc > cbest_auc else "아래", "위" if stat["p90"] > cbest_auc else "아래",
             "위" if stat["p75"] > cbest_auc else "아래", "위" if stat["median"] > cbest_auc else "아래"))
    E.log("[선택] corr(MMD²,AUC)=%.3f (Spearman %.3f) corr(TV,AUC)=%.3f | min-MMD²(seed%d) AUC순위 상위%d/16"
          % (corr_mmd_auc, sp_mmd_auc, corr_tv_auc, best_seed, rank_of_pick))

    eff_payload = {
        "encoding": "공통 10bit (1024), UNSW/ALL, M2 flow_iat. raw MWU. QCBM 재학습 없음.",
        "param_def": {"lowrank": "64r (32xr+rx32)", "bitgroup": "sum(2^b-1)", "qcbm": "L*nq*3=180",
                      "emp_joint": "점유 상태수", "marginal": "비트수(10)"},
        "consistency": {"emp_joint": auc_emp, "marginal": auc_marg, "occupied": occ,
                        "qcbm_best_seed": best_seed, "qcbm_best_auc": best_auc, "qcbm_best_tv": best_tv},
        "ref": {"marginal": REF_MARG, "emp_joint": REF_EMP},
        "qcbm_params": QCBM_PARAMS,
        "svd": svd, "nmf": nmf, "bitgroup": bg,
        "criterion1_reach_qcbm_auc": {"target_auc": best_auc, "result": reach},
        "criterion2_same_budget_180": {"qcbm_auc": best_auc, "classical_best": budget},
        "classical_best_le180": {"name": cbest["name"], "params": int(cbest["params"]), "auc": cbest_auc},
        "note": "사실 측정만. 판정은 웹 AI.",
    }
    with open(os.path.join(OUT, "param_efficiency_unsw.json"), "w") as f:
        json.dump(eff_payload, f, indent=2, ensure_ascii=False)

    rob_payload = {
        "encoding": "공통 10bit (1024), UNSW/ALL. raw MWU. QCBM 재학습 없음.",
        "qcbm_params": QCBM_PARAMS,
        "qcbm_auc_all_sorted": sorted(round(float(a), 4) for a in aucs),
        "qcbm_auc_stats": stat,
        "qcbm_best": {"seed": best_seed, "auc": best_auc, "tv": best_tv,
                      "select_rule": "min MMD² (라벨 무사용)", "auc_rank_top": rank_of_pick},
        "classical_best_le180": {"name": cbest["name"], "params": int(cbest["params"]), "auc": cbest_auc},
        "robustness": {
            "n_restarts_beat_classical_best": n_beat, "frac_beat": n_beat / len(aucs),
            "best_above": bool(best_auc > cbest_auc), "p90_above": bool(stat["p90"] > cbest_auc),
            "p75_above": bool(stat["p75"] > cbest_auc), "median_above": bool(stat["median"] > cbest_auc),
            "classical_params_to_reach_qcbm_median": reach_median,
        },
        "selection_robustness": {
            "corr_mmd2_auc": corr_mmd_auc, "spearman_mmd2_auc": sp_mmd_auc, "corr_tv_auc": corr_tv_auc,
            "min_mmd2_auc": best_auc, "min_mmd2_auc_rank_top": rank_of_pick,
            "note": "MMD²↔AUC 상관 0 근처면 라벨없는 선택이 고AUC 보장 못 함(Bot: 0.077).",
        },
        "per_restart": [{"seed": int(seeds[i]), "auc": float(aucs[i]), "mmd2": float(mmd2[i]),
                         "tv": float(tv_arr[i])} for i in range(len(aucs))],
        "note": "사실 측정만. 판정은 웹 AI.",
    }
    with open(os.path.join(OUT, "unsw_robustness.json"), "w") as f:
        json.dump(rob_payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/param_efficiency_unsw.json , results2/unsw_robustness.json")

    # --- 그림 ---
    try:
        fig, ax = plt.subplots(1, 2, figsize=(17, 6.3))
        a = ax[0]
        a.plot([r["params"] for r in svd], [r["auc"] for r in svd], "o-", color="steelblue", ms=4, label="SVD")
        a.plot([r["params"] for r in nmf], [r["auc"] for r in nmf], "^-", color="purple", ms=4, label="NMF")
        a.plot([r["params"] for r in bg], [r["auc"] for r in bg], "s-", color="darkorange", ms=5, label="bit-group")
        rng = np.random.default_rng(0)
        jit = QCBM_PARAMS * (1 + 0.04 * (rng.random(len(aucs)) - 0.5))
        a.scatter(jit, aucs, color="crimson", s=30, alpha=0.6, zorder=5, label="QCBM 16 restarts")
        a.scatter([QCBM_PARAMS], [best_auc], color="crimson", marker="*", s=220, edgecolor="k", zorder=7,
                  label="QCBM best (min-MMD², %.3f)" % best_auc)
        a.scatter([QCBM_PARAMS], [stat["median"]], color="black", marker="_", s=420, zorder=7,
                  label="QCBM median (%.3f)" % stat["median"])
        a.axhline(cbest_auc, color="darkorange", ls="--", alpha=.7, label="classical best <=180p (%.3f)" % cbest_auc)
        a.axhline(auc_emp, color="green", ls=":", alpha=.7, label="emp_joint (%.3f)" % auc_emp)
        a.axvline(QCBM_PARAMS, color="k", ls=":", alpha=.4)
        a.set_xscale("log"); a.set_xlabel("parameters (log)"); a.set_ylabel("AUC (raw MWU)")
        a.set_title("UNSW: QCBM 180p vs classical curve\n(%d/16 restarts beat classical-best %.3f)" % (n_beat, cbest_auc))
        a.legend(fontsize=7, loc="lower right"); a.grid(alpha=.3, which="both")
        a2 = ax[1]
        a2.hist(aucs, bins=np.linspace(min(0.55, aucs.min() - 0.02), max(0.9, aucs.max() + 0.02), 16),
                color="crimson", alpha=0.6, edgecolor="k")
        for val, c, lab in [(cbest_auc, "darkorange", "classical best <=180p"),
                            (stat["median"], "black", "QCBM median"), (best_auc, "crimson", "QCBM best"),
                            (auc_emp, "green", "emp_joint"), (REF_MARG, "gray", "marginal")]:
            a2.axvline(val, color=c, ls="--", lw=2, label="%s (%.3f)" % (lab, val))
        a2.set_xlabel("AUC"); a2.set_ylabel("# restarts")
        a2.set_title("UNSW QCBM 16-restart AUC distribution at 180p"); a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
        plt.suptitle("UNSW quantum-efficiency robustness: best-luck or distributional? (no retraining)")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT, "param_eff_unsw.png"), dpi=125); plt.close()
        E.log("그림 저장: results2/param_eff_unsw.png")
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)
    E.log("DONE")


if __name__ == "__main__":
    main()
