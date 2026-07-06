#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot 파라미터 효율 비교 — QCBM(216p) vs 고전(저랭크 SVD/NMF/비트그룹), AUC·TV 두 기준 (재학습 없음)
====================================================================================================
질문(후보 A 최종 사실 측정):
  (기준1, AUC) QCBM best AUC=0.7458 을 고전이 몇 파라미터로 내는가? 216보다 많은가/적은가?
  (기준2, 동일예산) params≤216 에서 고전 저랭크/NMF/비트그룹 AUC vs QCBM 0.746 — 누가 높은가?
배경: HOIC(단순·저랭크)에선 고전 압승(SVD r=2가 QCBM TV 도달). Bot은 고랭크(r=7)·강한 joint
      의존(marginal 0.287 역방향, emp_joint 0.875) — 고전 저랭크가 비싸질 수 있는 분포.

평가: s=-log p_benign, attack=positive, raw Mann-Whitney U (experiment2.auc_score, 방향반전 없음).
재사용(기존 소스 무수정, import):
  experiment2: tv_dist, auc_score, empirical_dist.
  diagnose_param_efficiency_b2: normalize_clip, nmf_factorize, group_product_dist (저랭크/NMF/비트그룹 + 파라미터 정의).
  train_qcbm_bot_resumable: build_bot() (q_train/st_tr/st_te/st_atk 재현, 학습 split과 동일).
QCBM 재학습 없음(results2/qcbm_bot_restarts.npz의 16 p_final 사용). 새 dependency 없음. 판정은 웹 AI.
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

import experiment2 as E                     # noqa: E402  tv_dist, auc_score, empirical_dist
import diagnose_param_efficiency_b2 as PB2  # noqa: E402  normalize_clip, nmf_factorize, group_product_dist
import train_qcbm_bot_resumable as TBOT     # noqa: E402  build_bot()

OUT = "results2"
NQ = 12
NSTATES = 1 << NQ
SIDE = 64
EPS = 1e-12
NPZ = os.path.join(OUT, "qcbm_bot_restarts.npz")
BOT_JSON = os.path.join(OUT, "qcbm_bot.json")
HOIC_JSON = os.path.join(OUT, "param_efficiency_b2.json")
QCBM_PARAMS = 6 * NQ * 3                    # 216
SVD_RMAX = 40
NMF_RMAX = 20
REF = {"marginal": 0.2870, "emp_joint": 0.8749, "qcbm_best_auc": 0.7458, "qcbm_best_tv": 0.4933}
TOL = 0.0015

# 비트그룹 스펙트럼 (PB2.bitgroup_curve와 동일 정의)
BITGROUPS = {
    "marginal(1x12)": [[i] for i in range(NQ)],
    "2bitx6":         [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
    "3bitx4":         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
    "4bitx3":         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
    "6+6":            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
    "8+4":            [list(range(8)), [8, 9, 10, 11]],
    "9+3":            [list(range(9)), [9, 10, 11]],
    "full(1x12)":     [list(range(NQ))],
}


def auc_of_dist(p, st_te, st_atk):
    """분포 p로 s=-log p, attack=positive, raw MWU AUC (anomaly_eval과 동일 recipe)."""
    p = np.asarray(p, dtype=float)
    s_te = -np.log(p[st_te] + EPS)
    s_atk = -np.log(p[st_atk] + EPS)
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


# ---------------------------------------------------------------------------
def svd_curve(q, st_te, st_atk):
    M = np.asarray(q, float).reshape(SIDE, SIDE)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rows = []
    for r in range(1, SVD_RMAX + 1):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        qa = PB2.normalize_clip(approx.reshape(-1))
        rows.append({"rank": r, "params": 128 * r, "tv": float(E.tv_dist(qa, q)),
                     "auc": auc_of_dist(qa, st_te, st_atk)})
    return rows


def nmf_curve(q, st_te, st_atk):
    M = np.asarray(q, float).reshape(SIDE, SIDE)
    rows = []
    for r in range(1, NMF_RMAX + 1):
        approx, ferr = PB2.nmf_factorize(M, r)
        qa = PB2.normalize_clip(approx.reshape(-1))
        rows.append({"rank": r, "params": 128 * r, "tv": float(E.tv_dist(qa, q)),
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
    """AUC>=target 인 행 중 최소 params. 비단조성 플래그도 반환."""
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
    """params<=budget 행 중 최대 AUC(동일예산 비교)."""
    elig = [r for r in rows if r["params"] <= budget]
    if not elig:
        return None
    b = max(elig, key=lambda r: r["auc"])
    return {"config": b.get("config") or ("r=%d" % b["rank"]), "params": int(b["params"]),
            "auc": b["auc"], "tv": b["tv"]}


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 80)
    E.log("Bot 파라미터 효율: QCBM(216p) vs 고전(SVD/NMF/비트그룹), AUC·TV (재학습 없음)")
    E.log("=" * 80)

    # --- 재현 + 정합 게이트 ---
    z = np.load(NPZ, allow_pickle=False)
    d = TBOT.build_bot()
    q = np.asarray(d["q_train"], float)
    st_tr, st_te, st_atk = d["st_tr"], d["st_te"], d["st_atk"]
    occ = int(d["occ"])
    if not np.allclose(q, z["q_train"]):
        E.log("!!! q_train 재현 불일치 -> 중단."); sys.exit(2)
    p_finals = z["p_finals"]
    rec_auc = np.array([auc_of_dist(p_finals[i], st_te, st_atk) for i in range(len(p_finals))])
    if float(np.max(np.abs(rec_auc - z["aucs"]))) > 1e-9:
        E.log("!!! QCBM AUC 재계산 불일치 -> 중단."); sys.exit(2)
    auc_emp = auc_of_dist(q, st_te, st_atk)
    auc_marg = auc_of_dist(marginal_dist(st_tr, NQ), st_te, st_atk)
    best_idx = int(np.argmin(z["mmd2"]))
    best_seed = int(z["seeds"][best_idx])
    best_auc = float(rec_auc[best_idx]); best_tv = float(z["tv"][best_idx])
    E.log("[정합] occ=%d | emp_joint=%.4f(기준%.4f) marginal=%.4f(기준%.4f) | QCBM best(seed%d) AUC=%.4f(기준%.4f) TV=%.4f"
          % (occ, auc_emp, REF["emp_joint"], auc_marg, REF["marginal"],
             best_seed, best_auc, REF["qcbm_best_auc"], best_tv))
    okgate = (abs(auc_emp - REF["emp_joint"]) <= TOL and abs(auc_marg - REF["marginal"]) <= TOL
              and abs(best_auc - REF["qcbm_best_auc"]) <= TOL)
    if not okgate:
        E.log("!!! 정합 불일치 -> 중단."); sys.exit(3)
    E.log("[정합] OK.")

    # --- 고전 곡선 (params, TV, AUC) ---
    E.log("고전 곡선 계산: SVD(r=1..%d) / NMF(r=1..%d) / 비트그룹..." % (SVD_RMAX, NMF_RMAX))
    svd = svd_curve(q, st_te, st_atk)
    nmf = nmf_curve(q, st_te, st_atk)
    bg = bitgroup_curve(st_tr, q, st_te, st_atk)

    # --- QCBM 16점 ---
    qcbm_pts = [{"seed": int(z["seeds"][i]), "params": QCBM_PARAMS,
                 "tv": float(z["tv"][i]), "auc": float(rec_auc[i])} for i in range(len(p_finals))]
    emp_point = {"params": occ, "tv": 0.0, "auc": auc_emp}
    marg_point = {"params": NQ, "tv": float(E.tv_dist(marginal_dist(st_tr, NQ), q)), "auc": auc_marg}

    # --- 두 기준 ---
    target = best_auc
    reach = {"SVD": reach_params(svd, target), "NMF": reach_params(nmf, target),
             "bitgroup": reach_params(bg, target)}
    budget = {"SVD": best_within_budget(svd, QCBM_PARAMS),
              "NMF": best_within_budget(nmf, QCBM_PARAMS),
              "bitgroup": best_within_budget(bg, QCBM_PARAMS)}

    # --- 로그 표 ---
    E.log("-" * 80)
    E.log("[SVD] params=128r | TV | AUC")
    for r in svd[:12]:
        E.log("  r=%-2d p=%-5d TV=%.4f AUC=%.4f" % (r["rank"], r["params"], r["tv"], r["auc"]))
    E.log("[NMF] params=128r | TV | AUC")
    for r in nmf[:10]:
        E.log("  r=%-2d p=%-5d TV=%.4f AUC=%.4f" % (r["rank"], r["params"], r["tv"], r["auc"]))
    E.log("[비트그룹] params=sum(2^b-1) | TV | AUC")
    for r in bg:
        E.log("  %-14s p=%-5d TV=%.4f AUC=%.4f" % (r["config"], r["params"], r["tv"], r["auc"]))
    E.log("-" * 80)
    E.log("[기준1] QCBM AUC=%.4f 도달 최소 고전 params (216 대비):" % target)
    for nm in ["SVD", "NMF", "bitgroup"]:
        rr = reach[nm]
        if rr["reached"]:
            E.log("  %-9s %-5d params (%s, AUC=%.4f) %s %s"
                  % (nm, rr["min_params"], rr["at"], rr["auc_there"],
                     ">216" if rr["min_params"] > QCBM_PARAMS else "<=216",
                     "[AUC 비단조]" if rr["nonmonotonic"] else ""))
        else:
            E.log("  %-9s 미도달 (max AUC=%.4f @%dp) %s"
                  % (nm, rr["max_auc"], rr["max_auc_params"], "[AUC 비단조]" if rr["nonmonotonic"] else ""))
    E.log("[기준2] params<=216 동일예산 고전 최대 AUC (vs QCBM %.4f):" % best_auc)
    for nm in ["SVD", "NMF", "bitgroup"]:
        b = budget[nm]
        if b:
            E.log("  %-9s %.4f (%s, %dp)  %s"
                  % (nm, b["auc"], b["config"], b["params"], "고전우위" if b["auc"] >= best_auc else "QCBM우위"))

    # --- HOIC 비교 (있으면) ---
    hoic_cmp = None
    if os.path.exists(HOIC_JSON):
        with open(HOIC_JSON) as f:
            h = json.load(f)
        hoic_cmp = {"qcbm_tv": h["qcbm_point"]["tv"], "qcbm_params": h["qcbm_point"]["params"],
                    "svd_reach_tv": h["qcbm_tv_reach"]["svd"], "occupied": h["q_train_occupied"]}
        E.log("[HOIC 참고] QCBM(216p, TV=%.3f) TV도달 SVD: %s (점유 %d). HOIC는 고전 저랭크 압승."
              % (hoic_cmp["qcbm_tv"], hoic_cmp["svd_reach_tv"], hoic_cmp["occupied"]))

    payload = {
        "encoding": "B2 12bit (4096), Bot. 평가 raw MWU(s=-log p), 방향반전 없음.",
        "param_def": {"lowrank": "128r (64xr+rx64)", "bitgroup": "sum_g(2^{b_g}-1)",
                      "emp_joint": "점유 상태수", "marginal": "비트수(12)", "qcbm": "L*nq*3=216"},
        "consistency": {"emp_joint": auc_emp, "marginal": auc_marg,
                        "qcbm_best_seed": best_seed, "qcbm_best_auc": best_auc, "qcbm_best_tv": best_tv,
                        "occupied": occ},
        "ref": REF,
        "svd": svd, "nmf": nmf, "bitgroup": bg,
        "qcbm_points": qcbm_pts, "emp_joint_point": emp_point, "marginal_point": marg_point,
        "criterion1_reach_qcbm_auc": {"target_auc": target, "result": reach,
                                      "note": "AUC>=target 최소 params. 216보다 크면 양자 효율우위 방향."},
        "criterion2_same_budget_216": {"qcbm_auc": best_auc, "classical_best": budget,
                                       "note": "params<=216 에서 고전 최대 AUC. QCBM보다 낮으면 양자 효율우위 방향."},
        "hoic_compare": hoic_cmp,
        "note": "사실 측정만. 판정은 웹 AI. QCBM 재학습 없음. 기존 소스 무수정.",
    }
    with open(os.path.join(OUT, "param_efficiency_bot.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/param_efficiency_bot.json")

    # --- 그림 (TV 패널 + AUC 패널) ---
    try:
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        for a, ykey, ylab in [(ax[0], "tv", "TV vs q_train"), (ax[1], "auc", "AUC (raw MWU)")]:
            a.plot([r["params"] for r in svd], [r[ykey] for r in svd], "o-", color="steelblue",
                   label="SVD low-rank (128r)", ms=4)
            a.plot([r["params"] for r in nmf], [r[ykey] for r in nmf], "^-", color="purple",
                   label="NMF low-rank (128r)", ms=4)
            a.plot([r["params"] for r in bg], [r[ykey] for r in bg], "s-", color="darkorange",
                   label="bit-group", ms=5)
            # QCBM 16점 + best
            a.scatter([QCBM_PARAMS] * len(qcbm_pts), [p[ykey] for p in qcbm_pts],
                      color="crimson", s=28, alpha=0.55, zorder=4, label="QCBM 16 restarts")
            a.scatter([QCBM_PARAMS], [best_auc if ykey == "auc" else best_tv], color="crimson",
                      s=160, marker="*", zorder=6, edgecolor="k", label="QCBM best (216p)")
            # 끝점
            a.scatter([emp_point["params"]], [emp_point[ykey]], color="green", s=70, marker="D",
                      zorder=5, label="emp_joint (occ=%d)" % occ)
            a.scatter([marg_point["params"]], [marg_point[ykey]], color="gray", s=70, marker="v",
                      zorder=5, label="marginal (12p)")
            a.set_xscale("log"); a.set_xlabel("parameters (log)"); a.set_ylabel(ylab)
            a.grid(alpha=.3, which="both")
            if ykey == "auc":
                a.axhline(best_auc, color="crimson", ls="--", alpha=.5)
                a.axvline(QCBM_PARAMS, color="k", ls=":", alpha=.4)
                a.set_title("AUC vs params (QCBM best AUC=%.3f dashed)" % best_auc)
                a.legend(fontsize=7, loc="lower right")
            else:
                a.set_yscale("log"); a.set_title("TV vs params")
                a.legend(fontsize=7)
        plt.suptitle("Bot parameter efficiency: QCBM (216p) vs classical (SVD/NMF/bit-group) — no retraining")
        plt.tight_layout()
        fp = os.path.join(OUT, "param_efficiency_bot.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
