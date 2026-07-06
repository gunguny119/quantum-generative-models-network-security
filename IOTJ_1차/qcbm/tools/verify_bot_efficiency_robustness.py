#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot 양자 효율 우위의 견고성 검증 — 216p QCBM 우위가 best 운인지 분포적인지 (재학습 없음)
=========================================================================================
직전 결과: 216p 동일 예산에서 QCBM best(seed7, AUC 0.7458) > 고전 216p-이하 최고(비트그룹 6+6, 0.7064).
그러나 QCBM 16 restart는 median 0.665(<0.706). 본 작업은 16 분포를 고전선과 비교해 견고성을 사실로 규명.

핵심 질문:
  (a) 고전 216p-이하 최고(0.7064)를 넘는 QCBM restart 수 / 16.
  (b) QCBM 분포(min/median/75th/90th/max/std)가 고전선 대비 어디에 위치(위/아래).
  (c) best 전용 우위인가, 상위분위(75/90th)·median에서도 성립인가.
  (d) 고전이 QCBM median(0.665)에 도달하는 데 필요한 params(곡선 역산).
  (e) 모델선택 견고성: 실제 선택규칙(min MMD², 라벨 무사용)이 고전초과 restart를 고르는가
      — corr(MMD², AUC), min-MMD² restart의 AUC 순위.
  (f) TV-AUC 해리: restart 전반에서 TV(분포적합)와 AUC(탐지)가 무관한지.

재사용(기존 소스 무수정, import): experiment2(auc_score, tv_dist), train_qcbm_bot_resumable(build_bot).
고전 곡선은 param_efficiency_bot.json 로드(재생성 안 함). raw MWU 유지(0.5 미만도 그대로). 판정은 웹 AI.
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

import experiment2 as E                  # noqa: E402  auc_score, tv_dist
import train_qcbm_bot_resumable as TBOT  # noqa: E402  build_bot()

OUT = "results2"
NQ = 12
EPS = 1e-12
QCBM_PARAMS = 6 * NQ * 3                  # 216
NPZ = os.path.join(OUT, "qcbm_bot_restarts.npz")
BOT_JSON = os.path.join(OUT, "qcbm_bot.json")
EFF_JSON = os.path.join(OUT, "param_efficiency_bot.json")
TOL = 1e-9


def auc_of_dist(p, st_te, st_atk):
    p = np.asarray(p, dtype=float)
    s_te = -np.log(p[st_te] + EPS)
    s_atk = -np.log(p[st_atk] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def classical_rows(eff):
    """모든 고전 점을 (name, params, tv, auc) 평탄화."""
    rows = []
    for r in eff["svd"]:
        rows.append({"name": "SVD r=%d" % r["rank"], "family": "SVD",
                     "params": r["params"], "tv": r["tv"], "auc": r["auc"]})
    for r in eff["nmf"]:
        rows.append({"name": "NMF r=%d" % r["rank"], "family": "NMF",
                     "params": r["params"], "tv": r["tv"], "auc": r["auc"]})
    for r in eff["bitgroup"]:
        rows.append({"name": r["config"], "family": "bitgroup",
                     "params": r["params"], "tv": r["tv"], "auc": r["auc"]})
    return rows


def min_params_to_reach(rows, target):
    """AUC>=target 인 고전 점 중 최소 params (보수적: 가장 적은 params로 도달)."""
    hit = [r for r in rows if r["auc"] >= target]
    if not hit:
        return None
    b = min(hit, key=lambda r: r["params"])
    return {"params": int(b["params"]), "name": b["name"], "auc": b["auc"], "family": b["family"]}


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 80)
    E.log("Bot 양자 효율 우위 견고성 검증 — 216p QCBM 분포 vs 고전 (재학습 없음)")
    E.log("=" * 80)

    z = np.load(NPZ, allow_pickle=False)
    with open(EFF_JSON) as f:
        eff = json.load(f)
    with open(BOT_JSON) as f:
        botj = json.load(f)

    # --- 재현 + 정합 게이트 (16 p_final로 AUC 재계산) ---
    d = TBOT.build_bot()
    q = np.asarray(d["q_train"], float)
    st_te, st_atk, st_tr = d["st_te"], d["st_atk"], d["st_tr"]
    if not np.allclose(q, z["q_train"]):
        E.log("!!! q_train 재현 불일치 -> 중단."); sys.exit(2)
    if int(d["occ"]) != int(botj["occupied_train_states"]):
        E.log("!!! 점유 불일치 occ=%d != %d -> 중단." % (d["occ"], botj["occupied_train_states"])); sys.exit(2)
    p_finals = z["p_finals"]; seeds = z["seeds"]; mmd2 = z["mmd2"]; tv_arr = z["tv"]
    rec = np.array([auc_of_dist(p_finals[i], st_te, st_atk) for i in range(len(p_finals))])
    diff = float(np.max(np.abs(rec - z["aucs"])))
    jdist = botj["auc_distribution"]
    ok = (diff <= TOL and abs(np.median(rec) - jdist["median"]) <= 1e-4
          and abs(rec.max() - jdist["max"]) <= 1e-4)
    E.log("[정합] 16 AUC 재계산 max|Δ|=%.2e | median 재=%.4f(json %.4f) max 재=%.4f(json %.4f) -> %s"
          % (diff, np.median(rec), jdist["median"], rec.max(), jdist["max"], "OK" if ok else "불일치"))
    if not ok:
        E.log("!!! 정합 불일치 -> 중단."); sys.exit(3)

    aucs = rec                          # 재계산값 사용(=저장값)
    best_idx = int(np.argmin(mmd2))     # 실제 선택규칙: 최소 MMD²
    best_seed = int(seeds[best_idx]); best_auc = float(aucs[best_idx]); best_tv = float(tv_arr[best_idx])

    # --- 분포 통계 ---
    stat = {"min": float(aucs.min()), "p25": float(np.percentile(aucs, 25)),
            "median": float(np.median(aucs)), "mean": float(aucs.mean()),
            "p75": float(np.percentile(aucs, 75)), "p90": float(np.percentile(aucs, 90)),
            "max": float(aucs.max()), "std": float(aucs.std())}

    # --- 고전 기준선 ---
    crows = classical_rows(eff)
    le216 = [r for r in crows if r["params"] <= QCBM_PARAMS]
    cbest = max(le216, key=lambda r: r["auc"])      # 216p-이하 고전 최고
    cbest_auc = float(cbest["auc"])
    E.log("[고전 기준] 216p-이하 최고 = %s (%dp, AUC=%.4f)" % (cbest["name"], cbest["params"], cbest_auc))

    # --- 견고성 지표 ---
    n_beat = int((aucs > cbest_auc).sum())          # (a)
    frac_beat = n_beat / len(aucs)
    best_above = best_auc > cbest_auc
    p75_above = stat["p75"] > cbest_auc
    p90_above = stat["p90"] > cbest_auc
    median_above = stat["median"] > cbest_auc

    # (d) 고전이 QCBM median 도달 필요 params (가족별 + 전체 최소)
    reach_median = {fam: min_params_to_reach([r for r in crows if r["family"] == fam], stat["median"])
                    for fam in ("SVD", "NMF", "bitgroup")}
    reach_median_any = min_params_to_reach(crows, stat["median"])
    # 참고: 고전이 QCBM best 도달 필요 params
    reach_best_any = min_params_to_reach(crows, best_auc)

    # (e) 모델선택 견고성
    corr_mmd_auc = float(np.corrcoef(mmd2, aucs)[0, 1])
    corr_tv_auc = float(np.corrcoef(tv_arr, aucs)[0, 1])
    rank_of_best = int((aucs >= best_auc).sum())    # min-MMD² restart의 AUC 순위(상위 몇 번째)
    expected_pick_auc = stat["median"]              # 선택규칙이 AUC와 무관하면 기대 ~ median

    E.log("-" * 80)
    E.log("[견고성] 고전최고(%.4f) 초과 restart = %d/16 (%.0f%%)" % (cbest_auc, n_beat, 100 * frac_beat))
    E.log("[분포] min %.4f | p25 %.4f | median %.4f | p75 %.4f | p90 %.4f | max %.4f | std %.4f"
          % (stat["min"], stat["p25"], stat["median"], stat["p75"], stat["p90"], stat["max"], stat["std"]))
    E.log("[위치] 고전최고(%.4f) 대비: best %s | p90 %s | p75 %s | median %s"
          % (cbest_auc, "위" if best_above else "아래", "위" if p90_above else "아래",
             "위" if p75_above else "아래", "위" if median_above else "아래"))
    E.log("[median 도달 고전 params] any-min=%s | %s"
          % (reach_median_any, {k: (v["params"] if v else None) for k, v in reach_median.items()}))
    E.log("[best 도달 고전 params] any-min=%s" % (reach_best_any,))
    E.log("[선택견고성] corr(MMD²,AUC)=%.3f corr(TV,AUC)=%.3f | min-MMD²(seed%d) AUC=%.4f 순위=상위%d/16"
          % (corr_mmd_auc, corr_tv_auc, best_seed, best_auc, rank_of_best))

    payload = {
        "encoding": "B2 12bit (4096), Bot. raw MWU(s=-log p), 방향반전 없음. QCBM 재학습 없음.",
        "consistency": {"auc_recompute_maxdiff": diff, "occ": int(d["occ"]),
                        "median_match": True, "ok": True},
        "qcbm_params": QCBM_PARAMS,
        "qcbm_auc_all_sorted": sorted(round(a, 4) for a in aucs.tolist()),
        "qcbm_auc_stats": stat,
        "qcbm_best": {"seed": best_seed, "auc": best_auc, "tv": best_tv,
                      "select_rule": "min MMD² (라벨 무사용)", "auc_rank_top": rank_of_best},
        "classical_best_le216": {"name": cbest["name"], "params": int(cbest["params"]), "auc": cbest_auc},
        "robustness": {
            "n_restarts_beat_classical_best": n_beat, "frac_beat": frac_beat,
            "best_above": bool(best_above), "p90_above": bool(p90_above),
            "p75_above": bool(p75_above), "median_above": bool(median_above),
            "classical_params_to_reach_qcbm_median": {"any_min": reach_median_any, "by_family": reach_median},
            "classical_params_to_reach_qcbm_best": reach_best_any,
        },
        "selection_robustness": {
            "corr_mmd2_auc": corr_mmd_auc, "corr_tv_auc": corr_tv_auc,
            "min_mmd2_auc": best_auc, "min_mmd2_auc_rank_top": rank_of_best,
            "expected_pick_auc_if_uncorrelated": expected_pick_auc,
            "note": "MMD²↔AUC 상관이 0에 가까우면 라벨없는 선택(min MMD²)이 고AUC를 보장 못 함.",
        },
        "tv_auc_dissociation": {
            "corr_tv_auc": corr_tv_auc, "qcbm_tv_range": [float(tv_arr.min()), float(tv_arr.max())],
            "per_restart": [{"seed": int(seeds[i]), "tv": float(tv_arr[i]), "auc": float(aucs[i])}
                            for i in range(len(aucs))],
            "note": "QCBM TV(0.49~0.70)는 같은 params 고전보다 나쁘나 AUC는 상위권 — 적합도≠탐지.",
        },
        "classical_curve_ref": {"le216_points": le216, "classical_best": cbest},
        "note": "사실 측정만. 판정은 웹 AI. 기존 소스/데이터/결과 무수정.",
    }
    with open(os.path.join(OUT, "bot_efficiency_robustness.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/bot_efficiency_robustness.json")

    # --- 그림 ---
    try:
        svd = eff["svd"]; nmf = eff["nmf"]; bg = eff["bitgroup"]
        fig, ax = plt.subplots(1, 2, figsize=(16, 6.2))
        # 패널1: 고전 AUC-params 곡선 + QCBM 216p 분포 overlay
        a = ax[0]
        a.plot([r["params"] for r in svd], [r["auc"] for r in svd], "o-", color="steelblue", ms=4, label="SVD")
        a.plot([r["params"] for r in nmf], [r["auc"] for r in nmf], "^-", color="purple", ms=4, label="NMF")
        a.plot([r["params"] for r in bg], [r["auc"] for r in bg], "s-", color="darkorange", ms=5, label="bit-group")
        # QCBM 16점: 216p 세로 산점(약간 jitter)
        rng = np.random.default_rng(0)
        jitter = QCBM_PARAMS * (1 + 0.04 * (rng.random(len(aucs)) - 0.5))
        a.scatter(jitter, aucs, color="crimson", s=30, alpha=0.6, zorder=5, label="QCBM 16 restarts")
        a.scatter([QCBM_PARAMS], [best_auc], color="crimson", marker="*", s=200, edgecolor="k",
                  zorder=7, label="QCBM best (min-MMD², %.3f)" % best_auc)
        a.scatter([QCBM_PARAMS], [stat["median"]], color="black", marker="_", s=400, zorder=7,
                  label="QCBM median (%.3f)" % stat["median"])
        a.axhline(cbest_auc, color="darkorange", ls="--", alpha=.7,
                  label="classical best <=216p (%.3f)" % cbest_auc)
        a.axvline(QCBM_PARAMS, color="k", ls=":", alpha=.4)
        a.set_xscale("log"); a.set_xlabel("parameters (log)"); a.set_ylabel("AUC (raw MWU)")
        a.set_title("Bot: QCBM 216p distribution vs classical AUC curve\n(%d/16 restarts beat classical-best %.3f)"
                    % (n_beat, cbest_auc))
        a.legend(fontsize=7, loc="lower right"); a.grid(alpha=.3, which="both")
        # 패널2: QCBM AUC 분포 히스토 + 기준선
        a2 = ax[1]
        a2.hist(aucs, bins=np.linspace(0.5, 0.85, 15), color="crimson", alpha=0.6, edgecolor="k")
        for val, c, lab in [(cbest_auc, "darkorange", "classical best <=216p"),
                            (stat["median"], "black", "QCBM median"),
                            (best_auc, "crimson", "QCBM best"),
                            (stat["p90"], "green", "QCBM p90")]:
            a2.axvline(val, color=c, ls="--", lw=2, label="%s (%.3f)" % (lab, val))
        a2.set_xlabel("AUC"); a2.set_ylabel("# restarts")
        a2.set_title("QCBM 16-restart AUC distribution at 216p"); a2.legend(fontsize=8); a2.grid(alpha=.3, axis="y")
        plt.suptitle("Bot quantum-efficiency robustness: is the 216p edge best-luck or distributional? (no retraining)")
        plt.tight_layout()
        fp = os.path.join(OUT, "bot_efficiency_robustness.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
