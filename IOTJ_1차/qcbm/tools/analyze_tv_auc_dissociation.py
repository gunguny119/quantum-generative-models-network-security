#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TV↔AUC 해리 메커니즘 분석 (재학습 없음, 저장 npz 분석)
======================================================
관찰: psh 제거 11비트 16 restart가 TV~0.10(분포 비슷하게 적합)인데 AUC 0.344~0.818로 산포.
질문: 왜? AUC는 분포 전체가 아니라 "공격이 떨어지는 소수 상태의 -log p 순위"로 정해진다.
      그 소수 상태에서 모델 확률이 갈리고, TV 오차가 어디 쌓이느냐가 AUC를 가르는가?

분석:
  (A) 공격 상태 집중도: 상위 k 상태가 공격의 몇 %.
  (B) restart별 (TV, AUC, 공격/benign 평균 -log p, 격차) + 공격집중 상태의 모델 확률 산포.
      극단(seed4 저AUC vs seed13 고AUC) 직접 비교.
  (C) TV 오차 |p-q_train|이 공격집중 상태 vs benign흔한 상태 중 어디 쌓이나, AUC와 연관.
  (D) MMD²/TV 순위 vs AUC 순위 Spearman 상관 (MMD-best 선택이 AUC를 얼마나 못 맞추나).

데이터: results2/qcbm_iat_no_psh_restarts.npz (16 p_final + q_train + st_te + st_atk). 재학습 없음.
재사용: experiment2(auc_score, tv_dist). 기존 소스 무수정. 판정/해석은 웹 AI.
"""
import os
import sys
import json

import numpy as np
from scipy.stats import spearmanr, pearsonr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if QCBM_DIR not in sys.path:
    sys.path.insert(0, QCBM_DIR)
os.chdir(QCBM_DIR)

import experiment2 as E  # noqa: E402  auc_score, tv_dist

OUT = "results2"
NPZ = os.path.join(OUT, "qcbm_iat_no_psh_restarts.npz")
NSTATES = 2048
EPS = 1e-12
MARG_REF, EMP_REF = 0.5136, 0.7773
PREV = {"min": 0.344, "median": 0.715, "max": 0.818, "std": 0.143, "exceed_emp": 4}


def auc_of(p, st_te, st_atk):
    s = np.concatenate([-np.log(p[st_te] + EPS), -np.log(p[st_atk] + EPS)])
    lab = np.concatenate([np.zeros(len(st_te), int), np.ones(len(st_atk), int)])
    return float(E.auc_score(s, lab))


def main():
    if not os.path.exists(NPZ):
        print("npz 없음:", NPZ); sys.exit(1)
    d = np.load(NPZ)
    pf = d["p_finals"]               # (16, 2048)
    q = d["q_train"]                 # emp_joint
    st_te = d["st_te"]; st_atk = d["st_atk"]
    seeds = d["seeds"]; tvs = d["tvs"]; mmd2 = d["mmd2"]
    n = len(pf)

    # --- 신뢰성: 16 AUC 재계산 + 이전값 일치 ---
    aucs = np.array([auc_of(pf[i], st_te, st_atk) for i in range(n)])
    stats = {"min": float(aucs.min()), "median": float(np.median(aucs)),
             "max": float(aucs.max()), "std": float(aucs.std())}
    exceed_emp = int((aucs > EMP_REF).sum())
    ok = (abs(stats["min"] - PREV["min"]) < 0.005 and abs(stats["max"] - PREV["max"]) < 0.005
          and exceed_emp == PREV["exceed_emp"])
    print("[검증] AUC min/med/max/std=%.3f/%.3f/%.3f/%.3f, emp초과 %d/16 -> 이전값 일치: %s"
          % (stats["min"], stats["median"], stats["max"], stats["std"], exceed_emp, ok))
    if not ok:
        print("!!! 이전값 불일치 -> 분석 신뢰성 전제 깨짐. 중단."); sys.exit(2)

    # 분포(state 단위) 카운트
    atk_cnt = np.bincount(st_atk, minlength=NSTATES).astype(float)
    ben_cnt = np.bincount(st_te, minlength=NSTATES).astype(float)
    atk_frac = atk_cnt / atk_cnt.sum()
    n_atk = st_atk.shape[0]

    # =========================================================
    # (A) 공격 상태 집중도
    # =========================================================
    order_atk = np.argsort(-atk_cnt)
    cum = np.cumsum(atk_cnt[order_atk]) / atk_cnt.sum()
    topk_cover = {k: float(cum[k - 1]) for k in [1, 3, 5, 10, 20]}
    n_atk_states = int((atk_cnt > 0).sum())
    print("[A] 공격 점유 상태=%d/2048 | 상위1=%.1f%% 상위3=%.1f%% 상위5=%.1f%% 상위10=%.1f%%"
          % (n_atk_states, 100 * topk_cover[1], 100 * topk_cover[3],
             100 * topk_cover[5], 100 * topk_cover[10]))
    TOPA = order_atk[:10]            # 공격 집중 상위 10 상태

    # =========================================================
    # (B) restart별 지표 + 공격집중 상태 모델확률 산포
    # =========================================================
    rows = []
    for i in range(n):
        p = pf[i]
        nlp_atk = float(np.mean(-np.log(p[st_atk] + EPS)))
        nlp_ben = float(np.mean(-np.log(p[st_te] + EPS)))
        rows.append({"seed": int(seeds[i]), "tv": float(tvs[i]), "mmd2": float(mmd2[i]),
                     "auc": float(aucs[i]),
                     "mean_neglogp_attack": nlp_atk, "mean_neglogp_benign": nlp_ben,
                     "gap_atk_minus_ben": nlp_atk - nlp_ben,
                     "p_top_attack_states": p[TOPA].tolist()})
    # 공격집중 상태에서 모델이 부여한 총확률(낮을수록 공격을 비정상으로 봄 -> 높은 AUC 기대)
    p_mass_on_topA = np.array([float(pf[i][TOPA].sum()) for i in range(n)])
    gap = np.array([r["gap_atk_minus_ben"] for r in rows])
    # 상관: AUC vs (격차), AUC vs (공격집중 상태 총확률)
    r_gap, _ = pearsonr(aucs, gap)
    r_mass, _ = pearsonr(aucs, p_mass_on_topA)
    print("[B] corr(AUC, 공격-benign 격차)=%.3f | corr(AUC, 공격집중상태 총확률)=%.3f" % (r_gap, r_mass))

    # 극단 비교: 저AUC vs 고AUC (TV 비슷)
    idx_lo = int(np.argmin(aucs)); idx_hi = int(np.argmax(aucs))
    print("[B] 극단: seed%d AUC=%.3f TV=%.3f 공격집중총p=%.4f | seed%d AUC=%.3f TV=%.3f 공격집중총p=%.4f"
          % (seeds[idx_lo], aucs[idx_lo], tvs[idx_lo], p_mass_on_topA[idx_lo],
             seeds[idx_hi], aucs[idx_hi], tvs[idx_hi], p_mass_on_topA[idx_hi]))
    # emp_joint(q)가 공격집중 상태에 주는 확률(기준)
    q_mass_topA = float(q[TOPA].sum())
    print("    (emp_joint q 공격집중총p=%.4f, AUC=%.4f)" % (q_mass_topA, EMP_REF))

    # =========================================================
    # (C) TV 오차 위치: |p-q| 가 공격집중 상태 vs benign흔한 상태 중 어디 쌓이나
    # =========================================================
    order_ben = np.argsort(-ben_cnt)
    TOPB = order_ben[:10]           # benign 흔한 상위 10 상태
    err_topA, err_topB, err_total = [], [], []
    for i in range(n):
        err = np.abs(pf[i] - q)
        err_topA.append(float(err[TOPA].sum()))
        err_topB.append(float(err[TOPB].sum()))
        err_total.append(float(err.sum()))            # = 2*TV
    err_topA = np.array(err_topA); err_topB = np.array(err_topB); err_total = np.array(err_total)
    # AUC와 "공격집중 상태 오차 비중"의 상관
    frac_err_A = err_topA / err_total
    r_errA, _ = pearsonr(aucs, frac_err_A)
    print("[C] corr(AUC, 공격집중상태 오차비중)=%.3f | 평균 오차비중 A=%.3f B(benign흔함)=%.3f"
          % (r_errA, float(frac_err_A.mean()), float((err_topB / err_total).mean())))

    # =========================================================
    # (D) MMD²/TV 순위 vs AUC 순위 (Spearman)
    # =========================================================
    sp_mmd, _ = spearmanr(mmd2, aucs)         # 낮은 mmd2가 좋은 모델 -> AUC와 음의 상관이면 정렬됨
    sp_tv, _ = spearmanr(tvs, aucs)
    print("[D] Spearman(MMD², AUC)=%.3f | Spearman(TV, AUC)=%.3f (음수=낮은오차가 높은AUC)" % (sp_mmd, sp_tv))
    # MMD-best가 AUC에서 몇 위인가
    best_mmd_idx = int(np.argmin(mmd2))
    auc_rank_of_bestmmd = int((aucs > aucs[best_mmd_idx]).sum()) + 1
    print("    MMD-best(seed%d) AUC=%.3f -> AUC 순위 %d/16 (최고 AUC=seed%d %.3f)"
          % (seeds[best_mmd_idx], aucs[best_mmd_idx], auc_rank_of_bestmmd,
             seeds[idx_hi], aucs[idx_hi]))

    # --- 저장 ---
    payload = {
        "verify_auc": {"recomputed": stats, "exceed_emp": exceed_emp, "matches_prev": bool(ok)},
        "A_attack_concentration": {"n_attack_states": n_atk_states,
                                   "topk_cover_frac": topk_cover,
                                   "top10_states": TOPA.tolist()},
        "B_restart_metrics": rows,
        "B_corr_auc_gap": float(r_gap),
        "B_corr_auc_attackmass": float(r_mass),
        "B_extremes": {"low": {"seed": int(seeds[idx_lo]), "auc": float(aucs[idx_lo]),
                               "tv": float(tvs[idx_lo]), "p_mass_topA": float(p_mass_on_topA[idx_lo])},
                       "high": {"seed": int(seeds[idx_hi]), "auc": float(aucs[idx_hi]),
                                "tv": float(tvs[idx_hi]), "p_mass_topA": float(p_mass_on_topA[idx_hi])},
                       "emp_joint": {"p_mass_topA": q_mass_topA, "auc": EMP_REF}},
        "C_error_location": {"corr_auc_frac_err_on_attack_states": float(r_errA),
                             "mean_frac_err_attackstates": float(frac_err_A.mean()),
                             "mean_frac_err_benigncommon": float((err_topB / err_total).mean())},
        "D_rank_corr": {"spearman_mmd2_auc": float(sp_mmd), "spearman_tv_auc": float(sp_tv),
                        "mmd_best_seed": int(seeds[best_mmd_idx]),
                        "mmd_best_auc": float(aucs[best_mmd_idx]),
                        "mmd_best_auc_rank": auc_rank_of_bestmmd,
                        "max_auc_seed": int(seeds[idx_hi]), "max_auc": float(aucs[idx_hi])},
        "note": "사실 측정만. 판정/해석은 웹 AI. QCBM 재학습 없음(npz 분석).",
    }
    with open(os.path.join(OUT, "tv_auc_dissociation.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("저장: results2/tv_auc_dissociation.json")

    # --- 그림 (4패널) ---
    try:
        fig, ax = plt.subplots(2, 2, figsize=(14, 11))
        # 1) TV vs AUC 산점
        ax[0][0].scatter(tvs, aucs, c="steelblue")
        for i in range(n):
            ax[0][0].annotate(str(seeds[i]), (tvs[i], aucs[i]), fontsize=7)
        ax[0][0].axhline(EMP_REF, color="crimson", ls="--", label="emp_joint %.3f" % EMP_REF)
        ax[0][0].axhline(MARG_REF, color="orange", ls=":", label="marginal %.3f" % MARG_REF)
        ax[0][0].set_xlabel("TV (적합)"); ax[0][0].set_ylabel("AUC (탐지)")
        ax[0][0].set_title("TV vs AUC 해리 (Spearman TV-AUC=%.2f)" % sp_tv)
        ax[0][0].legend(fontsize=8); ax[0][0].grid(alpha=.3)
        # 2) 공격집중 상태 총확률 vs AUC
        ax[0][1].scatter(p_mass_on_topA, aucs, c="purple")
        for i in range(n):
            ax[0][1].annotate(str(seeds[i]), (p_mass_on_topA[i], aucs[i]), fontsize=7)
        ax[0][1].axvline(q_mass_topA, color="green", ls="-.", label="emp_joint q")
        ax[0][1].set_xlabel("공격집중 상위10 상태에 모델이 준 총확률")
        ax[0][1].set_ylabel("AUC")
        ax[0][1].set_title("공격상태 확률 vs AUC (corr=%.2f)" % r_mass)
        ax[0][1].legend(fontsize=8); ax[0][1].grid(alpha=.3)
        # 3) 공격 집중도 누적
        ax[1][0].plot(range(1, 21), cum[:20] * 100, "o-", color="darkorange")
        ax[1][0].set_xlabel("상위 k 공격 상태"); ax[1][0].set_ylabel("공격 누적 %")
        ax[1][0].set_title("(A) 공격 상태 집중도 (상위10=%.0f%%)" % (100 * topk_cover[10]))
        ax[1][0].grid(alpha=.3)
        # 4) 공격집중 상태 오차비중 vs AUC
        ax[1][1].scatter(frac_err_A, aucs, c="teal")
        for i in range(n):
            ax[1][1].annotate(str(seeds[i]), (frac_err_A[i], aucs[i]), fontsize=7)
        ax[1][1].set_xlabel("|p-q| 중 공격집중 상태 오차 비중")
        ax[1][1].set_ylabel("AUC")
        ax[1][1].set_title("(C) 오차 위치 vs AUC (corr=%.2f)" % r_errA)
        ax[1][1].grid(alpha=.3)
        plt.suptitle("TV↔AUC dissociation: 분포 적합(TV)과 탐지(AUC)가 따로 노는 메커니즘")
        plt.tight_layout()
        fp = os.path.join(OUT, "tv_auc_dissociation.png")
        plt.savefig(fp, dpi=130); plt.close()
        print("그림 저장:", fp)
    except Exception as ex:
        print("그림 건너뜀:", repr(ex))

    print("DONE")


if __name__ == "__main__":
    main()
