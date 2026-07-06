#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
비지도 모델 선택 기준 탐색 — MMD² 외 라벨 없는 기준이 고AUC restart를 고르는가 (재학습 없음)
================================================================================
직전 결과: Bot QCBM 16 restart에서 우리가 best를 고르는 기준 min-MMD²가 탐지 AUC와 무관함:
  corr(MMD²,AUC)=0.077≈0, corr(TV,AUC)=−0.014≈0 (TV-AUC 해리). 실전(비지도)에는 공격 라벨이
  없어 AUC로 못 고르므로, "216p best가 고전을 이겨도 실전에서 그 best를 못 고른다"는 약점.
본 작업: 라벨/공격 데이터 일절 없이 benign(train/test)+학습분포 p_final만으로 계산되는 후보
  선택 기준들이 MMD²보다 고AUC restart를 더 잘 고르는지 탐색. 앙상블도 평가. 판정은 웹 AI.

[비지도 조건 — 엄수]
  선택 기준(S1~S6)과 앙상블 구성(E1~E3)은 st_atk(공격)·라벨을 절대 쓰지 않는다.
  AUC는 *기준의 성능 평가*에만 쓴다(기준 자체엔 미포함): "기준 X로 restart 고름 -> 그 AUC 사후확인".

재사용(기존 소스 무수정, import): experiment2(auc_score, tv_dist, kl_div, empirical_dist),
  train_qcbm_bot_resumable(build_bot). raw MWU 유지(0.5 미만도 그대로). scipy.stats만 추가 사용.
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

import experiment2 as E                  # noqa: E402  auc_score, tv_dist, kl_div, empirical_dist
import train_qcbm_bot_resumable as TBOT  # noqa: E402  build_bot()

OUT = "results2"
NQ = 12
NSTATES = 1 << NQ
EPS = 1e-12
QCBM_PARAMS = 6 * NQ * 3                  # 216
NPZ = os.path.join(OUT, "qcbm_bot_restarts.npz")
BOT_JSON = os.path.join(OUT, "qcbm_bot.json")
ROB_JSON = os.path.join(OUT, "bot_efficiency_robustness.json")
CLASSICAL_BEST = 0.7063764867284203       # 비트그룹 6+6 (126p), 고전 216p-이하 최고
TOL = 1e-9


def auc_of_dist(p, st_te, st_atk):
    """benign_test(label0) vs attack(label1), s=-log p. raw MWU. (AUC는 평가 전용)."""
    p = np.asarray(p, dtype=float)
    s_te = -np.log(p[st_te] + EPS)
    s_atk = -np.log(p[st_atk] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def held_out_nll(p, states):
    """held-out benign 상태의 평균 음의 로그가능도 (라벨 불필요)."""
    p = np.asarray(p, dtype=float)
    return float(np.mean(-np.log(p[states] + EPS)))


def entropy(p):
    p = np.asarray(p, dtype=float)
    m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def corr_pair(score, auc):
    """Pearson, Spearman (점수 vs AUC)."""
    pear = float(np.corrcoef(score, auc)[0, 1])
    sp = float(spearmanr(score, auc).correlation)
    return pear, sp


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 80)
    E.log("비지도 모델 선택 기준 탐색 — 라벨 없이 고AUC restart를 고를 수 있는가 (재학습 없음)")
    E.log("=" * 80)

    z = np.load(NPZ, allow_pickle=False)
    with open(BOT_JSON) as f:
        botj = json.load(f)

    # --- 재현 + 정합 게이트 (benign/attack states 재현, 16 AUC 재계산) ---
    d = TBOT.build_bot()
    q_train = np.asarray(d["q_train"], float)
    st_tr, st_te, st_atk = d["st_tr"], d["st_te"], d["st_atk"]
    if not np.allclose(q_train, z["q_train"]):
        E.log("!!! q_train 재현 불일치 -> 중단."); sys.exit(2)
    if int(d["occ"]) != int(botj["occupied_train_states"]):
        E.log("!!! 점유 불일치 occ=%d != %d -> 중단." % (d["occ"], botj["occupied_train_states"])); sys.exit(2)

    p_finals = z["p_finals"]; seeds = z["seeds"]
    n = len(p_finals)
    aucs = np.array([auc_of_dist(p_finals[i], st_te, st_atk) for i in range(n)])
    diff = float(np.max(np.abs(aucs - z["aucs"])))
    jdist = botj["auc_distribution"]
    ok = (diff <= TOL and abs(np.median(aucs) - jdist["median"]) <= 1e-4
          and abs(aucs.max() - jdist["max"]) <= 1e-4)
    E.log("[정합] 16 AUC 재계산 max|Δ|=%.2e | median 재=%.4f(json %.4f) -> %s"
          % (diff, np.median(aucs), jdist["median"], "OK" if ok else "불일치"))
    if not ok:
        E.log("!!! 정합 불일치 -> 중단."); sys.exit(3)
    E.log("[비지도조건] 선택 기준은 q_train(benign train)·st_te(benign test)·p_final만 사용. "
          "st_atk는 AUC 평가에만 사용(기준 미포함).")

    # ---------------------------------------------------------------------
    # 비지도 후보 점수 (공격 미사용). 각 점수에 'direction': 'min'/'max'
    #   = 실전에서 "좋다"고 보고 고를 방향(argmin/argmax).
    # ---------------------------------------------------------------------
    mmd2 = z["mmd2"].astype(float)                         # S1 학습 기준값(=final_mmd2)
    tv = np.array([E.tv_dist(p_finals[i], q_train) for i in range(n)])   # S2
    kl_qp = np.array([E.kl_div(q_train, p_finals[i]) for i in range(n)]) # S3a KL(q||p)
    kl_pq = np.array([E.kl_div(p_finals[i], q_train) for i in range(n)]) # S3b KL(p||q)
    nll_te = np.array([held_out_nll(p_finals[i], st_te) for i in range(n)])  # S4 held-out benign NLL
    nll_tr = np.array([held_out_nll(p_finals[i], st_tr) for i in range(n)])
    gen_gap = nll_te - nll_tr                              # S6 일반화 갭(>0: test가 더 나쁨)
    ent = np.array([entropy(p_finals[i]) for i in range(n)])             # S5a 엔트로피
    eff_supp = np.exp(ent)                                 # S5b 유효 support 크기
    # S2 저장값 정합(재사용 일관성)
    tv_match = float(np.max(np.abs(tv - z["tv"].astype(float))))
    E.log("[정합] TV 재계산 vs 저장 max|Δ|=%.2e" % tv_match)

    # 각 기준: (이름, 값, 선택방향). 방향은 "분포적합/그럴듯함이 좋다"는 비지도 직관.
    candidates = [
        ("S1_MMD2",        mmd2,    "min", "MMD²(p,q_train) — 학습 기준(기준선)"),
        ("S2_TV",          tv,      "min", "TV(p,q_train)"),
        ("S3a_KL_q||p",    kl_qp,   "min", "KL(q_train || p)"),
        ("S3b_KL_p||q",    kl_pq,   "min", "KL(p || q_train)"),
        ("S4_NLL_test",    nll_te,  "min", "held-out benign NLL = mean(-log p[st_te])"),
        ("S5a_entropy",    ent,     "min", "엔트로피 H(p) (구조적; 방향 불명, min 가정)"),
        ("S5b_eff_support",eff_supp,"min", "유효 support exp(H) (구조적; min 가정)"),
        ("S6_gen_gap",     gen_gap, "min", "일반화 갭 NLL_test - NLL_train"),
    ]

    rows = []
    for name, score, direction, desc in candidates:
        pear, sp = corr_pair(score, aucs)
        # 선택: direction대로 argmin/argmax. 추가로 반대방향도 참고.
        idx_min = int(np.argmin(score)); idx_max = int(np.argmax(score))
        pick_idx = idx_min if direction == "min" else idx_max
        pick_auc = float(aucs[pick_idx]); pick_seed = int(seeds[pick_idx])
        # 양방향 모두 보고(어느 방향이 고AUC를 고르는지 사실로)
        row = {
            "name": name, "desc": desc, "direction": direction,
            "pearson": pear, "spearman": sp,
            "pick_seed": pick_seed, "pick_auc": pick_auc,
            "beats_classical": bool(pick_auc > CLASSICAL_BEST),
            "argmin_seed": int(seeds[idx_min]), "argmin_auc": float(aucs[idx_min]),
            "argmax_seed": int(seeds[idx_max]), "argmax_auc": float(aucs[idx_max]),
        }
        rows.append(row)
        E.log("[기준] %-16s Pearson=%+.3f Spearman=%+.3f | %s-pick seed%d AUC=%.4f %s"
              % (name, pear, sp, direction, pick_seed, pick_auc,
                 "(>고전)" if row["beats_classical"] else "(<=고전)"))

    # ---------------------------------------------------------------------
    # 앙상블 (공격 미사용 — p_final만 결합) -> AUC 평가
    # ---------------------------------------------------------------------
    p_arith = p_finals.mean(axis=0); p_arith = p_arith / p_arith.sum()             # E1
    log_geo = np.mean(np.log(p_finals + EPS), axis=0)
    p_geo = np.exp(log_geo - log_geo.max()); p_geo = p_geo / p_geo.sum()           # E2
    auc_arith = auc_of_dist(p_arith, st_te, st_atk)
    auc_geo = auc_of_dist(p_geo, st_te, st_atk)

    # E3: 비지도 상위 k개 평균 (min-MMD², min-NLL_test 각각)
    K = 4
    ens = {"E1_arith_mean": auc_arith, "E2_geo_mean": auc_geo}
    for crit_name, crit in [("MMD2", mmd2), ("NLL_test", nll_te)]:
        topk = np.argsort(crit)[:K]                  # 작을수록 좋음
        p_top = p_finals[topk].mean(axis=0); p_top = p_top / p_top.sum()
        ens["E3_top%d_by_%s" % (K, crit_name)] = auc_of_dist(p_top, st_te, st_atk)

    med = float(np.median(aucs)); best = float(aucs.max())
    E.log("-" * 80)
    for k, v in ens.items():
        rel = "median %s / 고전 %s / best %s" % (
            "위" if v > med else "아래", "위" if v > CLASSICAL_BEST else "아래",
            "위" if v > best else "아래")
        E.log("[앙상블] %-22s AUC=%.4f (%s)" % (k, v, rel))

    # ---------------------------------------------------------------------
    # 저장
    # ---------------------------------------------------------------------
    # MMD² 기준선 corr 정합 확인(직전 결과 0.077)
    base_pear = next(r["pearson"] for r in rows if r["name"] == "S1_MMD2")
    payload = {
        "encoding": "B2 12bit (4096), Bot. raw MWU(s=-log p). QCBM 재학습 없음.",
        "unsup_condition": "선택 기준 S1~S6, 앙상블 E1~E3은 공격(st_atk)/라벨 미사용. "
                           "AUC는 기준 성능 평가에만 사용.",
        "consistency": {"auc_recompute_maxdiff": diff, "tv_recompute_maxdiff": tv_match,
                        "occ": int(d["occ"]), "ok": True,
                        "mmd2_pearson_vs_prior_0.077": base_pear},
        "classical_best_le216": CLASSICAL_BEST,
        "qcbm_auc": {"all_sorted": sorted(round(float(a), 4) for a in aucs),
                     "min": float(aucs.min()), "median": med, "max": best,
                     "best_seed": int(seeds[int(np.argmax(aucs))])},
        "criteria": rows,
        "best_predictor": max(rows, key=lambda r: abs(r["spearman"]))["name"],
        "ensembles": ens,
        "ensemble_vs": {"median": med, "classical": CLASSICAL_BEST, "best": best},
        "note": "사실 측정만. 판정은 웹 AI. 기존 소스/데이터/결과 무수정.",
    }
    with open(os.path.join(OUT, "unsup_selection.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/unsup_selection.json")

    # ---------------------------------------------------------------------
    # 그림: (좌) 각 기준 vs AUC 산점도(Spearman), (우) 앙상블 AUC 막대 + 기준선
    # ---------------------------------------------------------------------
    try:
        plot_scores = [("S1_MMD2", mmd2), ("S2_TV", tv), ("S3a_KL_q||p", kl_qp),
                       ("S4_NLL_test", nll_te), ("S5a_entropy", ent), ("S6_gen_gap", gen_gap)]
        fig = plt.figure(figsize=(17, 9))
        gs = fig.add_gridspec(2, 4)
        for i, (name, score) in enumerate(plot_scores):
            ax = fig.add_subplot(gs[i // 3, i % 3])
            sp = next(r["spearman"] for r in rows if r["name"] == name)
            pe = next(r["pearson"] for r in rows if r["name"] == name)
            ax.scatter(score, aucs, c="crimson", s=40, alpha=0.7, edgecolor="k")
            ax.axhline(CLASSICAL_BEST, color="darkorange", ls="--", alpha=.7, label="classical 0.706")
            ax.axhline(med, color="black", ls=":", alpha=.6, label="QCBM median")
            ax.set_xlabel(name); ax.set_ylabel("AUC")
            ax.set_title("%s\nPearson=%+.3f Spearman=%+.3f" % (name, pe, sp), fontsize=9)
            ax.grid(alpha=.3)
            if i == 0:
                ax.legend(fontsize=7)
        # 앙상블 막대 (오른쪽 큰 칸)
        axb = fig.add_subplot(gs[:, 3])
        labels = list(ens.keys()); vals = [ens[k] for k in labels]
        ypos = np.arange(len(labels))
        axb.barh(ypos, vals, color="steelblue", alpha=.8, edgecolor="k")
        axb.set_yticks(ypos); axb.set_yticklabels(labels, fontsize=8)
        axb.invert_yaxis()
        for v, c, lab in [(med, "black", "QCBM median %.3f" % med),
                          (CLASSICAL_BEST, "darkorange", "classical 0.706"),
                          (best, "crimson", "QCBM best %.3f" % best)]:
            axb.axvline(v, color=c, ls="--", lw=2, label=lab)
        for i, v in enumerate(vals):
            axb.text(v + 0.003, i, "%.3f" % v, va="center", fontsize=8)
        axb.set_xlim(0.5, max(0.85, best + 0.03))
        axb.set_xlabel("AUC (raw MWU)"); axb.set_title("Ensemble AUC vs baselines", fontsize=10)
        axb.legend(fontsize=7, loc="lower right"); axb.grid(alpha=.3, axis="x")
        plt.suptitle("Unsupervised model selection for Bot QCBM 16 restarts "
                     "(criteria use benign+p_final only; AUC = post-hoc eval)", fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.97])
        fp = os.path.join(OUT, "unsup_selection.png")
        plt.savefig(fp, dpi=120); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
