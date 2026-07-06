#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psh 제거 11비트 학습의 16 restart 분산 정찰 (재학습 없음, 로그 분석)
====================================================================
질문: QCBM AUC 0.8023 > emp_joint 0.7773(103.2%)이 '견고한 효과'인가 'best 선택의 운'인가?
배경(저장물 확인): qcbm_iat_no_psh.json에는 best 요약만(results 배열·p_final 없음). 16개 restart의
      p_final이 디스크에 없음 -> **경우 B**: 16개 AUC를 재학습 없이 복원 불가.
      대신 학습 로그(qcbm_iat_no_psh_train.log)에서 16 restart의 MMD²/TV 분포를 파싱해
      '품질 견고성'(best만 좋은가 다수가 좋은가)을 간접 측정.

이 작업은 QCBM을 재학습하지 않는다. 로그 파싱·통계만. 판정은 웹 AI.
AUC 분포(emp_joint 0.7773 초과 restart 수)는 p_final 미저장으로 불가 -> 1회 재실행 필요(다음 단계).
"""
import os
import re
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(QCBM_DIR)
OUT = "results2"
LOG = os.path.join(OUT, "qcbm_iat_no_psh_train.log")
JSON_IN = os.path.join(OUT, "qcbm_iat_no_psh.json")

# 기준선 (psh 제거 11비트)
MARG_AUC, EMP_AUC, QCBM_BEST_AUC = 0.5136, 0.7773, 0.8023
BEST_MMD2, BEST_TV = 6.6567e-05, 0.0622   # json train 요약과 대조용

DONE_RE = re.compile(
    r"seed\s+(\d+)\]\s*완료\s*MMD\^2=([\d.eE+\-]+)\s*TV=([\d.eE+\-]+)\s*\|grad\|_fin=([\d.eE+\-]+)")
FULL_START_RE = re.compile(r"\[full\]\s*R=\d+\s*EP=\d+")     # full 학습 시작 마커
FULL_END_RE = re.compile(r"\[full\]\s*wall=")               # full 종료(요약) 마커


def parse_full_restarts(lines):
    """full 학습 구간(시작 마커~종료 마커) 안의 '완료' 줄만 파싱 -> smoke 제외."""
    start = end = None
    for i, ln in enumerate(lines):
        if start is None and FULL_START_RE.search(ln):
            start = i
        elif start is not None and FULL_END_RE.search(ln):
            end = i
            break
    if start is None:
        return [], (None, None)
    seg = lines[start:(end if end is not None else len(lines))]
    rows = []
    for ln in seg:
        m = DONE_RE.search(ln)
        if m:
            rows.append({"seed": int(m.group(1)), "mmd2": float(m.group(2)),
                         "tv": float(m.group(3)), "grad_fin": float(m.group(4))})
    return rows, (start, end)


def stats(xs):
    a = np.asarray(xs, float)
    return {"min": float(a.min()), "median": float(np.median(a)),
            "max": float(a.max()), "mean": float(a.mean()), "std": float(a.std())}


def main():
    if not os.path.exists(LOG):
        print("로그 없음:", LOG); sys.exit(1)
    with open(LOG) as f:
        lines = f.readlines()
    rows, (s, e) = parse_full_restarts(lines)
    n = len(rows)
    print("full 학습 구간 라인 %s~%s, 파싱된 restart=%d" % (s, e, n))
    if n != 16:
        print("주의: full restart 파싱 수=%d (기대 16). 그대로 진행하되 .md에 기록 필요." % n)

    mmd2 = [r["mmd2"] for r in rows]
    tv = [r["tv"] for r in rows]
    seeds = [r["seed"] for r in rows]
    mmd2_stats, tv_stats = stats(mmd2), stats(tv)

    # best = 최소 MMD² (run_config의 선택 기준). 그 순위/위치.
    order = np.argsort(mmd2)                       # 오름차순(좋은 순)
    best_idx = int(order[0])
    best_row = rows[best_idx]
    best_rank_mmd2 = 1                              # 정의상 최소
    # best의 TV가 TV 분포에서 상위 몇 %인지
    tv_arr = np.asarray(tv)
    best_tv_pct = float((tv_arr <= best_row["tv"]).mean() * 100)

    # '견고성' 간접 지표: best와 비슷한 품질(국소최적 실패 아님) restart 수.
    # 기준: TV <= 0.15 (잘 학습된 분포; 직전 B2에서 실패 restart는 TV 0.5~0.75였음).
    well_learned = [r for r in rows if r["tv"] <= 0.15]
    failed = [r for r in rows if r["tv"] > 0.4]    # 명백한 국소최적 실패

    print("MMD2 stats:", {k: "%.2e" % v for k, v in mmd2_stats.items()})
    print("TV   stats:", {k: "%.4f" % v for k, v in tv_stats.items()})
    print("best restart: seed=%d MMD2=%.3e TV=%.4f (json best TV=%.4f 대조)"
          % (best_row["seed"], best_row["mmd2"], best_row["tv"], BEST_TV))
    print("잘 학습(TV<=0.15) restart: %d/%d | 명백 실패(TV>0.4): %d/%d"
          % (len(well_learned), n, len(failed), n))

    payload = {
        "case": "B (best p_final만 저장; 16개 AUC 재학습 없이 복원 불가)",
        "n_restarts_parsed": n,
        "restarts": rows,
        "mmd2_stats": mmd2_stats,
        "tv_stats": tv_stats,
        "best_restart": best_row,
        "best_tv_percentile_low_is_better": best_tv_pct,
        "well_learned_tv_le_0_15": len(well_learned),
        "failed_tv_gt_0_4": len(failed),
        "baseline_auc": {"marginal": MARG_AUC, "emp_joint": EMP_AUC, "qcbm_best": QCBM_BEST_AUC},
        "auc_per_restart": "UNAVAILABLE — p_final 미저장. emp_joint 초과 restart 수 측정엔 1회 재실행(16 p_final 저장) 필요.",
        "note": "재학습 없음. 로그 MMD²/TV 분포만. 판정은 웹 AI.",
    }
    with open(os.path.join(OUT, "no_psh_restart_analysis.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("저장: results2/no_psh_restart_analysis.json")

    # --- 그림: TV / MMD2 분포 (16 restart), best 표시 ---
    try:
        fig, ax = plt.subplots(1, 2, figsize=(13, 5))
        ax[0].hist(tv, bins=12, color="steelblue", edgecolor="k", alpha=.8)
        ax[0].axvline(best_row["tv"], color="crimson", ls="--", lw=2,
                      label="best TV=%.4f" % best_row["tv"])
        ax[0].axvline(0.15, color="gray", ls=":", label="TV=0.15 (well-learned)")
        ax[0].set_xlabel("TV (16 restarts)"); ax[0].set_ylabel("count")
        ax[0].set_title("no-psh 11q: TV distribution over 16 restarts")
        ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
        ax[1].hist(np.log10(mmd2), bins=12, color="darkorange", edgecolor="k", alpha=.8)
        ax[1].axvline(np.log10(best_row["mmd2"]), color="crimson", ls="--", lw=2,
                      label="best log10 MMD²")
        ax[1].set_xlabel("log10 MMD² (16 restarts)"); ax[1].set_ylabel("count")
        ax[1].set_title("MMD² distribution over 16 restarts")
        ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
        plt.suptitle("Restart robustness (best AUC 0.8023 > emp_joint 0.7773): "
                     "%d/%d well-learned (TV<=0.15)" % (len(well_learned), n))
        plt.tight_layout()
        fp = os.path.join(OUT, "no_psh_restart_analysis.png")
        plt.savefig(fp, dpi=130); plt.close()
        print("그림 저장:", fp)
    except Exception as ex:
        print("그림 건너뜀:", repr(ex))

    print("DONE")


if __name__ == "__main__":
    main()
