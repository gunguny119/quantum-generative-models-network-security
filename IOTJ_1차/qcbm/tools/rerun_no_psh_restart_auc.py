#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psh 제거 11비트 — 16 restart 각각의 AUC 계산·저장 재실행 (결정론적 재현)
========================================================================
목적: best AUC 0.8023 > emp_joint 0.7773 이 견고한가 best 운인가? 이전 학습을 동일 설정으로
      결정론적 재현(seed 고정)하되, 16 restart 각각의 p_final로 AUC를 계산해
      "emp_joint(0.7773) 초과 restart 수/16"을 직접 측정. 16 p_final도 디스크 저장(향후 재실행 불필요).

동일 설정(이전 본학습): nq=11, L=6, LR=0.1, RESTARTS=16, EPOCHS=300. -> best AUC 0.8023 재현 기대(검증).
재사용: train_qcbm_iat_no_psh(build_nopsh, SPEC, run_train), experiment2(run_config, anomaly_eval).
기존 소스 무수정(import + 모듈속성 대입). 평가 동일(s=-log p, auc_score, raw MWU).
"""
import os
import sys
import json

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E                          # noqa: E402
import train_qcbm_iat_no_psh as NP               # noqa: E402  build_nopsh, SPEC, run_train
from train_qcbm_iat_b2 import marginal_dist      # noqa: E402

OUT = "results2"
EPS = 1e-12
RESTARTS, EPOCHS = 16, 300       # 이전 본학습과 동일(qcbm_iat_no_psh.json 확인됨)
MARG_REF, EMP_REF, QCBM_BEST_REF = 0.5136, 0.7773, 0.8023
EXPECT_OCC = 90


def eval_auc_full(p_final, st_te, st_atk):
    """experiment2.anomaly_eval 동일 평가를 restart p_final에 적용."""
    fake = {"best": {"p_final": list(p_final), "final_mmd2": float("nan"), "tv": float("nan")}}
    an = E.anomaly_eval(fake, st_te, st_atk, "r")
    return an


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 80)
    E.log("psh 제거 11비트 재실행 — 16 restart AUC (결정론 재현, p_final 저장)")
    E.log("=" * 80)

    d = NP.build_nopsh()
    if d["occ"] != EXPECT_OCC:
        E.log("!!! 점유 %d != %d -> 인코딩 재현 어긋남. 중단." % (d["occ"], EXPECT_OCC)); sys.exit(2)

    # 기준선 재확인 (정합)
    q_marg = marginal_dist(d["st_tr"], NP.NQ)
    an_marg = eval_auc_full(q_marg, d["st_te"], d["st_atk"])                    # marginal
    an_emp = eval_auc_full(d["q_train"], d["st_te"], d["st_atk"])               # emp_joint
    E.log("기준선: marginal=%.4f emp_joint=%.4f (기대 %.4f/%.4f)"
          % (an_marg["auc"], an_emp["auc"], MARG_REF, EMP_REF))

    # --- 결정론 재학습 (이전과 동일 설정) ---
    E.log("[재학습] RESTARTS=%d EPOCHS=%d L=%d (결정론 재현)" % (RESTARTS, EPOCHS, NP.LAYERS))
    cfg = NP.run_train(d, RESTARTS, EPOCHS)
    results = cfg["results"]                       # final_mmd2 오름차순 정렬됨(run_config)
    E.log("재학습 완료: %d restart, best MMD^2=%.4e TV=%.4f"
          % (len(results), cfg["best"]["final_mmd2"], cfg["best"]["tv"]))

    # --- 각 restart AUC ---
    rows = []
    p_finals = []
    for r in results:
        an = eval_auc_full(r["p_final"], d["st_te"], d["st_atk"])
        rows.append({"seed": int(r["seed"]), "final_mmd2": float(r["final_mmd2"]),
                     "tv": float(r["tv"]), "auc": float(an["auc"]),
                     "youden_j": float(an["youden_j"]),
                     "logp_gap": float(an["mean_logp_ben"] - an["mean_logp_atk"])})
        p_finals.append(np.asarray(r["p_final"], dtype=float))
    aucs = np.array([x["auc"] for x in rows])

    # best = 최소 MMD² restart(run_config 기준) = results[0]
    best_by_mmd2 = rows[0]
    best_auc_repro = best_by_mmd2["auc"]
    max_auc = float(aucs.max())
    n_exceed_emp = int((aucs > EMP_REF).sum())
    n_exceed_marg = int((aucs > MARG_REF).sum())
    determinism_ok = abs(best_auc_repro - QCBM_BEST_REF) <= 0.001

    E.log("=" * 80)
    E.log("[결정론 검증] best(최소MMD², seed%d) AUC=%.4f (기대 0.8023) -> %s"
          % (best_by_mmd2["seed"], best_auc_repro, "일치" if determinism_ok else "불일치!"))
    E.log("[16 restart AUC] min=%.4f median=%.4f max=%.4f std=%.4f"
          % (aucs.min(), float(np.median(aucs)), aucs.max(), aucs.std()))
    E.log("  emp_joint(%.4f) 초과: %d/16 | marginal(%.4f) 초과: %d/16"
          % (EMP_REF, n_exceed_emp, MARG_REF, n_exceed_marg))
    # AUC 내림차순 표
    for x in sorted(rows, key=lambda r: r["auc"], reverse=True):
        flag = ">emp" if x["auc"] > EMP_REF else (">marg" if x["auc"] > MARG_REF else "")
        E.log("  seed%-3d AUC=%.4f TV=%.4f MMD2=%.2e %s"
              % (x["seed"], x["auc"], x["tv"], x["final_mmd2"], flag))

    # --- 저장: 16 p_final npz + json ---
    np.savez_compressed(os.path.join(OUT, "qcbm_iat_no_psh_restarts.npz"),
                        p_finals=np.array(p_finals),
                        seeds=np.array([x["seed"] for x in rows]),
                        aucs=aucs, tvs=np.array([x["tv"] for x in rows]),
                        mmd2=np.array([x["final_mmd2"] for x in rows]),
                        q_train=d["q_train"],
                        st_te=d["st_te"], st_atk=d["st_atk"])
    E.log("저장: results2/qcbm_iat_no_psh_restarts.npz (16 p_final + q_train + states)")

    payload = {
        "encoding": "11bit psh 제거 (proto2+syn1+size4+FwdIAT2+FlowIAT2)",
        "settings": {"nq": NP.NQ, "layers": NP.LAYERS, "lr": NP.LR,
                     "restarts": RESTARTS, "epochs": EPOCHS},
        "occupied": d["occ"],
        "baseline": {"marginal": an_marg["auc"], "emp_joint": an_emp["auc"],
                     "expected": {"marginal": MARG_REF, "emp_joint": EMP_REF}},
        "determinism": {"best_auc_reproduced": best_auc_repro,
                        "expected_best_auc": QCBM_BEST_REF, "ok": bool(determinism_ok),
                        "best_seed_by_mmd2": best_by_mmd2["seed"]},
        "restart_auc": rows,
        "auc_stats": {"min": float(aucs.min()), "median": float(np.median(aucs)),
                      "max": float(aucs.max()), "mean": float(aucs.mean()),
                      "std": float(aucs.std())},
        "n_exceed_emp_joint": n_exceed_emp,
        "n_exceed_marginal": n_exceed_marg,
        "best_auc": best_auc_repro, "max_auc": max_auc,
        "eval": "s=-log p(x)+1e-12, attack=positive, raw Mann-Whitney U",
        "note": "결정론 재현. 판정은 웹 AI. 기존 소스 무수정.",
    }
    with open(os.path.join(OUT, "no_psh_restart_auc.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/no_psh_restart_auc.json")

    # --- 그림 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(9, 5.5))
        order = np.argsort(-aucs)
        xs = np.arange(len(aucs))
        plt.bar(xs, aucs[order], color=["steelblue" if a > EMP_REF else
                ("slategray" if a > MARG_REF else "lightgray") for a in aucs[order]])
        plt.axhline(EMP_REF, color="crimson", ls="--", label="emp_joint 상한 %.4f" % EMP_REF)
        plt.axhline(MARG_REF, color="orange", ls=":", label="marginal %.4f" % MARG_REF)
        plt.axhline(QCBM_BEST_REF, color="green", ls="-.", alpha=.6,
                    label="이전 best %.4f" % QCBM_BEST_REF)
        plt.xticks(xs, [str(rows[i]["seed"]) for i in order], fontsize=8)
        plt.xlabel("restart (seed, AUC 내림차순)"); plt.ylabel("HOIC AUC")
        plt.ylim(0.45, 0.85)
        plt.title("no-psh 11q: 16 restart AUC — emp_joint(0.7773) 초과 %d/16" % n_exceed_emp)
        plt.legend(fontsize=8); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
        fp = os.path.join(OUT, "no_psh_restart_auc.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as ex:
        E.log("그림 건너뜀: %r" % ex)

    E.log("DONE")


if __name__ == "__main__":
    main()
