#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이상탐지 재평가: 얕은(L=3) vs 제대로 학습된(L=6) 8큐비트 QCBM
============================================================
배경(정정):
  - 이전 run2에서 "8q 확장 실패(TV 0.498)"라 결론냈으나, 이는 회로가 얕아서(L=3)였다.
  - depth_test에서 L=6으로 올리니 TV 0.141로 잘 학습됨이 확인됨.
  - 그러나 이상탐지(AUC 0.43, 무작위 이하)는 아직 얕은 L=3 모델로만 평가되었다.
  -> 본 실험: 제대로 학습된 L=6 깊은 모델로 이상탐지를 공정하게 재평가한다.

방법:
  - 8큐비트, proto(2)+syn(1)+psh(1)+size(4) = 256상태, depth_test와 동일 설정.
  - 동일한 train/test 분할(seed 7, benign 0.7 train), 동일 이산화기.
  - L=3, L=6 각각 16 무작위 재시작 -> best 모델 선택(experiment2 파이프라인 재활용).
  - 학습된 정상분포 q(x)로 benign(test) vs attack(Infilteration) 이상탐지:
    AUC, mean log p 격차, Youden J 비교.
  - 핵심 질문: 분포를 제대로 학습한 모델(L=6)에서 이상탐지가 개선되는가?

병렬: 독립 무작위 재시작 16개를 16코어에 1:1 프로세스 병렬(워커당 OMP=1).
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import json
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import experiment2 as E

# 동일 환경 유지
E.EPOCHS = 300
E.RESTARTS = 16
E.LR = 0.1
E.LOG_EVERY = 50
E.OUT = "results2"

SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
DEPTHS = [3, 6]   # 얕음 vs 깊음


def plot_l3_vs_l6(ans, out):
    """L=3 vs L=6 이상탐지 점수 분포 + ROC 비교 (한 그림)."""
    fig, ax = plt.subplots(2, 2, figsize=(13, 9))
    colmap = {3: ("L=3 (shallow)", "tab:gray", "tab:orange"),
              6: ("L=6 (deep)", "steelblue", "crimson")}
    for row, L in enumerate(DEPTHS):
        an = ans[L]
        title, cben, catk = colmap[L]
        bins = np.linspace(0, max(an["s_ben"].max(), an["s_atk"].max()) + 1e-6, 40)
        ax[row][0].hist(an["s_ben"], bins=bins, density=True, alpha=0.6,
                        label="Benign", color=cben)
        ax[row][0].hist(an["s_atk"], bins=bins, density=True, alpha=0.6,
                        label="Attack(Infil.)", color=catk)
        ax[row][0].set_xlabel("anomaly score  s(x) = -log p(x)")
        ax[row][0].set_ylabel("density")
        ax[row][0].set_title("%s  anomaly score (Benign vs Attack)" % title)
        ax[row][0].legend(); ax[row][0].grid(alpha=.3)
        ax[row][1].plot(an["fpr"], an["tpr"], color="darkgreen", lw=2,
                        label="AUC=%.3f" % an["auc"])
        ax[row][1].plot([0, 1], [0, 1], "k--", alpha=.5)
        ax[row][1].set_xlabel("FPR"); ax[row][1].set_ylabel("TPR")
        ax[row][1].set_title("%s  ROC" % title)
        ax[row][1].legend(); ax[row][1].grid(alpha=.3)
    plt.tight_layout()
    fp = os.path.join(out, "anomaly_L3_vs_L6.png")
    plt.savefig(fp, dpi=130); plt.close()
    E.log("그림 저장: %s" % fp)


def main():
    t_start = time.time()
    os.makedirs(E.OUT, exist_ok=True)
    E.log("=" * 70)
    E.log("이상탐지 재평가: 얕은(L=3) vs 제대로학습(L=6) 8큐비트 QCBM")
    E.log("=" * 70)

    benign, attack = E.load_frames()
    # experiment2와 동일한 분할 (seed 7, benign 0.7 train / 0.3 test)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(benign))
    cut = int(0.7 * len(benign))
    ben_tr = benign.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = benign.iloc[idx[cut:]].reset_index(drop=True)
    E.log("Benign train=%d / test=%d, attack(Infilteration)=%d"
          % (len(ben_tr), len(ben_te), len(attack)))

    # 동일 이산화기 (16버킷, train에 적합)
    edges = E.size_edges_from_benign(ben_tr, 16)
    st_tr = E.discretize(ben_tr, SPEC8, edges)
    q_train = E.empirical_dist(st_tr, 256)
    st_te = E.discretize(ben_te, SPEC8, edges)
    st_atk = E.discretize(attack, SPEC8, edges)
    E.log("8q 이산화: 256상태 중 %d개 점유 | benign_test=%d, attack=%d 샘플"
          % (int((q_train > 0).sum()), len(st_te), len(st_atk)))

    summary = {"depths": {}, "anomaly": {}}
    ans = {}

    for L in DEPTHS:
        E.LAYERS = L
        tag = "8q_L%d" % L
        E.log("#" * 70)
        E.log("########## 8큐비트  L=%d  학습 (16재시작 병렬) ##########" % L)
        cfg_out = E.run_config(SPEC8, q_train, edges)
        E.plot_training(cfg_out, tag)
        E.plot_dist(q_train, cfg_out["best"], tag)

        # 이상탐지 평가
        an = E.anomaly_eval(cfg_out, st_te, st_atk, tag)
        E.plot_anomaly(an, tag)
        ans[L] = an

        summary["depths"][L] = {
            "layers": L, "nparams": cfg_out["nparams"],
            "best_mmd2": cfg_out["best"]["final_mmd2"],
            "best_tv": cfg_out["best"]["tv"],
            "median_mmd2": cfg_out["median_mmd2"],
            "conv_ep": cfg_out["conv_ep"], "wall_sec": cfg_out["wall"],
            "gnorm0": cfg_out["gnorm0"], "gnormf": cfg_out["gnormf"],
        }
        summary["anomaly"][L] = {k: an[k] for k in
                                 ["auc", "mean_p_ben", "mean_p_atk",
                                  "mean_logp_ben", "mean_logp_atk", "youden_j"]}

    plot_l3_vs_l6(ans, E.OUT)

    # -------- 비교 표 --------
    E.log("=" * 70)
    E.log("[학습 품질] L=3 vs L=6 (분포를 제대로 학습했는가)")
    E.log("  %-5s %-7s %-9s %-7s %-7s %-8s %-9s" %
          ("L", "params", "bestMMD2", "TV", "conv", "wall", "grad0->f"))
    for L in DEPTHS:
        d = summary["depths"][L]
        E.log("  L=%-3d %-7d %-9.2e %-7.3f %-7d %-8.1f %.1e->%.1e" %
              (L, d["nparams"], d["best_mmd2"], d["best_tv"],
               d["conv_ep"], d["wall_sec"], d["gnorm0"], d["gnormf"]))
    E.log("=" * 70)
    E.log("[이상탐지] L=3 vs L=6 (공격이 정상보다 낮은 확률을 받는가)")
    E.log("  %-5s %-8s %-22s %-10s" %
          ("L", "AUC", "mean logp (ben vs atk)", "Youden J"))
    for L in DEPTHS:
        a = summary["anomaly"][L]
        gap = a["mean_logp_ben"] - a["mean_logp_atk"]
        E.log("  L=%-3d AUC=%.4f  benign=%.3f vs attack=%.3f (격차 %+.3f)  J=%.3f" %
              (L, a["auc"], a["mean_logp_ben"], a["mean_logp_atk"], gap, a["youden_j"]))

    # -------- 결론 --------
    a3, a6 = summary["anomaly"][3], summary["anomaly"][6]
    d6 = summary["depths"][6]
    auc3, auc6 = a3["auc"], a6["auc"]
    d_auc = auc6 - auc3

    def verdict(auc):
        if auc > 0.65: return "뚜렷한 신호"
        if auc > 0.55: return "약한 신호"
        if auc >= 0.45: return "신호 거의 없음(무작위 수준)"
        return "역신호(공격이 오히려 더 정상스러움)"

    E.log("=" * 70)
    E.log("결론: 제대로 학습된 L=6 모델(TV=%.3f)로 이상탐지를 공정 재평가한 결과" % d6["best_tv"])
    E.log("  - L=3(얕음): AUC=%.4f -> %s" % (auc3, verdict(auc3)))
    E.log("  - L=6(깊음): AUC=%.4f -> %s" % (auc6, verdict(auc6)))
    E.log("  - 깊이를 올려 분포를 제대로 학습했을 때 이상탐지 변화: ΔAUC=%+.4f" % d_auc)
    if auc6 > 0.55 and d_auc > 0.05:
        msg = ("개선됨 -> '얕은 모델이라 안 됐던 것'. "
               "제대로 학습하니 이상탐지 신호가 나타남.")
    elif auc6 <= 0.55:
        msg = ("L=6에서도 AUC가 무작위 근처/이하 -> "
               "'Infiltration은 분포를 잘 학습해도 잡기 어려운 은밀한 공격'(정직 보고). "
               "확장 실패가 아니라 공격 자체의 난이도.")
    else:
        msg = "소폭 변화. 결정적이지 않음(수치 그대로 보고)."
    E.log("  => %s" % msg)
    E.log("-" * 70)
    E.log("[최종 요약 1줄] 제대로 학습된(L=6) 모델로 이상탐지가 작동하는가? : %s"
          % ("예 (AUC %.3f, ΔAUC %+.3f)" % (auc6, d_auc) if (auc6 > 0.55 and d_auc > 0.05)
             else "아니오 (AUC %.3f, 무작위 수준 -> Infiltration이 본질적으로 어려운 공격)"
                  % auc6))

    summary["delta_auc_L6_minus_L3"] = float(d_auc)
    summary["total_runtime_sec"] = time.time() - t_start
    with open(os.path.join(E.OUT, "anomaly_redo.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    E.log("총 실행 시간: %.1fs (%.2f분)"
          % (summary["total_runtime_sec"], summary["total_runtime_sec"] / 60))
    E.log("결과물: %s/ (train_8q_L*.png, dist_8q_L*.png, anomaly_8q_L*.png, "
          "anomaly_L3_vs_L6.png, anomaly_redo.json)" % E.OUT)
    E.log("=" * 70)
    E.log("DONE")


if __name__ == "__main__":
    main()
