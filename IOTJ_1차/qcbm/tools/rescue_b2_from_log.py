#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B2 결과 박제(경로 A): 재학습 없이 tee 로그의 확정 스칼라값을 JSON으로 저장 + 실데이터 비교 그림.
=================================================================================================
배경: train_qcbm_iat_b2.py 가 학습/평가를 모두 끝낸 뒤, "[B1 대비]" 로그 한 줄의 % 이스케이프
      버그로 JSON 저장·그림 생성 직전에 죽음. 학습 결과는 results2/qcbm_iat_b2_train.log 에 무손실.
      best 모델/분포 배열은 디스크에 저장된 적이 없어(스크립트가 p_final을 저장 안 함) 분포/ROC/
      학습곡선 그림은 재학습 없이는 복원 불가. -> 스칼라 JSON 박제 + 실스칼라 비교 막대그림만 생성.

값 출처: results2/qcbm_iat_b2_train.log (tee), qcbm_iat_b2_smoke.json, iat_joint.json(동일 benign
         train의 IAT 분위수 경계). 재학습 없음.
"""
import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(QCBM_DIR)
OUT = "results2"

# --- 로그에서 확인된 확정 값 (results2/qcbm_iat_b2_train.log) ---
payload = {
    "encoding": "B2: proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2 = 12bit (4096)",
    "data": "data/cic_0221_ddos.csv",
    "attack": "ddos attack-hoic",
    "nq": 12, "nstates": 4096,
    "occupied_train_states": 94, "occ_frac": 94 / 4096,
    "split": {"seed": 7, "benign_train_frac": 0.7},
    "fwd_iat_log1p_inner_edges": [5.738988730390821, 5.906723318652891,
                                  7.8580608894435695],
    "flow_iat_log1p_inner_edges": [5.335131339670753, 5.502617829924897,
                                   7.454768172443553],
    "layers": 6, "lr": 0.1,
    "decision": {"smoke_wall_sec": 112.93808269500732,
                 "sec_per_epoch": 2.2587616539001463,
                 "est_300_sec": 677.6284961700438,
                 "chosen_restarts": 16, "chosen_epochs": 300,
                 "est_final_sec": 677.6284961700438},
    "chosen_restarts": 16, "chosen_epochs": 300,
    "full_wall_sec": 1319.8,
    "train": {"best_mmd2": 1.8865e-04, "best_tv": 0.0930, "best_kl": 0.1141,
              "median_mmd2": 3.9601e-04, "conv_ep": 284,
              "gnorm0": 8.110e-03, "gnormf": 1.252e-05, "nparams": 216,
              "best_seed": 14,
              "local_optima_seeds": [0, 1, 12]},  # TV≈0.50,0.50,0.75 (나머지 13개 TV 0.08~0.12)
    "auc": {"marginal": 0.8741, "emp_joint": 0.9258, "qcbm": 0.9185,
            "qcbm_youden_j": 0.8374, "qcbm_logp_gap": 7.475,
            "qcbm_mean_p_ben": 1.528e-02, "qcbm_mean_p_atk": 2.384e-03,
            "qcbm_mean_logp_ben": -4.260, "qcbm_mean_logp_atk": -11.735,
            "marginal_youden_j": 0.7558, "marginal_logp_gap": 3.874,
            "emp_joint_youden_j": 0.8448, "emp_joint_logp_gap": 8.054,
            "qcbm_beats_marginal": True,
            "qcbm_minus_marginal": 0.0445,
            "qcbm_over_emp_joint_frac": 0.992},
    "compare_b1": {"qcbm_minus_marginal": 0.0085, "over_emp_joint_frac": 0.995,
                   "joint_gain_upper": 0.0130,
                   "marginal": 0.8726, "emp_joint": 0.8856, "qcbm": 0.8811},
    "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (no direction selection)",
    "_rescue_note": ("재학습 없이 results2/qcbm_iat_b2_train.log 의 확정값으로 박제. "
                     "분포/ROC/학습곡선 그림은 모델 배열 미저장으로 복원 불가."),
}


def main():
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "qcbm_iat_b2.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print("저장: results2/qcbm_iat_b2.json")

    # --- 실스칼라 비교 막대그림 (B1 vs B2, 모두 로그의 실제 AUC) ---
    labels = ["marginal", "emp_joint(상한)", "QCBM"]
    b1 = [0.8726, 0.8856, 0.8811]
    b2 = [0.8741, 0.9258, 0.9185]
    x = np.arange(len(labels)); w = 0.38
    fig, ax = plt.subplots(figsize=(8, 5.2))
    ax.bar(x - w / 2, b1, w, label="B1 (10q, +FwdIAT)", color="steelblue")
    ax.bar(x + w / 2, b2, w, label="B2 (12q, +Fwd+FlowIAT)", color="crimson")
    for xi, v in zip(x - w / 2, b1):
        ax.text(xi, v + 0.002, "%.4f" % v, ha="center", va="bottom", fontsize=9)
    for xi, v in zip(x + w / 2, b2):
        ax.text(xi, v + 0.002, "%.4f" % v, ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_ylim(0.86, 0.94); ax.set_ylabel("HOIC anomaly AUC")
    ax.set_title("QCBM vs baselines — joint gain grows with time-axis features\n"
                 "QCBM-marginal: B1 +0.0085  ->  B2 +0.0445  (emp_joint 상한 99%대 실현)")
    ax.legend(); ax.grid(alpha=.3, axis="y")
    plt.tight_layout()
    fp = os.path.join(OUT, "compare_b1_b2_qcbm.png")
    plt.savefig(fp, dpi=130); plt.close()
    print("저장: %s (실스칼라 비교; 분포/ROC/학습곡선은 모델 미저장으로 복원 불가)" % fp)
    print("DONE")


if __name__ == "__main__":
    main()
