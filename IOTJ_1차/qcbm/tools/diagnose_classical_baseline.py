#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
고전 baseline 이상탐지 AUC 진단 (새 학습 없음)
=================================================
목적:
  QCBM(8q, MMD, joint 256상태)의 Infiltration AUC가 0.43(역신호)인 것이
  "양자가 못해서"인지 "이 데이터·이 feature로는 분포 기반 탐지가 원래 안 되는지"를
  가리기 위해, 동일한 데이터·인코딩·train/test 분할·평가로 고전 baseline 2종의 AUC를 잰다.

  (a) emp_joint : benign train의 256상태 경험적 분포 (분포 기반 탐지의 사실상 상한)
  (b) marginal  : 각 비트의 P(bit=1)만으로 만든 독립(곱) 분포 (CMC 논문 방식의 핵심)
  (c) QCBM L3/L6: results2/anomaly_redo.json에서 기존 수치를 읽어와 표에 함께 표시 (새 학습 X)

평가 방식은 experiment2.anomaly_eval과 "완전히 동일"하게 한다:
  s(x) = -log p(x), attack=positive, raw Mann-Whitney U AUC (방향 자동반전 없음).
  baseline 분포를 cfg_out["best"]["p_final"] 형태로 감싸 그대로 anomaly_eval에 넘긴다.

QCBM은 새로 학습하지 않는다. (a)(b)는 학습이 필요 없는 순수 계산이다.
"""
import os
import sys
import json
import math

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# experiment2 를 import 할 수 있도록 qcbm 루트를 path/cwd 로 맞춘다.
# (experiment2.DATA = "data/cic_thursday.csv" 가 상대경로이고, 출력도 results2/ 이므로
#  qcbm 디렉토리에서 동작하도록 chdir 한다.)
# ---------------------------------------------------------------------------
QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if QCBM_DIR not in sys.path:
    sys.path.insert(0, QCBM_DIR)
os.chdir(QCBM_DIR)

import experiment2 as E  # noqa: E402  (가드 있음 -> import 시 main() 자동실행 안 됨)

OUT = "results2"
NQ = 8
NSTATES = 1 << NQ  # 256
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}

# experiment2.discretize 의 비트 패킹은 features 나열 순서대로 MSB->LSB:
#   state = ((((proto<<1)|syn)<<1|psh)<<4)|size
#         = proto*64 + syn*32 + psh*16 + size
# 따라서 8비트 위치(bit0=LSB ... bit7=MSB):
#   bit7,bit6 = proto   bit5 = syn   bit4 = psh   bit3..bit0 = size
BIT_FEATURE = {7: "proto[hi]", 6: "proto[lo]", 5: "syn", 4: "psh",
               3: "size[b3]", 2: "size[b2]", 1: "size[b1]", 0: "size[b0]"}


def build_marginal_dist(states, nq=NQ):
    """benign train 상태들로부터 각 비트의 P(bit=1)을 추정,
       비트 독립(곱) 가정으로 256상태 분포를 구성한다."""
    states = np.asarray(states, dtype=int)
    bit_idx = np.arange(nq)                              # 0..7 (LSB..MSB)
    sample_bits = (states[:, None] >> bit_idx) & 1       # (N, nq)
    p_bit = sample_bits.mean(axis=0)                     # P(bit=1) per bit
    all_states = np.arange(1 << nq)
    state_bits = (all_states[:, None] >> bit_idx) & 1    # (256, nq)
    probs = np.where(state_bits == 1, p_bit, 1.0 - p_bit)
    q = probs.prod(axis=1)
    q = q / q.sum()
    return q, p_bit


def eval_dist(q, st_te, st_atk, tag):
    """experiment2.anomaly_eval 을 그대로 재사용해 동일 평가.
       anomaly_eval 은 cfg_out['best']['p_final'] 만 사용하므로 그 형태로 감싼다."""
    fake_cfg = {"best": {"p_final": np.asarray(q, dtype=float).tolist(),
                         "final_mmd2": float("nan"), "tv": float("nan")}}
    an = E.anomaly_eval(fake_cfg, st_te, st_atk, tag)
    return an


def main():
    E.log("=" * 70)
    E.log("고전 baseline 이상탐지 AUC 진단 (emp_joint / marginal vs QCBM, 새 학습 없음)")
    E.log("=" * 70)

    # --- 데이터 + experiment2.main()과 동일한 분할 재현 ---
    benign, attack = E.load_frames()
    rng = np.random.default_rng(7)                # experiment2/anomaly_redo 와 동일 seed
    idx = rng.permutation(len(benign))
    cut = int(0.7 * len(benign))
    ben_tr = benign.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = benign.iloc[idx[cut:]].reset_index(drop=True)
    E.log("Benign train=%d / test=%d, attack(Infilteration)=%d"
          % (len(ben_tr), len(ben_te), len(attack)))

    # --- 동일 이산화기 (16버킷, benign train에 적합) ---
    edges = E.size_edges_from_benign(ben_tr, 16)
    st_tr = E.discretize(ben_tr, SPEC8, edges)
    st_te = E.discretize(ben_te, SPEC8, edges)
    st_atk = E.discretize(attack, SPEC8, edges)
    E.log("8q 이산화: %d상태 중 %d개 점유 | benign_test=%d, attack=%d 샘플"
          % (NSTATES, int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum()),
             len(st_te), len(st_atk)))

    # --- (a) 경험적 joint 분포 ---
    q_emp = E.empirical_dist(st_tr, NSTATES)
    an_emp = eval_dist(q_emp, st_te, st_atk, "emp_joint")

    # --- (b) marginal 독립(곱) 분포 ---
    q_marg, p_bit = build_marginal_dist(st_tr, NQ)
    E.log("marginal P(bit=1) [bit7..bit0] = %s"
          % np.array2string(p_bit[::-1], precision=4, suppress_small=True))
    an_marg = eval_dist(q_marg, st_te, st_atk, "marginal")

    # --- (c) QCBM L3/L6: 기존 결과 json에서 읽기 (새 학습 없음) ---
    qcbm = {}
    redo_path = os.path.join(OUT, "anomaly_redo.json")
    if os.path.exists(redo_path):
        with open(redo_path) as f:
            redo = json.load(f)
        for L in ("3", "6"):
            a = redo.get("anomaly", {}).get(L)
            if a:
                qcbm["QCBM_L%s" % L] = a
        E.log("QCBM 참고수치 로드: %s" % redo_path)
    else:
        E.log("주의: %s 없음 -> QCBM 참고수치 생략" % redo_path)

    # --- 결과 정리 ---
    def row(tag, an):
        auc = an["auc"]
        return {
            "model": tag,
            "auc": auc,
            "auc_flipped": 1.0 - auc,            # 참고용(표시 전용, auc_score는 미수정)
            "youden_j": an["youden_j"],
            "mean_logp_ben": an["mean_logp_ben"],
            "mean_logp_atk": an["mean_logp_atk"],
            "logp_gap": an["mean_logp_ben"] - an["mean_logp_atk"],
            "mean_p_ben": an["mean_p_ben"],
            "mean_p_atk": an["mean_p_atk"],
        }

    table = [row("emp_joint", an_emp), row("marginal", an_marg)]
    for tag, a in qcbm.items():
        table.append({
            "model": tag,
            "auc": a["auc"],
            "auc_flipped": 1.0 - a["auc"],
            "youden_j": a["youden_j"],
            "mean_logp_ben": a["mean_logp_ben"],
            "mean_logp_atk": a["mean_logp_atk"],
            "logp_gap": a["mean_logp_ben"] - a["mean_logp_atk"],
            "mean_p_ben": a["mean_p_ben"],
            "mean_p_atk": a["mean_p_atk"],
        })

    # --- 표 출력 ---
    E.log("=" * 70)
    E.log("[비교표] 동일 데이터·인코딩·분할·평가(s=-log p, attack=positive, raw MWU AUC)")
    E.log("  %-11s %-8s %-8s %-9s %-10s %-10s %-8s" %
          ("model", "AUC", "1-AUC", "YoudenJ", "logp_ben", "logp_atk", "gap"))
    for r in table:
        E.log("  %-11s %-8.4f %-8.4f %-9.4f %-10.3f %-10.3f %+8.3f" %
              (r["model"], r["auc"], r["auc_flipped"], r["youden_j"],
               r["mean_logp_ben"], r["mean_logp_atk"], r["logp_gap"]))
    E.log("=" * 70)

    # --- json 저장 ---
    payload = {
        "setup": {
            "nq": NQ, "nstates": NSTATES,
            "features": SPEC8["features"],
            "split": {"seed": 7, "benign_train_frac": 0.7},
            "n_benign_train": int(len(ben_tr)), "n_benign_test": int(len(ben_te)),
            "n_attack": int(len(st_atk)),
            "attack_label": "infilteration",
            "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (no direction selection)",
        },
        "marginal_p_bit_lsb_to_msb": p_bit.tolist(),
        "bit_feature_map": BIT_FEATURE,
        "results": table,
    }
    out_json = os.path.join(OUT, "classical_baseline.json")
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("결과 저장: %s" % out_json)

    # --- (선택) ROC 비교 그림 ---
    try:
        plt.figure(figsize=(6.5, 6))
        for an, tag, c in [(an_emp, "emp_joint", "darkgreen"),
                           (an_marg, "marginal", "darkorange")]:
            plt.plot(an["fpr"], an["tpr"], lw=2, color=c,
                     label="%s (AUC=%.3f)" % (tag, an["auc"]))
        # QCBM은 ROC 좌표가 json에 없으므로 AUC만 범례 텍스트로 표기
        qcbm_txt = "  ".join("%s AUC=%.3f" % (k, v["auc"]) for k, v in qcbm.items())
        plt.plot([0, 1], [0, 1], "k--", alpha=.5, label="chance")
        plt.xlabel("FPR"); plt.ylabel("TPR")
        plt.title("Classical baseline ROC (8q, Infiltration)\n%s" % qcbm_txt)
        plt.legend(loc="lower right"); plt.grid(alpha=.3); plt.tight_layout()
        fp = os.path.join(OUT, "classical_baseline.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("ROC 그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
