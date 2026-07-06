#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
10큐빗 QCBM(MMD) 본 학습 — IAT 포함 B1 인코딩 (proto2+syn1+psh1+size4 + FwdIAT2 = 10bit, 1024상태)
==================================================================================================
목적:
  emp_joint(상한)에서 IAT 추가 시 joint 이득이 +0.013(B1)~+0.052(B2)로 나타났다.
  이것이 실제 QCBM(MMD, 표현력 제한) 학습으로 실현되는지 검증한다.
  핵심 질문: 학습된 QCBM의 HOIC 이상탐지 AUC가 marginal 천장(~0.873)을 넘고,
             emp_joint 상한(~0.886)에 얼마나 근접하는가?

설계(시간 안전):
  단계1 smoke: RESTARTS=1, EPOCHS=50 으로 1회 학습 -> wall 측정 -> 전체 시간 추정.
  단계2 full : 추정에 따라 RESTARTS/EPOCHS 확정 후 본 학습.
    - est_300 <= 3h        -> RESTARTS=16, EPOCHS=300
    - 3h < est_300 <= 4.5h -> RESTARTS=16, EPOCHS=200
    - est_300 > 4.5h       -> RESTARTS=8,  EPOCHS=200
    - 최소설정(200ep) 추정도 5h 초과면 full 중단 + 보고 (학습 시작 안 함).
  --smoke-only : 단계1만 수행하고 추정 출력 후 종료(시간 측정용; full 미실행).

학습 경로: experiment2.run_config 를 직접 사용 (내부에서 discretize 재호출 없음을 코드로 확인:
  run_config는 spec["nq"]/spec["name"]/q_train/전역설정만 사용. size_edges 인자는 미사용 leftover).
  -> 10bit SPEC + 1024차원 q_train(B1 인코딩, 스크립트 내 자체 생성)을 넘긴다.

기존 소스(experiment2.py)는 import + 모듈속성(E.LAYERS/EPOCHS/RESTARTS/LR/LOG_EVERY/OUT) 대입만,
파일 자체는 수정하지 않는다. QCBM 학습 외 새 기능 없음.
"""
import os
import sys
import json
import time
import argparse

import numpy as np
import pandas as pd

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if QCBM_DIR not in sys.path:
    sys.path.insert(0, QCBM_DIR)
os.chdir(QCBM_DIR)

import experiment2 as E  # noqa: E402

DATA = "data/cic_0221_ddos.csv"
OUT = "results2"
ATTACK = "ddos attack-hoic"
TAG = "qcbm_iat_b1"
NQ = 10
NSTATES = 1 << NQ          # 1024
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
SPEC_B1 = {"name": TAG, "nq": NQ,
           "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4),
                        ("fwd_iat", 2)]}  # 기록용(run_config는 nq/name만 사용)
IAT_BITS = 2
IAT_BUCKETS = 1 << IAT_BITS
# 학습 하이퍼파라미터 (고정)
LAYERS = 6
LR = 0.1
LOG_EVERY = 25
SMOKE_RESTARTS = 1
SMOKE_EPOCHS = 50
# 시간 임계 (초)
H3, H45, H5 = 3 * 3600, 4.5 * 3600, 5 * 3600


def log1p_quantile_edges(values, nbuckets):
    v = np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
    qs = np.linspace(0, 1, nbuckets + 1)[1:-1]
    return np.unique(np.quantile(v, qs))


def iat_to_bits(values, edges):
    v = np.log1p(np.clip(np.asarray(values, dtype=float), 0, None))
    return np.clip(np.digitize(v, edges), 0, IAT_BUCKETS - 1).astype(int)


def append_bits(state, add_val, add_bits):
    return (np.asarray(state, dtype=int) << add_bits) | np.asarray(add_val, dtype=int)


def marginal_dist(states, nq):
    states = np.asarray(states, dtype=int)
    bit_idx = np.arange(nq)
    p_bit = (((states[:, None] >> bit_idx) & 1).mean(axis=0))
    all_states = np.arange(1 << nq)
    sb = (all_states[:, None] >> bit_idx) & 1
    q = np.where(sb == 1, p_bit, 1.0 - p_bit).prod(axis=1)
    return q / q.sum()


def eval_dist(q, st_te, st_atk, tag):
    fake = {"best": {"p_final": np.asarray(q, dtype=float).tolist(),
                     "final_mmd2": float("nan"), "tv": float("nan")}}
    return E.anomaly_eval(fake, st_te, st_atk, tag)


def build_b1():
    """data 로드 -> benign 0.7/0.3 분할 -> B1(10bit) 상태 + q_train."""
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean",
            "Fwd IAT Mean", "Label"]
    E.log("CSV 로드: %s" % DATA)
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    for c in cols[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    lab = df["Label"].astype(str).str.strip().str.lower()
    ben = df[lab == "benign"].reset_index(drop=True)
    atk = df[lab == ATTACK].reset_index(drop=True)
    E.log("benign=%d  %s=%d" % (len(ben), ATTACK, len(atk)))

    rng = np.random.default_rng(7)
    idx = rng.permutation(len(ben))
    cut = int(0.7 * len(ben))
    ben_tr = ben.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = ben.iloc[idx[cut:]].reset_index(drop=True)

    size_edges = E.size_edges_from_benign(ben_tr, 16)
    fwd_edges = log1p_quantile_edges(ben_tr["Fwd IAT Mean"], IAT_BUCKETS)
    E.log("Fwd IAT Mean log1p inner edges = %s" % np.round(fwd_edges, 4).tolist())

    def enc(d):
        s8 = E.discretize(d, SPEC8, size_edges)
        return append_bits(s8, iat_to_bits(d["Fwd IAT Mean"], fwd_edges), IAT_BITS)

    st_tr, st_te, st_atk = enc(ben_tr), enc(ben_te), enc(atk)
    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    E.log("B1 점유 상태=%d/%d (%.1f%%) | benign train=%d test=%d attack=%d"
          % (occ, NSTATES, 100 * occ / NSTATES, len(st_tr), len(st_te), len(st_atk)))
    return dict(q_train=q_train, st_tr=st_tr, st_te=st_te, st_atk=st_atk, occ=occ,
                size_edges=size_edges, fwd_edges=fwd_edges)


def decide_settings(per_epoch):
    est_300 = per_epoch * 300
    if est_300 <= H3:
        R, EP = 16, 300
    elif est_300 <= H45:
        R, EP = 16, 200
    else:
        R, EP = 8, 200
    est_final = per_epoch * EP
    return R, EP, est_300, est_final


def run_train(d, restarts, epochs, layers):
    E.LAYERS = layers; E.EPOCHS = epochs; E.RESTARTS = restarts
    E.LR = LR; E.LOG_EVERY = LOG_EVERY; E.OUT = OUT
    return E.run_config(SPEC_B1, d["q_train"], d["size_edges"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true",
                    help="단계1(smoke)만 수행하고 추정 출력 후 종료(full 미실행)")
    args = ap.parse_args()

    if not os.path.exists(DATA):
        E.log("데이터 없음: %s -> 실행 불가." % DATA)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)

    E.log("=" * 80)
    E.log("10큐빗 QCBM(MMD) 본 학습 — B1 인코딩 (10bit/1024상태), 대상=HOIC")
    E.log("=" * 80)
    d = build_b1()

    # baseline(학습 불필요) AUC — 비교표용. emp_joint=q_train(경험적), marginal=비트독립 곱.
    an_emp = eval_dist(d["q_train"], d["st_te"], d["st_atk"], "emp_joint")
    an_marg = eval_dist(marginal_dist(d["st_tr"], NQ), d["st_te"], d["st_atk"], "marginal")
    auc_emp, auc_marg = an_emp["auc"], an_marg["auc"]
    E.log("[기준선 재계산] emp_joint AUC=%.4f | marginal AUC=%.4f (joint이득=%+.4f)"
          % (auc_emp, auc_marg, auc_emp - auc_marg))

    # === 단계1: smoke ===
    E.log("-" * 80)
    E.log("[단계1 smoke] RESTARTS=%d EPOCHS=%d L=%d : wall 측정"
          % (SMOKE_RESTARTS, SMOKE_EPOCHS, LAYERS))
    t0 = time.time()
    smoke = run_train(d, SMOKE_RESTARTS, SMOKE_EPOCHS, LAYERS)
    smoke_wall = time.time() - t0
    per_epoch = smoke_wall / SMOKE_EPOCHS
    R, EP, est_300, est_final = decide_settings(per_epoch)
    E.log("[smoke] wall=%.1fs (%.2fs/epoch, 1 restart) | smoke best MMD^2=%.4e TV=%.4f"
          % (smoke_wall, per_epoch, smoke["best"]["final_mmd2"], smoke["best"]["tv"]))
    E.log("[추정] 16재시작 병렬 가정(EP epochs): est_300=%.0fs(%.2fh) -> 확정 RESTARTS=%d EPOCHS=%d "
          "(est_final=%.0fs=%.2fh)"
          % (est_300, est_300 / 3600, R, EP, est_final, est_final / 3600))

    decision = {"smoke_wall_sec": smoke_wall, "sec_per_epoch": per_epoch,
                "est_300_sec": est_300, "chosen_restarts": R, "chosen_epochs": EP,
                "est_final_sec": est_final}

    if args.smoke_only:
        E.log("[--smoke-only] full 미실행. 위 추정으로 tmux 본 학습을 실행하세요.")
        with open(os.path.join(OUT, "qcbm_iat_b1_smoke.json"), "w") as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)
        E.log("DONE(smoke-only)")
        return

    if est_final > H5:
        E.log("[중단] 최소설정(EPOCHS=%d) 추정 %.2fh > 5h 예산. full 학습을 시작하지 않음."
              % (EP, est_final / 3600))
        with open(os.path.join(OUT, "qcbm_iat_b1_smoke.json"), "w") as f:
            json.dump({**decision, "aborted": True}, f, indent=2, ensure_ascii=False)
        E.log("DONE(aborted: over budget)")
        return

    # === 단계2: full ===
    E.log("=" * 80)
    E.log("[단계2 full] RESTARTS=%d EPOCHS=%d L=%d 본 학습 시작" % (R, EP, LAYERS))
    t1 = time.time()
    cfg = run_train(d, R, EP, LAYERS)
    full_wall = time.time() - t1
    best = cfg["best"]
    E.log("[full] 완료 wall=%.1fs (%.2fh) | best MMD^2=%.4e TV=%.4f 수렴~%dep grad %.2e->%.2e"
          % (full_wall, full_wall / 3600, best["final_mmd2"], best["tv"],
             cfg["conv_ep"], cfg["gnorm0"], cfg["gnormf"]))

    # === 이상탐지 평가 ===
    an_qcbm = E.anomaly_eval(cfg, d["st_te"], d["st_atk"], TAG)

    E.log("=" * 80)
    E.log("[핵심 비교] HOIC 이상탐지 AUC (s=-log p, attack=positive, raw MWU)")
    rows = [("marginal(CMC 방식, 기준선)", auc_marg, an_marg),
            ("emp_joint(상한)", auc_emp, an_emp),
            ("QCBM 10q L=%d (우리 모델)" % LAYERS, an_qcbm["auc"], an_qcbm)]
    for name, auc, an in rows:
        E.log("  %-32s AUC=%.4f Youden J=%.4f logp격차=%.3f"
              % (name, auc, an["youden_j"],
                 an["mean_logp_ben"] - an["mean_logp_atk"]))
    beats_marg = an_qcbm["auc"] > auc_marg
    close_to_emp = an_qcbm["auc"] / auc_emp if auc_emp else float("nan")
    E.log("  => QCBM > marginal(%.4f)? %s | emp_joint(%.4f) 근접도=%.1f%%"
          % (auc_marg, "예" if beats_marg else "아니오", auc_emp, 100 * close_to_emp))

    # === 그림 (기존 plot 함수 재사용; 파일명 train_/dist_/anomaly_<TAG>.png) ===
    try:
        E.plot_training(cfg, TAG)
        E.plot_dist(d["q_train"], best, TAG)
        E.plot_anomaly(an_qcbm, TAG)
        E.log("그림 저장: train_%s.png, dist_%s.png, anomaly_%s.png" % (TAG, TAG, TAG))
    except Exception as e:
        E.log("그림 저장 일부 실패: %r" % e)

    # === 저장 ===
    payload = {
        "encoding": "B1: proto2+syn1+psh1+size4 + FwdIAT2 = 10bit (1024)",
        "data": DATA, "attack": ATTACK, "nq": NQ, "nstates": NSTATES,
        "occupied_train_states": d["occ"],
        "split": {"seed": 7, "benign_train_frac": 0.7},
        "fwd_iat_log1p_inner_edges": d["fwd_edges"].tolist(),
        "layers": LAYERS, "lr": LR,
        "decision": decision, "chosen_restarts": R, "chosen_epochs": EP,
        "full_wall_sec": full_wall,
        "train": {"best_mmd2": best["final_mmd2"], "best_tv": best["tv"],
                  "best_kl": best["kl_q_p"], "median_mmd2": cfg["median_mmd2"],
                  "conv_ep": cfg["conv_ep"], "gnorm0": cfg["gnorm0"],
                  "gnormf": cfg["gnormf"], "nparams": cfg["nparams"]},
        "auc": {
            "marginal": auc_marg, "emp_joint": auc_emp,
            "qcbm": an_qcbm["auc"],
            "qcbm_youden_j": an_qcbm["youden_j"],
            "qcbm_logp_gap": an_qcbm["mean_logp_ben"] - an_qcbm["mean_logp_atk"],
            "qcbm_beats_marginal": bool(beats_marg),
            "qcbm_over_emp_joint_frac": float(close_to_emp),
        },
        "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (no direction selection)",
    }
    with open(os.path.join(OUT, "qcbm_iat_b1.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: %s" % os.path.join(OUT, "qcbm_iat_b1.json"))
    E.log("DONE")


if __name__ == "__main__":
    main()
