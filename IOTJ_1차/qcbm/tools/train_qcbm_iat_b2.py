#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12큐빗 QCBM(MMD) 본 학습 — B2 인코딩 (proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2 = 12bit, 4096상태)
====================================================================================================
목적:
  B1(10bit, +FwdIAT)에서 QCBM AUC 0.8811 > marginal 0.8726, emp_joint 상한 99.5% 실현됨.
  B2는 emp_joint 상한 joint 이득이 훨씬 큼(+0.052 vs B1 +0.013).
  가설: joint 이득이 클수록 QCBM이 marginal을 더 큰 폭으로 이긴다(단조 관계).
  핵심 질문: 학습된 QCBM의 HOIC AUC가 B2 marginal(~0.874)을 넘고 B1보다 큰 폭인가?
            emp_joint 상한(~0.926)에 얼마나 근접? 점유 2.3% 희소성의 영향은?

설계(시간 안전): B1과 동일.
  단계1 smoke: RESTARTS=1, EPOCHS=50 -> wall 측정 -> 추정.
  단계2 full:
    - est_300 <= 3h        -> 16/300
    - 3h < est_300 <= 4.5h -> 16/200
    - 4.5h < est_300       -> 8/200
    - est가 5h 근접/초과면 더 보수적으로 8/150.
    - 최소설정 추정도 5h 초과면 full 중단.
  --smoke-only : 단계1만 수행 후 종료.

학습 경로: experiment2.run_config 직접 사용(내부 discretize 재호출 없음, B1에서 검증).
  12bit SPEC + 자체 생성 4096차원 q_train 전달. experiment2.py 무수정(모듈속성 대입만).
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
TAG = "qcbm_iat_b2"
NQ = 12
NSTATES = 1 << NQ          # 4096
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
SPEC_B2 = {"name": TAG, "nq": NQ,
           "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4),
                        ("fwd_iat", 2), ("flow_iat", 2)]}  # 기록용
IAT_BITS = 2
IAT_BUCKETS = 1 << IAT_BITS
LAYERS = 6
LR = 0.1
LOG_EVERY = 25
SMOKE_RESTARTS = 1
SMOKE_EPOCHS = 50
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


def build_b2():
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean",
            "Fwd IAT Mean", "Flow IAT Mean", "Label"]
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
    flow_edges = log1p_quantile_edges(ben_tr["Flow IAT Mean"], IAT_BUCKETS)
    E.log("Fwd IAT  log1p inner edges = %s" % np.round(fwd_edges, 4).tolist())
    E.log("Flow IAT log1p inner edges = %s" % np.round(flow_edges, 4).tolist())

    def enc(d):
        s = E.discretize(d, SPEC8, size_edges)                       # 8bit (MSB)
        s = append_bits(s, iat_to_bits(d["Fwd IAT Mean"], fwd_edges), IAT_BITS)   # +Fwd
        s = append_bits(s, iat_to_bits(d["Flow IAT Mean"], flow_edges), IAT_BITS)  # +Flow (LSB)
        return s

    st_tr, st_te, st_atk = enc(ben_tr), enc(ben_te), enc(atk)
    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    E.log("B2 점유 상태=%d/%d (%.1f%%) | benign train=%d test=%d attack=%d"
          % (occ, NSTATES, 100 * occ / NSTATES, len(st_tr), len(st_te), len(st_atk)))
    return dict(q_train=q_train, st_tr=st_tr, st_te=st_te, st_atk=st_atk, occ=occ,
                size_edges=size_edges, fwd_edges=fwd_edges, flow_edges=flow_edges)


def decide_settings(per_epoch):
    est_300 = per_epoch * 300
    if est_300 <= H3:
        R, EP = 16, 300
    elif est_300 <= H45:
        R, EP = 16, 200
    else:
        R, EP = 8, 200
    # 5h 근접/초과 시 더 보수적으로
    if per_epoch * EP > 0.95 * H5:
        R, EP = 8, 150
    return R, EP, est_300, per_epoch * EP


def run_train(d, restarts, epochs, layers):
    E.LAYERS = layers; E.EPOCHS = epochs; E.RESTARTS = restarts
    E.LR = LR; E.LOG_EVERY = LOG_EVERY; E.OUT = OUT
    return E.run_config(SPEC_B2, d["q_train"], d["size_edges"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(DATA):
        E.log("데이터 없음: %s -> 실행 불가." % DATA)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)

    E.log("=" * 80)
    E.log("12큐빗 QCBM(MMD) 본 학습 — B2 인코딩 (12bit/4096상태), 대상=HOIC")
    E.log("=" * 80)
    d = build_b2()

    an_emp = eval_dist(d["q_train"], d["st_te"], d["st_atk"], "emp_joint")
    an_marg = eval_dist(marginal_dist(d["st_tr"], NQ), d["st_te"], d["st_atk"], "marginal")
    auc_emp, auc_marg = an_emp["auc"], an_marg["auc"]
    E.log("[기준선 재계산] emp_joint AUC=%.4f | marginal AUC=%.4f (joint이득=%+.4f)"
          % (auc_emp, auc_marg, auc_emp - auc_marg))

    # === 단계1 smoke ===
    E.log("-" * 80)
    E.log("[단계1 smoke] RESTARTS=%d EPOCHS=%d L=%d" % (SMOKE_RESTARTS, SMOKE_EPOCHS, LAYERS))
    t0 = time.time()
    smoke = run_train(d, SMOKE_RESTARTS, SMOKE_EPOCHS, LAYERS)
    smoke_wall = time.time() - t0
    per_epoch = smoke_wall / SMOKE_EPOCHS
    R, EP, est_300, est_final = decide_settings(per_epoch)
    E.log("[smoke] wall=%.1fs (%.2fs/epoch) | best MMD^2=%.4e TV=%.4f"
          % (smoke_wall, per_epoch, smoke["best"]["final_mmd2"], smoke["best"]["tv"]))
    E.log("[추정] est_300=%.0fs(%.2fh) -> 확정 RESTARTS=%d EPOCHS=%d (est_final=%.0fs=%.2fh)"
          % (est_300, est_300 / 3600, R, EP, est_final, est_final / 3600))

    decision = {"smoke_wall_sec": smoke_wall, "sec_per_epoch": per_epoch,
                "est_300_sec": est_300, "chosen_restarts": R, "chosen_epochs": EP,
                "est_final_sec": est_final}

    if args.smoke_only:
        E.log("[--smoke-only] full 미실행.")
        with open(os.path.join(OUT, "qcbm_iat_b2_smoke.json"), "w") as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)
        E.log("DONE(smoke-only)")
        return

    if est_final > H5:
        E.log("[중단] 최소설정 추정 %.2fh > 5h. full 미시작." % (est_final / 3600))
        with open(os.path.join(OUT, "qcbm_iat_b2_smoke.json"), "w") as f:
            json.dump({**decision, "aborted": True}, f, indent=2, ensure_ascii=False)
        E.log("DONE(aborted)")
        return

    # === 단계2 full ===
    E.log("=" * 80)
    E.log("[단계2 full] RESTARTS=%d EPOCHS=%d L=%d 본 학습 시작" % (R, EP, LAYERS))
    t1 = time.time()
    cfg = run_train(d, R, EP, LAYERS)
    full_wall = time.time() - t1
    best = cfg["best"]
    E.log("[full] 완료 wall=%.1fs (%.2fh) | best MMD^2=%.4e TV=%.4f 수렴~%dep grad %.2e->%.2e"
          % (full_wall, full_wall / 3600, best["final_mmd2"], best["tv"],
             cfg["conv_ep"], cfg["gnorm0"], cfg["gnormf"]))

    an_qcbm = E.anomaly_eval(cfg, d["st_te"], d["st_atk"], TAG)

    E.log("=" * 80)
    E.log("[핵심 비교] HOIC AUC (s=-log p, attack=positive, raw MWU)")
    for name, auc, an in [("marginal(기준선)", auc_marg, an_marg),
                          ("emp_joint(상한)", auc_emp, an_emp),
                          ("QCBM 12q L=%d" % LAYERS, an_qcbm["auc"], an_qcbm)]:
        E.log("  %-24s AUC=%.4f Youden J=%.4f logp격차=%.3f"
              % (name, auc, an["youden_j"], an["mean_logp_ben"] - an["mean_logp_atk"]))
    beats = an_qcbm["auc"] > auc_marg
    gain = an_qcbm["auc"] - auc_marg
    close = an_qcbm["auc"] / auc_emp if auc_emp else float("nan")
    E.log("  => QCBM > marginal(%.4f)? %s | QCBM-marginal=%+.4f | emp_joint(%.4f) 근접도=%.1f%%"
          % (auc_marg, "예" if beats else "아니오", gain, auc_emp, 100 * close))
    E.log("  [B1 대비] B1: QCBM-marg=+0.0085, 상한근접 99.5%% | B2: QCBM-marg=%+.4f, 상한근접 %.1f%%"
          % (gain, 100 * close))

    try:
        E.plot_training(cfg, TAG)
        E.plot_dist(d["q_train"], best, TAG)
        E.plot_anomaly(an_qcbm, TAG)
        E.log("그림 저장: train_%s.png, dist_%s.png, anomaly_%s.png" % (TAG, TAG, TAG))
    except Exception as e:
        E.log("그림 저장 일부 실패: %r" % e)

    payload = {
        "encoding": "B2: proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2 = 12bit (4096)",
        "data": DATA, "attack": ATTACK, "nq": NQ, "nstates": NSTATES,
        "occupied_train_states": d["occ"], "occ_frac": d["occ"] / NSTATES,
        "split": {"seed": 7, "benign_train_frac": 0.7},
        "fwd_iat_log1p_inner_edges": d["fwd_edges"].tolist(),
        "flow_iat_log1p_inner_edges": d["flow_edges"].tolist(),
        "layers": LAYERS, "lr": LR,
        "decision": decision, "chosen_restarts": R, "chosen_epochs": EP,
        "full_wall_sec": full_wall,
        "train": {"best_mmd2": best["final_mmd2"], "best_tv": best["tv"],
                  "best_kl": best["kl_q_p"], "median_mmd2": cfg["median_mmd2"],
                  "conv_ep": cfg["conv_ep"], "gnorm0": cfg["gnorm0"],
                  "gnormf": cfg["gnormf"], "nparams": cfg["nparams"]},
        "auc": {"marginal": auc_marg, "emp_joint": auc_emp, "qcbm": an_qcbm["auc"],
                "qcbm_youden_j": an_qcbm["youden_j"],
                "qcbm_logp_gap": an_qcbm["mean_logp_ben"] - an_qcbm["mean_logp_atk"],
                "qcbm_beats_marginal": bool(beats),
                "qcbm_minus_marginal": float(gain),
                "qcbm_over_emp_joint_frac": float(close)},
        "compare_b1": {"qcbm_minus_marginal": 0.0085, "over_emp_joint_frac": 0.9948,
                       "joint_gain_upper": 0.0130},
        "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (no direction selection)",
    }
    with open(os.path.join(OUT, "qcbm_iat_b2.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: %s" % os.path.join(OUT, "qcbm_iat_b2.json"))
    E.log("DONE")


if __name__ == "__main__":
    main()
