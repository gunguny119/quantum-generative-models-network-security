#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
psh 제거 11비트 QCBM(MMD) 본 학습 — 강신호 없는 순수 상호작용을 QCBM이 학습하는지
====================================================================================
인코딩: proto2+syn1+size4+FwdIAT2+FlowIAT2 = 11bit (2048상태). psh(1비트) 제거.
배경: psh 통제 분석에서 psh 제거 시 marginal 0.5136(붕괴)·emp_joint 0.7773(유지), joint이득 +0.2637.
      이 0.7773은 emp_joint(상한). QCBM(MMD)이 이 순수 상호작용을 학습으로 실현하는지 검증.
핵심: QCBM AUC > marginal(0.5136)? emp_joint 상한(0.7773) 근접도? (B1 99.5%, B2 99.2% 실현)

설계(시간 안전, B1·B2와 동일): smoke(R=1,EP=50)->추정->full.
  est_300<=3h -> 16/300 ; 3~4.5h -> 16/200 ; >4.5h -> 8/200. 5h 초과 금지.
  --smoke-only : 단계1만 후 종료.
학습 경로: experiment2.run_config 직접(내부 discretize 재호출 없음). 11bit SPEC + 자체 2048차원 q_train.
재사용: experiment2(run_config/build_kernel/anomaly_eval/empirical_dist),
        diagnose_iat_joint(log1p_quantile_edges), diagnose_encoding_info_loss(load_split/feature_levels),
        train_qcbm_iat_b2(marginal_dist). 기존 소스 무수정(모듈속성 대입만).
"""
import os
import sys
import json
import time
import argparse

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E                          # noqa: E402
import diagnose_iat_joint as DIAG                # noqa: E402
import diagnose_encoding_info_loss as INFO       # noqa: E402  load_split, feature_levels
from train_qcbm_iat_b2 import marginal_dist      # noqa: E402

OUT = "results2"
TAG = "qcbm_iat_no_psh"
NQ = 11
NSTATES = 1 << NQ                                # 2048
# psh 제거: MSB->LSB 순서
ORDER = [("proto", 2), ("syn", 1), ("size", 4), ("fwd_iat", 2), ("flow_iat", 2)]
SPEC = {"name": TAG, "nq": NQ, "features": ORDER}
LAYERS = 6
LR = 0.1
LOG_EVERY = 25
SMOKE_RESTARTS, SMOKE_EPOCHS = 1, 50
H3, H45, H5 = 3 * 3600, 4.5 * 3600, 5 * 3600
EPS = 1e-12
EXPECT_EMP, EXPECT_MARG, EXPECT_OCC = 0.7773, 0.5136, 90


def assemble(levels):
    state = None
    for name, bits in ORDER:
        v = np.asarray(levels[name], int)
        state = v if state is None else ((state << bits) | v)
    return state


def eval_auc(q, st_te, st_atk):
    s_te = -np.log(q[np.asarray(st_te, int)] + EPS)
    s_atk = -np.log(q[np.asarray(st_atk, int)] + EPS)
    scores = np.concatenate([s_te, s_atk])
    labels = np.concatenate([np.zeros(len(s_te), int), np.ones(len(s_atk), int)])
    return float(E.auc_score(scores, labels))


def build_nopsh():
    ben_tr, ben_te, atk = INFO.load_split()
    size_edges = E.size_edges_from_benign(ben_tr, 16)
    fwd_edges = DIAG.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], 4)
    flow_edges = DIAG.log1p_quantile_edges(ben_tr["Flow IAT Mean"], 4)
    E.log("Fwd IAT edges=%s | Flow IAT edges=%s"
          % (np.round(fwd_edges, 4).tolist(), np.round(flow_edges, 4).tolist()))

    def lv(df):
        return {n: np.asarray(INFO.feature_levels(df, n, size_edges, fwd_edges, flow_edges), int)
                for n, _ in ORDER}
    L_tr, L_te, L_atk = lv(ben_tr), lv(ben_te), lv(atk)
    st_tr, st_te, st_atk = assemble(L_tr), assemble(L_te), assemble(L_atk)
    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    E.log("11bit(psh 제거) 점유=%d/%d (%.1f%%) | train=%d test=%d atk=%d"
          % (occ, NSTATES, 100 * occ / NSTATES, len(st_tr), len(st_te), len(st_atk)))
    return dict(q_train=q_train, st_tr=st_tr, st_te=st_te, st_atk=st_atk, occ=occ,
                size_edges=size_edges)


def decide(per_epoch):
    est_300 = per_epoch * 300
    if est_300 <= H3:
        R, EP = 16, 300
    elif est_300 <= H45:
        R, EP = 16, 200
    else:
        R, EP = 8, 200
    return R, EP, est_300, per_epoch * EP


def run_train(d, restarts, epochs):
    E.LAYERS = LAYERS; E.EPOCHS = epochs; E.RESTARTS = restarts
    E.LR = LR; E.LOG_EVERY = LOG_EVERY; E.OUT = OUT
    return E.run_config(SPEC, d["q_train"], d["size_edges"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke-only", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 80)
    E.log("psh 제거 11비트 QCBM(MMD) 본 학습 (2048상태), 대상=HOIC")
    E.log("=" * 80)

    d = build_nopsh()

    # --- 정합 검증: emp_joint 0.7773 / marginal 0.5136 ---
    q_emp = d["q_train"]
    q_marg = marginal_dist(d["st_tr"], NQ)
    auc_emp = eval_auc(q_emp, d["st_te"], d["st_atk"])
    auc_marg = eval_auc(q_marg, d["st_te"], d["st_atk"])
    E.log("정합 검증: emp_joint=%.4f marginal=%.4f 점유=%d (기대 %.4f/%.4f/%d)"
          % (auc_emp, auc_marg, d["occ"], EXPECT_EMP, EXPECT_MARG, EXPECT_OCC))
    if abs(auc_emp - EXPECT_EMP) > 0.01 or abs(auc_marg - EXPECT_MARG) > 0.01:
        E.log("!!! 정합 실패 -> 11비트 인코딩 재현 어긋남. 중단."); sys.exit(2)
    E.log("정합 OK.")

    # --- smoke ---
    E.log("-" * 80)
    E.log("[smoke] R=%d EP=%d L=%d" % (SMOKE_RESTARTS, SMOKE_EPOCHS, LAYERS))
    t0 = time.time()
    sm = run_train(d, SMOKE_RESTARTS, SMOKE_EPOCHS)
    sw = time.time() - t0
    per = sw / SMOKE_EPOCHS
    R, EP, est300, estf = decide(per)
    E.log("[smoke] wall=%.1fs (%.2fs/ep) best MMD^2=%.4e TV=%.4f" %
          (sw, per, sm["best"]["final_mmd2"], sm["best"]["tv"]))
    E.log("[추정] est_300=%.0fs(%.2fh) -> 확정 R=%d EP=%d (est_final=%.0fs=%.2fh)"
          % (est300, est300 / 3600, R, EP, estf, estf / 3600))
    decision = {"smoke_wall_sec": sw, "sec_per_epoch": per, "est_300_sec": est300,
                "chosen_restarts": R, "chosen_epochs": EP, "est_final_sec": estf}

    if args.smoke_only:
        with open(os.path.join(OUT, "qcbm_iat_no_psh_smoke.json"), "w") as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)
        E.log("[--smoke-only] full 미실행. DONE(smoke-only)"); return
    if estf > H5:
        E.log("[중단] 추정 %.2fh > 5h. full 미시작." % (estf / 3600))
        with open(os.path.join(OUT, "qcbm_iat_no_psh_smoke.json"), "w") as f:
            json.dump({**decision, "aborted": True}, f, indent=2, ensure_ascii=False)
        return

    # --- full ---
    E.log("=" * 80)
    E.log("[full] R=%d EP=%d L=%d 본 학습" % (R, EP, LAYERS))
    t1 = time.time()
    cfg = run_train(d, R, EP)
    fw = time.time() - t1
    best = cfg["best"]
    E.log("[full] wall=%.1fs(%.2fh) best MMD^2=%.4e TV=%.4f 수렴~%dep grad %.2e->%.2e"
          % (fw, fw / 3600, best["final_mmd2"], best["tv"], cfg["conv_ep"],
             cfg["gnorm0"], cfg["gnormf"]))

    an = E.anomaly_eval(cfg, d["st_te"], d["st_atk"], TAG)
    qcbm_auc = an["auc"]
    E.log("=" * 80)
    E.log("[핵심 비교] HOIC AUC (s=-log p, attack=positive, raw MWU)")
    for nm, a, anr in [("marginal(psh제거 붕괴)", auc_marg, q_marg),
                       ("emp_joint(상한)", auc_emp, q_emp),
                       ("QCBM 11q L=%d" % LAYERS, qcbm_auc, None)]:
        if anr is not None:
            E.log("  %-22s AUC=%.4f" % (nm, a))
        else:
            E.log("  %-22s AUC=%.4f Youden J=%.4f logp격차=%.3f"
                  % (nm, a, an["youden_j"], an["mean_logp_ben"] - an["mean_logp_atk"]))
    beats = qcbm_auc > auc_marg
    close = qcbm_auc / auc_emp if auc_emp else float("nan")
    E.log("  => QCBM > marginal(%.4f)? %s | emp_joint(%.4f) 근접도=%.1f%%"
          % (auc_marg, "예" if beats else "아니오", auc_emp, 100 * close))

    try:
        E.plot_training(cfg, TAG); E.plot_dist(d["q_train"], best, TAG); E.plot_anomaly(an, TAG)
        E.log("그림 저장: train_%s.png dist_%s.png anomaly_%s.png" % (TAG, TAG, TAG))
    except Exception as e:
        E.log("그림 일부 실패: %r" % e)

    payload = {
        "encoding": "11bit: proto2+syn1+size4+FwdIAT2+FlowIAT2 (psh 제거)",
        "nq": NQ, "nstates": NSTATES, "occupied_train_states": d["occ"],
        "layers": LAYERS, "lr": LR, "decision": decision,
        "chosen_restarts": R, "chosen_epochs": EP, "full_wall_sec": fw,
        "train": {"best_mmd2": best["final_mmd2"], "best_tv": best["tv"],
                  "best_kl": best["kl_q_p"], "median_mmd2": cfg["median_mmd2"],
                  "conv_ep": cfg["conv_ep"], "gnorm0": cfg["gnorm0"],
                  "gnormf": cfg["gnormf"], "nparams": cfg["nparams"]},
        "auc": {"marginal": auc_marg, "emp_joint": auc_emp, "qcbm": qcbm_auc,
                "qcbm_youden_j": an["youden_j"],
                "qcbm_logp_gap": an["mean_logp_ben"] - an["mean_logp_atk"],
                "qcbm_beats_marginal": bool(beats),
                "qcbm_over_emp_joint_frac": float(close)},
        "compare_b1_b2": {"b1": {"qcbm": 0.8811, "marg": 0.8726, "emp": 0.8856, "over_emp": 0.995},
                          "b2": {"qcbm": 0.9185, "marg": 0.8741, "emp": 0.9258, "over_emp": 0.992}},
        "baseline_verify": {"emp_joint": auc_emp, "marginal": auc_marg,
                            "expected": {"emp": EXPECT_EMP, "marg": EXPECT_MARG, "occ": EXPECT_OCC}},
        "eval": "s=-log p(x)+1e-12, attack=positive, raw Mann-Whitney U (no direction selection)",
        "note": "판정은 웹 AI. QCBM 학습 1회(tmux). 기존 소스 무수정.",
    }
    with open(os.path.join(OUT, "qcbm_iat_no_psh.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/qcbm_iat_no_psh.json")
    E.log("DONE")


if __name__ == "__main__":
    main()
