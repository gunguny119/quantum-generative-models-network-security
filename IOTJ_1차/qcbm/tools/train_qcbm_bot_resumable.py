#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bot QCBM(MMD) 본 학습 — restart 단위 체크포인트로 중단 복원 가능 (기존 소스 무수정, import 재사용)
=====================================================================================================
데이터: data/cic_bot_0203.csv (benign / bot). 인코딩 B2 12bit(4096상태):
  proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2.  경계는 Bot benign train(seed=7,0.7)에서 생성.

복잡도 진단(results2/attack_complexity.json):
  Bot marginal AUC=0.287(독립곱 역방향), emp_joint AUC=0.875(상한), 고랭크(r=7)·조밀(8.81%).
핵심 질문: 학습된 QCBM AUC가 marginal(0.287)을 넘어 emp_joint(0.875)에 얼마나 근접하는가.

중단 복원:
  - 각 restart(seed) 완료 즉시 results2/bot_ckpt/restart_<seed>.npz 원자적 저장(p_final/weights/hist/AUC).
  - Pool.imap_unordered -> 완료되는 즉시 저장(중간저장). 중단돼도 끝난 restart는 보존.
  - 시작 시 bot_ckpt/ 스캔 -> 완료 seed 건너뛰고 남은 것만 학습.
  - EPOCHS 결정은 bot_ckpt/decision.json에 1회 저장 -> 재개 시 동일 사용.

실행(권장): tmux 안에서
  python3 -u tools/train_qcbm_bot_resumable.py 2>&1 | tee -a results2/qcbm_bot_train.log
환경변수(선택): BOT_RESTARTS, BOT_NWORKERS, BOT_EPOCHS(강제, smoke 결정 무시),
  BOT_CKPT_DIR(체크포인트 디렉토리), BOT_SMOKE_ONLY=1(smoke만).
판정/효율비교는 다음 단계. 기존 평가(raw MWU) 유지(방향반전 금지). 새 dependency 없음.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import glob
import json
import time
import math
from multiprocessing import Pool

import numpy as np
import pandas as pd

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E          # noqa: E402  build_kernel/_init/train_one/discretize/size_edges/empirical_dist/anomaly_eval/plot_*
import train_qcbm_iat_b2 as TB2  # noqa: E402  log1p_quantile_edges/iat_to_bits/append_bits/marginal_dist/eval_dist

DATA = "data/cic_bot_0203.csv"
OUT = "results2"
TAG = "qcbm_bot"
ATTACK = "bot"
BENIGN = "benign"
NQ = 12
NSTATES = 1 << NQ
LAYERS = 6                       # params = L*nq*3 = 216
LR = 0.1
LOG_EVERY = 50
SMOKE_EPOCHS = 50
SMOKE_SEED = 9999                # restart seed 0..R-1 와 충돌 안 함
SIZE_BUCKETS = 16
IAT_BITS = 2
IAT_BUCKETS = 1 << IAT_BITS
SPEC8 = {"name": "8q", "nq": 8,
         "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]}
SPEC_B2 = {"name": TAG, "nq": NQ,
           "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4),
                        ("fwd_iat", 2), ("flow_iat", 2)]}

RESTARTS = int(os.environ.get("BOT_RESTARTS", "16"))
NWORKERS = int(os.environ.get("BOT_NWORKERS", "4"))
FORCE_EPOCHS = os.environ.get("BOT_EPOCHS")             # 있으면 smoke 결정 무시
CKPT_DIR = os.environ.get("BOT_CKPT_DIR", os.path.join(OUT, "bot_ckpt"))
SMOKE_ONLY = os.environ.get("BOT_SMOKE_ONLY") == "1"
DECISION_JSON = os.path.join(CKPT_DIR, "decision.json")

# Bot 기준선(정합 검증 대상) — results2/attack_complexity.json
REF_EMP, REF_MARG = 0.8749, 0.2870
TOL = 0.0015
H3, H45, H5 = 3 * 3600, 4.5 * 3600, 5 * 3600


# ---------------------------------------------------------------------------
def build_bot():
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean",
            "Fwd IAT Mean", "Flow IAT Mean", "Label"]
    E.log("CSV 로드: %s" % DATA)
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    for c in cols[:-1]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    n0 = len(df)
    df = df.dropna(subset=cols)
    lab = df["Label"].astype(str).str.strip().str.lower()
    ben = df[lab == BENIGN].reset_index(drop=True)
    atk = df[lab == ATTACK].reset_index(drop=True)
    E.log("dropna 제거=%d | benign=%d  %s=%d" % (n0 - len(df), len(ben), ATTACK, len(atk)))
    if len(ben) == 0 or len(atk) == 0:
        E.log("!!! benign/attack 표본 0 -> 중단."); sys.exit(2)

    rng = np.random.default_rng(7)                  # 기존과 동일 seed/0.7
    idx = rng.permutation(len(ben)); cut = int(0.7 * len(ben))
    ben_tr = ben.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = ben.iloc[idx[cut:]].reset_index(drop=True)

    size_edges = E.size_edges_from_benign(ben_tr, SIZE_BUCKETS)
    fwd_edges = TB2.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], IAT_BUCKETS)
    flow_edges = TB2.log1p_quantile_edges(ben_tr["Flow IAT Mean"], IAT_BUCKETS)

    def enc(d):
        s = E.discretize(d, SPEC8, size_edges)
        s = TB2.append_bits(s, TB2.iat_to_bits(d["Fwd IAT Mean"], fwd_edges), IAT_BITS)
        s = TB2.append_bits(s, TB2.iat_to_bits(d["Flow IAT Mean"], flow_edges), IAT_BITS)
        return s

    st_tr, st_te, st_atk = enc(ben_tr), enc(ben_te), enc(atk)
    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    E.log("Bot 인코딩: 점유 %d/%d (%.2f%%) | benign train=%d test=%d attack=%d"
          % (occ, NSTATES, 100 * occ / NSTATES, len(st_tr), len(st_te), len(st_atk)))
    return dict(q_train=q_train, st_tr=st_tr, st_te=st_te, st_atk=st_atk, occ=occ,
                size_edges=size_edges, fwd_edges=fwd_edges, flow_edges=flow_edges)


def reproduce_check(d):
    auc_emp = TB2.eval_dist(d["q_train"], d["st_te"], d["st_atk"], "emp_joint")["auc"]
    auc_marg = TB2.eval_dist(TB2.marginal_dist(d["st_tr"], NQ), d["st_te"], d["st_atk"], "marginal")["auc"]
    E.log("[정합검증] emp_joint AUC=%.4f (기준 %.4f) | marginal AUC=%.4f (기준 %.4f)"
          % (auc_emp, REF_EMP, auc_marg, REF_MARG))
    ok = abs(auc_emp - REF_EMP) <= TOL and abs(auc_marg - REF_MARG) <= TOL
    if not ok:
        E.log("!!! Bot 인코딩 emp/marginal 재계산이 기준값과 불일치 -> 인코딩 재현 오류. 중단.")
        sys.exit(3)
    E.log("[정합검증] OK — Bot 인코딩이 복잡도 진단과 일치.")
    return auc_emp, auc_marg


# ---------------------------------------------------------------------------
# 체크포인트 (원자적 저장 / 스캔 / 로드)
# ---------------------------------------------------------------------------
def ckpt_path(seed):
    return os.path.join(CKPT_DIR, "restart_%d.npz" % seed)


def save_ckpt(seed, res, an, epochs):
    os.makedirs(CKPT_DIR, exist_ok=True)
    data = {
        "seed": np.int64(seed),
        "p_final": np.asarray(res["p_final"], dtype=np.float64),
        "weights": np.asarray(res["weights"], dtype=np.float64),
        "loss_hist": np.asarray(res["loss_hist"], dtype=np.float64),
        "gnorm_hist": np.asarray(res["gnorm_hist"], dtype=np.float64),
        "final_mmd2": np.float64(res["final_mmd2"]),
        "tv": np.float64(res["tv"]),
        "kl_q_p": np.float64(res["kl_q_p"]),
        "elapsed": np.float64(res["elapsed"]),
        "auc": np.float64(an["auc"]),
        "youden_j": np.float64(an["youden_j"]),
        "logp_gap": np.float64(an["mean_logp_ben"] - an["mean_logp_atk"]),
        "epochs": np.int64(epochs),
        "layers": np.int64(LAYERS),
    }
    tmp = ckpt_path(seed) + ".tmp"
    with open(tmp, "wb") as fh:
        np.savez(fh, **data)
    os.replace(tmp, ckpt_path(seed))               # 원자적 교체


def scan_done():
    done = {}
    for f in sorted(glob.glob(os.path.join(CKPT_DIR, "restart_*.npz"))):
        try:
            z = np.load(f, allow_pickle=False)
            s = int(z["seed"]); _ = float(z["auc"])   # 무결성 확인
            done[s] = f
        except Exception as e:
            E.log("[ckpt] 손상 체크포인트 무시(재학습): %s (%r)" % (f, e))
    return done


def load_restart(seed):
    z = np.load(ckpt_path(seed), allow_pickle=False)
    return {
        "seed": int(z["seed"]), "p_final": z["p_final"], "weights": z["weights"],
        "loss_hist": z["loss_hist"].tolist(), "gnorm_hist": z["gnorm_hist"].tolist(),
        "final_mmd2": float(z["final_mmd2"]), "tv": float(z["tv"]),
        "kl_q_p": float(z["kl_q_p"]), "elapsed": float(z["elapsed"]),
        "auc": float(z["auc"]), "youden_j": float(z["youden_j"]),
        "logp_gap": float(z["logp_gap"]), "epochs": int(z["epochs"]),
    }


# ---------------------------------------------------------------------------
def smoke_measure(q_train, K):
    E.log("-" * 78)
    E.log("[smoke] 1 restart x %d epoch 시간 측정 (seed=%d, 결과 폐기)" % (SMOKE_EPOCHS, SMOKE_SEED))
    t0 = time.time()
    with Pool(processes=1, initializer=E._init,
              initargs=(q_train, K, NQ, LAYERS, SMOKE_EPOCHS, LR, LOG_EVERY, "smoke")) as pool:
        _ = pool.map(E.train_one, [SMOKE_SEED])[0]
    wall = time.time() - t0
    per_epoch = wall / SMOKE_EPOCHS
    return per_epoch, wall


def decide_epochs(per_epoch):
    """NWORKERS 동시 -> waves=ceil(RESTARTS/NWORKERS). wave당 wall ~= per_epoch*EPOCHS (15% 여유)."""
    waves = math.ceil(RESTARTS / NWORKERS)
    safety = 1.15
    def est(ep):
        return waves * ep * per_epoch * safety
    for ep, cap in [(300, H3), (200, H45), (150, H5)]:
        if est(ep) <= cap:
            return ep, est(ep), waves
    # 150 epoch도 5h 초과면 중단
    return None, est(150), waves


# ---------------------------------------------------------------------------
def aggregate(d, an_emp, an_marg, epochs):
    seeds = sorted(scan_done().keys())
    restarts = [load_restart(s) for s in seeds]
    if not restarts:
        E.log("!!! 집계할 체크포인트 없음."); return
    restarts_by_mmd = sorted(restarts, key=lambda r: r["final_mmd2"])
    best = restarts_by_mmd[0]
    aucs = np.array([r["auc"] for r in restarts], dtype=float)
    mmds = np.array([r["final_mmd2"] for r in restarts], dtype=float)
    tvs = np.array([r["tv"] for r in restarts], dtype=float)
    best_auc = best["auc"]
    n_beat_marg = int((aucs > REF_MARG).sum())
    n_exceed_emp = int((aucs > REF_EMP).sum())
    closeness = best_auc / REF_EMP if REF_EMP else float("nan")

    E.log("=" * 78)
    E.log("[집계] 완료 restart=%d/%d | best(최소MMD²) seed=%d MMD²=%.4e TV=%.4f AUC=%.4f"
          % (len(restarts), RESTARTS, best["seed"], best["final_mmd2"], best["tv"], best_auc))
    E.log("[AUC 분포] min=%.4f median=%.4f max=%.4f std=%.4f"
          % (aucs.min(), np.median(aucs), aucs.max(), aucs.std()))
    E.log("[비교] marginal=%.4f | emp_joint=%.4f | QCBM best=%.4f (상한근접 %.1f%%)"
          % (REF_MARG, REF_EMP, best_auc, 100 * closeness))
    E.log("[비교] marginal 초과 restart=%d/%d | emp_joint 초과 restart=%d/%d"
          % (n_beat_marg, len(restarts), n_exceed_emp, len(restarts)))

    # 통합 npz (재분석용)
    np.savez(os.path.join(OUT, "%s_restarts.npz" % TAG),
             seeds=np.array(seeds),
             p_finals=np.stack([np.asarray(r["p_final"]) for r in restarts]),
             aucs=aucs, mmd2=mmds, tv=tvs,
             weights=np.stack([np.asarray(r["weights"]) for r in restarts]),
             q_train=np.asarray(d["q_train"]))

    payload = {
        "encoding": "B2 12bit: proto2 syn1 psh1 size4 + FwdIAT2 + FlowIAT2 (4096)",
        "data": DATA, "attack": ATTACK, "nq": NQ, "nstates": NSTATES,
        "occupied_train_states": d["occ"], "occ_frac": d["occ"] / NSTATES,
        "split": {"seed": 7, "benign_train_frac": 0.7},
        "fwd_iat_log1p_inner_edges": d["fwd_edges"].tolist(),
        "flow_iat_log1p_inner_edges": d["flow_edges"].tolist(),
        "layers": LAYERS, "lr": LR, "nparams": LAYERS * NQ * 3,
        "restarts_done": len(restarts), "restarts_target": RESTARTS,
        "epochs": epochs, "nworkers": NWORKERS,
        "eval": "s(x)=-log p(x), attack=positive, raw Mann-Whitney U (방향반전 없음)",
        "ref": {"marginal": REF_MARG, "emp_joint": REF_EMP},
        "auc_recompute": {"marginal": an_marg, "emp_joint": an_emp},
        "best_seed": int(best["seed"]), "best_auc": best_auc,
        "best_mmd2": best["final_mmd2"], "best_tv": best["tv"], "best_kl": best["kl_q_p"],
        "best_logp_gap": best["logp_gap"], "best_youden_j": best["youden_j"],
        "best_gnorm0": float(best["gnorm_hist"][0]), "best_gnormf": float(best["gnorm_hist"][-1]),
        "auc_distribution": {"min": float(aucs.min()), "median": float(np.median(aucs)),
                             "max": float(aucs.max()), "std": float(aucs.std()),
                             "all": aucs.tolist()},
        "mmd2_distribution": {"min": float(mmds.min()), "median": float(np.median(mmds)),
                              "max": float(mmds.max())},
        "qcbm_beats_marginal": bool(best_auc > REF_MARG),
        "qcbm_minus_marginal": float(best_auc - REF_MARG),
        "qcbm_over_emp_joint_frac": float(closeness),
        "n_restarts_beat_marginal": n_beat_marg,
        "n_restarts_exceed_emp_joint": n_exceed_emp,
        "per_restart": [{"seed": r["seed"], "auc": r["auc"], "mmd2": r["final_mmd2"],
                         "tv": r["tv"], "gnormf": float(r["gnorm_hist"][-1])} for r in restarts],
        "note": "학습 결과(사실). 판정/효율비교는 다음 단계. QCBM 학습됨, 기존 소스 무수정.",
    }
    with open(os.path.join(OUT, "%s.json" % TAG), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: %s/%s.json , %s/%s_restarts.npz" % (OUT, TAG, OUT, TAG))

    # 그림 (기존 plot 함수 재사용)
    try:
        cfg_out = {"results": restarts, "best": best}
        E.plot_training(cfg_out, TAG)
        E.plot_dist(np.asarray(d["q_train"]), best, TAG)
        an_best = E.anomaly_eval({"best": {"p_final": np.asarray(best["p_final"])}},
                                 d["st_te"], d["st_atk"], TAG)
        E.plot_anomaly(an_best, TAG)
        E.log("그림 저장: %s/{train,dist,anomaly}_%s.png" % (OUT, TAG))
    except Exception as e:
        E.log("그림 저장 일부 실패: %r" % e)
    return payload


# ---------------------------------------------------------------------------
def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CKPT_DIR, exist_ok=True)
    E.log("#" * 78)
    E.log("Bot QCBM(MMD) 본 학습 — restart 체크포인트 복원 (NWORKERS=%d, RESTARTS=%d, ckpt=%s)"
          % (NWORKERS, RESTARTS, CKPT_DIR))
    E.log("#" * 78)
    if not os.path.exists(DATA):
        E.log("데이터 없음: %s -> 중단." % DATA); sys.exit(1)

    d = build_bot()
    an_emp, an_marg = reproduce_check(d)            # 정합 게이트(불일치 시 종료)

    done = scan_done()
    todo = [s for s in range(RESTARTS) if s not in done]
    E.log("[재개 스캔] 완료=%d %s | 남은 restart=%d %s"
          % (len(done), sorted(done.keys()), len(todo), todo))

    # EPOCHS 결정 (decision.json 우선 -> FORCE -> smoke)
    decision = None
    if os.path.exists(DECISION_JSON):
        with open(DECISION_JSON) as f:
            decision = json.load(f)
        E.log("[decision] 기존 결정 사용: EPOCHS=%d (%s)" % (decision["epochs"], DECISION_JSON))

    need_K = (decision is None and FORCE_EPOCHS is None) or (len(todo) > 0) or SMOKE_ONLY
    K = E.build_kernel(NQ) if need_K else None
    if K is not None:
        E.log("MMD 커널 build_kernel(%d) 준비 완료." % NQ)

    if decision is None:
        if FORCE_EPOCHS is not None:
            epochs = int(FORCE_EPOCHS)
            decision = {"epochs": epochs, "source": "BOT_EPOCHS(force)", "nworkers": NWORKERS,
                        "restarts": RESTARTS}
            E.log("[decision] BOT_EPOCHS 강제: EPOCHS=%d" % epochs)
        else:
            per_epoch, wall = smoke_measure(d["q_train"], K)
            epochs, est, waves = decide_epochs(per_epoch)
            E.log("[smoke] wall=%.1fs (%.3fs/epoch) | waves=%d(NWORKERS=%d)" % (wall, per_epoch, waves, NWORKERS))
            if epochs is None:
                E.log("[중단] 최소(150ep) 추정 %.2fh > 5h. full 미시작." % (est / 3600))
                with open(DECISION_JSON, "w") as f:
                    json.dump({"aborted": True, "sec_per_epoch": per_epoch,
                               "est_min_sec": est, "waves": waves}, f, indent=2, ensure_ascii=False)
                sys.exit(4)
            decision = {"epochs": epochs, "sec_per_epoch": per_epoch, "smoke_wall": wall,
                        "est_full_sec": est, "waves": waves, "nworkers": NWORKERS, "restarts": RESTARTS,
                        "source": "smoke"}
            E.log("[decision] 확정 EPOCHS=%d (추정 full=%.0fs=%.2fh)" % (epochs, est, est / 3600))
        with open(DECISION_JSON, "w") as f:
            json.dump(decision, f, indent=2, ensure_ascii=False)

    epochs = decision["epochs"]

    if SMOKE_ONLY:
        E.log("[--smoke-only] full 미실행. EPOCHS 결정=%d 저장 완료." % epochs)
        E.log("DONE(smoke-only)")
        return

    # ---- 본 학습 (남은 restart만, 완료마다 체크포인트) ----
    if todo:
        E.log("=" * 78)
        E.log("[full] 남은 %d restart 학습 시작 (EPOCHS=%d, L=%d, NWORKERS=%d, 완료마다 ckpt저장)"
              % (len(todo), epochs, LAYERS, NWORKERS))
        t0 = time.time()
        nproc = min(NWORKERS, len(todo))
        with Pool(processes=nproc, initializer=E._init,
                  initargs=(d["q_train"], K, NQ, LAYERS, epochs, LR, LOG_EVERY, TAG)) as pool:
            for res in pool.imap_unordered(E.train_one, todo):
                seed = int(res["seed"])
                an = TB2.eval_dist(np.asarray(res["p_final"]), d["st_te"], d["st_atk"],
                                   "%s_s%d" % (TAG, seed))
                save_ckpt(seed, res, an, epochs)
                ndone = len(scan_done())
                E.log("[ckpt] seed=%d 저장 ✓ AUC=%.4f MMD²=%.4e TV=%.4f | 진행 %d/%d (%.0fs)"
                      % (seed, an["auc"], res["final_mmd2"], res["tv"], ndone, RESTARTS, time.time() - t0))
        E.log("[full] 학습 루프 종료 wall=%.1fs (%.2fh)" % (time.time() - t0, (time.time() - t0) / 3600))
    else:
        E.log("[full] 모든 restart 이미 완료 -> 집계만 수행.")

    # ---- 집계 + 저장 + 그림 ----
    aggregate(d, an_emp, an_marg, epochs)
    E.log("DONE")


if __name__ == "__main__":
    main()
