#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UNSW QCBM(MMD) 본 학습 — restart 단위 체크포인트로 중단 복원 (기존 소스 무수정, import 재사용)
=====================================================================================================
데이터: data/unsw_nb15_training-set.csv + testing-set.csv (benign=Normal/label0, attack=label1 ALL).
공통 10비트 인코딩(교차 진단에서 확정, syn/psh 제외): proto2+size4 + fwd_iat2 + flow_iat2 = 1024상태.
  매핑: proto→proto(tcp=6/udp=17/else=-1), size→smean, fwd_iat→sinpkt, flow_iat→M2=(sinpkt+dinpkt)/2.
  연속값 log1p 후 분위수 이산화, 경계는 UNSW benign train(seed=7, 0.7)에서 생성.

교차 진단 기준선(results2/cross_dataset_diagnosis / flow_iat_sensitivity M2, UNSW/ALL):
  marginal AUC≈0.7027, emp_joint AUC≈0.8705(상한). 학습 전 정합 게이트로 검증(불일치 시 중단).
핵심 질문: QCBM AUC가 marginal(0.703)을 넘어 emp_joint(0.871)에 얼마나 근접하는가. 180p에서 고전 대비
  best/median 우위가 견고한가, min-MMD² 선택이 AUC와 상관있는가(Bot: best운/median미달/corr0.077).

중단 복원(Bot 방식 동일):
  - 각 restart(seed) 완료 즉시 results2/unsw_ckpt/restart_<seed>.npz 원자적 저장.
  - Pool.imap_unordered -> 완료 즉시 저장. 시작 시 완료 seed 건너뜀. EPOCHS는 unsw_ckpt/decision.json 1회 저장.
실행(권장): tmux 안에서
  python3 -u tools/train_qcbm_unsw_resumable.py 2>&1 | tee -a results2/qcbm_unsw_train.log
환경변수(선택): UNSW_RESTARTS, UNSW_NWORKERS, UNSW_EPOCHS(강제), UNSW_CKPT_DIR, UNSW_SMOKE_ONLY=1.
raw MWU 유지(방향반전 금지). 새 dependency 없음. 판정/효율비교는 후속 분석 스크립트.
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

DATA_PATHS = ["data/unsw_nb15_training-set.csv", "data/unsw_nb15_testing-set.csv"]
OUT = "results2"
TAG = "qcbm_unsw"
NQ = 10
NSTATES = 1 << NQ                # 1024
LAYERS = 6                       # params = L*nq*3 = 180
LR = 0.1
LOG_EVERY = 50
SMOKE_EPOCHS = 50
SMOKE_SEED = 9999
SIZE_BUCKETS = 16
IAT_BITS = 2
IAT_BUCKETS = 1 << IAT_BITS
SPEC_PS = {"name": "ps6", "nq": 6, "features": [("proto", 2), ("size", 4)]}  # 6bit (MSB), 플래그 제외

RESTARTS = int(os.environ.get("UNSW_RESTARTS", "16"))
NWORKERS = int(os.environ.get("UNSW_NWORKERS", "8"))
FORCE_EPOCHS = os.environ.get("UNSW_EPOCHS")
CKPT_DIR = os.environ.get("UNSW_CKPT_DIR", os.path.join(OUT, "unsw_ckpt"))
SMOKE_ONLY = os.environ.get("UNSW_SMOKE_ONLY") == "1"
DECISION_JSON = os.path.join(CKPT_DIR, "decision.json")

# UNSW/ALL 기준선(정합 검증 대상) — cross_dataset_diagnosis / flow_iat_sensitivity(M2)
REF_EMP, REF_MARG = 0.8705, 0.7027
TOL = 0.003
H3, H45, H5 = 3 * 3600, 4.5 * 3600, 5 * 3600


# ---------------------------------------------------------------------------
def build_unsw():
    src = ["proto", "smean", "sinpkt", "dinpkt", "attack_cat", "label"]
    E.log("CSV 로드: %s" % DATA_PATHS)
    df = pd.concat([pd.read_csv(p, usecols=src, low_memory=False) for p in DATA_PATHS],
                   ignore_index=True)
    n0 = len(df)
    proto_num = df["proto"].astype(str).str.lower().map({"tcp": 6, "udp": 17}).fillna(-1).astype(int)
    sinpkt = pd.to_numeric(df["sinpkt"], errors="coerce")
    dinpkt = pd.to_numeric(df["dinpkt"], errors="coerce")
    can = pd.DataFrame({
        "Protocol": proto_num,
        "Pkt Len Mean": pd.to_numeric(df["smean"], errors="coerce"),
        "Fwd IAT Mean": sinpkt,
        "Flow IAT Mean": 0.5 * (sinpkt + dinpkt),                 # M2 = (sinpkt+dinpkt)/2
        "label": pd.to_numeric(df["label"], errors="coerce"),
    })
    cols = ["Protocol", "Pkt Len Mean", "Fwd IAT Mean", "Flow IAT Mean", "label"]
    can = can.dropna(subset=cols)
    ben = can[can["label"] == 0].reset_index(drop=True)
    atk = can[can["label"] == 1].reset_index(drop=True)
    E.log("dropna 제거=%d | benign(Normal)=%d  attack(ALL,label1)=%d" % (n0 - len(can), len(ben), len(atk)))
    if len(ben) == 0 or len(atk) == 0:
        E.log("!!! benign/attack 표본 0 -> 중단."); sys.exit(2)

    rng = np.random.default_rng(7)                  # 기존과 동일 seed/0.7
    idx = rng.permutation(len(ben)); cut = int(0.7 * len(ben))
    ben_tr = ben.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = ben.iloc[idx[cut:]].reset_index(drop=True)

    size_edges = E.size_edges_from_benign(ben_tr, SIZE_BUCKETS)
    fwd_edges = TB2.log1p_quantile_edges(ben_tr["Fwd IAT Mean"], IAT_BUCKETS)
    flow_edges = TB2.log1p_quantile_edges(ben_tr["Flow IAT Mean"], IAT_BUCKETS)

    def enc(dd):
        s = E.discretize(dd, SPEC_PS, size_edges)                                  # proto2+size4 (6bit MSB)
        s = TB2.append_bits(s, TB2.iat_to_bits(dd["Fwd IAT Mean"], fwd_edges), IAT_BITS)
        s = TB2.append_bits(s, TB2.iat_to_bits(dd["Flow IAT Mean"], flow_edges), IAT_BITS)
        return s

    st_tr, st_te, st_atk = enc(ben_tr), enc(ben_te), enc(atk)
    q_train = E.empirical_dist(st_tr, NSTATES)
    occ = int(np.bincount(st_tr, minlength=NSTATES).astype(bool).sum())
    E.log("UNSW 인코딩(10bit): 점유 %d/%d (%.2f%%) | benign train=%d test=%d attack=%d"
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
        E.log("!!! UNSW 인코딩 emp/marginal 재계산이 기준값과 불일치 -> 인코딩 재현 오류. 중단.")
        sys.exit(3)
    E.log("[정합검증] OK — UNSW 10bit 인코딩이 교차 진단(M2)과 일치.")
    return auc_emp, auc_marg


# ---------------------------------------------------------------------------
# 체크포인트 (원자적 저장 / 스캔 / 로드) — Bot 방식 동일
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
    os.replace(tmp, ckpt_path(seed))


def scan_done():
    done = {}
    for f in sorted(glob.glob(os.path.join(CKPT_DIR, "restart_*.npz"))):
        try:
            z = np.load(f, allow_pickle=False)
            s = int(z["seed"]); _ = float(z["auc"])
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
    return wall / SMOKE_EPOCHS, wall


def decide_epochs(per_epoch):
    waves = math.ceil(RESTARTS / NWORKERS)
    safety = 1.15
    def est(ep):
        return waves * ep * per_epoch * safety
    for ep, cap in [(300, H3), (200, H45), (150, H5)]:
        if est(ep) <= cap:
            return ep, est(ep), waves
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
    # min-MMD² 선택이 AUC와 상관있는가 (실전 선택 가능성)
    corr_mmd_auc = float(np.corrcoef(mmds, aucs)[0, 1]) if len(aucs) > 1 else float("nan")
    corr_tv_auc = float(np.corrcoef(tvs, aucs)[0, 1]) if len(aucs) > 1 else float("nan")
    best_auc_rank_top = int((aucs >= best_auc).sum())   # min-MMD² 모델의 AUC 순위(상위 몇 번째)

    E.log("=" * 78)
    E.log("[집계] 완료 restart=%d/%d | best(최소MMD²) seed=%d MMD²=%.4e TV=%.4f AUC=%.4f (AUC순위 상위%d/%d)"
          % (len(restarts), RESTARTS, best["seed"], best["final_mmd2"], best["tv"], best_auc,
             best_auc_rank_top, len(restarts)))
    E.log("[AUC 분포] min=%.4f median=%.4f max=%.4f std=%.4f"
          % (aucs.min(), np.median(aucs), aucs.max(), aucs.std()))
    E.log("[비교] marginal=%.4f | emp_joint=%.4f | QCBM best=%.4f (상한근접 %.1f%%)"
          % (REF_MARG, REF_EMP, best_auc, 100 * closeness))
    E.log("[비교] marginal 초과=%d/%d | emp_joint 초과=%d/%d | corr(MMD²,AUC)=%.3f corr(TV,AUC)=%.3f"
          % (n_beat_marg, len(restarts), n_exceed_emp, len(restarts), corr_mmd_auc, corr_tv_auc))

    np.savez(os.path.join(OUT, "%s_restarts.npz" % TAG),
             seeds=np.array(seeds),
             p_finals=np.stack([np.asarray(r["p_final"]) for r in restarts]),
             aucs=aucs, mmd2=mmds, tv=tvs,
             weights=np.stack([np.asarray(r["weights"]) for r in restarts]),
             q_train=np.asarray(d["q_train"]))

    payload = {
        "encoding": "공통 10bit: proto2 size4 + fwd_iat2 + flow_iat2 (1024), flow_iat=M2(sinpkt+dinpkt)/2",
        "data": DATA_PATHS, "attack": "ALL(label==1)", "nq": NQ, "nstates": NSTATES,
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
        "best_auc_rank_top": best_auc_rank_top,
        "auc_distribution": {"min": float(aucs.min()), "median": float(np.median(aucs)),
                             "max": float(aucs.max()), "std": float(aucs.std()),
                             "all": aucs.tolist()},
        "mmd2_distribution": {"min": float(mmds.min()), "median": float(np.median(mmds)),
                              "max": float(mmds.max())},
        "corr_mmd2_auc": corr_mmd_auc, "corr_tv_auc": corr_tv_auc,
        "qcbm_beats_marginal": bool(best_auc > REF_MARG),
        "qcbm_minus_marginal": float(best_auc - REF_MARG),
        "qcbm_over_emp_joint_frac": float(closeness),
        "n_restarts_beat_marginal": n_beat_marg,
        "n_restarts_exceed_emp_joint": n_exceed_emp,
        "per_restart": [{"seed": r["seed"], "auc": r["auc"], "mmd2": r["final_mmd2"],
                         "tv": r["tv"], "gnormf": float(r["gnorm_hist"][-1])} for r in restarts],
        "note": "학습 결과(사실). 180p 효율/견고성은 후속 분석. QCBM 학습됨, 기존 소스 무수정.",
    }
    with open(os.path.join(OUT, "%s.json" % TAG), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: %s/%s.json , %s/%s_restarts.npz" % (OUT, TAG, OUT, TAG))

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
    E.log("UNSW QCBM(MMD) 본 학습 — restart 체크포인트 복원 (NWORKERS=%d, RESTARTS=%d, ckpt=%s)"
          % (NWORKERS, RESTARTS, CKPT_DIR))
    E.log("#" * 78)
    for p in DATA_PATHS:
        if not os.path.exists(p):
            E.log("데이터 없음: %s -> 중단." % p); sys.exit(1)

    d = build_unsw()
    an_emp, an_marg = reproduce_check(d)            # 정합 게이트(불일치 시 종료)

    done = scan_done()
    todo = [s for s in range(RESTARTS) if s not in done]
    E.log("[재개 스캔] 완료=%d %s | 남은 restart=%d %s"
          % (len(done), sorted(done.keys()), len(todo), todo))

    decision = None
    if os.path.exists(DECISION_JSON):
        with open(DECISION_JSON) as f:
            decision = json.load(f)
        if decision.get("aborted"):
            E.log("[decision] 이전에 5h 초과로 중단됨(%s). 재검토 필요 -> 종료." % DECISION_JSON); sys.exit(4)
        E.log("[decision] 기존 결정 사용: EPOCHS=%d (%s)" % (decision["epochs"], DECISION_JSON))

    need_K = (decision is None and FORCE_EPOCHS is None) or (len(todo) > 0) or SMOKE_ONLY
    K = E.build_kernel(NQ) if need_K else None
    if K is not None:
        E.log("MMD 커널 build_kernel(%d) 준비 완료." % NQ)

    if decision is None:
        if FORCE_EPOCHS is not None:
            epochs = int(FORCE_EPOCHS)
            decision = {"epochs": epochs, "source": "UNSW_EPOCHS(force)", "nworkers": NWORKERS,
                        "restarts": RESTARTS}
            E.log("[decision] UNSW_EPOCHS 강제: EPOCHS=%d" % epochs)
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

    aggregate(d, an_emp, an_marg, epochs)
    E.log("DONE")


if __name__ == "__main__":
    main()
