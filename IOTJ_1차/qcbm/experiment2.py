#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
QCBM 확장 실험 2: (1) 6 -> 8 큐비트 확장성, (2) 공격 트래픽 이상탐지 첫 검증
---------------------------------------------------------------------------
데이터: CSE-CIC-IDS2018 Thursday-01-03-2018 (Benign + Infilteration 공격)

확장 1 (확장성):
  6q : proto(2) + syn(1) + size(3, 8버킷)              = 6bit(64상태)   [기존 재현]
  8q : proto(2) + syn(1) + psh(1) + size(4, 16버킷)    = 8bit(256상태)
  - Benign으로 학습, 두 경우 모두 16재시작 병렬, 동일 ansatz(L=3)
  - epoch별 손실 + gradient-norm 추적 -> 바렌 플래토(기울기 소실) 진단

확장 2 (이상탐지):
  - Benign(train)으로 학습한 QCBM의 분포 p(x)를 고정
  - Benign(test)와 공격(Infilteration) 샘플의 q(x)=p(x[state]) 평가
  - 이상점수 s(x) = -log p(x). 공격이 낮은 확률(=높은 점수)을 받는지,
    ROC AUC / 평균 log-prob 격차로 "신호 유무"만 정직하게 확인.

병렬: 독립 무작위 재시작 16개를 16코어에 1:1 프로세스 병렬(워커당 OMP=1).
"""

import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import json
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.stats import rankdata
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp

DATA = "data/cic_thursday.csv"
OUT = "results2"
LAYERS = 3
EPOCHS = 300
RESTARTS = 16
LR = 0.1
LOG_EVERY = 25

# 두 구성. features: (이름, 비트수). 비트 패킹은 나열 순서대로 MSB->LSB.
SPECS = [
    {"name": "6q", "nq": 6,
     "features": [("proto", 2), ("syn", 1), ("size", 3)]},
    {"name": "8q", "nq": 8,
     "features": [("proto", 2), ("syn", 1), ("psh", 1), ("size", 4)]},
]


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


# ---------------------------------------------------------------------------
# 데이터 이산화
# ---------------------------------------------------------------------------
def proto_code(p):
    if p == 6:  return 0   # TCP
    if p == 17: return 1   # UDP
    if p == 0:  return 2
    return 3


def load_frames():
    cols = ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean", "Label"]
    log("CSV 로드: %s" % cols)
    df = pd.read_csv(DATA, usecols=cols, low_memory=False)
    for c in ["Protocol", "SYN Flag Cnt", "PSH Flag Cnt", "Pkt Len Mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    lab = df["Label"].astype(str).str.strip().str.lower()
    benign = df[lab == "benign"].reset_index(drop=True)
    attack = df[lab == "infilteration"].reset_index(drop=True)
    log("Benign=%d  Infilteration(공격)=%d" % (len(benign), len(attack)))
    return benign, attack


def size_edges_from_benign(benign, nbuckets):
    """Benign에 맞춰 분위수 경계 생성(이산화기를 train에 적합). 양쪽으로 변환 공유."""
    vals = benign["Pkt Len Mean"].astype(float).to_numpy()
    qs = np.linspace(0, 1, nbuckets + 1)[1:-1]
    inner = np.unique(np.quantile(vals, qs))
    return inner  # np.digitize용 inner edges


def discretize(df, spec, size_edges):
    parts = []
    for name, bits in spec["features"]:
        if name == "proto":
            v = df["Protocol"].astype(int).map(proto_code).to_numpy()
        elif name == "syn":
            v = (df["SYN Flag Cnt"].astype(float) > 0).astype(int).to_numpy()
        elif name == "psh":
            v = (df["PSH Flag Cnt"].astype(float) > 0).astype(int).to_numpy()
        elif name == "size":
            v = np.digitize(df["Pkt Len Mean"].astype(float).to_numpy(), size_edges)
            v = np.clip(v, 0, (1 << bits) - 1)
        else:
            raise ValueError(name)
        parts.append((v.astype(int), bits))
    state = np.zeros(len(df), dtype=int)
    for v, bits in parts:           # MSB -> LSB
        state = (state << bits) | v
    return state


def empirical_dist(states, nstates):
    c = np.bincount(states, minlength=nstates).astype(float)
    return c / c.sum()


# ---------------------------------------------------------------------------
# MMD 커널
# ---------------------------------------------------------------------------
def build_kernel(nq, sigmas=(0.5, 1.0, 2.0, 4.0)):
    n = 1 << nq
    bits = np.array([[(s >> b) & 1 for b in range(nq)] for s in range(n)], float)
    d2 = np.sum((bits[:, None, :] - bits[None, :, :]) ** 2, axis=-1)
    K = np.zeros((n, n))
    for sg in sigmas:
        K += np.exp(-d2 / (2.0 * sg * sg))
    return K / len(sigmas)


# ---------------------------------------------------------------------------
# 회로
# ---------------------------------------------------------------------------
def make_circuit(nq, n_layers):
    dev = qml.device("lightning.qubit", wires=nq)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights, wires=range(nq))
        return qml.probs(wires=range(nq))
    return circuit


def wshape(nq, n_layers):
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=nq)


def kl_div(q, p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    m = q > 0
    return float(np.sum(q[m] * np.log(q[m] / p[m])))


def tv_dist(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


# ---------------------------------------------------------------------------
# 워커: 재시작 1개 학습 (수동 Adam -> epoch별 gradient norm 확보)
# ---------------------------------------------------------------------------
_G = {}


def _init(q, K, nq, n_layers, epochs, lr, log_every, tag):
    _G.update(q=q, K=K, nq=nq, n_layers=n_layers, epochs=epochs,
              lr=lr, log_every=log_every, tag=tag)


def train_one(seed):
    q, K = _G["q"], _G["K"]
    nq, n_layers = _G["nq"], _G["n_layers"]
    epochs, lr, log_every, tag = _G["epochs"], _G["lr"], _G["log_every"], _G["tag"]

    circuit = make_circuit(nq, n_layers)
    shp = wshape(nq, n_layers)
    rng = np.random.default_rng(2000 + seed)
    w = pnp.array(rng.uniform(0, 2 * np.pi, size=shp), requires_grad=True)

    qK = pnp.array(K @ q)
    qKq = float(q @ K @ q)

    def cost(ww):
        p = circuit(ww)
        return p @ (K @ p) - 2.0 * (p @ qK) + qKq

    gfun = qml.grad(cost)
    b1, b2, eps = 0.9, 0.999, 1e-8
    m = np.zeros(shp); v = np.zeros(shp)
    loss_hist = []; gnorm_hist = []
    t0 = time.time()
    for ep in range(epochs):
        g = np.array(gfun(w), dtype=float)
        loss = float(cost(w))
        gnorm = float(np.linalg.norm(g))
        loss_hist.append(loss); gnorm_hist.append(gnorm)
        m = b1 * m + (1 - b1) * g
        v = b2 * v + (1 - b2) * (g * g)
        mhat = m / (1 - b1 ** (ep + 1))
        vhat = v / (1 - b2 ** (ep + 1))
        w = pnp.array(np.array(w) - lr * mhat / (np.sqrt(vhat) + eps),
                      requires_grad=True)
        if ep % log_every == 0 or ep == epochs - 1:
            log("  [%s seed %2d] ep %4d/%d  MMD^2=%.5e  |grad|=%.3e  (%.1fs)"
                % (tag, seed, ep, epochs, loss, gnorm, time.time() - t0))

    p_final = np.array(circuit(w), dtype=float)
    p_final = np.clip(p_final, 0, None); p_final /= p_final.sum()
    res = {
        "seed": int(seed),
        "loss_hist": [float(x) for x in loss_hist],
        "gnorm_hist": [float(x) for x in gnorm_hist],
        "final_mmd2": float(loss_hist[-1]),
        "kl_q_p": kl_div(q, p_final),
        "tv": tv_dist(p_final, q),
        "p_final": p_final.tolist(),
        "weights": np.array(w).tolist(),
        "elapsed": time.time() - t0,
    }
    log("  [%s seed %2d] 완료 MMD^2=%.4e TV=%.4f |grad|_fin=%.3e (%.1fs)"
        % (tag, seed, res["final_mmd2"], res["tv"], gnorm_hist[-1], res["elapsed"]))
    return res


def run_config(spec, q_train, size_edges):
    nq = spec["nq"]; tag = spec["name"]
    K = build_kernel(nq)
    nparams = int(np.prod(wshape(nq, LAYERS)))
    log("-" * 70)
    log("[%s] 학습 시작: %d큐비트, %d상태, params=%d, 재시작=%d (16코어 병렬)"
        % (tag, nq, 1 << nq, nparams, RESTARTS))
    t0 = time.time()
    with Pool(processes=min(RESTARTS, 16), initializer=_init,
              initargs=(q_train, K, nq, LAYERS, EPOCHS, LR, LOG_EVERY, tag)) as pool:
        results = pool.map(train_one, list(range(RESTARTS)))
    wall = time.time() - t0
    results.sort(key=lambda r: r["final_mmd2"])
    best = results[0]
    mmds = [r["final_mmd2"] for r in results]
    # 수렴 epoch: best의 손실이 (최종*1.1) 이하로 처음 내려간 지점
    lh = np.array(best["loss_hist"])
    thr = lh[-1] * 1.1
    conv_ep = int(np.argmax(lh <= thr)) if np.any(lh <= thr) else EPOCHS
    log("[%s] 완료: best MMD^2=%.4e TV=%.4f KL=%.4f | 중앙값MMD^2=%.4e | "
        "수렴~%d ep | wall=%.1fs"
        % (tag, best["final_mmd2"], best["tv"], best["kl_q_p"],
           float(np.median(mmds)), conv_ep, wall))
    g0 = np.median([r["gnorm_hist"][0] for r in results])
    gf = np.median([r["gnorm_hist"][-1] for r in results])
    log("[%s] gradient-norm(중앙값): 초기=%.3e -> 최종=%.3e (바렌플래토 진단)"
        % (tag, g0, gf))
    return {"spec": spec, "results": results, "best": best, "wall": wall,
            "median_mmd2": float(np.median(mmds)), "worst_mmd2": float(np.max(mmds)),
            "conv_ep": conv_ep, "gnorm0": float(g0), "gnormf": float(gf),
            "nparams": nparams}


# ---------------------------------------------------------------------------
# 이상탐지 평가
# ---------------------------------------------------------------------------
def auc_score(scores, labels):
    """labels: 1=공격(positive). Mann-Whitney U 기반 AUC."""
    r = rankdata(scores)
    n1 = labels.sum(); n0 = len(labels) - n1
    return float((r[labels == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def roc_curve(scores, labels, npts=200):
    thr = np.quantile(scores, np.linspace(0, 1, npts))
    tpr = []; fpr = []
    pos = labels == 1; neg = labels == 0
    P = pos.sum(); N = neg.sum()
    for t in thr:
        pred = scores >= t
        tpr.append((pred & pos).sum() / P)
        fpr.append((pred & neg).sum() / N)
    return np.array(fpr), np.array(tpr)


def anomaly_eval(cfg_out, benign_test_states, attack_states, tag):
    p = np.array(cfg_out["best"]["p_final"])
    eps = 1e-12
    s_ben = -np.log(p[benign_test_states] + eps)
    s_atk = -np.log(p[attack_states] + eps)
    scores = np.concatenate([s_ben, s_atk])
    labels = np.concatenate([np.zeros(len(s_ben), int), np.ones(len(s_atk), int)])
    auc = auc_score(scores, labels)
    # 평균 확률/로그확률
    mean_p_ben = float(np.mean(p[benign_test_states]))
    mean_p_atk = float(np.mean(p[attack_states]))
    mean_lp_ben = float(np.mean(np.log(p[benign_test_states] + eps)))
    mean_lp_atk = float(np.mean(np.log(p[attack_states] + eps)))
    # 최적 임계값(Youden J)에서 분리도
    fpr, tpr = roc_curve(scores, labels)
    j = tpr - fpr
    best_j = float(np.max(j))
    log("[%s 이상탐지] AUC=%.4f | mean p(x): benign=%.3e vs attack=%.3e (배율 %.2fx)"
        % (tag, auc, mean_p_ben, mean_p_atk, mean_p_ben / max(mean_p_atk, 1e-30)))
    log("[%s 이상탐지] mean log p(x): benign=%.3f vs attack=%.3f (격차 %.3f) | "
        "Youden J=%.3f" % (tag, mean_lp_ben, mean_lp_atk,
                           mean_lp_ben - mean_lp_atk, best_j))
    return {"auc": auc, "mean_p_ben": mean_p_ben, "mean_p_atk": mean_p_atk,
            "mean_logp_ben": mean_lp_ben, "mean_logp_atk": mean_lp_atk,
            "youden_j": best_j, "s_ben": s_ben, "s_atk": s_atk,
            "fpr": fpr.tolist(), "tpr": tpr.tolist()}


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------
def plot_training(cfg_out, tag):
    res = cfg_out["results"]; best = cfg_out["best"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for r in res:
        ax[0].plot(r["loss_hist"], color="grey", alpha=0.3, lw=0.7)
    ax[0].plot(best["loss_hist"], color="crimson", lw=2, label="best")
    ax[0].set_yscale("log"); ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MMD^2")
    ax[0].set_title("%s loss (16 restarts)" % tag); ax[0].legend(); ax[0].grid(alpha=.3)
    for r in res:
        ax[1].plot(r["gnorm_hist"], color="grey", alpha=0.3, lw=0.7)
    ax[1].plot(best["gnorm_hist"], color="navy", lw=2, label="best")
    ax[1].set_yscale("log"); ax[1].set_xlabel("epoch"); ax[1].set_ylabel("|grad|")
    ax[1].set_title("%s gradient norm (barren-plateau check)" % tag)
    ax[1].legend(); ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "train_%s.png" % tag), dpi=130); plt.close()


def plot_dist(q, best, tag):
    p = np.array(best["p_final"]); n = len(q); x = np.arange(n)
    plt.figure(figsize=(min(14, 4 + n / 12), 4.5))
    plt.bar(x - 0.2, q, 0.4, label="real Benign", color="steelblue")
    plt.bar(x + 0.2, p, 0.4, label="QCBM", color="darkorange")
    plt.xlabel("state"); plt.ylabel("prob")
    plt.title("%s joint dist: real vs QCBM (MMD^2=%.2e, TV=%.3f)"
              % (tag, best["final_mmd2"], best["tv"]))
    plt.legend(); plt.grid(alpha=.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(OUT, "dist_%s.png" % tag), dpi=130); plt.close()


def plot_anomaly(an, tag):
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    bins = np.linspace(0, max(an["s_ben"].max(), an["s_atk"].max()) + 1e-6, 40)
    ax[0].hist(an["s_ben"], bins=bins, density=True, alpha=0.6,
               label="Benign", color="steelblue")
    ax[0].hist(an["s_atk"], bins=bins, density=True, alpha=0.6,
               label="Attack(Infil.)", color="crimson")
    ax[0].set_xlabel("anomaly score  s(x) = -log p(x)")
    ax[0].set_ylabel("density")
    ax[0].set_title("%s anomaly score: Benign vs Attack" % tag)
    ax[0].legend(); ax[0].grid(alpha=.3)
    ax[1].plot(an["fpr"], an["tpr"], color="darkgreen", lw=2)
    ax[1].plot([0, 1], [0, 1], "k--", alpha=.5)
    ax[1].set_xlabel("FPR"); ax[1].set_ylabel("TPR")
    ax[1].set_title("%s ROC  (AUC=%.3f)" % (tag, an["auc"]))
    ax[1].grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT, "anomaly_%s.png" % tag), dpi=130); plt.close()


# ---------------------------------------------------------------------------
def main():
    t_start = time.time()
    os.makedirs(OUT, exist_ok=True)
    log("=" * 70)
    log("QCBM 확장 실험: 6 vs 8 큐비트 + 공격 이상탐지")
    log("=" * 70)

    benign, attack = load_frames()
    # benign train/test 분할 (이산화기/학습은 train, 이상탐지 평가는 test)
    rng = np.random.default_rng(7)
    idx = rng.permutation(len(benign))
    cut = int(0.7 * len(benign))
    ben_tr = benign.iloc[idx[:cut]].reset_index(drop=True)
    ben_te = benign.iloc[idx[cut:]].reset_index(drop=True)
    log("Benign train=%d / test=%d, attack=%d" % (len(ben_tr), len(ben_te), len(attack)))

    summary = {"configs": {}, "anomaly": {}}
    config_outs = {}

    for spec in SPECS:
        tag = spec["name"]
        nbuckets = 1 << dict(spec["features"])["size"]
        edges = size_edges_from_benign(ben_tr, nbuckets)
        st_tr = discretize(ben_tr, spec, edges)
        q_train = empirical_dist(st_tr, 1 << spec["nq"])
        occ = int((q_train > 0).sum())
        log("[%s] 이산화: %d상태 중 %d개 점유 (size %d버킷)"
            % (tag, 1 << spec["nq"], occ, nbuckets))

        cfg_out = run_config(spec, q_train, edges)
        config_outs[tag] = (cfg_out, edges)
        plot_training(cfg_out, tag)
        plot_dist(q_train, cfg_out["best"], tag)

        summary["configs"][tag] = {
            "nq": spec["nq"], "nstates": 1 << spec["nq"], "occupied": occ,
            "nparams": cfg_out["nparams"], "best_mmd2": cfg_out["best"]["final_mmd2"],
            "best_tv": cfg_out["best"]["tv"], "best_kl": cfg_out["best"]["kl_q_p"],
            "median_mmd2": cfg_out["median_mmd2"], "conv_ep": cfg_out["conv_ep"],
            "wall_sec": cfg_out["wall"], "gnorm0": cfg_out["gnorm0"],
            "gnormf": cfg_out["gnormf"],
        }

        # 이상탐지 (각 구성)
        st_te = discretize(ben_te, spec, edges)
        st_atk = discretize(attack, spec, edges)
        an = anomaly_eval(cfg_out, st_te, st_atk, tag)
        plot_anomaly(an, tag)
        summary["anomaly"][tag] = {k: an[k] for k in
                                   ["auc", "mean_p_ben", "mean_p_atk",
                                    "mean_logp_ben", "mean_logp_atk", "youden_j"]}

    # 비교 표 로그
    log("=" * 70)
    log("[비교] 6 vs 8 큐비트 확장성")
    log("  %-4s %-7s %-7s %-9s %-9s %-7s %-8s %-9s" %
        ("conf", "states", "params", "bestMMD2", "bestTV", "conv_ep", "wall(s)", "grad0->f"))
    for tag in ["6q", "8q"]:
        c = summary["configs"][tag]
        log("  %-4s %-7d %-7d %-9.2e %-9.4f %-7d %-8.1f %.1e->%.1e" %
            (tag, c["nstates"], c["nparams"], c["best_mmd2"], c["best_tv"],
             c["conv_ep"], c["wall_sec"], c["gnorm0"], c["gnormf"]))
    log("=" * 70)
    log("[비교] 이상탐지 신호 (공격 vs 정상)")
    for tag in ["6q", "8q"]:
        a = summary["anomaly"][tag]
        log("  %-4s AUC=%.4f | mean logp benign=%.3f vs attack=%.3f (격차 %.3f)" %
            (tag, a["auc"], a["mean_logp_ben"], a["mean_logp_atk"],
             a["mean_logp_ben"] - a["mean_logp_atk"]))

    summary["total_runtime_sec"] = time.time() - t_start
    with open(os.path.join(OUT, "summary2.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 최종 결론
    c6, c8 = summary["configs"]["6q"], summary["configs"]["8q"]
    a8 = summary["anomaly"]["8q"]
    scaled = "예 (수렴 유지)" if c8["best_tv"] < 0.15 and c8["gnormf"] > 1e-3 else \
             "부분적 (수렴 약화/기울기 감소 관찰)"
    sig = ("뚜렷한 신호" if a8["auc"] > 0.65 else
           "약한 신호" if a8["auc"] > 0.55 else
           "신호 거의 없음")
    log("=" * 70)
    log("결론(1) 8큐비트 확장: %s | 8q TV=%.4f (6q TV=%.4f), wall %.0fs(6q %.0fs)"
        % (scaled, c8["best_tv"], c6["best_tv"], c8["wall_sec"], c6["wall_sec"]))
    log("결론(2) 이상탐지: 8q AUC=%.4f -> %s "
        "(공격 평균 logp가 정상보다 %.3f 낮음)"
        % (a8["auc"], sig, a8["mean_logp_ben"] - a8["mean_logp_atk"]))
    log("주의: Infiltration은 은밀한 공격이라 약한 분리도 정상적 결과(정직 보고).")
    log("총 실행 시간: %.1fs (%.2f분)" %
        (summary["total_runtime_sec"], summary["total_runtime_sec"] / 60))
    log("결과물: %s/ (train_*.png, dist_*.png, anomaly_*.png, summary2.json)" % OUT)
    log("=" * 70)
    log("DONE")


if __name__ == "__main__":
    main()
