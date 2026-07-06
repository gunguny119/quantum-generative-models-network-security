#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
6-qubit QCBM (Quantum Circuit Born Machine) 약식 검증 실험
-----------------------------------------------------------
데이터:  CSE-CIC-IDS2018 (Benign 트래픽) 이산 결합분포
  - Protocol            -> 2 bit  (TCP/UDP/0/기타)
  - SYN Flag Cnt > 0    -> 1 bit
  - Pkt Len Mean(8버킷) -> 3 bit
  => 6 bit = 6 qubit, 상태공간 2^6 = 64

모델:  lightning.qubit, StronglyEntanglingLayers (회전층 + CNOT 얽힘층)
손실:  MMD^2 (multi-bandwidth Gaussian kernel), 전체 확률벡터로 정확 계산
학습:  Adam, 다중 무작위 재시작(restart)을 16코어에 병렬 분배

병렬화 메모:
  6큐비트 상태벡터(64차원)는 너무 작아 OpenMP 스레드 병렬의 이득이 거의 없다.
  대신 '독립 재시작 N개'를 프로세스 병렬로 16코어에 꽉 채워 돌린다
  (워커당 OMP_NUM_THREADS=1 -> 오버서브스크립션 없이 16배 처리량 + 더 신뢰도 높은 결과).
"""

import os
# 프로세스 병렬이므로 워커는 단일 스레드. (numpy/lightning import 이전에 설정)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import time
import json
import argparse
import urllib.request
from datetime import datetime
from multiprocessing import Pool

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import pennylane as qml
from pennylane import numpy as pnp

# ----------------------------------------------------------------------------
N_QUBITS = 6
N_STATES = 2 ** N_QUBITS  # 64
PROTO_LABELS = {0: "TCP", 1: "UDP", 2: "proto0", 3: "other"}

DATA_URL = ("https://cse-cic-ids2018.s3.amazonaws.com/"
            "Processed%20Traffic%20Data%20for%20ML%20Algorithms/"
            "Thursday-01-03-2018_TrafficForML_CICFlowMeter.csv")

# 패킷크기(Pkt Len Mean) 고정 버킷 경계 -> 8단계(3bit). 현실적인 비대칭 marginal.
SIZE_EDGES = [0, 40, 80, 160, 320, 640, 1000, 1500, np.inf]


def log(msg):
    print("[%s] %s" % (datetime.now().strftime("%H:%M:%S"), msg), flush=True)


# ----------------------------------------------------------------------------
# 1. 데이터 준비
# ----------------------------------------------------------------------------
def download_if_needed(path):
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        log("CSV 이미 존재: %s (%.1f MB)" % (path, os.path.getsize(path) / 1e6))
        return True
    log("CSV 다운로드 시작 (S3, 무인증): %s" % DATA_URL)
    try:
        t0 = time.time()
        urllib.request.urlretrieve(DATA_URL, path)
        log("다운로드 완료: %.1f MB, %.1fs" %
            (os.path.getsize(path) / 1e6, time.time() - t0))
        return True
    except Exception as e:
        log("다운로드 실패: %r -> 합성 분포로 대체" % e)
        return False


def proto_to_2bit(p):
    if p == 6:   return 0   # TCP
    if p == 17:  return 1   # UDP
    if p == 0:   return 2   # proto 0
    return 3                # 기타


def discretize_from_csv(path):
    """실제 CIC CSV(Benign) -> 6bit 상태 정수 배열."""
    cols = ["Protocol", "SYN Flag Cnt", "Pkt Len Mean", "Label"]
    log("CSV 로드 (필요 컬럼만): %s" % cols)
    df = pd.read_csv(path, usecols=cols, low_memory=False)
    log("전체 행: %d" % len(df))
    # 수치 변환 (헤더 중복/오염 행 방어)
    for c in ["Protocol", "SYN Flag Cnt", "Pkt Len Mean"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=cols)
    benign = df[df["Label"].astype(str).str.strip().str.lower() == "benign"]
    log("Benign 행: %d (%.1f%%)" % (len(benign), 100.0 * len(benign) / max(1, len(df))))

    proto = benign["Protocol"].astype(int).map(proto_to_2bit).to_numpy()
    syn = (benign["SYN Flag Cnt"].astype(float) > 0).astype(int).to_numpy()
    size = np.digitize(benign["Pkt Len Mean"].astype(float).to_numpy(),
                       SIZE_EDGES[1:-1])  # 0..7
    size = np.clip(size, 0, 7)
    states = proto * 16 + syn * 8 + size
    return states.astype(int)


def synthetic_states(n=200_000, seed=0):
    """CSV가 없을 때: 상관(correlation)이 있는 합성 이산 결합분포."""
    rng = np.random.default_rng(seed)
    # protocol: TCP 우세
    proto = rng.choice([0, 1, 2, 3], size=n, p=[0.6, 0.25, 0.1, 0.05])
    # syn은 protocol에 상관 (TCP에서 SYN 빈도 높음)
    p_syn = np.where(proto == 0, 0.7, 0.2)
    syn = (rng.random(n) < p_syn).astype(int)
    # size는 protocol에 상관 (UDP는 작은 패킷 쪽으로)
    size = np.empty(n, dtype=int)
    for pr in range(4):
        m = proto == pr
        center = {0: 4, 1: 2, 2: 1, 3: 5}[pr]
        s = np.clip(np.round(rng.normal(center, 1.3, m.sum())), 0, 7).astype(int)
        size[m] = s
    return (proto * 16 + syn * 8 + size).astype(int)


def empirical_dist(states):
    counts = np.bincount(states, minlength=N_STATES).astype(float)
    q = counts / counts.sum()
    return q


# ----------------------------------------------------------------------------
# 2. MMD 커널 (multi-bandwidth Gaussian, 비트벡터 거리 기반)
# ----------------------------------------------------------------------------
def build_kernel(sigmas=(0.5, 1.0, 2.0, 4.0)):
    bits = np.array([[(s >> b) & 1 for b in range(N_QUBITS)]
                     for s in range(N_STATES)], dtype=float)
    d2 = np.sum((bits[:, None, :] - bits[None, :, :]) ** 2, axis=-1)  # 제곱 유클리드(=해밍)
    K = np.zeros((N_STATES, N_STATES))
    for sg in sigmas:
        K += np.exp(-d2 / (2.0 * sg * sg))
    return K / len(sigmas)


def mmd2(p, q, K):
    diff = p - q
    return diff @ K @ diff


# ----------------------------------------------------------------------------
# 3. QCBM 회로
# ----------------------------------------------------------------------------
def make_circuit(n_layers):
    dev = qml.device("lightning.qubit", wires=N_QUBITS)

    @qml.qnode(dev, diff_method="parameter-shift")
    def circuit(weights):
        qml.StronglyEntanglingLayers(weights, wires=range(N_QUBITS))
        return qml.probs(wires=range(N_QUBITS))

    return circuit


def weight_shape(n_layers):
    return qml.StronglyEntanglingLayers.shape(n_layers=n_layers, n_wires=N_QUBITS)


# ----------------------------------------------------------------------------
# 평가 지표
# ----------------------------------------------------------------------------
def kl_div(q, p, eps=1e-12):
    p = np.clip(p, eps, 1.0)
    mask = q > 0
    return float(np.sum(q[mask] * np.log(q[mask] / p[mask])))


def tv_dist(p, q):
    return float(0.5 * np.sum(np.abs(p - q)))


# ----------------------------------------------------------------------------
# 학습 (워커 1개 = 재시작 1개)
# ----------------------------------------------------------------------------
_Q = None
_K = None
_CFG = None


def _init_worker(q, K, cfg):
    global _Q, _K, _CFG
    _Q, _K, _CFG = q, K, cfg


def train_one(seed):
    q, K, cfg = _Q, _K, _CFG
    epochs = cfg["epochs"]
    n_layers = cfg["layers"]
    lr = cfg["lr"]

    circuit = make_circuit(n_layers)
    shp = weight_shape(n_layers)
    rng = np.random.default_rng(1000 + seed)
    weights = pnp.array(rng.uniform(0, 2 * np.pi, size=shp), requires_grad=True)

    qK = pnp.array(K @ q)            # 상수 벡터
    qKq = float(q @ K @ q)

    def cost(w):
        p = circuit(w)
        return p @ (K @ p) - 2.0 * (p @ qK) + qKq

    opt = qml.AdamOptimizer(stepsize=lr)
    history = []
    t0 = time.time()
    for ep in range(epochs):
        weights, loss = opt.step_and_cost(cost, weights)
        loss = float(loss)
        history.append(loss)
        if ep % cfg["log_every"] == 0 or ep == epochs - 1:
            log("  [seed %2d] epoch %4d/%d  MMD^2 = %.6e  (%.1fs)"
                % (seed, ep, epochs, loss, time.time() - t0))

    p_final = np.array(circuit(weights), dtype=float)
    p_final = np.clip(p_final, 0, None)
    p_final /= p_final.sum()
    result = {
        "seed": int(seed),
        "history": [float(x) for x in history],
        "final_mmd2": float(history[-1]),
        "kl_q_p": kl_div(q, p_final),
        "tv": tv_dist(p_final, q),
        "p_final": p_final.tolist(),
        "weights": np.array(weights).tolist(),
        "elapsed": time.time() - t0,
    }
    log("  [seed %2d] 완료  MMD^2=%.4e  KL=%.4f  TV=%.4f  (%.1fs)"
        % (seed, result["final_mmd2"], result["kl_q_p"], result["tv"],
           result["elapsed"]))
    return result


# ----------------------------------------------------------------------------
# 시각화
# ----------------------------------------------------------------------------
def decode_marginals(dist):
    proto = np.zeros(4); syn = np.zeros(2); size = np.zeros(8)
    for s in range(N_STATES):
        proto[s // 16] += dist[s]
        syn[(s // 8) % 2] += dist[s]
        size[s % 8] += dist[s]
    return proto, syn, size


def make_plots(q, results, best, outdir):
    # (1) 학습 곡선
    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(r["history"], color="grey", alpha=0.3, lw=0.8)
    plt.plot(best["history"], color="crimson", lw=2.0,
             label="best (seed %d)" % best["seed"])
    plt.yscale("log")
    plt.xlabel("epoch"); plt.ylabel("MMD$^2$ (log)")
    plt.title("QCBM 학습 곡선 (%d 재시작)" % len(results))
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "loss_curve.png"), dpi=130)
    plt.close()

    p = np.array(best["p_final"])
    # (2) 64-상태 결합분포 비교
    plt.figure(figsize=(12, 5))
    x = np.arange(N_STATES)
    plt.bar(x - 0.2, q, width=0.4, label="실제 Benign", color="steelblue")
    plt.bar(x + 0.2, p, width=0.4, label="QCBM 생성", color="darkorange")
    plt.xlabel("상태 (6bit = proto·16 + syn·8 + size)")
    plt.ylabel("확률")
    plt.title("결합분포 비교: 실제 vs QCBM  (MMD$^2$=%.3e, TV=%.3f)"
              % (best["final_mmd2"], best["tv"]))
    plt.legend(); plt.grid(alpha=0.3, axis="y"); plt.tight_layout()
    plt.savefig(os.path.join(outdir, "dist_compare.png"), dpi=130)
    plt.close()

    # (3) marginal 비교
    qp, qs, qz = decode_marginals(q)
    pp, ps, pz = decode_marginals(p)
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].bar(np.arange(4) - 0.2, qp, 0.4, label="실제")
    ax[0].bar(np.arange(4) + 0.2, pp, 0.4, label="QCBM")
    ax[0].set_xticks(range(4)); ax[0].set_xticklabels(list(PROTO_LABELS.values()))
    ax[0].set_title("Protocol marginal"); ax[0].legend()
    ax[1].bar(np.arange(2) - 0.2, qs, 0.4, label="실제")
    ax[1].bar(np.arange(2) + 0.2, ps, 0.4, label="QCBM")
    ax[1].set_xticks([0, 1]); ax[1].set_xticklabels(["SYN=0", "SYN>0"])
    ax[1].set_title("SYN flag marginal"); ax[1].legend()
    ax[2].bar(np.arange(8) - 0.2, qz, 0.4, label="실제")
    ax[2].bar(np.arange(8) + 0.2, pz, 0.4, label="QCBM")
    ax[2].set_title("Pkt size bucket marginal"); ax[2].set_xlabel("bucket 0..7")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "marginals.png"), dpi=130)
    plt.close()
    log("그림 저장: loss_curve.png, dist_compare.png, marginals.png")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--restarts", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--log-every", type=int, default=25)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--data", default="data/cic_thursday.csv")
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(args.outdir, exist_ok=True)

    log("=" * 70)
    log("6-qubit QCBM 약식 검증 실험 시작")
    log("재시작=%d  epochs=%d  layers=%d  lr=%g  (코어 16개 프로세스 병렬)"
        % (args.restarts, args.epochs, args.layers, args.lr))
    log("=" * 70)

    # --- 데이터 ---
    have = download_if_needed(args.data)
    if have:
        try:
            states = discretize_from_csv(args.data)
            source = "CSE-CIC-IDS2018 (Benign, 실제)"
        except Exception as e:
            log("CSV 파싱 실패: %r -> 합성 분포 사용" % e)
            states = synthetic_states(); source = "합성 이산 결합분포"
    else:
        states = synthetic_states(); source = "합성 이산 결합분포"

    q = empirical_dist(states)
    log("데이터 출처: %s | 표본 %d개 | 비어있지 않은 상태 %d/64"
        % (source, len(states), int((q > 0).sum())))
    qp, qs, qz = decode_marginals(q)
    log("Protocol marginal: %s" % np.round(qp, 3).tolist())
    log("SYN marginal:      %s" % np.round(qs, 3).tolist())
    log("Size marginal:     %s" % np.round(qz, 3).tolist())

    K = build_kernel()
    cfg = {"epochs": args.epochs, "layers": args.layers, "lr": args.lr,
           "log_every": args.log_every}

    n_params = int(np.prod(weight_shape(args.layers)))
    log("QCBM 파라미터 수: %d (StronglyEntanglingLayers L=%d)" % (n_params, args.layers))
    log("학습 시작: %d개 재시작을 %d 프로세스로 병렬 실행..." %
        (args.restarts, min(args.restarts, 16)))

    with Pool(processes=min(args.restarts, 16),
              initializer=_init_worker, initargs=(q, K, cfg)) as pool:
        results = pool.map(train_one, list(range(args.restarts)))

    results.sort(key=lambda r: r["final_mmd2"])
    best = results[0]
    mmds = [r["final_mmd2"] for r in results]

    log("=" * 70)
    log("학습 완료. 재시작 %d개 결과 요약" % len(results))
    log("  best MMD^2 = %.6e (seed %d)" % (best["final_mmd2"], best["seed"]))
    log("  MMD^2  중앙값=%.4e  최악=%.4e" %
        (float(np.median(mmds)), float(np.max(mmds))))
    log("  best KL(q||p) = %.5f   TV = %.5f" % (best["kl_q_p"], best["tv"]))

    make_plots(q, results, best, args.outdir)

    # 저장
    summary = {
        "source": source,
        "n_samples": int(len(states)),
        "n_qubits": N_QUBITS,
        "layers": args.layers,
        "n_params": n_params,
        "restarts": args.restarts,
        "epochs": args.epochs,
        "best_seed": best["seed"],
        "best_mmd2": best["final_mmd2"],
        "best_kl": best["kl_q_p"],
        "best_tv": best["tv"],
        "median_mmd2": float(np.median(mmds)),
        "worst_mmd2": float(np.max(mmds)),
        "all_final_mmd2": mmds,
        "total_runtime_sec": time.time() - t_start,
    }
    with open(os.path.join(args.outdir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    np.save(os.path.join(args.outdir, "best_weights.npy"),
            np.array(best["weights"]))
    np.save(os.path.join(args.outdir, "target_q.npy"), q)
    np.save(os.path.join(args.outdir, "best_p.npy"), np.array(best["p_final"]))

    # 최종 판정
    tv = best["tv"]
    verdict = ("매우 잘 학습됨 (거의 일치)" if tv < 0.03 else
               "잘 학습됨" if tv < 0.07 else
               "어느 정도 학습됨" if tv < 0.15 else
               "부분적으로만 학습됨")
    log("=" * 70)
    log("결론: 6큐비트 QCBM의 이산 결합분포 학습 결과 -> %s" % verdict)
    log("  (전변동거리 TV=%.4f, 즉 두 분포가 약 %.1f%% 일치)"
        % (tv, 100 * (1 - tv)))
    log("  총 실행 시간: %.1fs (%.2f분)" %
        (summary["total_runtime_sec"], summary["total_runtime_sec"] / 60))
    log("  결과물: %s/ (summary.json, *.png, *.npy)" % args.outdir)
    log("=" * 70)
    log("DONE")


if __name__ == "__main__":
    main()
