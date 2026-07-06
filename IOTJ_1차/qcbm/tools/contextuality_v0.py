#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
contextuality(k) 측정 v0 — malware 고차가 non-contextual인지(정의 b→a 폴백). 근거: arXiv:2507.11604 strong k-contextuality.

측정 전용(새 QCBM 학습 없음), 결정적(seed 고정), 무설치. 단정 금지·판정은 웹 AI.
개념·k추정 알고리즘 출처=2507.11604(greedy로 "E 재현 최소 은닉상태−1"=k̂). empirical-model 변환·도메인·차수대조=우리 정의.

★ k̂ 조작정의(v0): context×outcome empirical model 행렬 M 의 **추정 비음수 랭크**(rel Frobenius 오차 ≤ tol 되는 최소 성분수 r) − 1.
  = "E 를 재현하는 공유 은닉상태(잠재 혼합성분) 최소 개수 − 1"의 프록시. NMF(diagnose_param_efficiency_b2.nmf_factorize 재사용).
  ※ 진짜 Abramsky-Brandenburger contextuality 아님: **정적 결합분포는 전역 절편(global section)이 항상 존재→자명히 non-contextual**.
    def-b(가족=context)는 모든 변수를 공유하므로 no-signalling이 가족마다 주변분포 달라 **위반=지도적/signalling** 시나리오. → k̂는 잠재복잡도 프록시(§한계).

정의 b(우선, 가족=context): M[g,o]=P(o|family g), outcome=표적 k=10 부분집합 상태(가족 합집합 support). label 사용=지도적.
정의 a(폴백, feature블록=context): 10비트를 A(5)|B(5) 분할. M[a,b]=P(B=b|A=a). 랜덤 5|5 ×5(seed 고정) robustness.
대조: 부분집합 3종 (1)표적 고차 (2)저차 highvar (3)랜덤 × split{0,1,2}. k̂ vs 이미 측정된 ≥3차 비중 vs QCBM 상대성능.

사용: python tools/contextuality_v0.py [--splits 0,1,2] [--k 10] [--tol 0.05] [--smoke]
"""
import os
import sys
import json
import argparse

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

import diagnose_drift_auc as DA           # noqa: E402  _load_tr_te, laplace_emp
import diagnose_order_decomposition as OD  # noqa: E402  maxent_upto, bit_matrix, kl_div
import diagnose_drebin_order as DR         # noqa: E402  emp_dist
import diagnose_param_efficiency_b2 as PE2  # noqa: E402  nmf_factorize (재사용)

OUTDIR = os.path.join("results2", "contextuality")
OUT_JSON = os.path.join(OUTDIR, "ctx_v0.json")
OUT_PNG = os.path.join(OUTDIR, "ctx_v0.png")
TGT_JSON = os.path.join("results2", "drebin_targeted", "drebin_targeted.json")
QCC_JSON = os.path.join("results2", "qcbm_order_comp", "qcbm_order_comp.json")
K = 10
SEED = 4242
N_PART = 5            # 정의 a 랜덤 5|5 분할 수
EPS = 1e-12


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# k̂ = 추정 비음수 랭크 − 1  (rel Frobenius 오차 ≤ tol 되는 최소 성분수)
# ---------------------------------------------------------------------------
def nn_rank(M, tol, rmax=None):
    normM = np.linalg.norm(M)
    if normM == 0:
        return 1, [1.0]
    rmax = rmax or min(M.shape)
    errs = []
    for r in range(1, rmax + 1):
        _, err = PE2.nmf_factorize(M, r, iters=400, n_init=3)
        rel = err / normM
        errs.append(float(rel))
        if rel <= tol:
            return r, errs
    return rmax, errs


def emp_model_rows(Xrows_list, k, c):
    """각 context 의 평활 경험분포(길이 2^k) → 행렬 (n_ctx × 2^k), 이후 합집합 support 로 축소."""
    rows = []
    for Xc in Xrows_list:
        if len(Xc) == 0:
            rows.append(np.full(1 << k, 1.0 / (1 << k)))
            continue
        pe, _ = DR.emp_dist(Xc)
        rows.append(DA.laplace_emp(pe, len(Xc), k, c=c))
    M = np.array(rows)                                   # (n_ctx, 2^k)
    supp = np.where(M.sum(0) > (M.shape[0] * c * 1.5))[0]  # 관측 support(평활 바닥 초과)
    if len(supp) < 2:
        supp = np.argsort(-M.sum(0))[:max(2, min(64, M.shape[1]))]
    Ms = M[:, supp]
    Ms = Ms / Ms.sum(1, keepdims=True)
    return Ms, supp


# ---------------------------------------------------------------------------
# 정의 b: 가족=context
# ---------------------------------------------------------------------------
def def_b(Xtr, ytr, idx, k, c, tol):
    fams = sorted(set(ytr.tolist()))
    Xrows = [Xtr[ytr == g][:, idx].astype(np.int64) for g in fams]
    M, supp = emp_model_rows(Xrows, k, c)
    r, errs = nn_rank(M, tol)
    # no-signalling(주변 일치) 위반: 가족 간 per-bit 주변 L1 (지도적/signalling 정도)
    margs = np.array([Xc.mean(0) for Xc in Xrows if len(Xc) > 0])
    sig = float(np.mean([np.abs(margs[i] - margs[j]).sum()
                         for i in range(len(margs)) for j in range(i + 1, len(margs))]))
    return dict(khat=int(r - 1), nn_rank=int(r), n_ctx=len(fams), support=int(M.shape[1]),
                rel_errs=errs, signalling_L1=sig, note="families=context(supervised); no-signalling 위반=signalling")


# ---------------------------------------------------------------------------
# 정의 a: feature블록=context (랜덤 5|5 ×N)
# ---------------------------------------------------------------------------
def def_a(Xpool, idx, k, c, tol):
    X = Xpool[:, idx].astype(np.int64)
    rng = np.random.default_rng(SEED)
    khats = []; ranks = []; details = []
    half = k // 2
    for pi in range(N_PART):
        perm = rng.permutation(k)
        A = sorted(int(q) for q in perm[:half]); B = sorted(int(q) for q in perm[half:])
        codeA = np.zeros(len(X), dtype=np.int64); codeB = np.zeros(len(X), dtype=np.int64)
        for j, q in enumerate(A):
            codeA += (X[:, q] << j)
        for j, q in enumerate(B):
            codeB += (X[:, q] << j)
        nA, nB = 1 << len(A), 1 << len(B)
        M = np.zeros((nA, nB))
        for a, b in zip(codeA, codeB):
            M[a, b] += 1
        M += c                                  # Laplace
        M = M / M.sum(1, keepdims=True)          # P(B|A=a)
        r, errs = nn_rank(M, tol)
        khats.append(r - 1); ranks.append(r)
        details.append(dict(A=A, khat=int(r - 1), nn_rank=int(r)))
    return dict(khat_mean=float(np.mean(khats)), khat_min=int(min(khats)), khat_max=int(max(khats)),
                per_partition=details,
                note="feature block=context; 단일 결합분포→전역절편 존재=자명 non-contextual(§한계)")


# ---------------------------------------------------------------------------
def ge3_share(Xpool, idx, k, bm):
    pe, _ = DR.emp_dist(Xpool[:, idx].astype(np.int64))
    q1, _ = OD.maxent_upto(pe, k, bm, 1)
    q2, _ = OD.maxent_upto(pe, k, bm, 2)
    KL1 = OD.kl_div(pe, q1, EPS); KL2 = OD.kl_div(pe, q2, EPS)
    return float(KL2 / KL1) if KL1 > 0 else None


def load_subsets(split, k):
    d = json.load(open(TGT_JSON))
    tgt = {(int(r["split"]), int(kk)): r["subset"] for kk, recs in d["targeted"].items() for r in recs}
    hv = {(int(r["split"]), int(kk)): r["subset"] for kk, recs in d["per_family"].items() for r in recs}
    return tgt[(split, k)], hv[(split, k)]


def random_subset(Xpool, k, split):
    mean = Xpool.mean(0); nonc = np.where((mean > 0) & (mean < 1))[0]
    rng = np.random.default_rng(SEED + 1 + split)
    return sorted(int(i) for i in rng.choice(nonc, size=k, replace=False))


def load_qcbm_gap():
    try:
        d = json.load(open(QCC_JSON))
        g = {}
        for r in d["results"]:
            if r["target_version"] != "A":
                continue
            key = ("highord_targeted" if r["subtype"] == "highord_targeted" else "loword_highvar", r["split"])
            g[key] = r["gap_QCBM_M2"]
        return g
    except Exception:
        return {}


def run(splits, k, tol, do_smoke=False):
    os.makedirs(OUTDIR, exist_ok=True)
    bm = OD.bit_matrix(k)
    c = 1.0 / (1 << k)
    qgap = load_qcbm_gap()
    results = {}
    for split in splits:
        Xtr, ytr, Xte, yte = DA._load_tr_te(split)
        Xpool = np.vstack([Xtr, Xte])
        tgt, hv = load_subsets(split, k)
        rnd = random_subset(Xpool, k, split)
        subsets = {"targeted_highord": tgt, "highvar_loword": hv, "random": rnd}
        log("\n=== split %d (k=%d, tol=%.2f) ===" % (split, k, tol))
        sres = {}
        for name, idx in subsets.items():
            b = def_b(Xtr, ytr, idx, k, c, tol)
            a = def_a(Xpool, idx, k, c, tol)
            g3 = ge3_share(Xpool, idx, k, bm)
            qtype = "highord_targeted" if name == "targeted_highord" else ("loword_highvar" if name == "highvar_loword" else None)
            gap = qgap.get((qtype, split)) if qtype else None
            sres[name] = dict(subset=[int(i) for i in idx], ge3_share=g3,
                              def_b=b, def_a=a, qcbm_gap_M2=gap)
            log("  %-17s ≥3차=%.2f | def-b k̂=%d(rank%d,sig=%.2f) | def-a k̂=%.1f[%d-%d] | QCBM gapM2=%s"
                % (name, g3, b["khat"], b["nn_rank"], b["signalling_L1"],
                   a["khat_mean"], a["khat_min"], a["khat_max"],
                   ("%+.3f" % gap) if gap is not None else "N/A"))
        results[str(split)] = sres
        if do_smoke:
            break

    payload = dict(
        note=("contextuality(k) v0. k̂=context×outcome empirical model 의 추정 비음수랭크−1 (E 재현 최소 은닉상태 프록시, "
              "NMF greedy). 정의 b(가족=context,지도적,signalling)·a(feature블록=context,단일결합→자명 non-contextual). "
              "부분집합 3종(표적고차/저차/랜덤)×split. ★진짜 AB-contextuality 아님(정적결합=전역절편 존재)—잠재복잡도 프록시. 단정 금지."),
        params=dict(splits=splits, k=k, tol=tol, laplace_c=c, n_partitions=N_PART, seed=SEED,
                    khat_def="NMF non-negative rank (rel Frob err<=tol) minus 1"),
        results=results)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if not do_smoke:
        try:
            plot(results, splits)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def plot(results, splits):
    names = ["targeted_highord", "highvar_loword", "random"]
    cols = {"targeted_highord": "#C1442E", "highvar_loword": "#3B6EA5", "random": "#9AA0A6"}
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    # 왼쪽: ≥3차 vs def-a k̂ (점 = split×subset)
    for nm in names:
        xs = [results[str(s)][nm]["ge3_share"] for s in splits]
        ys = [results[str(s)][nm]["def_a"]["khat_mean"] for s in splits]
        ax[0].scatter(xs, ys, color=cols[nm], label=nm, s=60)
    ax[0].set_xlabel("≥3rd-order share"); ax[0].set_ylabel("def-a k̂ (NMF rank−1)")
    ax[0].set_title("order vs k̂ (def-a, feature-block context)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # 오른쪽: subset별 k̂ (def-b, def-a) 막대 (split 평균)
    x = np.arange(len(names)); w = 0.35
    kb = [np.mean([results[str(s)][nm]["def_b"]["khat"] for s in splits]) for nm in names]
    ka = [np.mean([results[str(s)][nm]["def_a"]["khat_mean"] for s in splits]) for nm in names]
    ax[1].bar(x - w / 2, kb, w, label="def-b (family)", color="#4C9F70")
    ax[1].bar(x + w / 2, ka, w, label="def-a (feature-block)", color="#E2A13B")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8); ax[1].set_ylabel("k̂ (split 평균)")
    ax[1].set_title("k̂ by subset (higher-order vs low-order vs random)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Contextuality(k) v0: is malware higher-order high-k? (NMF-rank proxy; static→trivially non-contextual)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="contextuality(k) v0 (NMF-rank 프록시). 측정 전용. 단정 금지.")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--tol", type=float, default=0.05)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run([0], args.k, args.tol, do_smoke=True); return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    run(splits, args.k, args.tol)


if __name__ == "__main__":
    main()
