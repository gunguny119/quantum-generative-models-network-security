#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
얽힘 지문 v0 — split1에서 "얽힘 스펙트럼 거리"가 NLL/고전 고차 통계를 이기는지 배치수준 미니 검증.

측정 전용(새 학습 없음). 결정적(seed 고정). 기존 도구 import(무수정):
  diagnose_drift_auc(DA): _load_tr_te, laplace_emp, state_codes, auc_rank
  diagnose_order_decomposition(OD): maxent_upto, bit_matrix, emp 유틸
  diagnose_drebin_order(DR): emp_dist
  부분집합: results2/drebin_targeted/drebin_targeted.json 표적 k=10 (diagnose_split1 과 동일).

상태: 배치 S 경험분포 p̂_S(Laplace c=1/2^k) → ψ_S[x]=√p̂_S(x) (2^10 실벡터, Born 인코딩, 위상 없음).
기준: ψ_train(train 전체). 얽힘 프로파일: (a)연속 절단 9개 + (b)고정 seed 랜덤 5|5 30개 = 39 절단.
  각 절단 ψ→2^|A|×2^|B| reshape→SVD→Schmidt λ² 스펙트럼·엔트로피 S.
점수: ENT-L2=엔트로피벡터(39D) L2거리(vs train), ENT-W=절단별 λ² 스펙트럼 1-Wasserstein 평균(vs train).
탐지: unseen(양성)/known-test(음성)에서 부트스트랩 배치 B=200, n∈{50,100,200}, 배치점수 거리→AUC.
베이스라인(동일 배치): NLL-full, NLL-M2, CONN3-dist(고전 고차), MARG-L1(저차 대조).

판정: split1서 ENT AUC가 NLL-full·NLL-M2 상회? + CONN3도 상회해야 "얽힘이 고전 고차 재포장 이상"(아니면 재포장 판정).
★ 단정 금지. 배치수준(단일 비트열은 얽힘 정의 안 됨)·ψ=√p̂ 위상없음·k10 한정·관측 정렬(인과 아님).

사용: python tools/ent_fingerprint_v0.py [--splits 0,1,2] [--B 200] [--ns 50,100,200] [--smoke]
"""
import os
import sys
import json
import itertools
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

import diagnose_drift_auc as DA           # noqa: E402  _load_tr_te, laplace_emp, state_codes, auc_rank
import diagnose_order_decomposition as OD  # noqa: E402  maxent_upto, bit_matrix
import diagnose_drebin_order as DR         # noqa: E402  emp_dist

OUTDIR = os.path.join("results2", "ent_fingerprint")
OUT_JSON = os.path.join(OUTDIR, "ent_v0.json")
OUT_PNG = os.path.join(OUTDIR, "ent_v0.png")
TGT_JSON = os.path.join("results2", "drebin_targeted", "drebin_targeted.json")
EPS = 1e-12
K = 10
SEED_CUTS = 777          # 랜덤 이분할 고정 seed
SEED_BOOT = 20260703     # 부트스트랩 고정 seed
N_RAND_CUTS = 30
SCORES = ["ENT_L2", "ENT_W", "NLL_full", "NLL_M2", "CONN3_dist", "MARG_L1"]


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 절단(bipartition) 정의: 연속 9 + 랜덤 5|5 30 = 39
# ---------------------------------------------------------------------------
def make_cuts(k):
    cuts = [tuple(range(m)) for m in range(1, k)]          # 연속: A=low m bits, m=1..k-1 (9개)
    rng = np.random.default_rng(SEED_CUTS)
    for _ in range(N_RAND_CUTS):
        perm = rng.permutation(k)
        cuts.append(tuple(sorted(int(q) for q in perm[:k // 2])))   # 랜덤 5|5
    return cuts


# ---------------------------------------------------------------------------
# ψ 의 한 절단 얽힘: reshape(텐서 축 재배열)→SVD→λ²(Schmidt), 엔트로피
# ---------------------------------------------------------------------------
def schmidt_spectrum(psi, k, A):
    """psi(2^k) 를 A|B 이분할로 reshape 후 SVD. 반환 λ²(내림차순), 엔트로피."""
    T = psi.reshape((2,) * k)                    # 축 j ↔ 비트 (k-1-j)  (C-order, 축0=MSB)
    axesA = [k - 1 - q for q in A]
    axesB = [k - 1 - q for q in range(k) if q not in A]
    M = np.transpose(T, axesA + axesB).reshape(1 << len(A), 1 << len(axesB))
    sv = np.linalg.svd(M, compute_uv=False)
    l2 = sv ** 2
    l2 = l2[l2 > 0]
    l2 = l2 / l2.sum()                           # 정규화(수치 안전)
    S = float(-np.sum(l2 * np.log(l2)))
    return np.sort(l2)[::-1], S


def ent_profile(psi, k, cuts):
    ents = np.zeros(len(cuts))
    spectra = []
    for i, A in enumerate(cuts):
        l2, S = schmidt_spectrum(psi, k, A)
        ents[i] = S
        spectra.append(l2)
    return ents, spectra


def wasserstein1_sorted(a, b):
    """정렬된 스펙트럼(확률질량, 합=1) 간 1-Wasserstein (rank 인덱스 단위간격 CDF 차이 합)."""
    n = max(len(a), len(b))
    aa = np.zeros(n); aa[:len(a)] = a
    bb = np.zeros(n); bb[:len(b)] = b
    return float(np.sum(np.abs(np.cumsum(aa) - np.cumsum(bb))))


# ---------------------------------------------------------------------------
# 배치 → ψ, 점수
# ---------------------------------------------------------------------------
def psi_of(Xbatch, k, c):
    p_emp, _ = DR.emp_dist(Xbatch)
    p = DA.laplace_emp(p_emp, len(Xbatch), k, c=c)
    return np.sqrt(p)


def moment_marg_conn3(Xbatch):
    """1차 활성(marginal) + 연결 3점 상관 벡터."""
    s = (1 - 2 * np.asarray(Xbatch, float))
    k = s.shape[1]
    m1 = s.mean(0)
    m2 = (s.T @ s) / len(s)
    triples = list(itertools.combinations(range(k), 3))
    v3 = np.zeros(len(triples))
    for t, (i, j, l) in enumerate(triples):
        m3 = float(np.mean(s[:, i] * s[:, j] * s[:, l]))
        v3[t] = (m3 - (m1[i] * m2[j, l] + m1[j] * m2[i, l] + m1[l] * m2[i, j])
                 + 2 * m1[i] * m1[j] * m1[l])
    return Xbatch.mean(0), v3       # marginal(활성률), conn3


def cosine_dist(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 1.0 - float(a @ b / (na * nb)) if na > 0 and nb > 0 else 1.0


def batch_scores(Xb, k, c, ref, cuts):
    """한 배치의 6개 점수(모두 클수록 이상). ref=train 기준 통계."""
    codes = DA.state_codes(Xb)
    # ENT
    psi = psi_of(Xb, k, c)
    ents, spectra = ent_profile(psi, k, cuts)
    ent_l2 = float(np.linalg.norm(ents - ref["ents"]))
    ent_w = float(np.mean([wasserstein1_sorted(spectra[i], ref["spectra"][i]) for i in range(len(cuts))]))
    # NLL
    nll_full = float(np.mean(-np.log(np.clip(ref["q_train"][codes], EPS, None))))
    nll_m2 = float(np.mean(-np.log(np.clip(ref["M2_train"][codes], EPS, None))))
    # CONN3 / MARG
    marg, conn3 = moment_marg_conn3(Xb)
    conn3_dist = cosine_dist(conn3, ref["conn3"])
    marg_l1 = float(np.abs(marg - ref["marg"]).sum())
    return dict(ENT_L2=ent_l2, ENT_W=ent_w, NLL_full=nll_full, NLL_M2=nll_m2,
                CONN3_dist=conn3_dist, MARG_L1=marg_l1)


def build_ref(Xtr_sub, k, c, cuts, bm):
    p_emp, _ = DR.emp_dist(Xtr_sub)
    q_train = DA.laplace_emp(p_emp, len(Xtr_sub), k, c=c)
    M2_train, _ = OD.maxent_upto(p_emp, k, bm, 2)
    psi = np.sqrt(q_train)
    ents, spectra = ent_profile(psi, k, cuts)
    marg, conn3 = moment_marg_conn3(Xtr_sub)
    return dict(q_train=q_train, M2_train=np.asarray(M2_train, float),
                ents=ents, spectra=spectra, marg=marg, conn3=conn3)


def bootstrap_auc(Xpos, Xneg, k, c, ref, cuts, B, n, rng):
    """양성/음성에서 B배치(크기 n) 부트스트랩 → 점수별 AUC."""
    pos = {s: [] for s in SCORES}
    neg = {s: [] for s in SCORES}
    for _ in range(B):
        ib = rng.integers(0, len(Xpos), size=n)
        sc = batch_scores(Xpos[ib], k, c, ref, cuts)
        for s in SCORES:
            pos[s].append(sc[s])
    for _ in range(B):
        ib = rng.integers(0, len(Xneg), size=n)
        sc = batch_scores(Xneg[ib], k, c, ref, cuts)
        for s in SCORES:
            neg[s].append(sc[s])
    aucs = {}
    for s in SCORES:
        scores = np.array(pos[s] + neg[s])
        mask = np.array([True] * B + [False] * B)
        aucs[s] = DA.auc_rank(scores, mask)
    return aucs


def load_subset(split, k):
    d = json.load(open(TGT_JSON))
    idx = {(int(r["split"]), int(kk)): r["subset"]
           for kk, recs in d["targeted"].items() for r in recs}
    return idx[(split, k)]


def run(splits, B, ns, do_smoke=False):
    os.makedirs(OUTDIR, exist_ok=True)
    k = K
    bm = OD.bit_matrix(k)
    cuts = make_cuts(k)
    c0 = 1.0 / (1 << k)
    log("절단=%d (연속9+랜덤5|5 %d) | ψ=√p̂(Laplace c=%.2e) | B=%d n=%s | seed_cuts=%d seed_boot=%d"
        % (len(cuts), N_RAND_CUTS, c0, B, ns, SEED_CUTS, SEED_BOOT))

    results = {}
    for split in splits:
        Xtr, ytr, Xte, yte = DA._load_tr_te(split)
        unseen = sorted(set(yte.tolist()) - set(ytr.tolist()))
        idx = load_subset(split, k)
        Xtr_s = Xtr[:, idx].astype(np.int64)
        Xpos = Xte[np.isin(yte, unseen)][:, idx].astype(np.int64)   # unseen 양성
        Xneg = Xte[~np.isin(yte, unseen)][:, idx].astype(np.int64)  # known 음성
        ref = build_ref(Xtr_s, k, c0, cuts, bm)
        log("\n=== split %d | unseen=%s | n_pos=%d n_neg=%d ===" % (split, unseen, len(Xpos), len(Xneg)))
        per_n = {}
        for n in ns:
            rng = np.random.default_rng(SEED_BOOT + split * 100 + n)
            aucs = bootstrap_auc(Xpos, Xneg, k, c0, ref, cuts, B, n, rng)
            per_n[str(n)] = aucs
            log("  n=%3d | ENT-L2=%.3f ENT-W=%.3f | NLL-full=%.3f NLL-M2=%.3f | CONN3=%.3f MARG=%.3f"
                % (n, aucs["ENT_L2"], aucs["ENT_W"], aucs["NLL_full"], aucs["NLL_M2"],
                   aucs["CONN3_dist"], aucs["MARG_L1"]))
        # 평활 민감도: n=100(있으면), c/2·2c 에서 ENT-L2/W AUC
        smooth = {}
        n_s = 100 if 100 in ns else ns[len(ns) // 2]
        for lab, cc in [("c_half", c0 / 2), ("c_2x", c0 * 2)]:
            ref_c = build_ref(Xtr_s, k, cc, cuts, bm)
            rng = np.random.default_rng(SEED_BOOT + split * 100 + n_s + 7)
            a = bootstrap_auc(Xpos, Xneg, k, cc, ref_c, cuts, B, n_s, rng)
            smooth[lab] = dict(ENT_L2=a["ENT_L2"], ENT_W=a["ENT_W"], NLL_full=a["NLL_full"], CONN3_dist=a["CONN3_dist"])
        log("  [평활민감 n=%d] c/2: ENT-L2=%.3f ENT-W=%.3f | 2c: ENT-L2=%.3f ENT-W=%.3f"
            % (n_s, smooth["c_half"]["ENT_L2"], smooth["c_half"]["ENT_W"],
               smooth["c_2x"]["ENT_L2"], smooth["c_2x"]["ENT_W"]))
        results[str(split)] = dict(unseen=unseen, n_pos=int(len(Xpos)), n_neg=int(len(Xneg)),
                                   per_n=per_n, smoothing=smooth)
        if do_smoke:
            break

    payload = dict(
        note=("얽힘 지문 v0. ψ=√p̂(Born, 위상없음) 의 39절단 Schmidt 스펙트럼/엔트로피 거리(ENT-L2/W) 가 "
              "배치수준 drift 탐지(unseen vs known)서 NLL-full/NLL-M2/CONN3-dist/MARG-L1 을 이기는지. "
              "split1 전용 여부 위해 0,2 병기. 단정 금지. 배치수준·위상없음·k10·관측정렬 한계."),
        params=dict(k=k, splits=splits, B=B, ns=ns, n_cuts=len(cuts), n_rand_cuts=N_RAND_CUTS,
                    laplace_c=c0, seed_cuts=SEED_CUTS, seed_boot=SEED_BOOT, scores=SCORES),
        results=results)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if not do_smoke:
        try:
            plot(results, splits, ns)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def plot(results, splits, ns):
    fig, ax = plt.subplots(1, len(splits), figsize=(6 * len(splits), 5.2), squeeze=False)
    ax = ax[0]
    x = np.arange(len(ns))
    style = {"ENT_L2": ("#7B2FBE", "o-"), "ENT_W": ("#3B6EA5", "s-"),
             "NLL_full": ("#4C9F70", "^--"), "NLL_M2": ("#9AA0A6", "v--"),
             "CONN3_dist": ("#E2A13B", "D-"), "MARG_L1": ("#C1442E", "x:")}
    for a, split in zip(ax, splits):
        r = results[str(split)]
        for s in SCORES:
            col, mk = style[s]
            a.plot(x, [r["per_n"][str(n)][s] for n in ns], mk, color=col, label=s, alpha=.85)
        a.axhline(0.5, color="black", lw=.8, ls=":")
        a.set_xticks(x); a.set_xticklabels(["n=%d" % n for n in ns])
        a.set_ylim(0.4, 1.02); a.set_ylabel("batch AUC (unseen vs known)")
        a.set_title("split%d (unseen=%s)" % (split, r["unseen"])); a.grid(alpha=.3)
        if split == splits[0]:
            a.legend(fontsize=7)
    plt.suptitle("Entanglement-fingerprint v0: ENT distance vs NLL / classical higher-order (batch-level drift)")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="얽힘 지문 v0(배치수준 얽힘거리 vs NLL/고전고차). 측정 전용. 단정 금지.")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--B", type=int, default=200)
    ap.add_argument("--ns", type=str, default="50,100,200")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    ns = [int(x) for x in args.ns.split(",") if x.strip()]
    if args.smoke:
        run([1], B=30, ns=[50], do_smoke=True); return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    run(splits, args.B, ns)


if __name__ == "__main__":
    main()
