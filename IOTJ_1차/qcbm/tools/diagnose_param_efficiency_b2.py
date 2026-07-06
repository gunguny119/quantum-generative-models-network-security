#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
양자 justification 2단계 — B2(12q, 4096상태) 파라미터 효율 + 공정성 검증 (학습 없음)
=====================================================================================
질문:
  (1) 스케일: B2(4096상태, 점유 94=2.3%)에서도 B1처럼 고전이 QCBM보다 효율적인가?
  (2) 1단계 공정성:
      (a) NMF(비음수 분해)도 SVD만큼 효율적인가? (SVD는 음수를 clip+정규화로 후처리 -> 고전을
          부당하게 도왔을 수 있음. QCBM은 Born rule로 항상 유효분포.)
      (b) SVD 저랭크가 clip 전 음수 질량을 얼마나 만들었나? (정량)
      (c) TV뿐 아니라 MMD^2(QCBM이 학습에 쓴 그 커널)로도 재면 결론이 바뀌나?
판정은 웹 AI. 본 작업은 사실 측정만.

고전 baseline (numpy/scipy만; 새 dependency 없음):
  (A) SVD 저랭크: q_train(4096) -> 64x64 reshape -> 랭크 r 근사 -> clip(0)+정규화. params=128r.
  (a) NMF 저랭크: 같은 64x64 -> 비음수 분해(곱셈 업데이트) -> 정규화. params=128r. (clip 불필요)
  (B) 비트그룹: 12비트를 연속 그룹으로 -> 그룹내 joint x 그룹간 독립. params=sum_g(2^{b_g}-1).
끝점: marginal(1bit×12, params 12), emp_joint(q_train, TV0, params=점유 94).
QCBM 점: (216, json best_tv 0.093, json best_mmd2 0.00018865). 재학습 없음(저장 스칼라 사용).

q_train: train_qcbm_iat_b2.build_b2() import 재사용(동일 seed/edges/패킹). 점유 94 검증.
tv_dist/build_kernel/empirical_dist: experiment2 import. 기존 소스 무수정.
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import experiment2 as E                  # noqa: E402  tv_dist, build_kernel, empirical_dist
import train_qcbm_iat_b2 as TB2          # noqa: E402  build_b2()

OUT = "results2"
NQ = 12
NSTATES = 1 << NQ                        # 4096
SIDE = 64                                # 64x64 reshape
EXPECT_OCC = 94
B2_JSON = os.path.join(OUT, "qcbm_iat_b2.json")
B1_JSON = os.path.join(OUT, "param_efficiency_b1.json")


def normalize_clip(p):
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    s = p.sum()
    return p / s if s > 0 else p


def mmd2(p, q, K):
    d = np.asarray(p, float) - np.asarray(q, float)
    return float(d @ (K @ d))


# ---------------------------------------------------------------------------
# (A) SVD 저랭크 (+ 음수량)
# ---------------------------------------------------------------------------
def svd_curve(q, K, side=SIDE, rmax=SIDE):
    M = q.reshape(side, side)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rows = []
    for r in range(1, rmax + 1):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        flat = approx.reshape(-1)
        neg_mass = float(-flat[flat < 0].sum())          # 음수 질량(절댓값 합)
        neg_cells = int((flat < 0).sum())
        qa = normalize_clip(flat)
        rows.append({"rank": r, "params": 128 * r, "tv": float(E.tv_dist(qa, q)),
                     "mmd2": mmd2(qa, q, K), "neg_mass": neg_mass,
                     "neg_cells": neg_cells})
    return rows


# ---------------------------------------------------------------------------
# (a) NMF 저랭크 (numpy 곱셈 업데이트, 비음수)
# ---------------------------------------------------------------------------
def nmf_factorize(M, r, iters=400, n_init=3, eps=1e-9):
    """비음수 행렬분해 M≈W@H. 여러 init 중 Frobenius 오차 최소 선택."""
    best = None; best_err = np.inf
    for init in range(n_init):
        rng = np.random.default_rng(1000 * r + init)
        W = rng.random((M.shape[0], r)) + eps
        H = rng.random((r, M.shape[1])) + eps
        for _ in range(iters):
            H *= (W.T @ M) / (W.T @ W @ H + eps)
            W *= (M @ H.T) / (W @ (H @ H.T) + eps)
        err = float(np.linalg.norm(M - W @ H))
        if err < best_err:
            best_err, best = err, (W.copy(), H.copy())
    W, H = best
    return W @ H, best_err


def nmf_curve(q, K, side=SIDE, rmax=SIDE):
    M = q.reshape(side, side)
    rows = []
    for r in range(1, rmax + 1):
        approx, ferr = nmf_factorize(M, r)
        qa = normalize_clip(approx.reshape(-1))          # 비음수지만 정규화만(클립 무효과)
        rows.append({"rank": r, "params": 128 * r, "tv": float(E.tv_dist(qa, q)),
                     "mmd2": mmd2(qa, q, K), "frob_err": ferr})
    return rows


# ---------------------------------------------------------------------------
# (B) 비트그룹
# ---------------------------------------------------------------------------
def group_index(states, bits):
    idx = np.zeros(len(states), dtype=int)
    for j, b in enumerate(bits):
        idx |= (((states >> b) & 1) << j)
    return idx


def group_product_dist(st_tr, grouping, nstates):
    all_states = np.arange(nstates)
    q = np.ones(nstates, dtype=float)
    params = 0
    for g in grouping:
        b = len(g)
        gd = np.bincount(group_index(st_tr, g), minlength=1 << b).astype(float)
        gd /= gd.sum()
        q *= gd[group_index(all_states, g)]
        params += (1 << b) - 1
    return q / q.sum(), params


def bitgroup_curve(st_tr, q, K, nq, nstates):
    groupings = {
        "marginal(1x12)": [[i] for i in range(nq)],
        "2bitx6":         [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [10, 11]],
        "3bitx4":         [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        "4bitx3":         [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11]],
        "6+6":            [[0, 1, 2, 3, 4, 5], [6, 7, 8, 9, 10, 11]],
        "8+4":            [list(range(8)), [8, 9, 10, 11]],
        "9+3":            [list(range(9)), [9, 10, 11]],
        "full(1x12)":     [list(range(nq))],
    }
    rows = []
    for name, gr in groupings.items():
        qd, params = group_product_dist(st_tr, gr, nstates)
        rows.append({"config": name, "grouping": gr, "params": int(params),
                     "tv": float(E.tv_dist(qd, q)), "mmd2": mmd2(qd, q, K)})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 78)
    E.log("B2 파라미터 효율 + 공정성(NMF/음수량/MMD): 고전 vs QCBM (학습 없음)")
    E.log("=" * 78)

    d = TB2.build_b2()
    q = np.asarray(d["q_train"], dtype=float)
    st_tr = np.asarray(d["st_tr"], dtype=int)
    occ = int(d["occ"])
    E.log("q_train 재현: 길이=%d, 점유=%d (기대 %d)" % (len(q), occ, EXPECT_OCC))
    if occ != EXPECT_OCC or len(q) != NSTATES:
        E.log("!!! 점유/차원 불일치 -> 중단."); sys.exit(2)
    E.log("검증 OK (점유=94, 4096차원).")

    E.log("MMD 커널 생성: build_kernel(%d) (%dx%d)..." % (NQ, NSTATES, NSTATES))
    K = E.build_kernel(NQ)

    with open(B2_JSON) as f:
        b2 = json.load(f)
    qcbm_pt = {"params": b2["train"]["nparams"], "tv": b2["train"]["best_tv"],
               "mmd2_stored": b2["train"]["best_mmd2"]}
    E.log("QCBM 점: params=%d TV=%.4f MMD^2(저장)=%.3e | AUC qcbm=%.4f marg=%.4f emp=%.4f"
          % (qcbm_pt["params"], qcbm_pt["tv"], qcbm_pt["mmd2_stored"],
             b2["auc"]["qcbm"], b2["auc"]["marginal"], b2["auc"]["emp_joint"]))

    E.log("고전 곡선 계산 (SVD / NMF / 비트그룹)...")
    svd = svd_curve(q, K)
    nmf = nmf_curve(q, K)
    bg = bitgroup_curve(st_tr, q, K, NQ, NSTATES)
    emp_mmd = mmd2(q, q, K)  # =0
    emp_point = {"params_occupied": occ, "params_generic": NSTATES - 1,
                 "tv": 0.0, "mmd2": emp_mmd}

    # --- 로그 표 (요약: r=1..6 + 끝부분) ---
    E.log("-" * 78)
    E.log("[SVD] 64x64 (params=128r) | TV | MMD^2 | 음수질량 | 음수셀")
    for rrow in svd[:8]:
        E.log("  r=%-2d p=%-4d TV=%.5f MMD2=%.3e negMass=%.4f negCells=%d"
              % (rrow["rank"], rrow["params"], rrow["tv"], rrow["mmd2"],
                 rrow["neg_mass"], rrow["neg_cells"]))
    E.log("[NMF] 64x64 (params=128r) | TV | MMD^2 | frobErr")
    for rrow in nmf[:8]:
        E.log("  r=%-2d p=%-4d TV=%.5f MMD2=%.3e frob=%.4f"
              % (rrow["rank"], rrow["params"], rrow["tv"], rrow["mmd2"], rrow["frob_err"]))
    E.log("[비트그룹] params=sum(2^b-1)")
    for rrow in bg:
        E.log("  %-14s p=%-5d TV=%.5f MMD2=%.3e"
              % (rrow["config"], rrow["params"], rrow["tv"], rrow["mmd2"]))
    E.log("  emp_joint: params(점유)=%d / 일반=%d TV=0 MMD2=%.1e" % (occ, NSTATES - 1, emp_mmd))

    # --- QCBM TV 도달 파라미터 (사실) ---
    qtv = qcbm_pt["tv"]
    def reach(rows):
        ok = [r for r in rows if r["tv"] <= qtv]
        return min(ok, key=lambda r: r["params"]) if ok else None
    svd_r, nmf_r, bg_r = reach(svd), reach(nmf), reach(bg)
    E.log("=" * 78)
    E.log("[사실] QCBM TV=%.4f (216p) 도달 고전 파라미터:" % qtv)
    for nm, rr in [("SVD", svd_r), ("NMF", nmf_r), ("비트그룹", bg_r)]:
        if rr:
            tag = rr.get("config") or ("r=%d" % rr["rank"])
            E.log("  %-8s params=%-4d (%s, TV=%.5f)" % (nm, rr["params"], tag, rr["tv"]))
        else:
            E.log("  %-8s 미도달" % nm)

    # --- B1 비교 ---
    b1cmp = None
    if os.path.exists(B1_JSON):
        with open(B1_JSON) as f:
            b1 = json.load(f)
        b1cmp = {"qcbm": b1["qcbm_point"],
                 "lowrank_reach": b1["qcbm_tv_reach"]["lowrank_min_params_to_reach"],
                 "bitgroup_reach": b1["qcbm_tv_reach"]["bitgroup_min_params_to_reach"],
                 "occupied": b1["q_train_occupied"]}

    # --- 저장 ---
    payload = {
        "encoding": "B2: proto2+syn1+psh1+size4 + FwdIAT2 + FlowIAT2 = 12bit (4096)",
        "q_train_occupied": occ, "nstates": NSTATES,
        "tv_metric": "experiment2.tv_dist = 0.5*sum|p-q|",
        "mmd_metric": "experiment2.build_kernel(12) (multi-bandwidth Gaussian), MMD^2=(p-q)K(p-q)",
        "qcbm_point": qcbm_pt,
        "qcbm_mmd2_note": "QCBM MMD^2는 json 저장 스칼라(p_final 미저장이라 재계산 불가). 고전 MMD^2는 동일 K로 새 계산.",
        "auc_ref": {k: b2["auc"][k] for k in ("qcbm", "marginal", "emp_joint")},
        "svd": {"param_def": "128r (64xr+rx64)", "curve": svd},
        "nmf": {"param_def": "128r, numpy 곱셈업데이트 400it x3init best", "curve": nmf},
        "bitgroup": {"param_def": "sum_g(2^{b_g}-1)", "curve": bg},
        "emp_joint": emp_point,
        "qcbm_tv_reach": {"qcbm_tv": qtv, "svd": svd_r, "nmf": nmf_r, "bitgroup": bg_r},
        "b1_compare": b1cmp,
        "note": "판정은 웹 AI. 사실(params vs TV/MMD^2)만 기록. QCBM 재학습 없음.",
    }
    with open(os.path.join(OUT, "param_efficiency_b2.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/param_efficiency_b2.json")

    # --- 그림 (TV 패널 + MMD 패널) ---
    try:
        fig, ax = plt.subplots(1, 2, figsize=(15, 6))
        for a, ykey, ylab in [(ax[0], "tv", "TV vs q_train"), (ax[1], "mmd2", "MMD^2 vs q_train")]:
            a.plot([r["params"] for r in svd], [r[ykey] for r in svd], "o-",
                   color="steelblue", label="SVD low-rank (128r)")
            a.plot([r["params"] for r in nmf], [r[ykey] for r in nmf], "^-",
                   color="purple", label="NMF low-rank (128r, 비음수)")
            a.plot([r["params"] for r in bg], [r[ykey] for r in bg], "s-",
                   color="darkorange", label="bit-group (sum 2^b-1)")
            yq = qcbm_pt["tv"] if ykey == "tv" else qcbm_pt["mmd2_stored"]
            a.scatter([qcbm_pt["params"]], [yq], color="crimson", s=140, marker="*",
                      zorder=5, label="QCBM B2 (216p)")
            a.axhline(yq, color="crimson", ls="--", alpha=.5)
            a.scatter([occ], [0.0 if ykey == "tv" else emp_mmd], color="green", s=70,
                      marker="D", zorder=5, label="emp_joint (occ=%d)" % occ)
            a.set_xscale("log"); a.set_yscale("log")
            a.set_xlabel("parameters (log)"); a.set_ylabel(ylab + " (log)")
            a.set_title(ylab); a.legend(fontsize=8); a.grid(alpha=.3, which="both")
        plt.suptitle("B2 param efficiency (classical vs QCBM) — TV & MMD, SVD vs NMF")
        plt.tight_layout()
        fp = os.path.join(OUT, "param_efficiency_b2.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
