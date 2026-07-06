#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
양자 justification 1단계 — B1(10q): 고전 생성모델 2종 vs QCBM 파라미터 효율 (학습 없음)
==========================================================================================
질문: QCBM은 B1에서 180개 파라미터로 TV=0.03694를 냈다(qcbm_iat_b1.json). 같은 q_train 분포를
      고전 생성모델이 같은 TV로 표현하려면 파라미터가 몇 개 필요한가? 180보다 많은가/적은가?
      (정확도 AUC가 아니라 '표현 효율'이 양자 정당성의 핵심 — Gao et al. PRX Quantum 2022.)

비교 대상(고전, numpy만; torch 등 새 dependency 없음):
  (A) 저랭크 행렬 분해: q_train(1024) -> 32x32 reshape -> SVD 랭크 r 근사 -> clip(0)+정규화.
      파라미터(보수적, 고전에 유리): params = 64r  (= 32r + r*32).
  (B) 비트그룹 분해: 10비트를 그룹으로 나눠 그룹 내부 joint × 그룹 간 독립.
      파라미터(보수적): params = sum_g (2^{b_g} - 1).
      marginal(전부 1비트, params=10) ~ emp_joint(전체 joint) 사이를 스윕.

끝점:
  - marginal  : 비트그룹 전부 1비트 (params=10), TV는 q_train으로 계산.
  - emp_joint : q_train 자체 (TV=0). params는 '저장해야 할 비0 셀 수' = 점유 상태수(79)로 보수 카운트
                (생성식 일반 카운트 2^10-1=1023도 참고로 함께 기록).
  - QCBM (B1) : (180, 0.03694) — qcbm_iat_b1.json에서 읽음. 재학습 없음.

q_train은 train_qcbm_iat_b1.build_b1()을 import 재사용(동일 seed=7/edges/패킹). 점유=79로 검증.
tv_dist는 experiment2.tv_dist를 import(정의 불일치 방지). 기존 소스 무수정.
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

import experiment2 as E                  # noqa: E402  tv_dist, empirical_dist
import train_qcbm_iat_b1 as TB1          # noqa: E402  build_b1() (q_train 재현)

OUT = "results2"
NQ = 10
NSTATES = 1 << NQ                        # 1024
EXPECT_OCC = 79                          # qcbm_iat_b1.json occupied_train_states
B1_JSON = os.path.join(OUT, "qcbm_iat_b1.json")


def normalize_clip(p):
    """QCBM p_final 처리와 동일: 음수 클립 후 재정규화."""
    p = np.clip(np.asarray(p, dtype=float), 0, None)
    s = p.sum()
    return p / s if s > 0 else p


# ---------------------------------------------------------------------------
# (A) 저랭크 행렬 분해
# ---------------------------------------------------------------------------
def lowrank_curve(q, side=32):
    assert side * side == len(q), "1024 != %d*%d" % (side, side)
    M = q.reshape(side, side)
    U, S, Vt = np.linalg.svd(M, full_matrices=False)
    rows = []
    for r in range(1, side + 1):
        approx = (U[:, :r] * S[:r]) @ Vt[:r, :]
        qa = normalize_clip(approx.reshape(-1))
        tv = E.tv_dist(qa, q)
        rows.append({"rank": r, "params": 64 * r, "tv": float(tv)})
    return rows


# ---------------------------------------------------------------------------
# (B) 비트그룹 분해
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
    q /= q.sum()
    return q, params


def bitgroup_curve(st_tr, q, nq, nstates):
    # 10비트(bit0..bit9)를 연속 그룹으로 분할한 여러 구성 (fine -> coarse).
    groupings = {
        "marginal(1x10)":   [[i] for i in range(nq)],
        "2bitx5":           [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]],
        "3+3+4":            [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9]],
        "5+5":              [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]],
        "7+3":              [[0, 1, 2, 3, 4, 5, 6], [7, 8, 9]],
        "8+2":              [list(range(8)), [8, 9]],
        "full(1x10)":       [list(range(nq))],
    }
    rows = []
    for name, gr in groupings.items():
        qd, params = group_product_dist(st_tr, gr, nstates)
        rows.append({"config": name, "grouping": gr, "params": int(params),
                     "tv": float(E.tv_dist(qd, q))})
    return rows


def main():
    os.makedirs(OUT, exist_ok=True)
    E.log("=" * 78)
    E.log("B1 파라미터 효율: 고전(저랭크/비트그룹) vs QCBM (학습 없음)")
    E.log("=" * 78)

    # --- q_train 재현 + 검증 ---
    d = TB1.build_b1()
    q = np.asarray(d["q_train"], dtype=float)
    st_tr = np.asarray(d["st_tr"], dtype=int)
    occ = int(d["occ"])
    E.log("q_train 재현: 길이=%d, 점유=%d (기대 %d)" % (len(q), occ, EXPECT_OCC))
    if occ != EXPECT_OCC or len(q) != NSTATES:
        E.log("!!! 점유/차원 불일치 -> 인코딩 재현 어긋남. 중단.")
        sys.exit(2)
    E.log("검증 OK (점유=79, 1024차원).")

    # --- QCBM 점 (json) ---
    with open(B1_JSON) as f:
        b1 = json.load(f)
    qcbm_pt = {"params": b1["train"]["nparams"], "tv": b1["train"]["best_tv"]}
    marg_auc = b1["auc"]["marginal"]; emp_auc = b1["auc"]["emp_joint"]; qcbm_auc = b1["auc"]["qcbm"]
    E.log("QCBM 점: params=%d, TV=%.5f (AUC qcbm=%.4f marg=%.4f emp=%.4f)"
          % (qcbm_pt["params"], qcbm_pt["tv"], qcbm_auc, marg_auc, emp_auc))

    # --- 고전 곡선 ---
    lr = lowrank_curve(q, side=32)
    bg = bitgroup_curve(st_tr, q, NQ, NSTATES)
    emp_point = {"config": "emp_joint(occupied)", "params_occupied": occ,
                 "params_generic_2^10-1": NSTATES - 1, "tv": 0.0}

    # --- 로그 표 ---
    E.log("-" * 78)
    E.log("[저랭크] 32x32 SVD (params=64r)")
    for r in lr:
        E.log("  r=%-2d params=%-4d TV=%.5f" % (r["rank"], r["params"], r["tv"]))
    E.log("-" * 78)
    E.log("[비트그룹] params=sum(2^b-1)")
    for r in bg:
        E.log("  %-14s params=%-4d TV=%.5f" % (r["config"], r["params"], r["tv"]))
    E.log("  emp_joint: params(점유)=%d / 일반(2^10-1)=%d  TV=0" % (occ, NSTATES - 1))

    # --- QCBM TV에 도달하는 데 필요한 고전 파라미터 수(사실) ---
    qtv = qcbm_pt["tv"]
    def first_reach(rows, key="params"):
        ok = [r for r in rows if r["tv"] <= qtv]
        return min(ok, key=lambda r: r[key]) if ok else None
    lr_reach = first_reach(lr)
    bg_reach = first_reach(bg)
    E.log("=" * 78)
    E.log("[사실] QCBM TV=%.5f (180 params) 도달에 필요한 고전 파라미터:" % qtv)
    E.log("  저랭크: %s" % ("params=%d (r=%d, TV=%.5f)"
          % (lr_reach["params"], lr_reach["rank"], lr_reach["tv"]) if lr_reach else "곡선 내 미도달"))
    E.log("  비트그룹: %s" % ("%s params=%d (TV=%.5f)"
          % (bg_reach["config"], bg_reach["params"], bg_reach["tv"]) if bg_reach else "구성 내 미도달"))

    # --- 저장 ---
    payload = {
        "encoding": "B1: proto2+syn1+psh1+size4 + FwdIAT2 = 10bit (1024)",
        "q_train_occupied": occ, "nstates": NSTATES,
        "tv_metric": "experiment2.tv_dist = 0.5*sum(|p-q|)",
        "qcbm_point": qcbm_pt,
        "auc_ref": {"qcbm": qcbm_auc, "marginal": marg_auc, "emp_joint": emp_auc},
        "lowrank": {"param_def": "64*r (32xr + rx32, 보수적)", "curve": lr},
        "bitgroup": {"param_def": "sum_g (2^{b_g} - 1)", "curve": bg},
        "emp_joint": emp_point,
        "qcbm_tv_reach": {
            "qcbm_tv": qtv,
            "lowrank_min_params_to_reach": lr_reach,
            "bitgroup_min_params_to_reach": bg_reach,
        },
        "note": ("판정(우위 유무)은 웹 AI. 본 파일은 사실(파라미터 vs TV)만 기록. "
                 "QCBM 재학습 없음(json 값 사용)."),
    }
    with open(os.path.join(OUT, "param_efficiency_b1.json"), "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    E.log("저장: results2/param_efficiency_b1.json")

    # --- 그림 ---
    try:
        plt.figure(figsize=(8.5, 6))
        lr_p = [r["params"] for r in lr]; lr_t = [r["tv"] for r in lr]
        plt.plot(lr_p, lr_t, "o-", color="steelblue", label="classical: low-rank (64r)")
        bg_p = [r["params"] for r in bg]; bg_t = [r["tv"] for r in bg]
        plt.plot(bg_p, bg_t, "s-", color="darkorange", label="classical: bit-group (sum 2^b-1)")
        # QCBM 점
        plt.scatter([qcbm_pt["params"]], [qcbm_pt["tv"]], color="crimson", s=120,
                    zorder=5, marker="*", label="QCBM B1 (180 params)")
        plt.axhline(qtv, color="crimson", ls="--", alpha=.5, label="QCBM TV=%.4f" % qtv)
        # emp_joint 끝점(점유 카운트)
        plt.scatter([occ], [0.0], color="green", s=70, zorder=5, marker="D",
                    label="emp_joint (occupied=%d, TV=0)" % occ)
        plt.xscale("log"); plt.yscale("log")
        plt.xlabel("parameters (log)"); plt.ylabel("TV vs q_train (log)")
        plt.title("B1 param efficiency: classical vs QCBM\n"
                  "How many classical params to reach QCBM's TV=%.4f @180?" % qtv)
        plt.legend(); plt.grid(alpha=.3, which="both")
        plt.tight_layout()
        fp = os.path.join(OUT, "param_efficiency_b1.png")
        plt.savefig(fp, dpi=130); plt.close()
        E.log("그림 저장: %s" % fp)
    except Exception as e:
        E.log("그림 저장 건너뜀: %r" % e)

    E.log("DONE")


if __name__ == "__main__":
    main()
