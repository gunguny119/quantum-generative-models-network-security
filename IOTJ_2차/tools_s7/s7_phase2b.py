#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate S7 / Phase B — 네 템플릿 정면 대조: "왜 양자인가"의 격리

■ 무수정 재사용: gate_s5_multifacet.{eval_kl_23, chsh_S_sub, werner_p, werner_23,
                 nc_floor_23, quantum_behavior_23, detection_auc_23}, gate_s2_detection.auc
■ 신규(스크래치패드, 승인됨)
   raw_scores_23   ← gate_s5_multifacet.py:320 detection_auc_23 의 원시점수 반환판.
                     boxes()/dev()/rng 소비 순서를 **원본과 동일하게** 복제하고
                     AUC 가 원본과 1e-12 이내 일치함을 verify_raw() 로 검증한 뒤 사용.
                     (실측: N ∈ {245,2000,4900} 전부 max_abs_diff = 0.0)
   marginal_template_23  신규. p̂(a|x)·p̂(b|y), 자유도 6.
   empirical_template_23 신규. 훈련 emp 그대로(포화모형), 자유도 27 = 9ctx × 3.

■ 평활(Laplace) — 경험표의 유일한 정칙화이므로 상수를 명시하고 민감도 병기
   기본  : 1/(per*4), per = n_train//9 = 1111  →  2.250225e-04  (boxes() 관례와 동일)
   민감도: 0.0, 1e-6

■ 마지널 곱의 y 평균 처리
   정상 behavior 는 무신호이므로 p(a|x) 가 y 에 무관해야 한다. 유한표본에서 생기는
   y 의존 잔차는 **y 에 대한 산술평균**으로 흡수한다(파티 B 도 대칭).
"""
import os
import sys
import json
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from concurrent.futures import ProcessPoolExecutor

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
sys.path.insert(0, "/home/elicer/IOTJ/IOTJ_2차/tools")

import gate_s5_multifacet as M
import gate_s2_detection as D2
import fastq
import s7_finite_template as F

REPO = "/home/elicer/IOTJ/IOTJ_2차"
SEED, V_NORMAL = M.SEED, 0.856
N_TRAIN, RESTARTS, TRAIN_SEEDS, REPS = 10000, 6, 10, 100
N_GRID = [245, 490, 980, 1960, 4900, 24500]
TAUS = [0.0, 1e-3, 1e-2]
BOOT = 2000
SMOOTH_DEFAULT = 1.0 / ((N_TRAIN // 9) * 4.0)
SMOOTHS = [SMOOTH_DEFAULT, 0.0, 1e-6]


def _fast():
    if M.quantum_behavior_23 is not fastq.quantum_behavior_23:
        fastq.install_23(M, verbose=False)


# ------------------------------------------------------------ 신규 템플릿
def empirical_template_23(e_true, n, rng, smooth=None):
    """포화모형: 컨텍스트별 p̂(a,b|x,y). 자유도 27. smooth=None → boxes() 관례."""
    per = max(n // 9, 1)
    s = (1.0 / (per * 4.0)) if smooth is None else smooth
    emp = np.zeros((9, 2, 2))
    flat = e_true.reshape(9, 4)
    for c in range(9):
        emp[c] = np.bincount(rng.choice(4, size=per, p=flat[c]), minlength=4).reshape(2, 2)
    emp += s
    return emp / emp.sum(axis=(1, 2), keepdims=True)


def marginal_template_23(emp):
    """p̂(a|x)·p̂(b|y). CTX9 인덱싱 ci = 3x + y. y(및 x) 평균으로 무신호 잔차 흡수. 자유도 6."""
    pa = np.zeros((3, 2)); pb = np.zeros((3, 2))
    for x in range(3):
        pa[x] = np.mean([emp[3 * x + y].sum(axis=1) for y in range(3)], axis=0)
    for y in range(3):
        pb[y] = np.mean([emp[3 * x + y].sum(axis=0) for x in range(3)], axis=0)
    pa /= pa.sum(axis=1, keepdims=True); pb /= pb.sum(axis=1, keepdims=True)
    out = np.zeros((9, 2, 2))
    for ci, (x, y) in enumerate(M.CTX9):
        out[ci] = np.outer(pa[x], pb[y])
    return out


# ------------------------------------------------------------ 원시점수 래퍼
def raw_scores_23(e_normal, e_atk, models, n_eval, reps, seed):
    """detection_auc_23(gate_s5_multifacet.py:320)의 원시점수 반환판.
       boxes() 구현·rng 소비 순서(normal 먼저, attack 나중)를 원본과 동일하게 유지."""
    rng = np.random.default_rng(seed)

    def boxes(e):
        out = []
        flat = e.reshape(9, 4)
        per = max(n_eval // 9, 1)
        for _ in range(reps):
            emp = np.zeros((9, 2, 2))
            for c in range(9):
                emp[c] = np.bincount(rng.choice(4, size=per, p=flat[c]), minlength=4).reshape(2, 2)
            emp += 1.0 / (per * 4.0)
            out.append(emp / emp.sum(axis=(1, 2), keepdims=True))
        return out

    nb, ab = boxes(e_normal), boxes(e_atk)          # 순서 = 원본
    res = {}
    for name, mdl in models.items():
        res[name] = (np.array([M.eval_kl_23(b, mdl) for b in ab]),
                     np.array([M.eval_kl_23(b, mdl) for b in nb]))
    res["S_test"] = (np.array([-abs(M.chsh_S_sub(b)) for b in ab]),
                     np.array([-abs(M.chsh_S_sub(b)) for b in nb]))
    return res


def fast_auc(pos, neg):
    a = np.concatenate([pos, neg])
    r = np.empty_like(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    sa = a[order]
    i = 0
    while i < len(sa):
        j = i
        while j + 1 < len(sa) and sa[j + 1] == sa[i]:
            j += 1
        r[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    n1 = len(pos)
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * len(neg)))


def verify_raw(tol=1e-12):
    """원본 detection_auc_23 과 AUC 가 tol 이내 일치하는지 (승인 조건)."""
    _fast()
    e_n = M.werner_23(M.CANON_23, V_NORMAL)
    ex = json.load(open(os.path.join(REPO, "results2/gate_s5/s5.json")))["verification"]["exact_slsqp"]
    e_a = M.werner_p(np.array(ex["p"]))
    nc, c_pop = M.nc_floor_23(e_n, e_n, np.random.default_rng(SEED + 2), 8)
    out = {}
    for N in (245, 2000, 4900):
        orig = M.detection_auc_23(e_n, e_a, e_n, c_pop, n_eval=N, reps=REPS, seed=SEED + 99)
        raw = raw_scores_23(e_n, e_a, dict(quantum=e_n, classical=c_pop), N, REPS, SEED + 99)
        d = {k: abs(fast_auc(*raw[k]) - orig[k]) for k in ("quantum", "classical", "S_test")}
        out["N=%d" % N] = dict(orig={k: orig[k] for k in ("quantum", "classical", "S_test")},
                               max_abs_diff=max(d.values()), per_key=d, pass_=max(d.values()) < tol)
    out["all_pass"] = all(v["pass_"] for v in out.values() if isinstance(v, dict))
    return out


if __name__ == "__main__":
    print(json.dumps(verify_raw(), ensure_ascii=False, indent=1))
