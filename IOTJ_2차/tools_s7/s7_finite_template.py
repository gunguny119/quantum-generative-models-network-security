#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate S7 Phase 3-0 — (2,3,2) 유한표본 템플릿 적합기 (신규 파일, tools/ 무수정)

■ 원본 대응 (사용자 조건: "원본 로직 복제 시 어느 함수의 대응인지 명시")
  _ce                    : gap_vs_cf_v0.py:209  _ce                  ← 그대로 재사용(shape 무관)
  eval_kl_23             : gate_s5_multifacet.py:97  eval_kl_23      ← 그대로 재사용
  nc_floor_23            : gate_s5_multifacet.py:107 nc_floor_23     ← 그대로 재사용
  detection_auc_23       : gate_s5_multifacet.py:320                 ← 그대로 재사용(무수정)

  sample_emp_23          ← gap_vs_cf_v0.py:193 sample_emp 의 (2,3,2) 대응.
                           gate_s5_multifacet.py:325-337 detection_auc_23 내부 boxes() 와
                           동일한 샘플링·Laplace 평활(1/(per*4)) 방식.
  _q_inits_23            ← gap_vs_cf_v0.py:160 _q_inits 의 (2,3,2) 대응 (9→13 파라, CANON_23).
  fit_quantum_params_23  ← gap_vs_cf_v0.py:224 fit_quantum 의 (2,3,2) 대응.
                           = gate_s2_detection.py:67 fit_quantum_params 의 (2,3,2) 대응.
                           (원본과 동일: Powell, 목적=_ce, maxiter=2000/xtol=1e-6/ftol=1e-9,
                            차이는 quantum_behavior_23(13파라) 사용 + 파라미터 벡터 반환)
  fit_nc_params_23       ← gate_s2_detection.py:80 fit_nc_params 의 (2,3,2) 대응.
                           원본 nc_floor_23 이 이미 emp 에 CE 최소화로 적합하므로 model 만 반환.
  verify_wrappers_23     ← gate_s2_detection.py:85 verify_wrappers 의 (2,3,2) 대응.

■ 설계 조건 (사용자 승인 조건 그대로)
  - detection_auc_23 수정 금지 → 적합된 템플릿을 q_model/c_model 인자로 전달.
  - 양자/고전 템플릿을 **동일한 훈련 emp** 로 적합 (공정 비교).
  - 참값 템플릿(q_model=e_normal, c_model=모집단 NC적합) 결과와 나란히 보고.
"""
import os
import sys

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")

import numpy as np
from scipy.optimize import minimize

sys.path.insert(0, "/home/elicer/IOTJ/IOTJ_2차/tools")
import gap_vs_cf_v0 as G                 # _ce, EPS
import gate_s5_multifacet as M           # quantum_behavior_23, nc_floor_23, eval_kl_23, CTX9, CANON_23, SEED
import fastq                             # 성능 전용(결과 불변, max|diff|=2.22e-16 검증됨)

SEED = M.SEED                            # 20260725 (R7)
V_NORMAL = 0.856
QPARAM_23 = 13


def install_fast():
    """gate_s5_multifacet._slsqp_one(478-479행)과 동일한 fastq 설치 관례."""
    if M.quantum_behavior_23 is not fastq.quantum_behavior_23:
        fastq.install_23(M, verbose=False)


# ---------------------------------------------------------------------------
# 샘플링 — gap_vs_cf_v0.sample_emp(193행)의 (2,3,2) 대응
# ---------------------------------------------------------------------------
def sample_emp_23(e_true, N, rng):
    """N=None → 모집단 참값 그대로. 정수 → 9컨텍스트 균등배분(per = N//9)."""
    if N is None:
        return e_true.copy()
    emp = np.zeros((9, 2, 2))
    per = max(N // 9, 1)
    flat = e_true.reshape(9, 4)
    for ci in range(9):
        cnt = np.bincount(rng.choice(4, size=per, p=flat[ci]), minlength=4)
        emp[ci] = cnt.reshape(2, 2)
    emp += 1.0 / (per * 4.0)                       # 원본과 동일한 약한 Laplace
    emp = emp / emp.sum(axis=(1, 2), keepdims=True)
    return emp


# ---------------------------------------------------------------------------
# 초기점 — gap_vs_cf_v0._q_inits(160행)의 (2,3,2) 대응
# ---------------------------------------------------------------------------
def _q_inits_23(rng, n):
    inits = [M.CANON_23.copy()]
    for _ in range(n - 1):
        inits.append(rng.uniform(0, np.pi, size=QPARAM_23))
    return inits


# ---------------------------------------------------------------------------
# 양자 템플릿 적합 — gap_vs_cf_v0.fit_quantum(224행) / gate_s2_detection.fit_quantum_params(67행)
# ---------------------------------------------------------------------------
def fit_quantum_params_23(emp, rng, restarts):
    """emp 에 CE 최소화로 quantum_behavior_23(13파라)를 적합, 최적 파라미터 벡터 반환."""
    install_fast()
    best = None
    for x0 in _q_inits_23(rng, restarts):
        r = minimize(lambda p: G._ce(emp, M.quantum_behavior_23(p)), x0, method="Powell",
                     options=dict(maxiter=2000, xtol=1e-6, ftol=1e-9))
        val = G._ce(emp, M.quantum_behavior_23(r.x))
        if best is None or val < best[0]:
            best = (val, r.x)
    return best[1]


def fit_quantum_model_23(emp, rng, restarts):
    return M.quantum_behavior_23(fit_quantum_params_23(emp, rng, restarts))


# ---------------------------------------------------------------------------
# 고전 템플릿 적합 — gate_s2_detection.fit_nc_params(80행)의 (2,3,2) 대응
# ---------------------------------------------------------------------------
def fit_nc_params_23(emp, rng, restarts):
    """원본 nc_floor_23(emp, e_true, rng, restarts) 은 emp 에 적합하고
       (KL(e_true‖model), model) 을 반환한다. 여기서는 model 만 꺼내 쓴다."""
    _, model = M.nc_floor_23(emp, emp, rng, restarts)
    return model


# ---------------------------------------------------------------------------
# Phase 3-0 검증 — gate_s2_detection.verify_wrappers(85행)의 (2,3,2) 대응
# ---------------------------------------------------------------------------
def verify_wrappers_23(restarts=4, n_big=200000):
    install_fast()
    out = {}
    e_normal = M.werner_23(M.CANON_23, V_NORMAL)
    e_pure = M.quantum_behavior_23(M.CANON_23)          # in-class 참값(순수상태)

    # V1 — nc 래퍼가 원본 반환값과 정합인가 (원본 대비 직접 검증)
    rng = np.random.default_rng(SEED + 55)
    emp = sample_emp_23(e_normal, 20000, rng)
    kl_orig, model_orig = M.nc_floor_23(emp, e_normal, np.random.default_rng(SEED + 72), restarts)
    model_wrap = fit_nc_params_23(emp, np.random.default_rng(SEED + 72), restarts)
    kl_wrap = M.eval_kl_23(e_normal, model_wrap)
    out["V1_nc_wrapper"] = dict(kl_orig=float(kl_orig), kl_wrap=float(kl_wrap),
                                diff=float(abs(kl_orig - kl_wrap)),
                                model_identical=bool(np.array_equal(model_orig, model_wrap)),
                                pass_=bool(abs(kl_orig - kl_wrap) < 1e-9))

    # V2 — 양자 in-class 회복: 모델클래스가 참값을 포함하면 큰 표본에서 회복돼야 함
    rng = np.random.default_rng(SEED + 56)
    emp_big = sample_emp_23(e_pure, n_big, rng)
    p_fit = fit_quantum_params_23(emp_big, np.random.default_rng(SEED + 73), restarts)
    kl_rec = M.eval_kl_23(e_pure, M.quantum_behavior_23(p_fit))
    out["V2_quantum_inclass_recovery"] = dict(n=n_big, kl_to_truth=float(kl_rec),
                                              pass_=bool(kl_rec < 1e-3))

    # V3 — 양자 out-of-class floor: 참값이 Werner(V=0.856)라 순수상태 클래스로는 못 맞춤
    rng = np.random.default_rng(SEED + 57)
    emp_big_n = sample_emp_23(e_normal, n_big, rng)
    p_fit_n = fit_quantum_params_23(emp_big_n, np.random.default_rng(SEED + 74), restarts)
    kl_floor = M.eval_kl_23(e_normal, M.quantum_behavior_23(p_fit_n))
    out["V3_quantum_modelclass_floor"] = dict(n=n_big, kl_to_normal=float(kl_floor),
                                              note="순수상태 13파라 클래스의 Werner(V=0.856) 대비 하한. "
                                                   "탐지 신호(공격 KL=0.010182)와 비교할 기준.")

    # V4 — 결정성: 같은 seed → 파라미터 비트 동일
    a = fit_quantum_params_23(emp_big_n, np.random.default_rng(SEED + 74), restarts)
    b = fit_quantum_params_23(emp_big_n, np.random.default_rng(SEED + 74), restarts)
    c1 = fit_nc_params_23(emp, np.random.default_rng(SEED + 72), restarts)
    c2 = fit_nc_params_23(emp, np.random.default_rng(SEED + 72), restarts)
    out["V4_determinism"] = dict(quantum_identical=bool(np.array_equal(a, b)),
                                 classical_identical=bool(np.array_equal(c1, c2)),
                                 pass_=bool(np.array_equal(a, b) and np.array_equal(c1, c2)))

    out["all_pass"] = bool(out["V1_nc_wrapper"]["pass_"] and
                           out["V2_quantum_inclass_recovery"]["pass_"] and
                           out["V4_determinism"]["pass_"])
    return out


if __name__ == "__main__":
    import json
    r = verify_wrappers_23(restarts=int(sys.argv[1]) if len(sys.argv) > 1 else 4,
                           n_big=int(sys.argv[2]) if len(sys.argv) > 2 else 200000)
    print(json.dumps(r, ensure_ascii=False, indent=1))
