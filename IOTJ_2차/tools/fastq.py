#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
fastq — quantum_behavior 의 **수치동일** 고속 대체 (성능 전용, 의미 변경 없음).

동기: B/A 실행 병목의 ~100% 가 G.quantum_behavior (679 us/call, Powell 이 fit 당 ~11k 회 호출).
   원본은 16개 kron(2x2,2x2)+trace(rho@M) 를 파이썬 루프로 돈다. 상태가 |psi>=c|00>+s|11> (실수)
   이므로 tr(rho M) = <psi|M|psi> 를 닫힌형으로 전개하면 복소 4x4 연산이 전부 사라진다.

   <psi|kron(PA,PB)|psi> = c^2 PA00 PB00 + s^2 PA11 PB11 + 2 c s Re(PA01 PB01)
   P(theta,phi,a) = 0.5(I + (-1)^a (nx X + ny Y + nz Z))
     -> P00 = 0.5(1+q nz), P11 = 0.5(1-q nz), P01 = 0.5 q (nx - i ny),  q=(-1)^a
     -> Re(PA01 PB01) = 0.25 qa qb (nax nbx - nay nby)

   즉 prob = 0.25[ c^2 (1+qa naz)(1+qb nbz) + s^2 (1-qa naz)(1-qb nbz) ]
             + 0.5 c s qa qb (nax nbx - nay nby)

정합: R1 위배 아님 — gap_vs_cf_v0.py 는 **무수정**. install() 은 등가성을 런타임 검증(<1e-12)한 뒤에만
   G.quantum_behavior 를 교체한다. 검증 실패 시 예외를 던지고 교체하지 않는다(원본 그대로 동작).
   JSON params 에 fastq_installed 를 기록해 투명하게 남긴다.
"""
import numpy as np

CTX22 = [(0, 0), (0, 1), (1, 0), (1, 1)]
CTX33 = [(x, y) for x in range(3) for y in range(3)]


def _bloch(thetas, phis):
    """(m,) 각도 -> (m,3) Bloch 단위벡터."""
    st, ct = np.sin(thetas), np.cos(thetas)
    return np.stack([st * np.cos(phis), st * np.sin(phis), ct], axis=1)


def _behavior(th_state, nA, nB, ctx):
    """nA (mA,3), nB (mB,3) -> e[len(ctx),2,2]. 원본과 동일하게 clip(>=0) 후 정규화."""
    c, s = np.cos(th_state), np.sin(th_state)
    q = np.array([1.0, -1.0])                       # (-1)^a, a=0,1
    xs = np.array([t[0] for t in ctx]); ys = np.array([t[1] for t in ctx])
    az = nA[xs, 2][:, None, None] * q[None, :, None]        # (ctx,a,1)
    bz = nB[ys, 2][:, None, None] * q[None, None, :]        # (ctx,1,b)
    xx = (nA[xs, 0] * nB[ys, 0])[:, None, None]             # (ctx,1,1)
    yy = (nA[xs, 1] * nB[ys, 1])[:, None, None]
    qq = q[None, :, None] * q[None, None, :]                # (1,a,b)
    e = 0.25 * (c * c * (1.0 + az) * (1.0 + bz) + s * s * (1.0 - az) * (1.0 - bz)) \
        + 0.5 * c * s * qq * (xx - yy)
    e = np.maximum(e, 0.0)
    return e / e.sum(axis=(1, 2), keepdims=True)


def quantum_behavior(p):
    """G.quantum_behavior 수치동일 대체. p = [th_state, A0th,A0ph, A1th,A1ph, B0th,B0ph, B1th,B1ph]."""
    p = np.asarray(p, dtype=float)
    nA = _bloch(p[[1, 3]], p[[2, 4]])
    nB = _bloch(p[[5, 7]], p[[6, 8]])
    return _behavior(p[0], nA, nB, CTX22)


def quantum_behavior_23(p):
    """gate_s5.quantum_behavior_23 수치동일 대체. 13 파라미터, 9 컨텍스트."""
    p = np.asarray(p, dtype=float)
    nA = _bloch(p[[1, 3, 5]], p[[2, 4, 6]])
    nB = _bloch(p[[7, 9, 11]], p[[8, 10, 12]])
    return _behavior(p[0], nA, nB, CTX33)


def verify(G, n=400, seed=12345, tol=1e-12):
    """원본 대비 최대 절대오차. (canon + 무작위 각도)"""
    rng = np.random.default_rng(seed)
    worst = 0.0
    tests = [np.array([np.pi / 4, 0.0, 0.0, np.pi / 2, 0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0])]
    tests += [rng.uniform(-2 * np.pi, 2 * np.pi, size=9) for _ in range(n)]
    for p in tests:
        worst = max(worst, float(np.abs(G.quantum_behavior(p) - quantum_behavior(p)).max()))
    return worst, bool(worst < tol)


def verify_23(mod, n=400, seed=12345, tol=1e-12):
    rng = np.random.default_rng(seed)
    worst = 0.0
    tests = [mod.CANON_23] + [rng.uniform(-2 * np.pi, 2 * np.pi, size=13) for _ in range(n)]
    for p in tests:
        worst = max(worst, float(np.abs(mod.quantum_behavior_23(p) - quantum_behavior_23(p)).max()))
    return worst, bool(worst < tol)


def install(G, verbose=True):
    """등가성 검증 후에만 G.quantum_behavior 교체. 반환: dict(검증기록)."""
    worst, ok = verify(G)
    if not ok:
        raise RuntimeError("fastq.install 중단: 원본 대비 최대오차 %.3e (>1e-12)" % worst)
    G.quantum_behavior = quantum_behavior
    if verbose:
        print("[fastq] G.quantum_behavior 교체 (max|diff|=%.2e < 1e-12, 수치동일)" % worst, flush=True)
    return dict(installed=True, max_abs_diff=worst, target="gap_vs_cf_v0.quantum_behavior")


def install_23(mod, verbose=True):
    worst, ok = verify_23(mod)
    if not ok:
        raise RuntimeError("fastq.install_23 중단: 최대오차 %.3e" % worst)
    mod.quantum_behavior_23 = quantum_behavior_23
    if verbose:
        print("[fastq] quantum_behavior_23 교체 (max|diff|=%.2e)" % worst, flush=True)
    return dict(installed=True, max_abs_diff=worst, target="gate_s5.quantum_behavior_23")


if __name__ == "__main__":
    import os
    import sys
    import time
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gap_vs_cf_v0 as G
    w, ok = verify(G)
    p = np.array([np.pi / 4, 0.0, 0.0, np.pi / 2, 0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0])
    t = time.time(); [G.quantum_behavior(p) for _ in range(2000)]; t_orig = (time.time() - t) / 2000
    t = time.time(); [quantum_behavior(p) for _ in range(2000)]; t_fast = (time.time() - t) / 2000
    print("max|diff| = %.3e  (pass=%s)" % (w, ok))
    print("orig %.1f us/call, fast %.1f us/call  -> %.1fx" % (t_orig * 1e6, t_fast * 1e6, t_orig / t_fast))
