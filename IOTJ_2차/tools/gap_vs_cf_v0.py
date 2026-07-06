#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
gap-vs-CF 임계곡선 v0 — contextuality 연속 노브에서 (메모리매칭) 양자/고전 학습격차가 켜지는 지점 실측.

측정/학습 실험. 결정적(seed 고정), 무설치(numpy·scipy만; CF-LP는 gate1_iat_contextuality 와 동일 linprog 정식화 재사용,
양자모델은 QCBM계열 소형 2큐빗 파라미터화 모델을 순수 numpy 로 재구현=결정적). 단정 금지.

★ "우위"의 정의(모든 산출물에 명기): 여기서 gap 은 **메모리(파라미터) 매칭된 모델클래스 분리**이지 quantum supremacy 가 아님.
  고전 사다리 = no-signalling 국소(=비맥락) 은닉변수 잠재모델(Fine 정리로 국소 폴리토프에 갇힘) + 암기용 CPT(참조).
  양자 = 2큐빗 empirical model(양자 집합 ⊋ 국소집합, 단 Tsirelson 한계로 초양자 PR 는 실현 불가).

노브:  e(λ) = λ·e_SC + (1−λ)·e_NC,  e_SC=PR-box(2,2,2),  e_NC=균등잡음(비맥락).
이론:  CF(λ) = max(0, 2λ−1)  (ABM 2017, arXiv:1705.07918 Thm — 잡음 PR 상자의 contextual fraction).
       우리 CF-LP 로 λ마다 CF 실측 → 이론과 대조(불일치>0.01 보고) = self-check.
보조노브: 등방(isotropic) 양자실현 한계 λ_Q=1/√2 (CHSH 2√2). CF(λ_Q)=√2−1. 그 아래=양자실현, 위=초양자(비물리).
        ※ 명세의 "CF_max≈0.207" 값은 본 등방혼합선 계산과 불일치(√2−1≈0.414) → §self-check 에 그대로 보고.

과제(순차/스트리밍, 각 스텝 독립): 컨텍스트 c_t~unif{(x,y)} → 모델이 outcome (a,b) 분포 출력. 타깃=e(λ) 조건분포.
       학습표본 N∈{∞(모집단),500,5000}, 평가=held-out 컨텍스트별 KL(타깃‖모델) 평균, seed 5.
모델: 고전 잠재사다리 H∈{1(=독립),2,3,4,6,8}(파라미터 5H−1) + CPT(암기,12) ; 양자 2큐빗(9 파라미터).
gap(λ) := KL_best_classical_at_matched_memory(λ) − KL_quantum(λ)   (±20% 파라미터 셀끼리만; overall-best 도 병기).

판정: P1 임계 λ_emp(부트스트랩 CI) / P2 λ_emp vs λ*=0.5 / P3 곡선형태(선형·구간선형·시그모이드 AIC) /
      P4 인과대조(e_SC→CF=0 완전상관 비맥락상자 치환 시 gap 소멸?) / P5 실데이터 앵커(KCBS 0.2629·보안 ≈0).

한계: (2,2,2) 소규모·시뮬·메모리매칭 정의 의존·finite-sample·CF-LP 는 상한/근사. 관측 법칙이지 supremacy 아님.

사용: python tools/gap_vs_cf_v0.py [--gate0] [--smoke] [--seeds 5] [--full]
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)                       # IOTJ_2차
os.chdir(ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                     # noqa: E402

OUTDIR = os.path.join("results2", "gap_cf")
OUT_JSON = os.path.join(OUTDIR, "gap_cf.json")
OUT_PNG = os.path.join(OUTDIR, "gap_cf.png")

SEED = 20260703
EPS = 1e-12
CTX = [(0, 0), (0, 1), (1, 0), (1, 1)]              # (x,y) 컨텍스트 4개
KCBS_CF = 0.2629                                    # 실측 앵커 (b_to_a.json, PMC8827658)
SEC_CF = 4.5e-05                                    # 보안데이터 앵커 (gate1_ctx Tinba ≈0)
LAM_Q = 1.0 / np.sqrt(2.0)                          # 등방 양자실현 한계 λ_Q=1/√2
H_LADDER = [1, 2, 3, 4, 6, 8]


def log(m):
    print(m, flush=True)


def clsparam(H):
    return 5 * H - 1


QPARAM = 9                                          # 2큐빗: θ_state + 4측정×2각 = 9


# ===========================================================================
# empirical models  e[ctx_index, a, b]   (ctx_index -> CTX 순서)
# ===========================================================================
def pr_box():
    e = np.zeros((4, 2, 2))
    for ci, (x, y) in enumerate(CTX):
        for a in (0, 1):
            for b in (0, 1):
                e[ci, a, b] = 0.5 if ((a ^ b) == (x & y)) else 0.0
    return e


def uniform_box():
    return np.full((4, 2, 2), 0.25)


def perfect_corr_box():
    """CF=0 최대상관 비맥락 대조군: 모든 컨텍스트에서 a=b (설정 무관). CHSH=2=국소한계."""
    e = np.zeros((4, 2, 2))
    for ci in range(4):
        e[ci, 0, 0] = 0.5
        e[ci, 1, 1] = 0.5
    return e


def mix(e_sc, e_nc, lam):
    return lam * e_sc + (1.0 - lam) * e_nc


# ===========================================================================
# CF-LP (2,2,2)  — gate1_iat_contextuality.contextual_fraction 와 동일 정식화(전역절편 LP)
#   전역 결정론 배정 g ∈ {0,1}^4 = (A0,A1,B0,B1).  max Σ b_g s.t. ∀(ctx,a,b) Σ_{g일치} b_g ≤ e ; CF=1−maxΣ
# ===========================================================================
def contextual_fraction(e):
    assigns = list(itertools.product((0, 1), repeat=4))   # (A0,A1,B0,B1)
    G = len(assigns)                                       # 16
    A_rows = []
    b_ub = []
    for ci, (x, y) in enumerate(CTX):
        for a in (0, 1):
            for b in (0, 1):
                row = np.zeros(G)
                for gi, g in enumerate(assigns):
                    if g[x] == a and g[2 + y] == b:
                        row[gi] = 1.0
                A_rows.append(row)
                b_ub.append(e[ci, a, b])
    res = linprog(-np.ones(G), A_ub=np.array(A_rows), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * G, method="highs")
    if not res.success:
        return None
    ncf = min(max(float(-res.fun), 0.0), 1.0)
    return float(1.0 - ncf)


# ===========================================================================
# 2큐빗 양자 empirical model (QCBM 계열 소형 파라미터화; 순수 numpy·결정적)
#   state |ψ> = cosθ|00> + sinθ|11> ; 측정 x/y = Bloch 방향(θ,φ) 사영. 국소유니터리 흡수 → 최소 충실 파라미터.
# ===========================================================================
_I2 = np.eye(2, dtype=complex)
_SX = np.array([[0, 1], [1, 0]], dtype=complex)
_SY = np.array([[0, -1j], [1j, 0]], dtype=complex)
_SZ = np.array([[1, 0], [0, -1]], dtype=complex)


def _proj(theta, phi, a):
    n = np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])
    sig = n[0] * _SX + n[1] * _SY + n[2] * _SZ
    return 0.5 * (_I2 + ((-1) ** a) * sig)


def quantum_behavior(p):
    th = p[0]
    psi = np.array([np.cos(th), 0, 0, np.sin(th)], dtype=complex)
    rho = np.outer(psi, psi.conj())
    # 측정각: A0(1,2) A1(3,4) B0(5,6) B1(7,8)
    PA = {(x, a): _proj(p[1 + 2 * x], p[2 + 2 * x], a) for x in (0, 1) for a in (0, 1)}
    PB = {(y, b): _proj(p[5 + 2 * y], p[6 + 2 * y], b) for y in (0, 1) for b in (0, 1)}
    e = np.zeros((4, 2, 2))
    for ci, (x, y) in enumerate(CTX):
        for a in (0, 1):
            for b in (0, 1):
                M = np.kron(PA[(x, a)], PB[(y, b)])
                e[ci, a, b] = max(float(np.real(np.trace(rho @ M))), 0.0)
    e = e / e.sum(axis=(1, 2), keepdims=True)
    return e


def _q_inits(rng, n):
    canon = np.array([np.pi / 4,
                      0.0, 0.0, np.pi / 2, 0.0,          # A0=Z, A1=X
                      np.pi / 4, 0.0, -np.pi / 4, 0.0])  # B0, B1
    inits = [canon]
    for _ in range(n - 1):
        inits.append(rng.uniform(0, np.pi, size=QPARAM))
    return inits


# ===========================================================================
# 고전 잠재(=국소/비맥락) 모델  P(a,b|x,y)=Σ_h π_h P(a|x,h)P(b|y,h)  — H 은닉상태 (파라미터 5H−1)
# ===========================================================================
def classical_behavior(p, H):
    z = np.clip(p[:H], -30, 30)
    pi = np.exp(z - z.max())
    pi = pi / pi.sum()
    u = np.clip(p[H:H + 2 * H], -30, 30).reshape(2, H)          # P(a=1|x,h)
    w = np.clip(p[H + 2 * H:H + 4 * H], -30, 30).reshape(2, H)  # P(b=1|y,h)
    pa1 = 1.0 / (1.0 + np.exp(-u))            # (2=x, H)
    pb1 = 1.0 / (1.0 + np.exp(-w))            # (2=y, H)
    pA = np.stack([1 - pa1, pa1])             # (a, x, H)
    pB = np.stack([1 - pb1, pb1])             # (b, y, H)
    xs = np.array([c[0] for c in CTX]); ys = np.array([c[1] for c in CTX])
    PAx = pA[:, xs, :]                        # (a, ctx, H)
    PBy = pB[:, ys, :]                        # (b, ctx, H)
    e = np.einsum('act,bct,t->cab', PAx, PBy, pi)   # (ctx, a, b)
    return e


# ===========================================================================
# 학습/평가
# ===========================================================================
def sample_emp(e_true, N, rng):
    """N∈{None(=모집단, 타깃 그대로), 정수}. 컨텍스트별 N/4 표본."""
    if N is None:
        return e_true.copy()
    emp = np.zeros((4, 2, 2))
    per = max(N // 4, 1)
    flat = e_true.reshape(4, 4)
    for ci in range(4):
        draws = rng.choice(4, size=per, p=flat[ci])
        for d in draws:
            emp[ci, d // 2, d % 2] += 1.0
    emp += 1.0 / (per * 4.0)                    # 약한 Laplace(0확률 방지)
    emp = emp / emp.sum(axis=(1, 2), keepdims=True)
    return emp


def _ce(emp, model):
    m = np.clip(model, EPS, 1.0)
    return -float(np.sum(emp * np.log(m)))     # cross-entropy = MLE 목적(상수차)


def eval_kl(e_true, model):
    kl = 0.0
    for ci in range(4):
        p = e_true[ci]
        q = np.clip(model[ci], EPS, 1.0)
        mask = p > 0
        kl += float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return kl / 4.0


def fit_quantum(emp, e_true, rng, restarts):
    best = None
    for x0 in _q_inits(rng, restarts):
        r = minimize(lambda p: _ce(emp, quantum_behavior(p)), x0, method="Powell",
                     options=dict(maxiter=2000, xtol=1e-6, ftol=1e-9))
        val = _ce(emp, quantum_behavior(r.x))
        if best is None or val < best[0]:
            best = (val, r.x)
    return eval_kl(e_true, quantum_behavior(best[1]))


def fit_classical(emp, e_true, H, rng, restarts):
    ndim = 5 * H
    best = None
    for _ in range(restarts):
        x0 = rng.normal(0, 1.0, size=ndim)
        r = minimize(lambda p: _ce(emp, classical_behavior(p, H)), x0, method="Powell",
                     options=dict(maxiter=2000, xtol=1e-6, ftol=1e-9))
        val = _ce(emp, classical_behavior(r.x, H))
        if best is None or val < best[0]:
            best = (val, r.x)
    return eval_kl(e_true, classical_behavior(best[1], H))


def cpt_kl(emp, e_true):
    """CPT(암기): 모델=경험분포 그대로(MLE). 파라미터 12."""
    return eval_kl(e_true, emp)


# --- 비맥락 폴리토프 최적적합 (메모리 무제한 국소 은닉변수) : CF=0 이면 KL=0 ---
_NC_VERTS = None


def _nc_vertices():
    """국소 결정론 꼭짓점 16개 D_g[ctx,a,b] (g=(A0,A1,B0,B1))."""
    global _NC_VERTS
    if _NC_VERTS is None:
        V = []
        for g in itertools.product((0, 1), repeat=4):
            D = np.zeros((4, 2, 2))
            for ci, (x, y) in enumerate(CTX):
                D[ci, g[x], g[2 + y]] = 1.0
            V.append(D)
        _NC_VERTS = np.array(V)                    # (16,4,2,2)
    return _NC_VERTS


def nc_floor(emp, e_true, rng, restarts=4):
    """min_w KL(emp‖Σ w_g D_g) over simplex → held-out KL(true‖best-NC). 비맥락(=고전 국소) 하한."""
    V = _nc_vertices().reshape(16, -1)             # (16,16)
    empf = emp.reshape(-1)

    def loss(z):
        w = np.exp(z - z.max()); w = w / w.sum()
        m = np.clip(w @ V, EPS, 1.0)
        return -float(empf @ np.log(m))            # cross-entropy

    best = None
    for _ in range(restarts):
        r = minimize(loss, rng.normal(0, 1.0, size=16), method="Powell",
                     options=dict(maxiter=6000, xtol=1e-7, ftol=1e-10))
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    w = np.exp(best[1] - best[1].max()); w = w / w.sum()
    model = (w @ V).reshape(4, 2, 2)
    return eval_kl(e_true, model)


# ===========================================================================
# 한 (λ, e_SC) 셀 실행: seed 반복 → KL_q, KL_class[H], gap
# ===========================================================================
def run_cell(e_sc, e_nc, lam, N, seeds, restarts_q, restarts_c, Hs):
    e_true = mix(e_sc, e_nc, lam)
    cf = contextual_fraction(e_true)
    rows = {"q": [], "cpt": [], "nc": [], **{f"H{H}": [] for H in Hs}}
    for s in range(seeds):
        rng = np.random.default_rng(SEED + 1000 * s + int(round(lam * 1e4)))
        emp = sample_emp(e_true, N, rng)
        rows["q"].append(fit_quantum(emp, e_true, rng, restarts_q))
        rows["cpt"].append(cpt_kl(emp, e_true))
        rows["nc"].append(nc_floor(emp, e_true, rng, restarts_c + 1))
        for H in Hs:
            rows[f"H{H}"].append(fit_classical(emp, e_true, H, rng, restarts_c))
    agg = {k: dict(mean=float(np.mean(v)), std=float(np.std(v)), vals=[float(x) for x in v])
           for k, v in rows.items()}
    # 매칭메모리(±20%) 셀의 best-classical
    matched = [H for H in Hs if abs(clsparam(H) - QPARAM) / QPARAM <= 0.20]
    if not matched:
        matched = [min(Hs, key=lambda H: abs(clsparam(H) - QPARAM))]
    kl_q = agg["q"]["mean"]
    kl_cm = min(agg[f"H{H}"]["mean"] for H in matched)          # matched best
    kl_call = min(agg[f"H{H}"]["mean"] for H in Hs)             # overall best latent (보수적)
    kl_nc = agg["nc"]["mean"]                                   # 비맥락 폴리토프 하한(무제한 메모리)
    # gap 의 seed 분산(부트스트랩용): matched-best H 의 seed별 − q seed별
    Hbest = min(matched, key=lambda H: agg[f"H{H}"]["mean"])
    gap_seedwise = [agg[f"H{Hbest}"]["vals"][i] - agg["q"]["vals"][i] for i in range(seeds)]
    gap_ctx_seedwise = [agg["nc"]["vals"][i] - agg["q"]["vals"][i] for i in range(seeds)]
    return dict(lam=float(lam), cf=cf, N=("inf" if N is None else N),
                kl_quantum=kl_q, kl_class_matched=kl_cm, kl_class_overall=kl_call, kl_nc_floor=kl_nc,
                matched_H=matched, matched_Hbest=int(Hbest),
                gap_matched=float(kl_cm - kl_q), gap_overall=float(kl_call - kl_q),
                gap_contextual=float(kl_nc - kl_q),
                gap_seedwise=[float(g) for g in gap_seedwise],
                gap_ctx_seedwise=[float(g) for g in gap_ctx_seedwise],
                cpt_kl=agg["cpt"]["mean"], per_model=agg)


# ===========================================================================
# 판정 P1..P3
# ===========================================================================
def bootstrap_threshold(cells, key="gap_ctx_seedwise", nboot=2000, tol=1e-3):
    """gap(λ)>tol 로 처음 넘는 λ_emp: 각 λ의 seed별 gap 재표집 → 임계 λ 분포."""
    lams = np.array([c["lam"] for c in cells])
    order = np.argsort(lams)
    lams = lams[order]
    gseed = [np.array(cells[i][key]) for i in order]
    rng = np.random.default_rng(SEED)
    thr = []
    S = len(gseed[0])
    for _ in range(nboot):
        means = []
        for g in gseed:
            idx = rng.integers(0, S, size=S)
            means.append(g[idx].mean())
        means = np.array(means)
        pos = np.where(means > tol)[0]
        thr.append(float(lams[pos[0]]) if len(pos) else float("nan"))
    thr = np.array(thr)
    thr = thr[~np.isnan(thr)]
    if len(thr) == 0:
        return None
    return dict(lam_emp_median=float(np.median(thr)),
                ci95=[float(np.percentile(thr, 2.5)), float(np.percentile(thr, 97.5))],
                frac_no_threshold=float(1.0 - len(thr) / nboot))


def fit_shape(cells, key="gap_contextual"):
    """CF>0 영역 gap~CF 형태: 선형/구간선형/시그모이드 AIC 비교."""
    xs = np.array([c["cf"] for c in cells])
    ys = np.array([c[key] for c in cells])
    m = xs > 1e-9
    x = xs[m]
    y = ys[m]
    if len(x) < 4:
        return {"note": "CF>0 점 부족"}
    n = len(x)

    def aic(resid, k):
        rss = float(np.sum(resid ** 2))
        rss = max(rss, 1e-12)
        return n * np.log(rss / n) + 2 * k

    # 선형
    A = np.vstack([x, np.ones_like(x)]).T
    cL, *_ = np.linalg.lstsq(A, y, rcond=None)
    aic_lin = aic(y - A @ cL, 2)
    # 구간선형(원점 근방 0 → 꺾임): 최소자승 노드 탐색
    best_pw = None
    for xb in np.linspace(x.min(), x.max(), 15)[1:-1]:
        relu = np.maximum(x - xb, 0.0)
        Ap = np.vstack([relu, np.ones_like(x)]).T
        cP, *_ = np.linalg.lstsq(Ap, y, rcond=None)
        a = aic(y - Ap @ cP, 3)
        if best_pw is None or a < best_pw[0]:
            best_pw = (a, float(xb))
    # 시그모이드
    def sig(p):
        L, k, x0 = p
        return L / (1 + np.exp(-k * (x - x0)))
    r = minimize(lambda p: np.sum((y - sig(p)) ** 2),
                 [y.max() if y.max() > 0 else 1.0, 10.0, float(np.median(x))],
                 method="Nelder-Mead", options=dict(maxiter=5000))
    aic_sig = aic(y - sig(r.x), 3)
    scores = {"linear": aic_lin, "piecewise_linear": best_pw[0], "sigmoid": aic_sig}
    return dict(aic=scores, best=min(scores, key=scores.get),
                linear_slope=float(cL[0]), piecewise_knot_cf=best_pw[1])


# ===========================================================================
# 병렬 실행: 셀은 완전 독립(seed 는 SEED+lam+seed_idx 로 결정 → 순서 무관 결정성 유지).
# ===========================================================================
def _cell_task(t):
    e_sc, e_nc, lam, N, seeds, rq, rc = t
    return run_cell(e_sc, e_nc, lam, N, seeds, rq, rc, H_LADDER)


def run_many(tasks, jobs):
    if jobs <= 1:
        return [_cell_task(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        return list(ex.map(_cell_task, tasks))


def main():
    ap = argparse.ArgumentParser(description="gap-vs-CF 임계곡선 v0. 학습/측정. 단정 금지.")
    ap.add_argument("--gate0", action="store_true", help="λ=1.0 사망관문만 실행")
    ap.add_argument("--smoke", action="store_true", help="극소 격자 빠른 점검")
    ap.add_argument("--full", action="store_true", help="전체 λ 격자(모집단+500+5000)")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--jobs", type=int, default=1, help="병렬 프로세스 수(셀 단위). 1=직렬(결정성 동일)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    e_sc = pr_box()
    e_nc = uniform_box()

    # ---- Gate 0: λ=1.0, CF=1, gap>0 존재? (1σ 이상) ----
    if args.gate0 or args.smoke:
        seeds = 3 if args.smoke else args.seeds
        c = run_cell(e_sc, e_nc, 1.0, None, seeds, 4, 3, H_LADDER)
        g = c["gap_matched"]
        sd = float(np.std(c["gap_seedwise"]))
        alive = (g > sd) and (g > 0)
        log("\n=== GATE 0 (λ=1.0, CF=%.3f) ===" % c["cf"])
        log("  KL_quantum=%.4f  KL_class_matched(H=%s)=%.4f  gap=%.4f (1σ=%.4f)"
            % (c["kl_quantum"], c["matched_Hbest"], c["kl_class_matched"], g, sd))
        log("  판정: %s" % ("PASS (gap>1σ>0) → 곡선 진행 허용" if alive
                            else "FAIL (gap≈0) → 정지. 메모리매칭/과제 재검토"))
        gate0 = dict(cell=c, gap_1sigma=sd, alive=bool(alive))
        if args.gate0:
            json.dump(gate0, open(os.path.join(OUTDIR, "gate0.json"), "w"),
                      indent=1, ensure_ascii=False)
            log("[저장] %s" % os.path.join(OUTDIR, "gate0.json"))
            return
        if not alive:
            log("[STOP] Gate0 실패 — 전체 스윕 생략."); return
    else:
        gate0 = None

    # ---- λ 격자 ----
    base = [round(x, 3) for x in np.arange(0.0, 1.0001, 0.1)]
    dense = [0.45, 0.475, 0.5, 0.525, 0.55, 0.575, 0.6]
    lams = sorted(set([round(l, 3) for l in base + dense]))
    if args.smoke:
        lams = [0.0, 0.5, 0.6, 0.7, 1.0]
        Ns = [None]
        seeds = 3
    else:
        Ns = [None, 5000, 500] if args.full else [None, 5000]
        seeds = args.seeds
    rq, rc = (3, 2) if args.smoke else (3, 2)

    sweeps = {}
    self_check = []
    for N in Ns:
        tag = "inf" if N is None else str(N)
        log("\n===== SWEEP  N=%s  (seeds=%d, jobs=%d) =====" % (tag, seeds, args.jobs))
        cells = run_many([(e_sc, e_nc, lam, N, seeds, rq, rc) for lam in lams], args.jobs)
        for lam, c in zip(lams, cells):
            cf_th = max(0.0, 2 * lam - 1)
            self_check.append(dict(N=tag, lam=lam, cf_lp=c["cf"], cf_theory=cf_th,
                                   diff=abs(c["cf"] - cf_th)))
            log("  λ=%.3f CF=%.3f(이론%.3f) | KLq=%.4f KLc*=%.4f(H%d) | gap=%+.4f (σ%.4f)"
                % (lam, c["cf"], cf_th, c["kl_quantum"], c["kl_class_matched"],
                   c["matched_Hbest"], c["gap_matched"], float(np.std(c["gap_seedwise"]))))
        sweeps[tag] = cells

    # ---- P4 인과대조: e_SC → 완전상관 비맥락상자 ----
    log("\n===== P4 인과대조 (e_SC=완전상관 CF=0 상자) =====")
    e_ctrl = perfect_corr_box()
    ctrl_lams = [0.0, 0.5, 1.0] if args.smoke else [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]
    ctrl_N = None if args.smoke else 5000
    ctrl_cells = run_many([(e_ctrl, e_nc, lam, ctrl_N, seeds, rq, rc) for lam in ctrl_lams], args.jobs)
    for lam, c in zip(ctrl_lams, ctrl_cells):
        log("  λ=%.3f CF=%.3f | gap=%+.4f" % (lam, c["cf"], c["gap_matched"]))

    # ---- 판정 ----
    prim_tag = "inf"
    prim = sweeps[prim_tag]
    # 주지표=gap_contextual(양자 vs 무제한메모리 비맥락 폴리토프): 순수 contextuality 분리, CF=0 이면 0.
    # 보조=gap_matched(양자9 vs 고전잠재9 동일파라미터): 표현력 교란 포함(문헌 letter 그대로).
    p1_ctx = bootstrap_threshold(prim, key="gap_ctx_seedwise")
    p1_mem = bootstrap_threshold(prim, key="gap_seedwise")
    p3 = fit_shape(prim, key="gap_contextual")
    ctrl_maxgap = float(max(abs(c["gap_contextual"]) for c in ctrl_cells))
    verdicts = dict(
        P1_threshold=dict(contextual=p1_ctx, matched_memory=p1_mem,
                          note="contextual=주지표(비맥락 폴리토프 기준). matched_memory=동일 파라미터(표현력 교란 포함)."),
        P2_position=dict(lam_star=0.5,
                         lam_emp_contextual=(p1_ctx["lam_emp_median"] if p1_ctx else None),
                         lam_emp_matched=(p1_mem["lam_emp_median"] if p1_mem else None),
                         reading=("일치→CF가 학습격차 스위치(첫 실증): 주지표 λ_emp≈0.5=CF 경계"
                                  if p1_ctx and abs(p1_ctx["lam_emp_median"] - 0.5) <= 0.06
                                  else ("지연(dead zone: contextuality 존재하나 착취 불가)"
                                        if p1_ctx and p1_ctx["lam_emp_median"] > 0.5 else "미결/조기")),
                         caveat="matched_memory gap 은 CF=0 에서도 양수(표현력 차이) → 순수 contextuality 스위치는 주지표로 판정."),
        P3_shape=p3,
        P4_causal_control=dict(max_abs_gap_contextual=ctrl_maxgap,
                               vanished=bool(ctrl_maxgap < 0.02),
                               reading=("gap 소멸→상관 아니라 contextuality 가 원인"
                                        if ctrl_maxgap < 0.02 else "gap 잔존→상관 교란 존재(그대로 보고)")),
        P5_anchors=dict(KCBS_CF=KCBS_CF, security_CF=SEC_CF,
                        lamQ_isotropic=float(LAM_Q), CF_at_lamQ=float(np.sqrt(2) - 1),
                        note="앵커는 CF 축 참조점(재학습 없음). 명세 CF_max≈0.207 vs 등방계산 √2−1≈0.414 불일치 보고."),
    )

    payload = dict(
        note=("gap-vs-CF v0. (2,2,2) contextuality 노브 e(λ)=λ·PR+(1−λ)·균등. "
              "gap=메모리매칭 고전(국소 은닉변수 잠재모델)−양자(2큐빗) held-out KL. "
              "★모델클래스 분리이지 supremacy 아님. CF-LP=gate1 동일 정식화. 소규모·시뮬·단정 금지."),
        params=dict(seed=SEED, seeds=seeds, lams=lams, Ns=[("inf" if n is None else n) for n in Ns],
                    quantum_params=QPARAM, classical_params={f"H{H}": clsparam(H) for H in H_LADDER},
                    cpt_params=12, lamQ_isotropic=float(LAM_Q),
                    matched_rule="±20% 파라미터 (QPARAM=9 → H=2 =9 정확매칭)"),
        gate0=gate0,
        sweeps=sweeps,
        control_perfect_corr=ctrl_cells,
        self_check_cf=self_check,
        self_check_maxdiff=float(max(s["diff"] for s in self_check)),
        verdicts=verdicts,
        limits=("(2,2,2) 소규모·시뮬레이션·메모리매칭 정의 의존·finite-sample. PR box 는 초양자(비물리)—"
                "λ_Q=1/√2 위는 양자실현 불가. CF-LP 는 상한/근사. 관측 법칙이지 quantum supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s (CF self-check maxdiff=%.4f)" % (OUT_JSON, payload["self_check_maxdiff"]))
    try:
        plot(payload)
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(payload):
    prim = payload["sweeps"]["inf"]
    cf = np.array([c["cf"] for c in prim])
    lam = np.array([c["lam"] for c in prim])
    gap = np.array([c["gap_contextual"] for c in prim])
    gstd = np.array([np.std(c["gap_ctx_seedwise"]) for c in prim])
    gap_mem = np.array([c["gap_matched"] for c in prim])
    order = np.argsort(cf)
    cf_o, gap_o = cf[order], gap[order]

    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # ---- 좌: gap vs CF ----
    cf_lamQ = float(np.sqrt(2) - 1)
    ax[0].axvspan(0, cf_lamQ, color="#DfeeF7", alpha=0.6, label="quantum-realizable (≤ Tsirelson, CF≤√2−1)")
    ax[0].axvspan(cf_lamQ, 1.0, color="#F7E4DF", alpha=0.6, label="super-quantum (PR, nonphysical)")
    ax[0].axvline(0.0, color="k", lw=1, ls=":", label="CF=0 boundary (λ*=0.5)")
    ax[0].errorbar(cf, gap, yerr=gstd, fmt="o-", color="#7B2FBE", ms=6, capsize=3,
                   label="gap_contextual (vs best-NC, N=∞)")
    ax[0].plot(cf, gap_mem, "x--", color="#B08BD9", ms=6, alpha=0.7,
               label="gap_matched (9-param class, incl. expressivity)")
    for N in ("5000", "500"):
        if N in payload["sweeps"]:
            s = payload["sweeps"][N]
            ax[0].plot([c["cf"] for c in s], [c["gap_contextual"] for c in s], "s:", ms=4,
                       alpha=0.6, label="gap_contextual (N=%s)" % N)
    p1 = payload["verdicts"]["P1_threshold"]["contextual"]
    if p1:
        cf_emp = max(0.0, 2 * p1["lam_emp_median"] - 1)
        ax[0].axvline(cf_emp, color="#C1442E", lw=2,
                      label="λ_emp=%.3f (CF=%.3f)" % (p1["lam_emp_median"], cf_emp))
    ax[0].scatter([KCBS_CF], [0], marker="v", s=120, color="#1a7f37", zorder=5, label="KCBS real (CF=0.263)")
    ax[0].scatter([SEC_CF], [0], marker="v", s=120, color="#555", zorder=5, label="security data (CF≈0)")
    # 적합곡선
    sh = payload["verdicts"]["P3_shape"]
    if "linear_slope" in sh:
        xx = np.linspace(0, cf_o.max(), 50)
        ax[0].plot(xx, np.clip(sh["linear_slope"] * xx + (gap_o[cf_o > 0].mean() - sh["linear_slope"] * cf_o[cf_o > 0].mean()) if np.any(cf_o > 0) else 0 * xx, None, None),
                   color="#888", lw=1, ls="-.", alpha=0.6, label="linear fit (slope=%.2f)" % sh["linear_slope"])
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("contextual fraction  CF(λ)  [LP]")
    ax[0].set_ylabel("gap = KL_class(matched) − KL_quantum")
    ax[0].set_title("Learning gap vs contextuality  (2,2,2) knob\nmodel-class separation @ matched memory — NOT supremacy")
    ax[0].legend(fontsize=7, loc="upper left")
    ax[0].grid(alpha=0.3)

    # ---- 우: CF self-check (LP vs 이론) + KL 곡선 ----
    axr = ax[1]
    axr.plot(lam, cf, "o-", color="#3B6EA5", label="CF (our LP)")
    axr.plot(lam, np.maximum(0, 2 * lam - 1), "--", color="#C1442E", label="CF theory max(0,2λ−1)")
    axr.axvline(LAM_Q, color="green", ls=":", label="λ_Q=1/√2 (Tsirelson)")
    axr.set_xlabel("λ (contextuality knob)")
    axr.set_ylabel("CF", color="#3B6EA5")
    axr.legend(fontsize=8, loc="upper left")
    axr.grid(alpha=0.3)
    ax2 = axr.twinx()
    ax2.plot(lam, [c["kl_quantum"] for c in prim], "^-", color="#7B2FBE", alpha=0.8, label="KL quantum")
    ax2.plot(lam, [c["kl_class_matched"] for c in prim], "v-", color="#E2A13B", alpha=0.8, label="KL class(matched)")
    ax2.set_ylabel("held-out KL", color="#7B2FBE")
    ax2.legend(fontsize=8, loc="upper right")
    axr.set_title("CF self-check (LP vs ABM theory) & KL curves")

    plt.suptitle("gap-vs-CF v0 — where does the (memory-matched) quantum/classical learning gap switch on?", fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    plt.close()
    log("[저장] %s" % OUT_PNG)


if __name__ == "__main__":
    main()
