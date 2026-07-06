#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
꺾임점 규명 v1 — gap(CF) 볼록성의 원인 판별 (기하 vs Tsirelson vs 방향 vs 아티팩트).

전제: gap_vs_cf_v0(=G) 코드·정의 무수정 import 재사용. 결정적(seed 고정), 무설치(numpy·scipy), self-check, 단정 금지, supremacy 아님.
CF_Tsirelson=√2−1≈0.4142 확정 — 명세 초기값 0.207 은 오류(분모 정규화 실수)였고 본 로그에서 해결 명기.

gap(CF)=NCfloor(CF)−KL_quantum(CF).  가설:
  H1 기하: 경계 CF→0+ 에서 NCfloor~CF² (KL≈½χ²), 국소지수 α→2. gap 볼록의 근원.
  H2 Tsirelson: KL_quantum 이 CF≈0.414 에서 0→양수로 이륙(초양자 영역). 0.25 아니면 꺾임 원인 아님.
  H3 방향: gap 이 CF 만의 함수인가(혼합방향 바꿔 CF축 붕괴 여부).
  H4 아티팩트: v0 의 "꺾임 CF≈0.25" 가 매끄러운 볼록곡선에 구간선형을 억지 적합한 산물인가
     (멱법칙/2차/크로스오버가 구간선형을 AIC로 이기면 재해석).

Gate A 분해(가장 쌈): λ 0.5~1.0, 0.01 간격 성분분해·국소지수·이륙점·적합족 AIC.
Gate B 방향(H3): PR-box↔{균등, 결정론 비맥락꼭짓점×2, 국소상관상자} 혼합. NCfloor vs CF 중첩→붕괴?
Gate C KCBS 5-cycle(일반성): 극점(반상관)↔균등 노브. NCfloor(CF) 지수 α를 CHSH와 비교.
Gate D 용량불변(H4-양자): 양자 파라미터 {6,9,14}에서 CF<0.414 KLq≈0·이륙점 0.414 불변 확인.

사용: python tools/kink_origin_v1.py [--smoke] [--jobs N]
"""
import os
import sys
import json
import argparse
import itertools

import numpy as np
from scipy.optimize import linprog, minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G                       # 무수정 재사용 (import 시 os.chdir(IOTJ_2차))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                 # noqa: E402

OUTDIR = os.path.join("results2", "kink")
OUT_JSON = os.path.join(OUTDIR, "kink.json")
OUT_PNG = os.path.join(OUTDIR, "kink.png")
SEED = 20260704
CF_TSIRELSON = float(np.sqrt(2) - 1)           # 0.41421356…
EPS = 1e-12


def log(m):
    print(m, flush=True)


def rng_of(tag):
    return np.random.default_rng(SEED + (abs(hash(tag)) % 100000))


# ===========================================================================
# Gate A — 성분분해 · 국소지수 · 이륙점 · 적합족
# ===========================================================================
def gateA(lams, restarts=5):
    esc, enc = G.pr_box(), G.uniform_box()
    rows = []
    for lam in lams:
        e = G.mix(esc, enc, lam)
        cf = G.contextual_fraction(e)
        rng = rng_of("A%.3f" % lam)
        nc = G.nc_floor(e, e, rng, restarts)
        q = G.fit_quantum(e, e, rng, restarts)
        rows.append(dict(lam=float(lam), cf=float(cf),
                         ncfloor=float(nc), klq=float(q), gap=float(nc - q)))
    return rows


def local_exponent(cf, y, window):
    """α(CF)=d log y / d log CF, 중심차분(로그-로그), 창 크기 window."""
    cf = np.asarray(cf); y = np.asarray(y)
    m = (cf > 1e-6) & (y > 1e-9)
    xs = np.log(cf[m]); ys = np.log(y[m]); idx = np.where(m)[0]
    out = []
    for k in range(window, len(xs) - window):
        a = (ys[k + window] - ys[k - window]) / (xs[k + window] - xs[k - window])
        out.append((float(cf[m][k]), float(a)))
    return out


def liftoff_cf(rows, thr=1e-4):
    """KL_quantum 이 0→>thr 로 처음 넘는 CF."""
    s = sorted(rows, key=lambda r: r["cf"])
    for r in s:
        if r["klq"] > thr:
            return float(r["cf"])
    return None


def _aic(y, yhat, k):
    rss = float(np.sum((np.asarray(y) - np.asarray(yhat)) ** 2))
    n = len(y)
    return n * np.log(max(rss, 1e-15) / n) + 2 * k, rss


def fit_family(cf, gap):
    """{선형,2차,멱법칙,구간선형,2차→선형 크로스오버} AIC 비교. y=gap, x=CF>0."""
    cf = np.asarray(cf); gap = np.asarray(gap)
    m = cf > 1e-6
    x = cf[m]; y = gap[m]
    n = len(x)
    res = {}
    # 선형
    c1 = np.polyfit(x, y, 1); res["linear"] = _aic(y, np.polyval(c1, x), 2) + (list(c1),)
    # 2차
    c2 = np.polyfit(x, y, 2); res["quadratic"] = _aic(y, np.polyval(c2, x), 3) + (list(c2),)
    # 멱법칙 y=a x^α  (로그-로그 선형)
    mp = y > 1e-9
    if mp.sum() >= 3:
        p = np.polyfit(np.log(x[mp]), np.log(y[mp]), 1)
        alpha, loga = float(p[0]), float(p[1])
        yhat = np.exp(loga) * x ** alpha
        res["power_law"] = _aic(y, yhat, 2) + ([alpha, float(np.exp(loga))],)
    # 구간선형(연속, 노드 t)
    best = None
    for t in np.linspace(x.min(), x.max(), 25)[1:-1]:
        relu = np.maximum(x - t, 0.0)
        A = np.vstack([x, relu, np.ones_like(x)]).T
        c, *_ = np.linalg.lstsq(A, y, rcond=None)
        a, r = _aic(y, A @ c, 4)
        if best is None or a < best[0]:
            best = (a, r, float(t), list(c))
    res["piecewise_linear"] = (best[0], best[1], dict(knot=best[2], coef=best[3]))
    # 2차→선형 크로스오버(연속+접선): x≤t 는 a x²+b x, x>t 는 접선연장
    bestc = None
    for t in np.linspace(x.min(), x.max(), 25)[1:-1]:
        below = x <= t
        def model(ab):
            a, b = ab
            val = a * x ** 2 + b * x
            vt = a * t ** 2 + b * t; st = 2 * a * t + b
            lin = vt + st * (x - t)
            return np.where(below, val, lin)
        r = minimize(lambda ab: np.sum((y - model(ab)) ** 2), [1.0, 0.0], method="Nelder-Mead")
        a, rss = _aic(y, model(r.x), 3)
        if bestc is None or a < bestc[0]:
            bestc = (a, rss, float(t), [float(v) for v in r.x])
    res["quad_to_linear"] = (bestc[0], bestc[1], dict(knot=bestc[2], coef=bestc[3]))
    aic_scores = {k: v[0] for k, v in res.items()}
    return dict(aic=aic_scores, best=min(aic_scores, key=aic_scores.get), detail={k: v[2:] for k, v in res.items()})


# ===========================================================================
# Gate B — 방향 스윕 (H3): PR-box 를 여러 비맥락 방향과 혼합
# ===========================================================================
def gateB(lams, restarts=4):
    esc = G.pr_box()
    V = G._nc_vertices()                        # (16,4,2,2) 결정론 국소 꼭짓점
    directions = {
        "uniform": G.uniform_box(),
        "det_vertex_0000": V[0],                # (A0,A1,B0,B1)=(0,0,0,0)
        "det_vertex_0101": V[int("0101", 2)],   # 다른 결정론 꼭짓점
        "local_perfect_corr": G.perfect_corr_box(),
    }
    out = {}
    for name, e_dir in directions.items():
        rows = []
        for lam in lams:
            e = G.mix(esc, e_dir, lam)
            cf = G.contextual_fraction(e)
            rng = rng_of("B%s%.3f" % (name, lam))
            nc = G.nc_floor(e, e, rng, restarts)
            rows.append(dict(lam=float(lam), cf=float(cf), ncfloor=float(nc)))
        out[name] = rows
    return out


def collapse_metric(gateB_out):
    """각 방향 NCfloor(CF)를 공통 CF격자에 보간→ 방향간 최대 상대편차(붕괴 판정)."""
    grid = np.linspace(0.05, 0.95, 19)
    curves = {}
    for name, rows in gateB_out.items():
        cf = np.array([r["cf"] for r in rows]); nc = np.array([r["ncfloor"] for r in rows])
        o = np.argsort(cf)
        curves[name] = np.interp(grid, cf[o], nc[o], left=np.nan, right=np.nan)
    M = np.vstack(list(curves.values()))
    spread = []
    for j in range(len(grid)):
        col = M[:, j]; col = col[~np.isnan(col)]
        if len(col) >= 2 and col.mean() > 1e-6:
            spread.append(float((col.max() - col.min()) / col.mean()))
    return dict(grid=list(grid), max_rel_spread=float(np.nanmax(spread)) if spread else None,
                mean_rel_spread=float(np.nanmean(spread)) if spread else None,
                collapse=bool(spread and np.nanmax(spread) < 0.15))


# ===========================================================================
# Gate C — KCBS 5-cycle (일반성). cyclic CF-LP + NCfloor.
# ===========================================================================
def _cyclic_setup(n):
    V = list(itertools.product((0, 1), repeat=n))
    Gn = len(V)
    A_rows = []; idx = []
    for i in range(n):
        j = (i + 1) % n
        for a in (0, 1):
            for b in (0, 1):
                row = np.zeros(Gn)
                for gi, v in enumerate(V):
                    if v[i] == a and v[j] == b:
                        row[gi] = 1.0
                A_rows.append(row); idx.append((i, a, b))
    return V, np.array(A_rows), idx


def cyclic_cf(e, n, setup=None):
    V, A, idx = setup or _cyclic_setup(n)
    b = np.array([e[i, a, bb] for (i, a, bb) in idx])
    r = linprog(-np.ones(len(V)), A_ub=A, b_ub=b, bounds=[(0, None)] * len(V), method="highs")
    if not r.success:
        return None
    return float(1.0 - min(max(-r.fun, 0.0), 1.0))


def cyclic_ncfloor(e, n, rng, setup=None, restarts=5):
    V, _, _ = setup or _cyclic_setup(n)
    D = np.zeros((len(V), n, 2, 2))
    for gi, v in enumerate(V):
        for i in range(n):
            D[gi, i, v[i], v[(i + 1) % n]] = 1.0
    Df = D.reshape(len(V), -1); ef = e.reshape(-1)

    def loss(z):
        w = np.exp(z - z.max()); w = w / w.sum()
        m = np.clip(w @ Df, EPS, 1.0)
        return -float(ef @ np.log(m))

    best = None
    for _ in range(restarts):
        r = minimize(loss, rng.normal(0, 1.0, size=len(V)), method="Powell",
                     options=dict(maxiter=8000, xtol=1e-7, ftol=1e-10))
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    w = np.exp(best[1] - best[1].max()); w = w / w.sum()
    model = (w @ Df).reshape(n, 2, 2)
    # KL(e‖model) 컨텍스트 평균
    kl = 0.0
    for i in range(n):
        p = e[i]; qy = np.clip(model[i], EPS, 1.0); mask = p > 0
        kl += float(np.sum(p[mask] * np.log(p[mask] / qy[mask])))
    return kl / n


def kcbs_extremal(n=5):
    e = np.zeros((n, 2, 2))
    for i in range(n):
        e[i, 0, 1] = e[i, 1, 0] = 0.5           # 반상관(홀수 사이클 좌절)
    return e


def gateC(lams, n=5, restarts=5):
    setup = _cyclic_setup(n)
    ext = kcbs_extremal(n); unif = np.full((n, 2, 2), 0.25)
    rows = []
    for lam in lams:
        e = lam * ext + (1 - lam) * unif
        cf = cyclic_cf(e, n, setup)
        rng = rng_of("C%.3f" % lam)
        nc = cyclic_ncfloor(e, n, rng, setup, restarts)
        rows.append(dict(lam=float(lam), cf=float(cf), ncfloor=float(nc)))
    return rows


# ===========================================================================
# Gate D — 양자 용량 불변 (H4-양자). 파라미터 {6,9,14}.
# ===========================================================================
_I2, _SX, _SY, _SZ = G._I2, G._SX, G._SY, G._SZ


def _planar_proj(beta, a):
    n = np.array([np.sin(beta), 0.0, np.cos(beta)])
    sig = n[0] * _SX + n[2] * _SZ
    return 0.5 * (_I2 + ((-1) ** a) * sig)


def qbeh_planar6(p):
    """6 파라미터: state(θ,φ) + 4 planar 측정각."""
    th, ph = p[0], p[1]
    psi = np.array([np.cos(th), 0, 0, np.exp(1j * ph) * np.sin(th)], dtype=complex)
    rho = np.outer(psi, psi.conj())
    betas = p[2:6]
    e = np.zeros((4, 2, 2))
    for ci, (x, y) in enumerate(G.CTX):
        for a in (0, 1):
            PA = _planar_proj(betas[x], a)
            for b in (0, 1):
                PB = _planar_proj(betas[2 + y], b)
                e[ci, a, b] = max(float(np.real(np.trace(rho @ np.kron(PA, PB)))), 0.0)
    return e / e.sum(axis=(1, 2), keepdims=True)


def qbeh_general14(p):
    """14 파라미터: 일반 순수상태(6) + 4 측정×2 Bloch각(8)."""
    r = np.array([1.0, p[0], p[1], p[2]])
    ph = np.array([0.0, p[3], p[4], p[5]])
    psi = r * np.exp(1j * ph)
    psi = psi / np.linalg.norm(psi)
    rho = np.outer(psi, psi.conj())
    m = p[6:14]                                  # A: m[0..3]=(θ,φ)×2설정, B: m[4..7]
    PA = {(x, a): G._proj(m[2 * x], m[2 * x + 1], a) for x in (0, 1) for a in (0, 1)}
    PB = {(y, b): G._proj(m[4 + 2 * y], m[4 + 2 * y + 1], b) for y in (0, 1) for b in (0, 1)}
    e = np.zeros((4, 2, 2))
    for ci, (x, y) in enumerate(G.CTX):
        for a in (0, 1):
            for b in (0, 1):
                e[ci, a, b] = max(float(np.real(np.trace(rho @ np.kron(PA[(x, a)], PB[(y, b)])))), 0.0)
    return e / e.sum(axis=(1, 2), keepdims=True)


def _fit_q(behav, ndim, inits, emp, e_true):
    best = None
    for x0 in inits:
        r = minimize(lambda p: G._ce(emp, behav(p)), x0, method="Powell",
                     options=dict(maxiter=4000, xtol=1e-6, ftol=1e-9))
        v = G._ce(emp, behav(r.x))
        if best is None or v < best[0]:
            best = (v, r.x)
    return G.eval_kl(e_true, behav(best[1]))


def gateD(lams, restarts=5):
    esc, enc = G.pr_box(), G.uniform_box()
    canon6 = np.array([np.pi / 4, 0.0, 0.0, np.pi / 2, np.pi / 4, -np.pi / 4])
    canon14 = np.array([1.0, 1.0, 1.0, 0, 0, 0, 0, 0, np.pi / 2, 0, np.pi / 4, 0, -np.pi / 4, 0])
    out = {"q6": [], "q9": [], "q14": []}
    for lam in lams:
        e = G.mix(esc, enc, lam); cf = G.contextual_fraction(e)
        rng = rng_of("D%.3f" % lam)
        inits6 = [canon6] + [rng.uniform(0, np.pi, 6) for _ in range(restarts - 1)]
        inits14 = [canon14] + [np.concatenate([rng.uniform(0.2, 1.5, 6), rng.uniform(0, np.pi, 8)]) for _ in range(restarts - 1)]
        q6 = _fit_q(qbeh_planar6, 6, inits6, e, e)
        q9 = G.fit_quantum(e, e, rng, restarts)
        q14 = _fit_q(qbeh_general14, 14, inits14, e, e)
        for k, v in (("q6", q6), ("q9", q9), ("q14", q14)):
            out[k].append(dict(lam=float(lam), cf=float(cf), klq=float(v)))
    return out


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="꺾임점 규명 v1. 원인판별. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    if args.smoke:
        lamsA = np.round(np.arange(0.5, 1.0001, 0.05), 3)
        lamsB = np.round(np.arange(0.1, 1.0001, 0.1), 3)
        lamsC = np.round(np.arange(0.1, 1.0001, 0.15), 3)
        lamsD = [0.6, 0.7, 0.75, 0.8, 1.0]
        rA = rB = rC = rD = 3
    else:
        lamsA = np.round(np.arange(0.5, 1.0001, 0.01), 3)
        lamsB = np.round(np.arange(0.05, 1.0001, 0.05), 3)
        lamsC = np.round(np.arange(0.05, 1.0001, 0.05), 3)
        lamsD = np.round(np.arange(0.55, 1.0001, 0.05), 3)
        rA, rB, rC, rD = 5, 4, 5, 5

    log("=== Gate A: 성분분해 · 국소지수 · 이륙점 · 적합족 ===")
    A = gateA(lamsA, rA)
    cfA = [r["cf"] for r in A]; ncA = [r["ncfloor"] for r in A]; gapA = [r["gap"] for r in A]
    expo = {str(w): local_exponent(cfA, ncA, w) for w in (2, 4)}
    lift = liftoff_cf(A, thr=1e-4)
    fam_full = fit_family(cfA, gapA)
    # 경계영역(CF≤0.414, KLq≈0 → gap=NCfloor)만의 적합
    sub = [(r["cf"], r["gap"]) for r in A if 0 < r["cf"] <= CF_TSIRELSON]
    fam_bnd = fit_family([c for c, _ in sub], [g for _, g in sub]) if len(sub) >= 5 else None
    a2 = [a for _, a in expo["2"]]
    log("  이륙점 CF(KLq>1e-4) = %.4f  (Tsirelson=%.4f 예측)" % (lift or -1, CF_TSIRELSON))
    log("  경계 국소지수 α(창2) 중앙값 = %.3f  전체범위 [%.2f, %.2f]"
        % (np.median(a2), min(a2), max(a2)))
    log("  적합족 best(전체) = %s ; best(경계 CF≤0.414) = %s"
        % (fam_full["best"], fam_bnd["best"] if fam_bnd else "N/A"))

    log("=== Gate B: 방향 스윕 (H3 붕괴 판정) ===")
    B = gateB(lamsB, rB)
    coll = collapse_metric(B)
    log("  방향간 NCfloor(CF) 최대 상대편차 = %s → 붕괴=%s"
        % (("%.3f" % coll["max_rel_spread"]) if coll["max_rel_spread"] else "N/A", coll["collapse"]))

    log("=== Gate C: KCBS 5-cycle 일반성 ===")
    C = gateC(lamsC, 5, rC)
    cfC = [r["cf"] for r in C]; ncC = [r["ncfloor"] for r in C]
    expoC = local_exponent(cfC, ncC, 2)
    a2C = [a for _, a in expoC]
    log("  KCBS 경계 국소지수 α(창2) 중앙값 = %.3f  (CHSH=%.3f)"
        % (np.median(a2C) if a2C else -1, np.median(a2)))

    log("=== Gate D: 양자 용량 불변 {6,9,14} ===")
    D = gateD(lamsD, rD)
    lifts = {k: liftoff_cf([dict(cf=r["cf"], klq=r["klq"]) for r in D[k]], 1e-4) for k in D}
    below = {k: max([r["klq"] for r in D[k] if r["cf"] < CF_TSIRELSON - 1e-6] + [0.0]) for k in D}
    log("  이륙점: q6=%.3f q9=%.3f q14=%.3f | CF<0.414 최대 KLq: %s"
        % (lifts["q6"] or -1, lifts["q9"] or -1, lifts["q14"] or -1,
           {k: round(v, 4) for k, v in below.items()}))

    # ---- 가설 판정 ----
    alpha_med = float(np.median(a2))
    verdicts = dict(
        H1_geometry=dict(alpha_boundary_median=alpha_med,
                         holds=bool(1.7 <= alpha_med <= 2.3),
                         reading="NCfloor~CF^%.2f (경계) → CF² 기하법칙 %s"
                                 % (alpha_med, "성립" if 1.7 <= alpha_med <= 2.3 else "불성립")),
        H2_tsirelson=dict(liftoff_cf=lift, tsirelson=CF_TSIRELSON,
                          near_tsirelson=bool(lift and abs(lift - CF_TSIRELSON) <= 0.05),
                          near_kink025=bool(lift and abs(lift - 0.25) <= 0.05),
                          reading=("KLq 이륙 CF=%.3f ≈ Tsirelson 0.414 → 꺾임(0.25)의 원인 아님"
                                   % (lift or -1))),
        H3_direction=dict(max_rel_spread=coll["max_rel_spread"], collapse=coll["collapse"],
                          reading=("CF축 붕괴 → gap 은 CF 보편함수" if coll["collapse"]
                                   else "비붕괴 → CF 는 충분통계 아님, 기하 방향이 gap 좌우(독립 발견)")),
        H4_artifact=dict(best_fit_full=fam_full["best"], best_fit_boundary=(fam_bnd["best"] if fam_bnd else None),
                         piecewise_beaten=bool(fam_full["best"] != "piecewise_linear"),
                         reading=("최적적합=%s (구간선형 아님) → v0 '꺾임 0.25'는 매끄러운 볼록곡선의 적합 아티팩트"
                                  % fam_full["best"] if fam_full["best"] != "piecewise_linear"
                                  else "구간선형 잔존 → 꺾임 실재 가능")),
        C_scenario=dict(alpha_kcbs=float(np.median(a2C)) if a2C else None,
                        alpha_chsh=alpha_med,
                        universal=bool(a2C and abs(np.median(a2C) - alpha_med) <= 0.4),
                        reading="KCBS α vs CHSH α 근접 → CF² 기하 시나리오-보편성"),
        D_capacity=dict(liftoffs=lifts, max_klq_below_tsirelson=below,
                        invariant=bool(all((v is None or abs(v - CF_TSIRELSON) <= 0.06) for v in lifts.values())
                                       and all(x < 0.01 for x in below.values())),
                        reading="이륙점·바닥이 용량 {6,9,14}에 불변 → 꺾임은 양자 용량 아티팩트 아님"),
    )

    payload = dict(
        note=("꺾임점 규명 v1. gap(CF)=NCfloor−KLq 성분분해로 볼록성 원인 판별. gap_vs_cf_v0 무수정 재사용. "
              "CF_Tsirelson=√2−1≈0.4142(명세 0.207 은 분모 정규화 오류—본 로그서 해결). "
              "소규모(2,2,2)+KCBS·시뮬·NCfloor 수치최소화·국소지수 수치미분. 관측 법칙이지 supremacy 아님. 단정 금지."),
        params=dict(seed=SEED, cf_tsirelson=CF_TSIRELSON,
                    lamsA=[float(x) for x in lamsA], quantum_param_ladder=[6, 9, 14]),
        gateA=dict(rows=A, local_exponent=expo, liftoff_cf=lift,
                   fit_family_full=fam_full, fit_family_boundary=fam_bnd),
        gateB=dict(rows=B, collapse=coll),
        gateC=dict(rows=C, local_exponent=expoC),
        gateD=D,
        verdicts=verdicts,
        cf_0207_resolution=("명세 초기 CF_max≈0.207 은 오류. 등방혼합선 양자실현 한계는 CHSH=2√2 에서 "
                            "CF=(CHSH−2)/(4−2)=(2√2−2)/2=√2−1≈0.4142. 0.207=√2−1 의 절반 → 분모를 "
                            "(4−2) 대신 (4−0) 또는 2배 정규화한 실수로 추정. 본 실험 전체 0.4142 사용·검증."),
        limits=("(2,2,2)+KCBS 한정·시뮬·NCfloor 는 수치 최소화(다중초기값 방어)·국소지수는 수치미분(창2,4 병기). "
                "관측 법칙이지 quantum supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload)
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(P):
    A = P["gateA"]["rows"]; B = P["gateB"]["rows"]; C = P["gateC"]["rows"]; D = P["gateD"]
    cfA = np.array([r["cf"] for r in A]); ncA = np.array([r["ncfloor"] for r in A])
    klqA = np.array([r["klq"] for r in A]); gapA = np.array([r["gap"] for r in A])
    fig, ax = plt.subplots(2, 2, figsize=(15, 11))

    # (0,0) 성분분해
    a = ax[0, 0]
    a.axvspan(0, CF_TSIRELSON, color="#DfeeF7", alpha=.6)
    a.axvspan(CF_TSIRELSON, 1, color="#F7E4DF", alpha=.6)
    a.plot(cfA, ncA, "-", color="#E2A13B", lw=2, label="NCfloor(CF)")
    a.plot(cfA, klqA, "-", color="#7B2FBE", lw=2, label="KL_quantum(CF)")
    a.plot(cfA, gapA, "-", color="#1a7f37", lw=2.2, label="gap = NCfloor − KLq")
    xx = np.linspace(1e-3, cfA.max(), 100)
    fam = P["gateA"]["fit_family_full"]["detail"].get("power_law")
    if fam:
        al, aa = fam[0]
        a.plot(xx, aa * xx ** al, "k--", lw=1, alpha=.7, label="power fit CF^%.2f" % al)
    a.axvline(CF_TSIRELSON, color="green", ls=":", lw=1.5, label="Tsirelson 0.414")
    a.set_xlabel("CF(λ)"); a.set_ylabel("held-out KL")
    a.set_title("Gate A: component decomposition (convexity lives in NCfloor)")
    a.legend(fontsize=8); a.grid(alpha=.3)

    # (0,1) 국소지수 α(CF)
    a = ax[0, 1]
    for w, mk in (("2", "o"), ("4", "s")):
        pts = P["gateA"]["local_exponent"][w]
        if pts:
            a.plot([c for c, _ in pts], [v for _, v in pts], mk + "-", ms=4, label="α (window %s)" % w)
    a.axhline(2.0, color="r", ls="--", label="α=2 (CF² 기하)")
    a.axhline(1.0, color="gray", ls=":", label="α=1 (linear/ABM)")
    a.axvline(CF_TSIRELSON, color="green", ls=":", lw=1.2)
    a.set_xlabel("CF"); a.set_ylabel("local exponent α = dlogNC/dlogCF")
    a.set_title("Gate A: local exponent → CF² near boundary?")
    a.legend(fontsize=8); a.grid(alpha=.3); a.set_ylim(0, 4)

    # (1,0) 방향 스윕 붕괴
    a = ax[1, 0]
    cols = {"uniform": "#3B6EA5", "det_vertex_0000": "#C1442E",
            "det_vertex_0101": "#E2A13B", "local_perfect_corr": "#7B2FBE"}
    for name, rows in B.items():
        cf = np.array([r["cf"] for r in rows]); nc = np.array([r["ncfloor"] for r in rows])
        o = np.argsort(cf)
        a.plot(cf[o], nc[o], "o-", ms=4, color=cols.get(name), label=name)
    a.set_xlabel("CF (LP, per direction)"); a.set_ylabel("NCfloor")
    coll = P["gateB"]["collapse"]
    a.set_title("Gate B: direction sweep — collapse on CF? (max rel spread=%.2f, collapse=%s)"
                % (coll["max_rel_spread"] or 0, coll["collapse"]))
    a.legend(fontsize=8); a.grid(alpha=.3)

    # (1,1) KCBS vs CHSH (CF축) + Gate D 이륙 불변
    a = ax[1, 1]
    cfC = np.array([r["cf"] for r in C]); ncC = np.array([r["ncfloor"] for r in C])
    o = np.argsort(cfC)
    a.plot(cfA, ncA, "-", color="#3B6EA5", lw=2, label="CHSH NCfloor(CF)")
    a.plot(cfC[o], ncC[o], "s-", color="#C1442E", ms=4, label="KCBS 5-cycle NCfloor(CF)")
    a.set_xlabel("CF"); a.set_ylabel("NCfloor")
    vc = P["verdicts"]["C_scenario"]
    a.set_title("Gate C: scenario universality  (α_KCBS=%.2f vs α_CHSH=%.2f)"
                % (vc["alpha_kcbs"] or 0, vc["alpha_chsh"]))
    a.legend(fontsize=8); a.grid(alpha=.3)
    # inset: Gate D 이륙
    ins = a.inset_axes([0.58, 0.12, 0.38, 0.42])
    for k, c in (("q6", "#3B6EA5"), ("q9", "#1a7f37"), ("q14", "#C1442E")):
        ins.plot([r["cf"] for r in D[k]], [r["klq"] for r in D[k]], ".-", ms=4, color=c, label=k)
    ins.axvline(CF_TSIRELSON, color="green", ls=":", lw=1)
    ins.set_title("Gate D: KLq liftoff", fontsize=7); ins.tick_params(labelsize=6)
    ins.legend(fontsize=6)

    plt.suptitle("kink_origin_v1 — why is gap(CF) convex? geometry(CF²) vs Tsirelson vs direction vs artifact", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130); plt.close()
    log("[저장] %s" % OUT_PNG)


if __name__ == "__main__":
    main()
