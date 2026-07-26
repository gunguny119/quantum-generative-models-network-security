#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP C (gate_s5) — 다중 패싯 / 미감시 방향: 불가능성 정리 접점 논증이 실제 DI-QKD 기하((2,3,2))에서도 성립하는가.

동기: 실제 DI-QKD 는 (2,2,2)가 아님. 앨리스는 A₀,A₁(CHSH)+A₂(키생성). **A₂는 S 계산에 안 들어감** →
  S 의 등위면에 A₂ 방향 제약이 없어 **고전껍질이 A₂ 방향으로 삐져나올 수 있음** → abstract 원래 약속
  (양자 고유 탐지 우위) 복구 후보. (2,2,2) 불가능성의 접점 논증은 CF=(S−2)/2 아핀성에 의존.

정합: R1′ (2,3,2)는 신규 LP·파라미터화 필요(G 트랙 확장, K.cyclic_* 미사용). R2 신규 변환기 build_box_23.
  R6 커버리지 우위 주장 금지. R7 동일데이터.

구성(신규): contextual_fraction_23(64꼭짓점 LP), nc_floor_23, quantum_behavior_23(13 파라미터),
  build_box_23(36셀), chsh_S_sub(9컨텍스트 중 CHSH 4개만). ★self-check: 3번째 설정=1번째면 (2,2,2)로 축약.

사용:
  python tools/gate_s5_multifacet.py               # C-1: 인프라 self-check (축약 검증)
  python tools/gate_s5_multifacet.py --experiment  # C-2: 미감시 방향 껍질 탐색(사람 확인 후)
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
import gap_vs_cf_v0 as G                             # _proj, contextual_fraction, quantum_behavior(2,2,2)
import gate_r2_anchor_strengthen as R2
import gate_s2_detection as D2                       # auc (Mann-Whitney)
import fastq

FASTQ23 = None

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                      # noqa: E402

OUTDIR = os.path.join("results2", "gate_s5")
OUT_JSON = os.path.join(OUTDIR, "s5.json")
OUT_PNG = os.path.join(OUTDIR, "s5_shell.png")
SEED = 20260725
EPS = 1e-12
CTX9 = [(x, y) for x in range(3) for y in range(3)]   # 9 컨텍스트
CHSH_IDX = {(0, 0): 0, (0, 1): 1, (1, 0): 3, (1, 1): 4}   # CTX9 중 CHSH 4개


def log(m):
    print(m, flush=True)


# ===========================================================================
# (2,3,2) CF-LP  — 64 결정론 꼭짓점 (A0,A1,A2,B0,B1,B2)
# ===========================================================================
def contextual_fraction_23(e):
    assigns = list(itertools.product((0, 1), repeat=6))   # 64
    Gn = 64
    A_rows = []
    b_ub = []
    for ci, (x, y) in enumerate(CTX9):
        for a in (0, 1):
            for b in (0, 1):
                row = np.zeros(Gn)
                for gi, g in enumerate(assigns):
                    if g[x] == a and g[3 + y] == b:
                        row[gi] = 1.0
                A_rows.append(row)
                b_ub.append(e[ci, a, b])
    res = linprog(-np.ones(Gn), A_ub=np.array(A_rows), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * Gn, method="highs")
    if not res.success:
        return None
    ncf = min(max(float(-res.fun), 0.0), 1.0)
    return float(1.0 - ncf)


_NC23 = None


def _nc_vertices_23():
    global _NC23
    if _NC23 is None:
        V = []
        for g in itertools.product((0, 1), repeat=6):
            D = np.zeros((9, 2, 2))
            for ci, (x, y) in enumerate(CTX9):
                D[ci, g[x], g[3 + y]] = 1.0
            V.append(D)
        _NC23 = np.array(V)                            # (64,9,2,2)
    return _NC23


def eval_kl_23(e_true, model):
    kl = 0.0
    for ci in range(9):
        p = e_true[ci]
        q = np.clip(model[ci], EPS, 1.0)
        mask = p > 0
        kl += float(np.sum(p[mask] * np.log(p[mask] / q[mask])))
    return kl / 9.0


def nc_floor_23(emp, e_true, rng, restarts=4):
    V = _nc_vertices_23().reshape(64, -1)
    empf = emp.reshape(-1)

    def loss(z):
        w = np.exp(z - z.max()); w = w / w.sum()
        m = np.clip(w @ V, EPS, 1.0)
        return -float(empf @ np.log(m))

    best = None
    for _ in range(restarts):
        r = minimize(loss, rng.normal(0, 1.0, size=64), method="Powell",
                     options=dict(maxiter=8000, xtol=1e-7, ftol=1e-10))
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    w = np.exp(best[1] - best[1].max()); w = w / w.sum()
    model = (w @ V).reshape(9, 2, 2)
    return eval_kl_23(e_true, model), (w @ V).reshape(9, 2, 2)


# ===========================================================================
# (2,3,2) 양자 behavior — 상태 1 + party 3설정×(θ,φ) = 13 파라미터
#   p = [θ_state, A0θ,A0φ, A1θ,A1φ, A2θ,A2φ, B0θ,B0φ, B1θ,B1φ, B2θ,B2φ]
# ===========================================================================
def quantum_behavior_23(p):
    th = p[0]
    psi = np.array([np.cos(th), 0, 0, np.sin(th)], dtype=complex)
    rho = np.outer(psi, psi.conj())
    PA = {(x, a): G._proj(p[1 + 2 * x], p[2 + 2 * x], a) for x in (0, 1, 2) for a in (0, 1)}
    PB = {(y, b): G._proj(p[7 + 2 * y], p[8 + 2 * y], b) for y in (0, 1, 2) for b in (0, 1)}
    e = np.zeros((9, 2, 2))
    for ci, (x, y) in enumerate(CTX9):
        for a in (0, 1):
            for b in (0, 1):
                M = np.kron(PA[(x, a)], PB[(y, b)])
                e[ci, a, b] = max(float(np.real(np.trace(rho @ M))), 0.0)
    e = e / e.sum(axis=(1, 2), keepdims=True)
    return e


def build_box_23(a, b, xo, yo):
    """(setting a,b∈{0,1,2}; outcome ±1) → e[9,2,2]. 반환 box, per-ctx N."""
    box = np.zeros((9, 2, 2)); Ns = np.zeros(9, int)
    for ci, (sa, sb) in enumerate(CTX9):
        m = (a == sa) & (b == sb)
        Ns[ci] = int(m.sum())
        for xv in (+1, -1):
            for yv in (+1, -1):
                box[ci, 0 if xv == 1 else 1, 0 if yv == 1 else 1] = int((m & (xo == xv) & (yo == yv)).sum())
    box_n = box / np.clip(box.sum(axis=(1, 2), keepdims=True), 1, None)
    return box_n, Ns


def chsh_S_sub(e):
    """CHSH S: CTX9 중 (x,y∈{0,1}) 4개만 사용(A₂/B₂ 무시)."""
    E = {}
    for (x, y), ci in CHSH_IDX.items():
        m = e[ci]
        E[(x, y)] = float(m[0, 0] - m[0, 1] - m[1, 0] + m[1, 1])
    return E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)]


CANON_23 = np.array([np.pi / 4,
                     0.0, 0.0, np.pi / 2, 0.0, 0.0, 0.0,        # A0=Z, A1=X, A2=Z(=A0, 키생성)
                     np.pi / 4, 0.0, -np.pi / 4, 0.0, np.pi / 4, 0.0])   # B0, B1, B2=B0


def werner_23(p, Vvis):
    return Vvis * quantum_behavior_23(p) + (1.0 - Vvis) * np.full((9, 2, 2), 0.25)


# ===========================================================================
# C-1 self-check: (2,3,2) → (2,2,2) 축약
# ===========================================================================
def run_self_check():
    checks = {}
    # (1) 좌절/균등
    frust = np.zeros((9, 2, 2))                          # PR-유사: 각 컨텍스트 반대칭? 대신 균등·완전상관 검사
    uni = np.full((9, 2, 2), 0.25)
    checks["uniform_cf23"] = dict(value=contextual_fraction_23(uni), pass_=bool(abs(contextual_fraction_23(uni)) < 1e-9))

    # (2) 축약: A2=A0,B2=B0 인 (2,3,2) 양자박스 → CF23 == (2,2,2) canon CF
    #     canon_23 은 A2=A0(=Z), B2=B0(=45°). 3번째 설정이 1번째 복제 → 맥락성 불변.
    e23 = quantum_behavior_23(CANON_23)
    cf23 = contextual_fraction_23(e23)
    # 대응 (2,2,2) canon (동일 A0,A1,B0,B1)
    canon22 = np.array([np.pi / 4, 0.0, 0.0, np.pi / 2, 0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0])
    e22 = G.quantum_behavior(canon22)
    cf22 = G.contextual_fraction(e22)
    checks["reduction_cf"] = dict(cf23=float(cf23), cf22=float(cf22), diff=float(abs(cf23 - cf22)),
                                  pass_=bool(abs(cf23 - cf22) < 1e-3))

    # (3) chsh_S_sub 가 (2,2,2) S 와 일치
    S23 = chsh_S_sub(e23); S22 = R2.chsh_S(e22)
    checks["reduction_S"] = dict(S23=float(S23), S22=float(S22), diff=float(abs(S23 - S22)),
                                 pass_=bool(abs(S23 - S22) < 1e-6))

    # (4) NCfloor 정확성(올바른 불변량): 국소상자→NCfloor≈0, 맥락상자(canon)→NCfloor>0.
    #     ※ '축약=NCfloor22 일치'는 성립 안 함 — A2=A0,B2=B0 임베딩은 9컨텍스트 중복((0,0)×4,(0,1)×2,(1,0)×2,(1,1)×1)을
    #        만들어 eval_kl_23 의 /9 가중이 eval_kl_22 의 /4 와 다름(CHSH-위반 (1,1) 가중↓). 값 차이는 가중의 정확한 귀결.
    nc_uni, _ = nc_floor_23(uni, uni, np.random.default_rng(SEED + 3), 6)          # 균등=국소혼합 → ≈0
    w_rand = np.random.default_rng(SEED + 4).dirichlet(np.ones(64))
    e_local = (w_rand @ _nc_vertices_23().reshape(64, -1)).reshape(9, 2, 2)         # 임의 NC 혼합 → NCfloor≈0
    nc_loc, _ = nc_floor_23(e_local, e_local, np.random.default_rng(SEED + 5), 6)
    nc23_ctx, _ = nc_floor_23(e23, e23, np.random.default_rng(SEED + 1), 6)         # canon(맥락) → >0
    checks["ncfloor_local_zero"] = dict(uniform=float(nc_uni), random_nc=float(nc_loc),
                                        pass_=bool(nc_uni < 1e-4 and nc_loc < 1e-4))
    checks["ncfloor_contextual_positive"] = dict(canon=float(nc23_ctx),
                                                 pass_=bool(nc23_ctx > 1e-3))

    # (5) build_box_23 왕복
    rng = np.random.default_rng(SEED + 7)
    N = 60000
    ci_all = rng.integers(0, 9, size=N)
    a = np.array([CTX9[c][0] for c in ci_all]); b = np.array([CTX9[c][1] for c in ci_all])
    flat = e23.reshape(9, 4)
    xo = np.zeros(N, int); yo = np.zeros(N, int)
    for c in range(9):
        m = ci_all == c
        d = rng.choice(4, size=int(m.sum()), p=flat[c])
        ai = d // 2; bi = d % 2
        xo[m] = np.where(ai == 0, 1, -1); yo[m] = np.where(bi == 0, 1, -1)
    box, Ns = build_box_23(a, b, xo, yo)
    checks["build_box_23_roundtrip"] = dict(cf=float(contextual_fraction_23(box)),
                                            max_abs_dev=float(np.abs(box - e23).max()),
                                            pass_=bool(np.abs(box - e23).max() < 0.02))

    checks["all_pass"] = bool(all(checks[k].get("pass_", True) for k in checks if k != "all_pass"))
    return checks, dict(cf23_canon=float(cf23), S23_canon=float(S23))


# ===========================================================================
# C-2: 미감시 방향 껍질 탐색 (사람 확인 후 --experiment)
# ===========================================================================
S_TOL = 0.01            # S-검정 통과 허용오차
SHELL_TOL = 1e-3        # 고전 껍질(KL‖c_model = NCfloor) 허용오차
FAR_TOL = 1e-3          # "정상과 다르다" 판정(KL→normal)


def werner_p(p14):
    """공격 후보도 물리적으로 실현가능해야 함 → 가시도 V 를 자유 파라미터로(Werner 혼합은 양자집합 안).
       p14 = [13 각도 파라미터, V]."""
    V = float(min(max(p14[13], 0.0), 1.0))
    return werner_23(np.asarray(p14[:13], dtype=float), V)


def _shell_stats(e, e_normal, c_model, ncfloor, S_target):
    kln = eval_kl_23(e, e_normal)
    sdev = abs(chsh_S_sub(e) - S_target)
    shdev = abs(eval_kl_23(e, c_model) - ncfloor)
    return dict(kl_to_normal=float(kln), S_sub=float(chsh_S_sub(e)), s_dev=float(sdev),
                kl_to_c=float(eval_kl_23(e, c_model)), shell_dev=float(shdev),
                feasible=bool(sdev <= S_TOL and shdev <= SHELL_TOL))


def find_unmonitored_shell(e_normal, S_target, c_model, ncfloor, seed, restarts=40):
    """(a) 연속 최적화. S(=CHSH 4컨텍스트) 고정·고전껍질(KL‖c_model=NCfloor) 유지하며 정상과 먼 e′ 탐색.
       탐색공간 = Werner 혼합된 (2,3,2) 양자이미지(14 파라미터). A₂ 는 S 에 안 들어가므로 그 방향이 열려 있는지 관찰."""
    rng = np.random.default_rng(seed)
    wS, wSH = 300.0, 4000.0
    canon14 = np.concatenate([CANON_23, [0.856]])

    def loss(p):
        e = werner_p(p)
        return (-eval_kl_23(e, e_normal) + wS * (chsh_S_sub(e) - S_target) ** 2
                + wSH * (eval_kl_23(e, c_model) - ncfloor) ** 2)

    inits = [canon14.copy()]
    for _ in range(restarts - 1):
        base = canon14.copy()
        base[1:13] += rng.uniform(-np.pi, np.pi, size=12)
        base[13] = rng.uniform(0.5, 1.0)
        inits.append(base)
    best = None
    for x0 in inits:
        r = minimize(loss, x0, method="Powell", options=dict(maxiter=6000, xtol=1e-7, ftol=1e-10))
        e = werner_p(r.x)
        st = _shell_stats(e, e_normal, c_model, ncfloor, S_target)
        score = st["kl_to_normal"] if st["feasible"] else \
            st["kl_to_normal"] - 10 * (max(0, st["s_dev"] - S_TOL) + max(0, st["shell_dev"] - SHELL_TOL))
        if best is None or score > best[0]:
            st["cf"] = float(contextual_fraction_23(e))
            best = (score, r.x, e, st)
    return best[2], best[3]


def scan_a2(e_normal, S_target, c_model, ncfloor, base_p=None, Vn=0.856, npts=241):
    """(b) ★결정적 검사 — A₂ 방향 1D 스캔.
       A₂ 의 Bloch θ 만 [0,π] 로 훑는다. A₂ 는 S 에 안 들어가므로 S 는 구조적으로 불변이어야 하고(검증),
       남는 자유도는 KL(e′‖c_model) 뿐이다. 이 곡선이 NCfloor 를 정상점(δ=0) 말고 또 지나면 껍질이 삐져나온 것."""
    p0 = (CANON_23 if base_p is None else base_p).copy()
    th = np.linspace(0.0, np.pi, npts)
    S_arr, klc_arr, kln_arr, cf_arr = [], [], [], []
    for t in th:
        p = p0.copy(); p[5] = t                        # A₂ 의 θ (p[5],p[6] = A2θ,A2φ)
        e = werner_23(p, Vn)
        S_arr.append(chsh_S_sub(e)); klc_arr.append(eval_kl_23(e, c_model))
        kln_arr.append(eval_kl_23(e, e_normal)); cf_arr.append(contextual_fraction_23(e))
    S_arr = np.array(S_arr); klc_arr = np.array(klc_arr)
    kln_arr = np.array(kln_arr); cf_arr = np.array(cf_arr)
    hits = [i for i in range(npts)
            if abs(klc_arr[i] - ncfloor) <= SHELL_TOL and kln_arr[i] > FAR_TOL and abs(S_arr[i] - S_target) <= S_TOL]
    return dict(theta=th.tolist(), S=S_arr.tolist(), kl_to_c=klc_arr.tolist(),
                kl_to_normal=kln_arr.tolist(), cf=cf_arr.tolist(),
                S_invariant=bool(np.abs(S_arr - S_target).max() < 1e-9),
                S_max_dev=float(np.abs(S_arr - S_target).max()),
                klc_min=float(klc_arr.min()), klc_max=float(klc_arr.max()), ncfloor=float(ncfloor),
                n_shell_hits=len(hits),
                hits=[dict(theta=float(th[i]), kl_to_normal=float(kln_arr[i]), kl_to_c=float(klc_arr[i]),
                           cf=float(cf_arr[i])) for i in hits[:12]],
                best_hit=(max(hits, key=lambda i: kln_arr[i]) if hits else None))


def detection_auc_23(e_normal, e_atk, q_model, c_model, n_eval=2000, reps=200, seed=SEED + 99):
    """결과1 후속 — 삐져나온 껍질 점이 실제로 탐지되는가(양자 템플릿 vs 고전 템플릿, deviance 스코어).
       R6 준수: 커버리지 우위 주장 아님. 이 한 공격점에 대한 스코어 비교일 뿐."""
    rng = np.random.default_rng(seed)

    def boxes(e):
        out = []
        flat = e.reshape(9, 4)
        per = max(n_eval // 9, 1)
        for _ in range(reps):
            emp = np.zeros((9, 2, 2))
            for c in range(9):
                d = rng.choice(4, size=per, p=flat[c])
                cnt = np.bincount(d, minlength=4).reshape(2, 2)
                emp[c] = cnt
            emp += 1.0 / (per * 4.0)
            out.append(emp / emp.sum(axis=(1, 2), keepdims=True))
        return out

    def dev(emp, model):                                  # KL(emp‖model) = deviance(표본엔트로피 제거)
        return eval_kl_23(emp, model)

    nb, ab = boxes(e_normal), boxes(e_atk)
    res = {}
    for name, mdl in (("quantum", q_model), ("classical", c_model)):
        res[name] = float(D2.auc([dev(b, mdl) for b in ab], [dev(b, mdl) for b in nb]))
    res["S_test"] = float(D2.auc([-abs(chsh_S_sub(b)) for b in ab], [-abs(chsh_S_sub(b)) for b in nb]))
    res["n_eval"], res["reps"] = n_eval, reps
    return res


def run_experiment():
    Vn = 0.856
    e_normal = werner_23(CANON_23, Vn)
    S_t = chsh_S_sub(e_normal)
    nc, c_model = nc_floor_23(e_normal, e_normal, np.random.default_rng(SEED + 2), 8)
    log("정상(2,3,2): S_sub=%.4f CF23=%.4f NCfloor23=%.5f" % (S_t, contextual_fraction_23(e_normal), nc))

    # --- (b) 결정적 1D 스캔 먼저 ---
    scan = scan_a2(e_normal, S_t, c_model, nc, Vn=Vn)
    log("  [b] A₂ 1D 스캔: S 불변=%s(max dev %.2e) | KL(‖c) 범위 [%.5f, %.5f], NCfloor=%.5f | 껍질교점 %d개"
        % (scan["S_invariant"], scan["S_max_dev"], scan["klc_min"], scan["klc_max"], nc, scan["n_shell_hits"]))
    if scan["hits"]:
        h = max(scan["hits"], key=lambda d: d["kl_to_normal"])
        log("      최원 교점: θ_A2=%.4f rad, KL→normal=%.5f, CF=%.4f" % (h["theta"], h["kl_to_normal"], h["cf"]))

    # --- (a) 연속 최적화(전 방향, 14 파라미터) ---
    e_atk, info = find_unmonitored_shell(e_normal, S_t, c_model, nc, SEED + 20)
    log("  [a] 연속 최적화: KL→normal=%.5f S_sub=%.4f(dev%.4f) KL→c=%.5f(dev%.5f) CF=%.4f feasible=%s"
        % (info["kl_to_normal"], info["S_sub"], info["s_dev"], info["kl_to_c"], info["shell_dev"],
           info["cf"], info["feasible"]))

    bulges = bool((info["feasible"] and info["kl_to_normal"] > 0.01) or scan["n_shell_hits"] > 0)
    log("  >>> 껍질이 미감시(A₂) 방향으로 삐져나옴 = %s" % bulges)

    out = dict(v_normal=Vn, S_sub_normal=float(S_t), ncfloor23=float(nc),
               cf23_normal=float(contextual_fraction_23(e_normal)),
               scan_a2=scan, attack_info=info, shell_bulges=bulges)

    # --- 결과1 후속: 탐지 AUC ---
    if bulges:
        if scan["n_shell_hits"] > 0:
            h = max(scan["hits"], key=lambda d: d["kl_to_normal"])
            p = CANON_23.copy(); p[5] = h["theta"]
            e_use = werner_23(p, Vn); src = "scan_a2"
        else:
            e_use, src = e_atk, "continuous"
        q_model = e_normal            # 양자 템플릿(정상 운용점의 양자 모델 = 참 behavior)
        auc = detection_auc_23(e_normal, e_use, q_model, c_model)
        auc["attack_source"] = src
        log("  [후속] 탐지 AUC — quantum=%.3f classical=%.3f S-test=%.3f (attack=%s)"
            % (auc["quantum"], auc["classical"], auc["S_test"], src))
        out["detection"] = auc

    # --- 결과2 규명: A₂=B₀ 대칭이 원인인가? A₂ 를 일반각으로 둔 정상점에서 재스캔 ---
    p_alt = CANON_23.copy(); p_alt[5] = 0.7            # A₂ 를 A₀ 와 다른 일반각으로
    e_alt = werner_23(p_alt, Vn)
    S_alt = chsh_S_sub(e_alt)
    nc_alt, c_alt = nc_floor_23(e_alt, e_alt, np.random.default_rng(SEED + 3), 8)
    scan_alt = scan_a2(e_alt, S_alt, c_alt, nc_alt, base_p=p_alt, Vn=Vn)
    log("  [대칭규명] A₂=일반각(0.7) 정상점: NCfloor=%.5f, S 불변=%s, 껍질교점 %d개"
        % (nc_alt, scan_alt["S_invariant"], scan_alt["n_shell_hits"]))
    out["asymmetry_probe"] = dict(theta_a2=0.7, ncfloor=float(nc_alt), S_sub=float(S_alt), scan=scan_alt,
                                  note="A₂=B₀ 동일각 대칭이 결과의 원인인지 확인(결과2 시 규명 요건)")

    out["reading"] = ("결과1: 껍질 삐져나옴 → (2,2,2) 불가능성이 (2,3,2)서 깨짐. S-검정도 고전 템플릿도 못 보는 "
                      "A₂-변조가 존재 → abstract 원래 약속(양자 고유 탐지 우위) 복구 후보. 단 존재증명이며 "
                      "키 보안이 아니라 장비 무결성 대상."
                      if bulges else
                      "결과2: 여전히 공집합 → 불가능성이 (2,3,2)로 확장(더 강한 구조정리). "
                      "A₂ 1D 스캔에서 KL(‖c_model) 이 NCfloor 를 정상점 외에는 지나지 않음.")
    return out


# ===========================================================================
# C-2b 검증 — "삐져나옴" 판정이 허용오차 산물인가, 실재하는가.
#   s2c(2,2,2) 정리는 s_tol=0.01·shell_tol=1e-3 에서 max KL→normal = 1.4e-3(그리드), 연속최적 8.6e-5 였다.
#   같은 허용오차에서 (2,3,2) 가 더 큰 값을 주더라도, δ→0 에서 사라지면 접점의 평탄부일 뿐이다.
#   ★결정적 검사: 껍질 허용오차 δ 를 조여가며 max KL→normal 이 살아남는지 + 등식제약(SLSQP) 해가 있는지.
# ===========================================================================
DELTA_SWEEP = [1e-3, 3e-4, 1e-4, 3e-5, 1e-5]


JOBS = 1


def _pmap(fn, tasks):
    if JOBS <= 1 or len(tasks) < 2:
        return [fn(t) for t in tasks]
    from concurrent.futures import ProcessPoolExecutor
    with ProcessPoolExecutor(max_workers=JOBS) as ex:
        return list(ex.map(fn, tasks))


def _far_one(t):
    """δ-스윕의 restart 1개(병렬 워커). 벌점 Powell → 제약 만족 시 (kl_to_normal, p) 반환."""
    delta, x0, e_normal, S_target, c_model, ncfloor = t
    if quantum_behavior_23 is not fastq.quantum_behavior_23:
        fastq.install_23(sys.modules[__name__], verbose=False)
    wS, wSH = 1e4, 1.0 / (delta ** 2)

    def loss(p):
        e = werner_p(p)
        return (-eval_kl_23(e, e_normal) + wS * (chsh_S_sub(e) - S_target) ** 2
                + wSH * (eval_kl_23(e, c_model) - ncfloor) ** 2)

    r = minimize(loss, np.asarray(x0), method="Powell",
                 options=dict(maxiter=20000, xtol=1e-10, ftol=1e-13))
    e = werner_p(r.x)
    st = _shell_stats(e, e_normal, c_model, ncfloor, S_target)
    if st["shell_dev"] <= delta and st["s_dev"] <= 1e-3:
        return (float(st["kl_to_normal"]), r.x.tolist(), st)
    return None


def _max_far_at_delta(e_normal, S_target, c_model, ncfloor, delta, seed, restarts=12):
    """|KL(‖c)−floor| ≤ δ, |S−S_t| ≤ 1e-3 를 만족하며 KL→normal 최대. 벌점가중을 δ 에 맞춰 조임."""
    rng = np.random.default_rng(seed)
    canon14 = np.concatenate([CANON_23, [0.856]])
    inits = [canon14.copy()]
    for _ in range(restarts - 1):
        x0 = canon14.copy()
        x0[1:13] += rng.uniform(-np.pi, np.pi, size=12)
        x0[13] = rng.uniform(0.5, 1.0)
        inits.append(x0)
    res = [r for r in _pmap(_far_one, [(delta, x0, e_normal, S_target, c_model, ncfloor) for x0 in inits])
           if r is not None]
    if not res:
        return dict(delta=delta, feasible_found=False, max_kl_to_normal=0.0)
    best = max(res, key=lambda r: r[0])
    out = dict(delta=delta, feasible_found=True, max_kl_to_normal=float(best[0]), n_feasible=len(res))
    out.update({k: best[2][k] for k in ("s_dev", "shell_dev", "kl_to_c")})
    out["p"] = best[1]
    return out


def _slsqp_one(t):
    x0, e_normal, S_target, c_model, ncfloor = t
    if quantum_behavior_23 is not fastq.quantum_behavior_23:
        fastq.install_23(sys.modules[__name__], verbose=False)
    bounds = [(-4 * np.pi, 4 * np.pi)] * 13 + [(0.0, 1.0)]
    cons = [dict(type="eq", fun=lambda p: eval_kl_23(werner_p(p), c_model) - ncfloor),
            dict(type="eq", fun=lambda p: chsh_S_sub(werner_p(p)) - S_target)]
    try:
        r = minimize(lambda p: -eval_kl_23(werner_p(p), e_normal), np.asarray(x0), method="SLSQP",
                     bounds=bounds, constraints=cons, options=dict(maxiter=400, ftol=1e-12))
    except Exception:
        return None
    st = _shell_stats(werner_p(r.x), e_normal, c_model, ncfloor, S_target)
    resid = max(st["shell_dev"], st["s_dev"])
    if resid < 1e-6:
        return (float(st["kl_to_normal"]), r.x.tolist(), st, float(resid))
    return None


def _slsqp_exact(e_normal, S_target, c_model, ncfloor, seed, restarts=13):
    """등식제약 그대로: max KL→normal s.t. KL(‖c)=floor, S=S_t. 잔차까지 보고(허용오차 없는 판정)."""
    rng = np.random.default_rng(seed)
    canon14 = np.concatenate([CANON_23, [0.856]])
    inits = [canon14.copy()]
    for _ in range(restarts - 1):
        x0 = canon14.copy()
        x0[1:13] += rng.uniform(-np.pi, np.pi, size=12)
        x0[13] = rng.uniform(0.6, 1.0)
        inits.append(x0)
    res = [r for r in _pmap(_slsqp_one, [(x0, e_normal, S_target, c_model, ncfloor) for x0 in inits])
           if r is not None]
    if not res:
        return dict(converged=False, note="등식제약(잔차<1e-6)을 만족하는 해 없음")
    best = max(res, key=lambda r: r[0])
    return dict(converged=True, kl_to_normal=float(best[0]), max_residual=float(best[3]),
                n_converged=len(res), shell_dev=float(best[2]["shell_dev"]), s_dev=float(best[2]["s_dev"]),
                cf=float(contextual_fraction_23(werner_p(np.array(best[1])))), p=best[1])


def _context_decomp(e_atk, e_normal):
    """차이가 어디 사는가: 감시(CHSH, x,y≤1) 컨텍스트 vs 미감시(A₂/B₂ 관여) 컨텍스트."""
    per = []
    for ci, (x, y) in enumerate(CTX9):
        p = e_atk[ci]; q = np.clip(e_normal[ci], EPS, 1.0)
        m = p > 0
        per.append(float(np.sum(p[m] * np.log(p[m] / q[m]))))
    per = np.array(per)
    mon = [ci for ci, (x, y) in enumerate(CTX9) if x < 2 and y < 2]
    unmon = [ci for ci in range(9) if ci not in mon]
    tot = per.sum()
    return dict(per_context=per.tolist(),
                monitored_sum=float(per[mon].sum()), unmonitored_sum=float(per[unmon].sum()),
                unmonitored_share=float(per[unmon].sum() / tot) if tot > 0 else 0.0,
                note="unmonitored_share≈1 이면 변조가 실제로 A₂/B₂(미감시) 방향에 산다")


def run_verification(ex):
    """C-2 판정의 엄밀 검증. ex = run_experiment() 결과."""
    Vn = ex["v_normal"]
    e_normal = werner_23(CANON_23, Vn)
    S_t = chsh_S_sub(e_normal)
    nc, c_model = nc_floor_23(e_normal, e_normal, np.random.default_rng(SEED + 2), 8)

    log("\n=== C-2b 검증: 껍질 허용오차 δ 조이기 (s2c(2,2,2) 기준: 같은 tol 에서 max KL→normal=1.4e-3) ===")
    sweep = []
    for d in DELTA_SWEEP:
        r = _max_far_at_delta(e_normal, S_t, c_model, nc, d, SEED + 500 + int(-np.log10(d) * 10))
        sweep.append(r)
        log("  δ=%.0e → max KL→normal=%.6f (feasible=%s, shell_dev=%.2e)"
            % (d, r["max_kl_to_normal"], r["feasible_found"], r.get("shell_dev", float("nan"))))

    log("  등식제약(SLSQP, 잔차<1e-6) 직접 해:")
    ex_sol = _slsqp_exact(e_normal, S_t, c_model, nc, SEED + 600)
    if ex_sol["converged"]:
        log("    수렴: KL→normal=%.6f, 최대잔차=%.2e, CF=%.4f"
            % (ex_sol["kl_to_normal"], ex_sol["max_residual"], ex_sol["cf"]))
    else:
        log("    " + ex_sol["note"])

    # δ→0 에서 살아남는가
    tight = [r["max_kl_to_normal"] for r in sweep if r["delta"] <= 1e-4]
    survives = bool(tight and max(tight) > 1e-3)
    confirmed = bool(survives or (ex_sol["converged"] and ex_sol["kl_to_normal"] > 1e-3))

    out = dict(delta_sweep=sweep, exact_slsqp=ex_sol, survives_tight_tolerance=survives,
               bulge_confirmed=confirmed,
               s2c_22_reference=dict(max_kln_on_shell_grid=0.0013997633958794226,
                                     continuous_best_kl_to_normal=8.575399616906381e-05,
                                     s_tol=0.01, shell_tol=0.001,
                                     note="(2,2,2) 강한 불가능성 정리의 동일 허용오차 최대값 — 비교 기준선"))

    # 최강 공격점의 컨텍스트 분해 + 탐지 AUC(약한 scan 점이 아니라 최강점으로)
    p_best = None
    if ex_sol.get("converged"):
        p_best, src = np.array(ex_sol["p"]), "slsqp_exact"
    else:
        cand = [r for r in sweep if r.get("feasible_found") and r["delta"] <= 1e-4 and r.get("p")]
        if cand:
            b = max(cand, key=lambda r: r["max_kl_to_normal"]); p_best, src = np.array(b["p"]), "delta<=1e-4"
    if p_best is not None:
        e_best = werner_p(p_best)
        out["strongest_attack"] = dict(source=src, **_shell_stats(e_best, e_normal, c_model, nc, S_t))
        out["context_decomposition"] = _context_decomp(e_best, e_normal)
        log("  변조 위치: 감시(CHSH) 컨텍스트 KL=%.6f vs 미감시(A₂/B₂) KL=%.6f (미감시 비중 %.1f%%)"
            % (out["context_decomposition"]["monitored_sum"], out["context_decomposition"]["unmonitored_sum"],
               100 * out["context_decomposition"]["unmonitored_share"]))
        auc = detection_auc_23(e_normal, e_best, e_normal, c_model)
        auc["attack_source"] = src
        out["detection_strongest"] = auc
        log("  최강점 탐지 AUC — quantum=%.3f classical=%.3f S-test=%.3f"
            % (auc["quantum"], auc["classical"], auc["S_test"]))

    log("  >>> 검증 결론: 껍질 삐져나옴 %s" % ("확정(δ→0 에서 생존)" if confirmed else "미확정(허용오차 산물)"))
    out["reading"] = ("검증 통과: 등식제약/조인 δ 에서도 정상과 유의하게 다른 껍질 점이 존재 → (2,3,2)에서 불가능성 정리가 깨진다."
                      if confirmed else
                      "검증 실패: δ→0 에서 KL→normal 이 사라짐 → C-2 의 'feasible' 은 접점 평탄부의 허용오차 산물이며, "
                      "불가능성 정리는 (2,3,2)에서도 유지된다(결과2).")
    return out


def plot_shell(P):
    ex = P.get("experiment")
    if not ex:
        return
    sc = ex["scan_a2"]; sa = ex["asymmetry_probe"]["scan"]
    th = np.array(sc["theta"]); nc = ex["ncfloor23"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    ax[0].plot(th, sc["kl_to_c"], color="#7B2FBE", lw=2, label="KL(e′‖c_model)")
    ax[0].axhline(nc, color="#C1442E", ls="--", lw=1.5, label="NCfloor (classical blind shell)")
    ax[0].plot(th, sc["kl_to_normal"], color="#2E86C1", lw=1.2, alpha=0.8, label="KL(e′‖e_normal)")
    ax[0].axvline(CANON_23[5], color="k", ls=":", lw=1, label="normal A₂ (=A₀)")
    for h in sc["hits"]:
        ax[0].plot(h["theta"], h["kl_to_c"], "r*", ms=13)
    ax[0].set_xlabel("A₂ Bloch angle θ (rad) — unmonitored direction (not in S)")
    ax[0].set_ylabel("KL divergence")
    ax[0].set_title("A₂ 1-D scan: S invariant (max dev %.1e)\nshell crossings ≠ normal: %d"
                    % (sc["S_max_dev"], sc["n_shell_hits"]))
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    ax[1].plot(th, sa["kl_to_c"], color="#7B2FBE", lw=2, label="KL(e′‖c_model), A₂=0.7 (generic)")
    ax[1].axhline(sa["ncfloor"], color="#C1442E", ls="--", lw=1.5, label="NCfloor (generic A₂)")
    ax[1].axvline(0.7, color="k", ls=":", lw=1, label="normal A₂")
    for h in sa["hits"]:
        ax[1].plot(h["theta"], h["kl_to_c"], "r*", ms=13)
    ax[1].set_xlabel("A₂ Bloch angle θ (rad)"); ax[1].set_ylabel("KL divergence")
    ax[1].set_title("Symmetry probe: A₂ decoupled from B₀\nshell crossings: %d" % sa["n_shell_hits"])
    ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3)
    plt.suptitle("STEP C — (2,3,2) unmonitored-direction shell (device integrity, NOT key security; NOT supremacy)",
                 fontsize=12)
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=130); plt.close()


def main():
    ap = argparse.ArgumentParser(description="STEP C (2,3,2) 다중패싯. 기본=C-1 self-check, --experiment=C-2.")
    ap.add_argument("--experiment", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    # 성능 전용: quantum_behavior_23 을 수치동일(<1e-12 검증) 닫힌형으로 교체(~11x). 실패 시 예외.
    global FASTQ23, JOBS
    JOBS = args.jobs
    FASTQ23 = fastq.install_23(sys.modules[__name__])

    log("=== STEP C-1: (2,3,2) 인프라 self-check (축약 검증) ===")
    checks, meta = run_self_check()
    for k, v in checks.items():
        if k == "all_pass":
            continue
        log("  %-24s : %s" % (k, {kk: (round(vv, 5) if isinstance(vv, float) else vv) for kk, vv in v.items()}))
    log("  >>> C-1 self-check ALL PASS = %s" % checks["all_pass"])

    payload = dict(
        note=("STEP C (2,3,2) 다중패싯 인프라. A₂(키생성)는 S 계산 제외 → 미감시 방향. C-1=축약 self-check. "
              "C-2(--experiment)=미감시 껍질 탐색. ★R6 커버리지 우위 주장 금지. 최소 시나리오. supremacy 아님."),
        params=dict(seed=SEED, canon_23=CANON_23.tolist(), n_contexts=9, n_vertices=64, n_qparams=13,
                    tolerances=dict(S_TOL=S_TOL, SHELL_TOL=SHELL_TOL, FAR_TOL=FAR_TOL), fastq=FASTQ23),
        self_check=checks, canon_meta=meta,
        limits=("(2,3,2)는 여전히 최소 시나리오(일반정리 아님). A₂=B₀ 동일각(키생성). build_box_23 신규(원본 build_box 와 구분). "
                "R6: 탐지 커버리지 우위 주장 금지. supremacy 아님. 키보안 아닌 장비무결성."),
    )
    if args.experiment:
        if not checks["all_pass"]:
            log("[STOP] C-1 self-check 실패 → C-2 진입 금지(LP 버그 수정 먼저).")
        else:
            log("\n=== STEP C-2: 미감시 방향 껍질 탐색 ===")
            payload["experiment"] = run_experiment()
            payload["verification"] = run_verification(payload["experiment"])
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if args.experiment:
        try:
            plot_shell(payload)
            log("[저장] %s" % OUT_PNG)
        except Exception as e:
            log("[plot WARN] %r" % e)


if __name__ == "__main__":
    main()
