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
def find_unmonitored_shell(e_normal, S_target, c_model, ncfloor, seed, restarts=40):
    """S(=CHSH 4컨텍스트) 고정·고전껍질(KL‖c_model=NCfloor) 유지하며 정상과 먼 e′ 를 (2,3,2) 양자이미지에서.
       A₂ 방향은 S 에 안 들어가므로 그쪽으로 삐져나오는지 관찰."""
    rng = np.random.default_rng(seed)
    wS, wSH = 300.0, 4000.0

    def loss(p):
        e = quantum_behavior_23(p)
        L = -eval_kl_23(e, e_normal) + wS * (chsh_S_sub(e) - S_target) ** 2 \
            + wSH * (eval_kl_23(e, c_model) - ncfloor) ** 2
        return L

    inits = [CANON_23.copy()]
    for _ in range(restarts - 1):
        base = CANON_23.copy()
        base[1:] += rng.uniform(-np.pi, np.pi, size=12)
        inits.append(base)
    best = None
    for x0 in inits:
        r = minimize(loss, x0, method="Powell", options=dict(maxiter=6000, xtol=1e-7, ftol=1e-10))
        e = quantum_behavior_23(r.x)
        kln = eval_kl_23(e, e_normal); sdev = abs(chsh_S_sub(e) - S_target)
        shdev = abs(eval_kl_23(e, c_model) - ncfloor)
        feasible = sdev <= 0.01 and shdev <= 0.001
        score = kln if feasible else kln - 10 * (max(0, sdev - 0.01) + max(0, shdev - 0.001))
        if best is None or score > best[0]:
            best = (score, r.x, e, dict(kl_to_normal=float(kln), S_sub=float(chsh_S_sub(e)),
                                        kl_to_c=float(eval_kl_23(e, c_model)), s_dev=float(sdev),
                                        shell_dev=float(shdev), cf=float(contextual_fraction_23(e)),
                                        feasible=bool(feasible)))
    return best[2], best[3]


def run_experiment():
    Vn = 0.856
    e_normal = werner_23(CANON_23, Vn)
    S_t = chsh_S_sub(e_normal)
    nc, c_model = nc_floor_23(e_normal, e_normal, np.random.default_rng(SEED + 2), 8)
    log("정상(2,3,2): S_sub=%.4f CF23=%.4f NCfloor23=%.5f" % (S_t, contextual_fraction_23(e_normal), nc))
    e_atk, info = find_unmonitored_shell(e_normal, S_t, c_model, nc, SEED + 20)
    log("  미감시 껍질 탐색: KL→normal=%.5f S_sub=%.4f(dev%.4f) KL→c=%.5f(dev%.5f, NCfloor=%.5f) CF=%.4f feasible=%s"
        % (info["kl_to_normal"], info["S_sub"], info["s_dev"], info["kl_to_c"], info["shell_dev"], nc,
           info["cf"], info["feasible"]))
    bulges = bool(info["feasible"] and info["kl_to_normal"] > 0.01)
    log("  >>> 껍질이 미감시(A₂) 방향으로 삐져나옴 = %s" % bulges)
    return dict(v_normal=Vn, S_sub_normal=float(S_t), ncfloor23=float(nc),
                attack_info=info, shell_bulges=bulges,
                reading=("결과1: 껍질 삐져나옴 → (2,2,2) 불가능성이 (2,3,2)서 깨짐. 양자 템플릿이 S·고전 둘 다 못 보는 "
                         "A₂-변조 포착 후보 → abstract 원래 약속 복구 검토(탐지 실험 후속)."
                         if bulges else
                         "결과2: 여전히 공집합 → 불가능성이 (2,3,2)로 확장(더 강한 구조정리). A₂=B₀ 대칭 유지 때문인지 규명."))


def main():
    ap = argparse.ArgumentParser(description="STEP C (2,3,2) 다중패싯. 기본=C-1 self-check, --experiment=C-2.")
    ap.add_argument("--experiment", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

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
        params=dict(seed=SEED, canon_23=CANON_23.tolist(), n_contexts=9, n_vertices=64, n_qparams=13),
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
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)


if __name__ == "__main__":
    main()
