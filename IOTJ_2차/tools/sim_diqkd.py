#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DI-QKD 시뮬레이터 (bipartite CHSH (2,2,2), G 트랙 고정) + Delft 재현 검증 (STEP 1).

목적: 후속 탐지/유한표본 실험의 데이터 공급원을 만들고, 그것이 **실측 Delft Bell 데이터를
      재현하는지** STOP 게이트로 검증. 실패 시 STEP 2 진입 금지.

정합 규칙(프롬프트 R1~R5):
  R1  bipartite G 트랙만 사용 (gap_vs_cf_v0). K.cyclic_* 미사용.
  R2  16셀 변환기는 gate_r2.build_box 하나만 사용(재구현 금지). 시뮬은 (a,b,xo,yo) 4-튜플을 뱉는다.
  R3  훈련/평가 분리는 STEP 2 대상. STEP 1(법칙/재현 검증)은 R2와 동일하게 in-sample.
  R5  결정성(seed), results2/gate_s1/ 아래 JSON+PNG, note/params/self_check/limits 포함.

핵심 단순화(정확·고속): 모든 측정확률은 Bloch 벡터로  P(o|û) = ½(1+(-1)^o û·⟨σ⟩)  로 표현.
  - Werner:      e = V·quantum_behavior(p) + (1−V)·uniform_box   (측정이 ρ에 선형, _proj trace=1
                 → behavior(I/4)=uniform. 등방혼합. G.mix 와 동일 형태.)
  - 국소잡음:    p_flip / p_dark 는 각 당사자 결과에 대한 단일파티 stochastic map → behavior 에 folding.
  - blinding/mimic: 국소 결정론 vertex 혼합(Σ w_g D_g) = NC behavior → CF=0. per-trial latent 샘플과 동분포.
  - intercept_resend: 이브가 무작위 기저로 한쪽 큐빗을 측정·재전송 → 분리가능 상태. 이브 기저 평균을
                 Bloch 로 해석적 계산(고정 seed·캐시). S≤2·CF 작음.

사용: python tools/sim_diqkd.py            # self-check + Delft 재현 STOP 게이트
      python tools/sim_diqkd.py --quick    # 빠른 점검(reps 축소)
      python tools/sim_diqkd.py --reps 1000 --restarts 3 --jobs 6
"""
import os
import sys
import json
import argparse
import functools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G                       # 무수정 재사용 (import 시 os.chdir(IOTJ_2차))
import gate_r2_anchor_strengthen as R2         # build_box, chsh_S, no_signalling_L1_chsh (무수정)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

OUTDIR = os.path.join("results2", "gate_s1")
OUT_JSON = os.path.join(OUTDIR, "s1.json")
OUT_PNG = os.path.join(OUTDIR, "s1.png")

SEED = 20260724
EPS = 1e-12
CTX = [(0, 0), (0, 1), (1, 0), (1, 1)]
# 표준 CHSH 최적각(gap_vs_cf_v0._q_inits canon): θ_state=π/4, A0=Z,A1=X, B0=(θ=π/4),B1=(θ=−π/4)
CANON_STATE = np.pi / 4
CANON_ANGLES = np.array([0.0, 0.0, np.pi / 2, 0.0, np.pi / 4, 0.0, -np.pi / 4, 0.0])   # A0,A1,B0,B1 (θ,φ)×4
EVE_SEED = 777                                 # intercept-resend 이브 기저 평균용 고정 seed(물리모델, 실험난수 아님)
EVE_NDIR = 40000

# Delft 실측 참조값 (results2/gate_r2/r2.json — raw data는 gitignore로 부재, 검증된 요약통계 사용)
DELFT_PER_CTX_N = [53, 79, 62, 51]
DELFT_N = 245
DELFT_S = 2.4224998704613707
DELFT_CF = 0.21124993523068514
DELFT_GAP = 0.00846309086408372
DELFT_NCFLOOR = 0.012920218563118282
DELFT_KLQ = 0.004457127699034561
DELFT_NOSIG = 0.08947063189321118


def log(m):
    print(m, flush=True)


# ===========================================================================
# nc_floor 가중치 반환판 (gap_vs_cf_v0.nc_floor 271–291행 복제 + w 반환). 원본 무수정.
# ===========================================================================
def nc_floor_weights(emp, rng, restarts=4):
    """min_w CE(emp‖Σ w_g D_g) over simplex → 최적 vertex 가중치 w(16,)와 model(4,2,2) 반환."""
    from scipy.optimize import minimize
    V = G._nc_vertices().reshape(16, -1)          # (16,16)
    empf = emp.reshape(-1)

    def loss(z):
        w = np.exp(z - z.max()); w = w / w.sum()
        m = np.clip(w @ V, EPS, 1.0)
        return -float(empf @ np.log(m))

    best = None
    for _ in range(restarts):
        r = minimize(loss, rng.normal(0, 1.0, size=16), method="Powell",
                     options=dict(maxiter=6000, xtol=1e-7, ftol=1e-10))
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    w = np.exp(best[1] - best[1].max()); w = w / w.sum()
    model = (w @ V).reshape(4, 2, 2)
    return w, model


# ===========================================================================
# Bloch 헬퍼
# ===========================================================================
def _dir(theta, phi):
    return np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


def _alice_dirs(angles):
    return {0: _dir(angles[0], angles[1]), 1: _dir(angles[2], angles[3])}


def _bob_dirs(angles):
    return {0: _dir(angles[4], angles[5]), 1: _dir(angles[6], angles[7])}


# ===========================================================================
# behavior 빌더
# ===========================================================================
def quantum_e(state_params, angles, V):
    """Werner 등방혼합 양자 behavior: V·quantum_behavior(p) + (1−V)·uniform."""
    p = np.concatenate([[state_params], angles])
    qb = G.quantum_behavior(p)
    return V * qb + (1.0 - V) * G.uniform_box()


def _single_party_channel(p_flip, p_dark):
    """단일 결과에 대한 2x2 stochastic map T[o'|o] (flip→dark 순). o=0↔+1, o=1↔−1."""
    # flip: o->1-o w.p. p_flip
    F = np.array([[1 - p_flip, p_flip], [p_flip, 1 - p_flip]])
    # dark: o->uniform w.p. p_dark
    D = (1 - p_dark) * np.eye(2) + p_dark * 0.5 * np.ones((2, 2))
    return D @ F                                   # 먼저 flip 그다음 dark: T = D·F


def apply_local_noise(e, p_flip, p_dark):
    """양쪽 당사자에 단일파티 채널 적용(비맥락성 증가 불가)."""
    if p_flip == 0.0 and p_dark == 0.0:
        return e
    T = _single_party_channel(p_flip, p_dark)      # (out', out)
    # e[ctx, a, b] -> Σ_{a,b} T[a'|a] T[b'|b] e[ctx,a,b]
    return np.einsum('ia,jb,cab->cij', T, T, e)


def intercept_resend_e(angles, n_dir=EVE_NDIR, seed=EVE_SEED):
    """이브가 무작위 기저로 한쪽 큐빗 측정·재전송(|Φ+⟩). 이브 기저 균등평균을 Bloch 로 해석적 계산."""
    return _intercept_cached(tuple(np.round(angles, 10)), n_dir, seed)


@functools.lru_cache(maxsize=64)
def _intercept_cached(angles_key, n_dir, seed):
    angles = np.array(angles_key)
    Adir = _alice_dirs(angles); Bdir = _bob_dirs(angles)
    rng = np.random.default_rng(seed)
    # 균등 구면 방향 n (이브 기저), 각 방향에서 s=±1 (이브 결과) 등확률
    z = rng.uniform(-1, 1, size=n_dir)
    ph = rng.uniform(0, 2 * np.pi, size=n_dir)
    r = np.sqrt(1 - z * z)
    n = np.stack([r * np.cos(ph), r * np.sin(ph), z], axis=1)   # (n_dir,3)
    e = np.zeros((4, 2, 2))
    for ci, (x, y) in enumerate(CTX):
        A = Adir[x]; B = Bdir[y]
        for s in (+1, -1):
            # Bob 재전송 상태 |n,s> Bloch = s·n ; Alice 붕괴 conj(|n,s>) Bloch = s·(nx,−ny,nz)
            bloch_bob = s * n                                   # (n_dir,3)
            bloch_ali = s * n * np.array([1.0, -1.0, 1.0])
            projA = bloch_ali @ A                               # (n_dir,)  = â·s_A
            projB = bloch_bob @ B
            for ai in (0, 1):
                Pa = 0.5 * (1 + ((-1) ** ai) * projA)
                for bi in (0, 1):
                    Pb = 0.5 * (1 + ((-1) ** bi) * projB)
                    e[ci, ai, bi] += 0.5 * np.mean(Pa * Pb)     # (1/2) over s, mean over n
    e = np.clip(e, 0.0, None)
    e = e / e.sum(axis=(1, 2), keepdims=True)
    return e


# blinding: 국소 결정론 vertex 혼합. "loud" 공격 = CHSH 국소한계 S=2 포화(PR 의 최적 NC 근사).
@functools.lru_cache(maxsize=4)
def _blinding_e(seed):
    w, model = nc_floor_weights(G.pr_box(), np.random.default_rng(seed), restarts=6)
    return model


def build_behavior(attack, state_params, angles, noise, rng):
    """attack+noise → 16셀 behavior e_true. 모든 공격이 behavior 로 환원(샘플은 이 behavior 에서)."""
    V = float(noise.get("visibility", 1.0))
    p_flip = float(noise.get("p_flip", 0.0))
    p_dark = float(noise.get("p_dark", 0.0))

    def _none():
        return apply_local_noise(quantum_e(state_params, angles, V), p_flip, p_dark)

    if attack == "none":
        e = _none()
    elif attack == "intercept_resend":
        e = apply_local_noise(intercept_resend_e(np.asarray(angles)), p_flip, p_dark)
    elif attack == "blinding":
        e = _blinding_e(SEED)                                  # CF=0 국소(S=2 포화)
    elif attack == "mimic_ncfloor":
        target = _none()                                       # 정상 운용점
        _, e = nc_floor_weights(target, np.random.default_rng(SEED + 13), restarts=6)
    elif attack == "partial":
        p_atk = float(noise.get("p_atk", 0.25))
        sub = noise.get("sub_attack", "intercept_resend")
        e_atk = build_behavior(sub, state_params, angles,
                               {k: v for k, v in noise.items() if k not in ("p_atk", "sub_attack")}, rng)
        e = p_atk * e_atk + (1 - p_atk) * _none()
    else:
        raise ValueError("unknown attack: %s" % attack)
    e = np.clip(e, 0.0, None)
    return e / e.sum(axis=(1, 2), keepdims=True)


def _sample_from_behavior(e, n_trials, rng, per_ctx_N=None):
    """behavior → (a,b,xo,yo). per_ctx_N 주면 컨텍스트별 시행수 고정, 아니면 설정을 균등 난수로."""
    flat = e.reshape(4, 4)
    if per_ctx_N is not None:
        counts = list(per_ctx_N)
    else:
        ci_all = rng.integers(0, 4, size=n_trials)
        counts = [int((ci_all == c).sum()) for c in range(4)]
    a_l, b_l, xo_l, yo_l = [], [], [], []
    for c in range(4):
        nc = counts[c]
        if nc == 0:
            continue
        d = rng.choice(4, size=nc, p=flat[c])
        ai = d // 2; bi = d % 2
        a_l.append(np.full(nc, CTX[c][0]))
        b_l.append(np.full(nc, CTX[c][1]))
        xo_l.append(np.where(ai == 0, 1, -1))
        yo_l.append(np.where(bi == 0, 1, -1))
    a = np.concatenate(a_l); b = np.concatenate(b_l)
    xo = np.concatenate(xo_l); yo = np.concatenate(yo_l)
    perm = rng.permutation(len(a))
    return a[perm], b[perm], xo[perm], yo[perm]


def simulate_diqkd(n_trials, state_params, angles, noise, attack, rng):
    """
    Returns: a, b, xo, yo  (np.ndarray, 각 길이 n_trials)
      a, b in {0,1};  xo, yo in {+1,-1}
    noise: dict(visibility, p_flip, p_dark[, p_atk, sub_attack, per_ctx_N])
      per_ctx_N (선택): 컨텍스트별 시행수 고정 [N00,N01,N10,N11] (Delft 재현용).
    """
    e = build_behavior(attack, state_params, angles, noise, rng)
    per_ctx_N = noise.get("per_ctx_N", None)
    return _sample_from_behavior(e, n_trials, rng, per_ctx_N=per_ctx_N)


# ===========================================================================
# self-check
# ===========================================================================
def run_self_check():
    rng = np.random.default_rng(SEED)
    checks = {}

    # (1) blinding → CF ≈ 0
    e_bl = build_behavior("blinding", CANON_STATE, CANON_ANGLES, {}, rng)
    cf_bl = G.contextual_fraction(e_bl)
    checks["blinding_cf"] = dict(value=float(cf_bl), S=float(R2.chsh_S(e_bl)),
                                 pass_=bool(abs(cf_bl) < 1e-6))

    # (2) mimic_ncfloor → CF ≈ 0
    e_mm = build_behavior("mimic_ncfloor", CANON_STATE, CANON_ANGLES,
                          {"visibility": 0.856}, rng)
    cf_mm = G.contextual_fraction(e_mm)
    checks["mimic_cf"] = dict(value=float(cf_mm), pass_=bool(abs(cf_mm) < 1e-6))

    # (3) none, noise 없음, n=100000 → S ≈ 2√2
    a, b, xo, yo = simulate_diqkd(100000, CANON_STATE, CANON_ANGLES, {}, "none",
                                  np.random.default_rng(SEED + 1))
    box, Ns = R2.build_box(a, b, xo, yo)
    S_none = R2.chsh_S(box)
    checks["none_S"] = dict(value=float(S_none), target=float(2 * np.sqrt(2)),
                            pass_=bool(abs(S_none - 2 * np.sqrt(2)) < 0.05))

    # (보조) intercept_resend → CF 작음·S≤2
    e_ir = build_behavior("intercept_resend", CANON_STATE, CANON_ANGLES, {}, rng)
    checks["intercept_resend"] = dict(cf=float(G.contextual_fraction(e_ir)),
                                      S=float(R2.chsh_S(e_ir)),
                                      no_sig=float(R2.no_signalling_L1_chsh(e_ir)),
                                      pass_=bool(R2.chsh_S(e_ir) <= 2.0 + 1e-6))

    checks["all_pass"] = bool(checks["blinding_cf"]["pass_"] and checks["mimic_cf"]["pass_"]
                              and checks["none_S"]["pass_"] and checks["intercept_resend"]["pass_"])
    return checks


# ===========================================================================
# Delft 재현 STOP 게이트
# ===========================================================================
def fit_visibility_to_S(target_S=DELFT_S):
    """canon 각도 고정, visibility 1개 자유파라미터로 S 매칭. S 는 V 에 선형(uniform S=0)."""
    p = np.concatenate([[CANON_STATE], CANON_ANGLES])
    S_canon = R2.chsh_S(G.quantum_behavior(p))
    V = target_S / S_canon
    e = quantum_e(CANON_STATE, CANON_ANGLES, V)
    return float(V), float(R2.chsh_S(e)), float(G.contextual_fraction(e))


def _delft_rep(arg):
    """한 번의 독립 생성 → (cf, S, ncfloor, klq, gap, no_sig)."""
    behavior, per_ctx_N, seed, restarts = arg
    rng = np.random.default_rng(seed)
    a, b, xo, yo = _sample_from_behavior(behavior, sum(per_ctx_N), rng, per_ctx_N=per_ctx_N)
    box, Ns = R2.build_box(a, b, xo, yo)
    cf = G.contextual_fraction(box)
    S = R2.chsh_S(box)
    nsig = R2.no_signalling_L1_chsh(box)
    nc = G.nc_floor(box, box, np.random.default_rng(seed + 101), restarts)
    klq = G.fit_quantum(box, box, np.random.default_rng(seed + 202), restarts)
    return (float(cf), float(S), float(nc), float(klq), float(nc - klq), float(nsig))


def validate_delft(reps, restarts, jobs):
    V, S_fit, cf_fit = fit_visibility_to_S()
    behavior = quantum_e(CANON_STATE, CANON_ANGLES, V)
    log("  피팅: visibility V=%.5f → behavior S=%.4f, CF=%.4f (canon 각도 고정)" % (V, S_fit, cf_fit))
    args = [(behavior, DELFT_PER_CTX_N, SEED + 3000 + i, restarts) for i in range(reps)]
    if jobs <= 1:
        out = [_delft_rep(t) for t in args]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=jobs) as ex:
            out = list(ex.map(_delft_rep, args, chunksize=max(1, reps // (jobs * 4))))
    out = np.array(out)                        # (reps,6)
    cf, S, nc, klq, gap, nsig = [out[:, j] for j in range(6)]

    def band(v):
        return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

    cf_band, gap_band, S_band = band(cf), band(gap), band(S)
    cf_ok = cf_band[0] <= DELFT_CF <= cf_band[1]
    gap_ok = gap_band[0] <= DELFT_GAP <= gap_band[1]
    S_med = float(np.median(S))
    S_ok = abs(S_med - DELFT_S) < 0.05
    passed = bool(cf_ok and gap_ok and S_ok)

    # 실패 원인 후보(정직 보고)
    fail_reasons = []
    if not passed:
        if abs(float(np.median(nsig)) - DELFT_NOSIG) > 0.02:
            fail_reasons.append("(i) 노이즈 모델 부족: no-signalling 잔차 미재현 "
                                "(sim median=%.4f vs real=%.4f)"
                                % (float(np.median(nsig)), DELFT_NOSIG))
        if not S_ok:
            fail_reasons.append("(ii) 각도 파라미터화 불일치: S 중앙값=%.4f" % S_med)
        if not gap_ok or not cf_ok:
            fail_reasons.append("(iii) 유한표본 효과 과소평가: gap/cf 밴드가 실측 미포함")

    return dict(
        fit=dict(visibility=V, behavior_S=S_fit, behavior_CF=cf_fit,
                 note="canon 각도 고정, visibility 1개 자유파라미터로 S 매칭"),
        reps=reps, restarts=restarts, per_ctx_N=DELFT_PER_CTX_N,
        sim=dict(
            cf=dict(median=float(np.median(cf)), band95=cf_band, mean=float(cf.mean())),
            gap=dict(median=float(np.median(gap)), band95=gap_band, mean=float(gap.mean())),
            S=dict(median=S_med, band95=S_band),
            ncfloor=dict(median=float(np.median(nc))),
            klq=dict(median=float(np.median(klq))),
            no_sig=dict(median=float(np.median(nsig))),
        ),
        real=dict(cf=DELFT_CF, gap=DELFT_GAP, S=DELFT_S, ncfloor=DELFT_NCFLOOR,
                  klq=DELFT_KLQ, no_sig=DELFT_NOSIG,
                  source="results2/gate_r2/r2.json (raw data gitignored; 검증된 요약통계)"),
        pass_conditions=dict(cf_in_band=bool(cf_ok), gap_in_band=bool(gap_ok),
                             S_median_within_0p05=bool(S_ok)),
        passed=passed, fail_reasons=fail_reasons,
        _raw=dict(cf=cf.tolist(), gap=gap.tolist(), S=S.tolist()),   # plot용(JSON 저장 시 제거)
    )


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="DI-QKD 시뮬레이터 + Delft 재현 STOP 게이트 (STEP 1).")
    ap.add_argument("--reps", type=int, default=1000)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--quick", action="store_true", help="빠른 점검(reps=60, restarts=2)")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.restarts = 60, 2
    os.makedirs(OUTDIR, exist_ok=True)

    log("=== STEP 1: DI-QKD 시뮬레이터 self-check ===")
    sc = run_self_check()
    log("  blinding CF=%.2e (기대0) S=%.3f | mimic CF=%.2e (기대0)"
        % (sc["blinding_cf"]["value"], sc["blinding_cf"]["S"], sc["mimic_cf"]["value"]))
    log("  none n=1e5 S=%.4f (기대 2√2=%.4f) | intercept CF=%.4f S=%.3f (≤2)"
        % (sc["none_S"]["value"], 2 * np.sqrt(2), sc["intercept_resend"]["cf"], sc["intercept_resend"]["S"]))
    log("  self-check ALL PASS = %s" % sc["all_pass"])

    log("\n=== STEP 1 STOP 게이트: Delft Hensen2015 재현 (reps=%d, restarts=%d, jobs=%d) ==="
        % (args.reps, args.restarts, args.jobs))
    val = validate_delft(args.reps, args.restarts, args.jobs)
    s = val["sim"]
    log("  sim CF median=%.4f band95=%s  (real=%.4f) → %s"
        % (s["cf"]["median"], _fmt(s["cf"]["band95"]), DELFT_CF, val["pass_conditions"]["cf_in_band"]))
    log("  sim gap median=%.5f band95=%s (real=%.5f) → %s"
        % (s["gap"]["median"], _fmt(s["gap"]["band95"]), DELFT_GAP, val["pass_conditions"]["gap_in_band"]))
    log("  sim S median=%.4f (real=%.4f, |Δ|<0.05) → %s"
        % (s["S"]["median"], DELFT_S, val["pass_conditions"]["S_median_within_0p05"]))
    log("  sim no-sig median=%.4f (real=%.4f; 이상적 등방시뮬은 0 예상)"
        % (s["no_sig"]["median"], DELFT_NOSIG))
    log("\n  >>> STOP 게이트 판정: %s" % ("PASS → STEP 2 진입 허용" if val["passed"]
                                        else "FAIL → STEP 2 진입 금지"))
    if not val["passed"]:
        for r in val["fail_reasons"]:
            log("      실패원인 후보 " + r)

    plot(val, sc)

    raw = val.pop("_raw")
    payload = dict(
        note=("STEP 1: DI-QKD 시뮬레이터(bipartite CHSH, G 트랙) + Delft 재현 STOP 게이트. "
              "모든 공격을 16셀 behavior 로 환원해 build_box(gate_r2) 로 (a,b,xo,yo)→box. "
              "★모델클래스 분리이지 supremacy 아님. (2,2,2)·시뮬·소규모 한정. 단정 금지."),
        params=dict(seed=SEED, reps=args.reps, restarts=args.restarts,
                    canon_state=float(CANON_STATE), canon_angles=CANON_ANGLES.tolist(),
                    eve_seed=EVE_SEED, eve_ndir=EVE_NDIR,
                    delft_per_ctx_N=DELFT_PER_CTX_N),
        self_check=sc,
        delft_validation=val,
        limits=("이상적 등방(Werner) 시뮬은 Delft no-signalling 잔차(0.089)를 재현하지 않음 → "
                "sim KL_q≈0(box가 정확히 양자실현), real KL_q=0.0045. gap 은 유한표본 밴드로 비교. "
                "raw Delft 데이터는 gitignore 로 부재—검증된 r2.json 요약통계를 참조 타깃으로 사용(정직 기록). "
                "G 트랙(16 vertex NCfloor) 고정, K.cyclic_* 미사용. 관측이지 supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    log("[저장] %s" % OUT_PNG)


def _fmt(band):
    return "[%.5f, %.5f]" % (band[0], band[1])


def plot(val, sc):
    raw = val["_raw"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 5))
    specs = [("cf", DELFT_CF, "contextual fraction CF", "#3B6EA5"),
             ("gap", DELFT_GAP, "gap = NCfloor − KL_q", "#7B2FBE"),
             ("S", DELFT_S, "CHSH S", "#E2A13B")]
    for k, (key, real, xlabel, col) in enumerate(specs):
        v = np.array(raw[key])
        ax[k].hist(v, bins=40, color=col, alpha=0.7, edgecolor="white")
        lo, hi = np.percentile(v, 2.5), np.percentile(v, 97.5)
        ax[k].axvspan(lo, hi, color=col, alpha=0.12, label="sim 95%%=[%.4f,%.4f]" % (lo, hi))
        ax[k].axvline(real, color="#C1442E", lw=2.5, label="Delft real=%.4f" % real)
        ax[k].axvline(np.median(v), color="k", lw=1.2, ls="--", label="sim median=%.4f" % np.median(v))
        inside = lo <= real <= hi if key != "S" else abs(np.median(v) - real) < 0.05
        ax[k].set_xlabel(xlabel)
        ax[k].set_ylabel("count")
        ax[k].set_title("%s  →  %s" % (xlabel, "INCLUDES real ✓" if inside else "MISS ✗"))
        ax[k].legend(fontsize=8)
        ax[k].grid(alpha=0.3)
    verdict = "PASS" if val["passed"] else "FAIL"
    plt.suptitle("STEP 1 — Delft Hensen2015 reproduction STOP gate: %s  "
                 "(sim: Werner V=%.3f, per-ctx N=%s, %d reps)  — NOT supremacy"
                 % (verdict, val["fit"]["visibility"], DELFT_PER_CTX_N, val["reps"]), fontsize=11)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130)
    plt.close()


if __name__ == "__main__":
    main()
