"""Gate S8 Phase B-2: 유령 비트. V=0.70·n=245·Delft확장 배분·reps=100·레벨2·x*=A0.

대조축(PB2-4): 동일 경로 내 **plugin 입력 vs 양자적합 입력**.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np

R2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R2, "tools"))
sys.path.insert(0, os.path.join(R2, "tools_s7"))
sys.path.insert(0, os.path.dirname(__file__))
import npa_guess as N              # noqa: E402
import gate_s5_multifacet as M     # noqa: E402
import s7_finite_template as T     # noqa: E402

CH = [(0, 0, +1), (0, 1, +1), (1, 0, +1), (1, 1, -1)]
SEED = 20260725
V_GHOST, N_TOT, REPS, RESTARTS = 0.70, 245, 100, 3
DELFT4 = np.array([53.0, 79.0, 62.0, 51.0])          # gate_s3_estimation.py:54, (2,2,2) 4컨텍스트
OUT = os.path.join(R2, "results2", "gate_s8", "s8_phaseB2.json")

# --- Delft 배분의 (2,3,2) 9컨텍스트 확장 규칙 (사전 고정) --------------------
#   CHSH 4컨텍스트 (0,0)(0,1)(1,0)(1,1) 은 Delft 실측 비율을 그대로 쓰고,
#   나머지 5컨텍스트는 Delft 4개의 산술평균 가중치를 준 뒤,
#   전체 합이 정확히 N_TOT 가 되도록 정규화·최대잉여법으로 정수화한다.
CHSH_CTX = {(0, 0), (0, 1), (1, 0), (1, 1)}


def delft9_alloc(n_tot=N_TOT):
    w = np.empty(9)
    d = {(0, 0): DELFT4[0], (0, 1): DELFT4[1], (1, 0): DELFT4[2], (1, 1): DELFT4[3]}
    for ci, xy in enumerate(M.CTX9):
        w[ci] = d[xy] if xy in CHSH_CTX else DELFT4.mean()
    q = w / w.sum() * n_tot
    base = np.floor(q).astype(int)
    rem = n_tot - base.sum()
    order = np.argsort(-(q - base))
    base[order[:rem]] += 1
    return base


ALLOC9 = delft9_alloc()


def sample_delft9(e_true, rng):
    """Delft확장 배분으로 경험박스. sample_emp_23(s7_finite_template.py:58)과 동일한
    약 Laplace 관례를 컨텍스트별 시행수에 맞춰 적용."""
    emp = np.zeros((9, 2, 2))
    flat = e_true.reshape(9, 4)
    for ci in range(9):
        per = int(ALLOC9[ci])
        cnt = np.bincount(rng.choice(4, size=per, p=flat[ci]), minlength=4).astype(float)
        emp[ci] = cnt.reshape(2, 2) + 1.0 / (per * 4.0)
    return emp / emp.sum(axis=(1, 2), keepdims=True)


def to_p(e):
    p = np.zeros((3, 3, 2, 2))
    for ci, (x, y) in enumerate(M.CTX9):
        p[x, y] = e[ci]
    return p


def _sg(o):
    return 1.0 if o == 0 else -1.0


def moments(p):
    mA = np.zeros((3, 3)); mB = np.zeros((3, 3)); c = np.zeros((3, 3))
    for x in range(3):
        for y in range(3):
            for a in (0, 1):
                for b in (0, 1):
                    v = p[x, y, a, b]
                    mA[x, y] += _sg(a) * v
                    mB[x, y] += _sg(b) * v
                    c[x, y] += _sg(a) * _sg(b) * v
    return mA, mB, c


def ns_residual_l1(p):
    """무신호 위반 L1: A 주변확률의 y 의존 + B 주변확률의 x 의존."""
    mA, mB, _ = moments(p)
    r = float(np.abs(mA - mA.mean(axis=1, keepdims=True)).sum()
              + np.abs(mB - mB.mean(axis=0, keepdims=True)).sum())
    return r


def ns_project(p):
    """무신호 아핀부분공간으로의 정확한 최소제곱 사영.
    {1, a, b, ab}/4 가 컨텍스트별 직교기저이므로 제약성분(주변확률)의 평균이 곧 LS 해다.
    반환: (사영박스, 음수셀 개수, 최소셀값, 클리핑 총량)."""
    mA, mB, c = moments(p)
    aX = mA.mean(axis=1)          # ⟨Ax⟩ = y 평균
    bY = mB.mean(axis=0)          # ⟨By⟩ = x 평균
    q = np.zeros_like(p)
    for x in range(3):
        for y in range(3):
            for a in (0, 1):
                for b in (0, 1):
                    q[x, y, a, b] = (1 + _sg(a) * aX[x] + _sg(b) * bY[y]
                                     + _sg(a) * _sg(b) * c[x, y]) / 4.0
    neg = int((q < 0).sum())
    qmin = float(q.min())
    clipped = float(np.clip(-q, 0, None).sum())
    q = np.clip(q, 0.0, None)
    q = q / q.sum(axis=(2, 3), keepdims=True)      # 컨텍스트별 재정규화
    return q, neg, qmin, clipped


def chsh_from_p(p):
    _, _, c = moments(p)
    return float(c[0, 0] + c[0, 1] + c[1, 0] - c[1, 1])


def rep_job(i):
    rng = np.random.default_rng(SEED + 1000 * i)
    e_true = M.werner_23(M.CANON_23, V_GHOST)
    emp = sample_delft9(e_true, rng)
    p_hat = to_p(emp)
    out = dict(rep=i, S_hat=chsh_from_p(p_hat), ns_l1=ns_residual_l1(p_hat),
               cf_plugin=float(M.contextual_fraction_23(emp)))
    out["S_gt2"] = bool(out["S_hat"] > 2.0)

    # (B) 양자적합 입력 — 같은 표본
    e_q = T.fit_quantum_model_23(emp, np.random.default_rng(SEED + 1000 * i + 7), RESTARTS)
    p_q = to_p(e_q)
    out["S_qfit"] = chsh_from_p(p_q)
    out["ns_l1_qfit"] = ns_residual_l1(p_q)
    out["cf_qfit"] = float(M.contextual_fraction_23(e_q))

    g = N.GuessSDP(3, 3, "2", 0)

    def hS(S):
        r = g.solve(S_obs=S, mode="S", chsh_idx=CH, solver="SCS", eps=1e-8, max_iters=200000)
        return r["h_min"], r["status"]

    def hF(p):
        try:
            r = g.solve(p_obs=p, mode="full", solver="SCS", eps=1e-8, max_iters=200000)
            return r["h_min"], r["status"]
        except Exception as ex:
            return None, "EXC:" + type(ex).__name__

    out["hS_plugin"], out["st_hS_plugin"] = hS(out["S_hat"])
    out["hS_qfit"], out["st_hS_qfit"] = hS(out["S_qfit"])
    # (A)-(ii): 원시 p̂ 먼저
    h, st = hF(p_hat)
    out["hF_plugin_raw"], out["st_hF_plugin_raw"] = h, st
    out["raw_infeasible"] = bool(h is None or "infeasible" in str(st).lower())
    # 사영 후 재시도
    q, neg, qmin, clipped = ns_project(p_hat)
    out.update(proj_neg_cells=neg, proj_min_cell=qmin, proj_clipped_mass=clipped,
               proj_ns_l1=ns_residual_l1(q), S_proj=chsh_from_p(q))
    out["hF_plugin_proj"], out["st_hF_plugin_proj"] = hF(q)
    # (B) 사영 없이 직접
    out["hF_qfit"], out["st_hF_qfit"] = hF(p_q)
    return out


def summarize(vals):
    v = np.array([x for x in vals if x is not None], float)
    if v.size == 0:
        return dict(n=0)
    return dict(n=int(v.size), median=float(np.median(v)),
                q025=float(np.quantile(v, .025)), q975=float(np.quantile(v, .975)),
                mean=float(v.mean()), max=float(v.max()),
                frac_gt_0p01=float((v > 0.01).mean()))


def paired_boot(a, b, rng, B=2000):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = ~(np.isnan(a) | np.isnan(b))
    d = a[m] - b[m]
    if d.size == 0:
        return dict(n=0)
    idx = rng.integers(0, d.size, size=(B, d.size))
    meds = np.median(d[idx], axis=1)
    return dict(n=int(d.size), median_diff=float(np.median(d)),
                ci95=[float(np.quantile(meds, .025)), float(np.quantile(meds, .975))],
                frac_positive=float((d > 0).mean()))


def main():
    print("ALLOC9 =", ALLOC9.tolist(), "합", int(ALLOC9.sum()), flush=True)
    with ProcessPoolExecutor(max_workers=14) as ex:
        rows = list(ex.map(rep_job, range(REPS)))
    rows.sort(key=lambda r: r["rep"])
    for r in rows[:3]:
        print(" rep%d Ŝ=%.4f nsL1=%.4f neg=%d hS_pl=%.4f hF_pl=%.4f hS_qf=%.4f hF_qf=%.4f"
              % (r["rep"], r["S_hat"], r["ns_l1"], r["proj_neg_cells"],
                 r["hS_plugin"] or -1, r["hF_plugin_proj"] or -1,
                 r["hS_qfit"] or -1, r["hF_qfit"] or -1), flush=True)

    g = lambda k: [r[k] for r in rows]
    rng = np.random.default_rng(SEED + 999)
    summ = {k: summarize(g(k)) for k in
            ["hS_plugin", "hS_qfit", "hF_plugin_proj", "hF_qfit"]}
    cmp_ = dict(
        path_S_plugin_minus_qfit=paired_boot(g("hS_plugin"), g("hS_qfit"), rng),
        path_full_plugin_minus_qfit=paired_boot(g("hF_plugin_proj"), g("hF_qfit"), rng),
        amplification_full_minus_S_on_plugin=paired_boot(g("hF_plugin_proj"), g("hS_plugin"), rng),
    )
    diag = dict(
        raw_infeasible_frac=float(np.mean([r["raw_infeasible"] for r in rows])),
        S_gt2_frac=float(np.mean([r["S_gt2"] for r in rows])),
        ns_l1_plugin=summarize(g("ns_l1")), ns_l1_qfit=summarize(g("ns_l1_qfit")),
        proj_neg_cells=summarize([float(r["proj_neg_cells"]) for r in rows]),
        proj_clipped_mass=summarize(g("proj_clipped_mass")),
        proj_min_cell=summarize(g("proj_min_cell")),
        cf_plugin=summarize(g("cf_plugin")), cf_qfit=summarize(g("cf_qfit")),
    )
    json.dump(dict(
        note=("Gate S8 Phase B-2 유령 비트. V=0.70(S_true=1.9799<2, 참 H_min=0). "
              "대조축은 동일 경로 내 plugin 입력 vs 양자적합 입력(PB2-4). "
              "레벨 2·x*=A0·CANON_23."),
        params=dict(V=V_GHOST, n_total=N_TOT, reps=REPS, seed_rule="SEED+1000*i", SEED=SEED,
                    restarts_qfit=RESTARTS, level="2", x_star=0, solver="SCS", eps=1e-8,
                    delft4_source="gate_s3_estimation.py:54 DELFT_RATIO",
                    delft9_alloc=ALLOC9.tolist(),
                    delft9_rule=("CHSH 4컨텍스트는 Delft 실측비율 유지, 나머지 5컨텍스트는 "
                                 "Delft 4개 산술평균 가중치. 합이 n_total 이 되도록 정규화 후 "
                                 "최대잉여법 정수화."),
                    ns_projection=("무신호 아핀부분공간으로의 폐형 최소제곱 사영: 컨텍스트별 "
                                   "{1,a,b,ab} 직교기저에서 ⟨Ax⟩=y평균, ⟨By⟩=x평균, ⟨AxBy⟩ 불변. "
                                   "음수셀은 0으로 클리핑 후 컨텍스트별 재정규화(양 기록)."),
                    S_true=float(M.chsh_S_sub(M.werner_23(M.CANON_23, V_GHOST)))),
        preregistration=dict(
            PB2_1="V=0.70 에서 참 H_min = 0 (양 경로·양 입력).",
            PB2_2="plugin 입력은 무신호 위반(s3 실측 잔차 0.1454) → 원시 상태로 전체통계 SDP 등식제약과 비양립 → infeasible 예측.",
            PB2_3="양자적합 입력은 무신호를 구성상 만족 → 사영 불필요, 유령 비트 중앙값 ≈ 0.",
            PB2_4="대조축은 'S vs full' 이 아니라 동일 경로 내 'plugin 입력 vs 양자적합 입력'."),
        diagnostics=diag, summary=summ, comparisons=cmp_, rows=rows,
        limits=("점추정·i.i.d.·점근. 유한키 조합가능 보안 아님. 마틴게일 단측한계를 쓰는 성숙 "
                "CHSH 프로토콜은 이 병리를 피한다 — 표적은 CF-LP/plugin 관행이다. "
                "정렬(CANON_23) 기하 한정·x*=A0 한정·V=0.70 단일점. NPA 레벨2 상계(레벨3 미확인). "
                "NS 사영은 최소제곱+클리핑이므로 클리핑이 발생한 rep 의 유령 비트에는 사영 인공물이 "
                "섞일 수 있다(발생량 diagnostics 에 기록)."),
    ), open(OUT, "w"), ensure_ascii=False, indent=1)
    print("[저장] %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
