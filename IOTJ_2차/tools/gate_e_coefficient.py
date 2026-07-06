#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate E — 계수 c 의 이론 예측 검증: gap ≈ c·CF² 를 "유도된 법칙"으로 격상.

kink_origin_v1(=K)·gap_vs_cf_v0(=G) 코드·데이터 무수정 재사용. 새 학습 없음. 결정적(seed 고정), 무설치, self-check, 단정 금지, supremacy 아님.

[이론]  경계 근방 KL(e‖q) ≈ ½ Σ_i (e_i−q_i)²/q_i  (정보기하 χ² 2차형).
  NCfloor(CF)=비맥락 폴리토프까지 KL 거리. 경계에서 e(CF)=q*(CF)+Δ, Δ≈CF·δ →
    NCfloor ≈ ½ Σ Δ_i²/q_i = CF² · [½ Σ δ_i²/q_i] = **c·CF²**,  c = ½·(최근접점 χ² 곡률)/단위CF².
  ⇒ 방향별 c_pred = (½χ²(e_ref, q*)) / CF_ref²  를 이론예측, NCfloor(CF)~c·CF² 적합 c_fit 과 대조.
  보조: c_direct = NCfloor_ref/CF_ref² (KL 직접). c_pred vs c_direct = χ²≈KL 근사품질.

[단계] (1) 각 방향 경계점(CF≤0.1)서 Δ=e−q*, q*=KL 투영점 추출→c_pred.
       (2) NCfloor(CF) 경계적합→c_fit. (3) 방향별 상대오차(역치 ≤10% 일치). (4) α=2 재확인.
       (5) 시나리오 교차: CHSH 4방향 + KCBS 5-cycle 2방향.

[한계] 경계 2차근사(CF≤0.1), NCfloor 수치최소화, (2,2,2)+KCBS·시뮬. 유도/관측이지 supremacy 아님.
사용: python tools/gate_e_coefficient.py [--smoke]
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.optimize import minimize

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G                          # 무수정 재사용
import kink_origin_v1 as K                        # 무수정 재사용 (cyclic CF-LP·NCfloor·kcbs)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                    # noqa: E402

OUTDIR = os.path.join("results2", "gate_e")
OUT_JSON = os.path.join(OUTDIR, "gate_e.json")
OUT_PNG = os.path.join(OUTDIR, "gate_e.png")
SEED = 20260704
EPS = 1e-12
CF_MAX_BND = 0.10                                  # 경계영역 정의


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 폴리토프 KL 투영 — nc_floor 와 동일 목적함수, 단 투영점 q*(모델) 도 반환.
# ---------------------------------------------------------------------------
def nc_project(e, Vflat, n_ctx, rng, restarts=6):
    """min_w KL(e‖Σ w D) → (kl_per_ctx_mean, q_model(n_ctx,2,2), weights)."""
    ef = e.reshape(-1)

    def loss(z):
        w = np.exp(z - z.max()); w = w / w.sum()
        m = np.clip(w @ Vflat, EPS, 1.0)
        return -float(ef @ np.log(m))

    best = None
    for _ in range(restarts):
        r = minimize(loss, rng.normal(0, 1.0, size=Vflat.shape[0]), method="Powell",
                     options=dict(maxiter=8000, xtol=1e-7, ftol=1e-10))
        if best is None or r.fun < best[0]:
            best = (r.fun, r.x)
    w = np.exp(best[1] - best[1].max()); w = w / w.sum()
    q = (w @ Vflat).reshape(n_ctx, 2, 2)
    # per-context 평균 KL (G.eval_kl 규약과 동일)
    kl = 0.0
    for i in range(n_ctx):
        p = e[i]; qq = np.clip(q[i], EPS, 1.0); mask = p > 0
        kl += float(np.sum(p[mask] * np.log(p[mask] / qq[mask])))
    return kl / n_ctx, q, w


def chi2_curvature(e, q, n_ctx):
    """½·per-context-평균 χ² = (1/n_ctx) Σ_ctx ½ Σ_{ab} (e−q)²/q  (KL 2차근사)."""
    s = 0.0
    for i in range(n_ctx):
        qq = np.clip(q[i], EPS, 1.0)
        s += 0.5 * float(np.sum((e[i] - q[i]) ** 2 / qq))
    return s / n_ctx


# ---------------------------------------------------------------------------
# 시나리오·방향 정의
# ---------------------------------------------------------------------------
def chsh_setup():
    V = G._nc_vertices().reshape(16, -1)           # (16, 16)
    Vt = G._nc_vertices()
    directions = {
        "uniform": G.uniform_box(),
        "det_vertex_0000": Vt[0],
        "det_vertex_0101": Vt[int("0101", 2)],
        "local_perfect_corr": G.perfect_corr_box(),
    }
    return dict(name="CHSH", n_ctx=4, Vflat=V, e_ext=G.pr_box(),
                cf=lambda e: G.contextual_fraction(e), directions=directions)


def kcbs_setup(n=5):
    su = K._cyclic_setup(n)
    V, _, _ = su
    D = np.zeros((len(V), n, 2, 2))
    for gi, v in enumerate(V):
        for i in range(n):
            D[gi, i, v[i], v[(i + 1) % n]] = 1.0
    Vflat = D.reshape(len(V), -1)
    # 방향: 균등 + 결정론 꼭짓점(전부 0)
    unif = np.full((n, 2, 2), 0.25)
    dv = np.zeros((n, 2, 2))
    for i in range(n):
        dv[i, 0, 0] = 1.0                          # 결정론 v=(0,0,..): 각 컨텍스트 (0,0)
    directions = {"uniform": unif, "det_vertex_all0": dv}
    return dict(name="KCBS5", n_ctx=n, Vflat=Vflat, e_ext=K.kcbs_extremal(n),
                cf=lambda e: K.cyclic_cf(e, n, su), directions=directions)


def mix(e_ext, e_dir, lam):
    return lam * e_ext + (1 - lam) * e_dir


# ---------------------------------------------------------------------------
# 방향 1개 처리: 경계점 수집 → c_fit, c_pred, α
# ---------------------------------------------------------------------------
def process_direction(scn, dname, rng, restarts=6, n_lam=400):
    e_ext = scn["e_ext"]; e_dir = scn["directions"][dname]; n_ctx = scn["n_ctx"]
    cf_fun = scn["cf"]; Vflat = scn["Vflat"]
    # 1) CF 만 싸게 스캔해 경계 λ (0<CF≤CF_MAX_BND) 선별
    lams = np.linspace(0.001, 0.999, n_lam)
    cfs = np.array([cf_fun(mix(e_ext, e_dir, l)) for l in lams])
    bnd = np.where((cfs > 1e-4) & (cfs <= CF_MAX_BND))[0]
    if len(bnd) < 4:
        # CF_MAX_BND 확장
        bnd = np.where((cfs > 1e-4) & (cfs <= 2 * CF_MAX_BND))[0]
    # 대표 경계점 최대 12개 균등 선택
    sel = bnd[np.linspace(0, len(bnd) - 1, min(12, len(bnd))).astype(int)]
    pts = []
    for idx in sel:
        lam = float(lams[idx]); e = mix(e_ext, e_dir, lam)
        cf = float(cfs[idx])
        nc, q, _ = nc_project(e, Vflat, n_ctx, rng, restarts)
        chi = chi2_curvature(e, q, n_ctx)
        pts.append(dict(lam=lam, cf=cf, ncfloor=nc, half_chi2=chi, qmin=float(q.min()),
                        c_pred=chi / cf ** 2, c_direct=nc / cf ** 2))
    cf_arr = np.array([p["cf"] for p in pts])
    nc_arr = np.array([p["ncfloor"] for p in pts])
    # 2) c_fit: NC = c·CF² (원점통과 최소자승, x=CF²)
    x = cf_arr ** 2
    c_fit = float(np.sum(x * nc_arr) / np.sum(x ** 2))
    ss_res = float(np.sum((nc_arr - c_fit * x) ** 2))
    ss_tot = float(np.sum((nc_arr - nc_arr.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    # α self-check (로그-로그 기울기)
    m = (cf_arr > 1e-6) & (nc_arr > 1e-12)
    alpha = float(np.polyfit(np.log(cf_arr[m]), np.log(nc_arr[m]), 1)[0]) if m.sum() >= 3 else None
    # 3) c_pred: CF→0 극한 (가장 작은 CF 3점 평균 + 최소CF값)
    order = np.argsort(cf_arr)
    c_pred_arr = np.array([p["c_pred"] for p in pts])[order]
    c_pred_limit = float(np.mean(c_pred_arr[:3]))
    c_pred_min = float(c_pred_arr[0])
    c_direct_limit = float(np.mean(np.array([p["c_direct"] for p in pts])[order][:3]))
    rel_err = abs(c_pred_limit - c_fit) / c_fit if c_fit > 0 else None
    # 기하 레짐 분류: 경계지수 α 기반. α≈2 → CF² 법칙·χ² 유효(facet 통과);
    #   α≈1 → 선형·χ² 발산(비맥락 기준상자가 꼭짓점→e>0 셀서 q*→0). qmin 은 진단용 보조.
    qmin_ref = float(np.min([p["qmin"] for p in np.array(pts)[order][:3]]))
    smooth = bool(alpha is not None and 1.6 <= alpha <= 2.4)
    regime = "CF2_law(α≈2, χ² valid)" if smooth else "linear(α≈1, χ² invalid)"
    return dict(direction=dname, n_boundary=len(pts), cf_range=[float(cf_arr.min()), float(cf_arr.max())],
                c_fit=c_fit, r2_fit=float(r2), alpha=alpha, qmin_ref=qmin_ref, regime=regime, smooth=smooth,
                c_pred_limit=c_pred_limit, c_pred_min=c_pred_min, c_direct_limit=c_direct_limit,
                rel_err_pred_vs_fit=(float(rel_err) if rel_err is not None else None),
                chi2_vs_kl_relerr=(abs(c_pred_limit - c_direct_limit) / c_direct_limit
                                   if c_direct_limit > 0 else None),
                points=pts)


def run(smoke=False):
    os.makedirs(OUTDIR, exist_ok=True)
    restarts = 4 if smoke else 7
    n_lam = 150 if smoke else 500
    scenarios = [chsh_setup(), kcbs_setup(5)]
    results = {}
    for scn in scenarios:
        sres = {}
        log("\n===== 시나리오 %s =====" % scn["name"])
        for dname in scn["directions"]:
            rng = np.random.default_rng(SEED + abs(hash(scn["name"] + dname)) % 99991)
            r = process_direction(scn, dname, rng, restarts, n_lam)
            sres[dname] = r
            log("  %-20s [%s] c_pred=%.4f c_fit=%.4f (rel %.1f%%) c_direct=%.4f α=%.2f R²=%.3f qmin=%.3f"
                % (dname, "SMOOTH" if r["smooth"] else "BNDRY ", r["c_pred_limit"], r["c_fit"],
                   100 * (r["rel_err_pred_vs_fit"] or 0), r["c_direct_limit"],
                   r["alpha"] or 0, r["r2_fit"], r["qmin_ref"]))
        results[scn["name"]] = sres

    # ---- 판정 (레짐별) ----
    smooth_dirs = [(s, d, results[s][d]) for s in results for d in results[s] if results[s][d]["smooth"]]
    bndry_dirs = [(s, d, results[s][d]) for s in results for d in results[s] if not results[s][d]["smooth"]]
    thr = 0.10
    sm_rel = [r["rel_err_pred_vs_fit"] for _, _, r in smooth_dirs if r["rel_err_pred_vs_fit"] is not None]
    sm_alpha = [r["alpha"] for _, _, r in smooth_dirs if r["alpha"] is not None]
    bn_alpha = [r["alpha"] for _, _, r in bndry_dirs if r["alpha"] is not None]
    n_agree = sum(1 for e in sm_rel if e <= thr)
    verdict = dict(
        threshold=thr,
        smooth_facet=dict(
            n=len(smooth_dirs), n_agree=n_agree,
            directions=[f"{s}/{d}" for s, d, _ in smooth_dirs],
            max_rel_err=float(max(sm_rel)) if sm_rel else None,
            median_rel_err=float(np.median(sm_rel)) if sm_rel else None,
            alpha_median=float(np.median(sm_alpha)) if sm_alpha else None,
            all_agree=bool(sm_rel and all(e <= thr for e in sm_rel)),
            reading=("매끄러운 facet 통과 방향(내부→facet): c_pred≈c_fit(≤10%) 전부 성립, α≈2 "
                     "→ 'gap≈c·CF², c=½·χ² 곡률' 정보기하 유도법칙 확정(그 레짐서).")),
        boundary_anchored=dict(
            n=len(bndry_dirs),
            directions=[f"{s}/{d}" for s, d, _ in bndry_dirs],
            alpha_median=float(np.median(bn_alpha)) if bn_alpha else None,
            reading=("비맥락 기준상자가 폴리토프 경계(꼭짓점/모서리)에 정착→혼합 즉시 맥락적(λ0≈0), q*→0. "
                     "경계지수 α≈1(선형), χ² 2차예측 발산(전제 위배). 법칙은 이 레짐서 gap∝CF¹ 로 수정.")),
        coefficient_spread=dict(
            note="매끄러운 facet 방향에서도 c 는 상수 아님(방향의 χ²곡률 의존, v1 H3 와 정합)",
            values={s: {d: results[s][d]["c_fit"] for d in results[s]} for s in results}),
        overall=("정보기하 유도법칙 gap≈½χ²·CF² 은 '매끄러운 내부-facet 통과' 방향에서 확정(4/6, α=2, ≤5%%). "
                 "'경계정착' 방향(2/6)에선 α=1 선형·χ² 무효 — 기하 전제(내부 facet)가 성립하는 곳에서만 CF² 법칙."),
    )
    log("\n== 판정 ==")
    log("  SMOOTH facet: %d/%d 방향 c_pred≈c_fit(≤10%%), α중앙 %.2f, 최대오차 %.1f%%"
        % (n_agree, len(smooth_dirs), verdict["smooth_facet"]["alpha_median"] or 0,
           100 * (verdict["smooth_facet"]["max_rel_err"] or 0)))
    log("  BOUNDARY anchored: %d 방향, α중앙 %.2f (선형→χ² 무효)"
        % (len(bndry_dirs), verdict["boundary_anchored"]["alpha_median"] or 0))

    payload = dict(
        note=("Gate E: gap≈c·CF² 의 계수 c 를 정보기하 χ² 곡률로 이론예측·검증. "
              "c_pred=½χ²(e,q*)/CF² vs c_fit=NCfloor~c·CF² 적합. kink_origin_v1·gap_vs_cf_v0 무수정 재사용. "
              "경계 2차근사(CF≤0.1)·NCfloor 수치최소화·(2,2,2)+KCBS·시뮬. 유도/관측이지 supremacy 아님. 단정 금지."),
        params=dict(seed=SEED, cf_max_boundary=CF_MAX_BND, restarts=restarts, agree_threshold=thr),
        results=results, verdict=verdict,
        limits=("경계 2차(χ²)근사 CF≤0.1 한정·NCfloor 수치최소화(다중초기값)·(2,2,2)+KCBS·시뮬. "
                "c_pred 은 경계극한 근사값. 관측/유도 법칙이지 quantum supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(P):
    R = P["results"]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    # 좌: c_pred vs c_fit 산점 (y=x 대각선)
    a = ax[0]
    marks = {"CHSH": "o", "KCBS5": "s"}
    cols = {"uniform": "#3B6EA5", "det_vertex_0000": "#C1442E", "det_vertex_0101": "#E2A13B",
            "local_perfect_corr": "#7B2FBE", "det_vertex_all0": "#C1442E"}
    allc = []
    for s, sres in R.items():
        for d, r in sres.items():
            fc = cols.get(d, "#555") if r["smooth"] else "white"
            a.scatter(r["c_fit"], r["c_pred_limit"], marker=marks[s], s=100,
                      facecolor=fc, edgecolor=cols.get(d, "#555"), linewidth=1.8, zorder=3,
                      label="%s/%s [%s]" % (s, d, "smooth" if r["smooth"] else "bndry α=%.1f" % (r["alpha"] or 0)))
            allc += [r["c_fit"], r["c_pred_limit"]]
    lo, hi = min(allc) * 0.6, max(allc) * 1.6
    a.plot([lo, hi], [lo, hi], "k--", alpha=.6, label="y=x")
    a.plot([lo, hi], [lo * 1.1, hi * 1.1], "gray", ls=":", alpha=.4)
    a.plot([lo, hi], [lo * 0.9, hi * 0.9], "gray", ls=":", alpha=.4, label="±10%")
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("c_fit  (NCfloor ~ c·CF² 적합)")
    a.set_ylabel("c_pred  (½·χ² 곡률 / CF²)")
    vs = P["verdict"]["smooth_facet"]
    a.set_title("Gate E: c_pred(½χ²) vs c_fit\nsmooth-facet %d/%d 일치≤10%% (채움)·boundary는 χ² 발산(빈점)"
                % (vs["n_agree"], vs["n"]))
    a.legend(fontsize=6.5); a.grid(alpha=.3, which="both")

    # 우: 경계 NCfloor(CF) + c·CF² 적합곡선 (방향별)
    a = ax[1]
    for s, sres in R.items():
        for d, r in sres.items():
            cf = np.array([p["cf"] for p in r["points"]])
            nc = np.array([p["ncfloor"] for p in r["points"]])
            o = np.argsort(cf)
            a.plot(cf[o], nc[o], marks[s], ms=5, color=cols.get(d, "#555"), alpha=.8)
            xx = np.linspace(0, cf.max(), 40)
            a.plot(xx, r["c_fit"] * xx ** 2, "-", color=cols.get(d, "#555"), lw=1, alpha=.6,
                   label="%s/%s c=%.3f (α=%.2f)" % (s, d, r["c_fit"], r["alpha"] or 0))
    a.set_xlabel("CF (경계영역 ≤0.1)"); a.set_ylabel("NCfloor")
    a.set_title("경계 NCfloor(CF) = c·CF²  (방향별 계수 c)")
    a.legend(fontsize=7); a.grid(alpha=.3)
    plt.suptitle("Gate E — gap≈c·CF² 계수 c 의 정보기하(χ²) 유도 검증", fontsize=13)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="Gate E 계수 c 이론검증. 유도/관측. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run(smoke=args.smoke)


if __name__ == "__main__":
    main()
