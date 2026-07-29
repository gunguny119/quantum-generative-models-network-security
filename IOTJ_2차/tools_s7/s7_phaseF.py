#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate S7 / Phase F — 교정 예산 창 (few-shot fingerprinting window)

■ 사전등록 예측 (실행 전 고정, 결과와 무관하게 기록)
  P1: n_train ∈ [~500, ~1500] 에 Δ_qe = AUC_q − AUC_emp 의 창이 열린다.
      근거 산술 — 양자 13/(2n)=0.0102 → n≈637, 경험표 27/(2n)=0.0102 → n≈1324.
  P2 (승인된 교정판): 훈련오차 ≈ **floor + d/(2·n_train)**.
      원안 "d/(2n) 단독"은 NC·마지널이 오분류(misspecified)라 구성상 FAIL한다
      (NC 의 하한 = NCfloor = 0.0060078851; gate_s4 A-3 이 (2,2,2)에서 관측한 현상).
      n=10000 저장값 대조: 양자 0.98 / 경험표 1.21 / NC 0.97 (비 = 실측/예측).
  P3: NC 템플릿은 n_train 과 무관하게 blind(≈0.5) 를 유지한다.

■ 성공 판정 (사전 고정, 사후 변경 금지) — 어느 격자점에서 셋 동시 성립 시 "창 실재"
  (i)  Δ_qe 페어드 부트스트랩(B=2000) CI 가 0 배제, 훈련시드 ≥7/10
  (ii) Δ_qe 중앙값 ≥ 0.05
  (iii) 그 점에서 AUC_q ≥ 0.75

■ 평활 (승인된 교정판): `boxes()` 관례 1/(per·4) 는 **n 의존**이다.
  n_train=10000 → 2.250e-04 지만 n_train=245 → per=27 → **9.259e-03** (37배).
  따라서 n-의존 기본값과 고정값 {0, 2.25e-4, 1e-3} 을 **둘 다** 재고,
  민감도 대상에 n_train=245 도 포함한다.

■ 무수정 재사용: gate_s5_multifacet.{eval_kl_23, werner_23, werner_p, nc_floor_23,
  quantum_behavior_23, chsh_S_sub}, s7_finite_template.{fit_quantum_params_23,
  fit_nc_params_23, sample_emp_23}, s7_phase2b.{raw_scores_23, marginal_template_23}
  (raw_scores_23 은 Phase B 에서 원본 detection_auc_23 과 max_abs_diff=0.0 검증됨)
■ 신규: fit_quantum_diag (restart 별 CE 반환 — 적합실패/표본한계 구분용)
"""
import os
import sys
import json
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("MPLBACKEND", "Agg")
import numpy as np
from scipy.optimize import minimize
from concurrent.futures import ProcessPoolExecutor

SCRATCH = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRATCH)
sys.path.insert(0, "/home/elicer/IOTJ/IOTJ_2차/tools")
import gap_vs_cf_v0 as G
import gate_s5_multifacet as M
import fastq
import s7_finite_template as F
import s7_phase2b as B

REPO = "/home/elicer/IOTJ/IOTJ_2차"
SEED, V_NORMAL = M.SEED, 0.856
N_TRAIN_GRID = [245, 490, 700, 1000, 1400, 2000, 3000, 5000, 10000]
N_TRAIN_DET = [245, 490, 700, 1000, 1400, 2000, 3000, 10000]
N_EVAL = [245, 980, 2450]
TRAIN_SEEDS, RESTARTS, REPS, BOOT = 10, 6, 100, 2000
SMOOTH_FIXED = [0.0, 2.25e-4, 1e-3]
SMOOTH_SENS_N = [245, 490, 1000]
JOBS = 14
DOF = dict(quantum=13, empirical=27, classical=15, marginal=6)


def _fast():
    if M.quantum_behavior_23 is not fastq.quantum_behavior_23:
        fastq.install_23(M, verbose=False)


def fit_quantum_diag(emp, rng, restarts):
    """F.fit_quantum_params_23 과 동일 로직 + restart 별 CE 반환(적합 진단용)."""
    _fast()
    ces, best = [], None
    for x0 in F._q_inits_23(rng, restarts):
        r = minimize(lambda p: G._ce(emp, M.quantum_behavior_23(p)), x0, method="Powell",
                     options=dict(maxiter=2000, xtol=1e-6, ftol=1e-9))
        val = G._ce(emp, M.quantum_behavior_23(r.x))
        ces.append(float(val))
        if best is None or val < best[0]:
            best = (val, r.x)
    return best[1], ces


def emp_template(counts_emp, per, smooth):
    e = counts_emp + smooth
    return e / e.sum(axis=(1, 2), keepdims=True)


def _counts(e_true, n, rng):
    per = max(n // 9, 1)
    c = np.zeros((9, 2, 2))
    flat = e_true.reshape(9, 4)
    for i in range(9):
        c[i] = np.bincount(rng.choice(4, size=per, p=flat[i]), minlength=4).reshape(2, 2)
    return c, per


def _tpl_one(t):
    """(n_train, seed) → 세 템플릿 + 경험표 평활 변형 + 진단."""
    nt, ts = t
    _fast()
    e_n = M.werner_23(M.CANON_23, V_NORMAL)
    rng = np.random.default_rng(SEED + 2000 + 1000 * ts + 7 * nt)
    counts, per = _counts(e_n, nt, rng)
    default_s = 1.0 / (per * 4.0)                       # boxes() 관례 (n 의존)
    emp_default = emp_template(counts, per, default_s)

    pq, ces = fit_quantum_diag(emp_default, np.random.default_rng(SEED + 3000 + 1000 * ts + 7 * nt), RESTARTS)
    q = M.quantum_behavior_23(pq)
    c = F.fit_nc_params_23(emp_default, np.random.default_rng(SEED + 4000 + 1000 * ts + 7 * nt), RESTARTS)
    mg = B.marginal_template_23(emp_default)

    variants = {"default": emp_default.tolist()}
    for s in SMOOTH_FIXED:
        variants["s%g" % s] = emp_template(counts, per, s).tolist()

    err = dict(quantum=float(M.eval_kl_23(e_n, q)),
               empirical=float(M.eval_kl_23(e_n, emp_default)),
               classical=float(M.eval_kl_23(e_n, c)),
               marginal=float(M.eval_kl_23(e_n, mg)))
    err_var = {k: float(M.eval_kl_23(e_n, np.array(v))) for k, v in variants.items()}
    return (nt, ts, q.tolist(), emp_default.tolist(), c.tolist(), mg.tolist(),
            err, err_var, dict(default_smooth=default_s, per_ctx=per,
                               ce_restarts=ces, ce_spread=float(max(ces) - min(ces))))


def auc_np(pos, neg_sorted):
    lo = np.searchsorted(neg_sorted, pos, side="left")
    hi = np.searchsorted(neg_sorted, pos, side="right")
    return float((lo + hi).sum() / (2.0 * len(pos) * len(neg_sorted)))


def auc_pair(pos, neg):
    return auc_np(pos, np.sort(neg))


def paired_boot(raw, k1, k2, boot, rng):
    p1, n1 = raw[k1]; p2, n2 = raw[k2]
    na, nn = len(p1), len(n1)
    d = np.empty(boot)
    for b in range(boot):
        ia = rng.integers(0, na, na); inn = rng.integers(0, nn, nn)
        d[b] = auc_pair(p1[ia], n1[inn]) - auc_pair(p2[ia], n2[inn])
    lo, hi = float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))
    return dict(delta=auc_pair(p1, n1) - auc_pair(p2, n2), ci_lo=lo, ci_hi=hi,
                excludes_zero=bool(lo > 0 or hi < 0))


def fit_floor_plus_c_over_n(ns, ys):
    """y = floor + c/n 최소제곱(선형: 설계행렬 [1, 1/n])."""
    n = np.asarray(ns, float); y = np.asarray(ys, float)
    A = np.vstack([np.ones_like(n), 1.0 / n]).T
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ sol
    r2 = 1 - ((y - yh) ** 2).sum() / max(((y - y.mean()) ** 2).sum(), 1e-300)
    return dict(floor=float(sol[0]), c=float(sol[1]), r2=float(r2), n_points=len(ns))


def main():
    t0 = time.time()
    _fast()
    e_n = M.werner_23(M.CANON_23, V_NORMAL)
    ex = json.load(open(os.path.join(REPO, "results2/gate_s5/s5.json")))["verification"]["exact_slsqp"]
    e_atk = M.werner_p(np.array(ex["p"]))
    nc_pop, _ = M.nc_floor_23(e_n, e_n, np.random.default_rng(SEED + 2), 8)

    out = dict(
        preregistered=dict(
            P1="n_train ∈ [~500, ~1500] 에 Δ_qe 창이 열린다 (양자 n≈637, 경험표 n≈1324)",
            P2="훈련오차 ≈ floor + d/(2·n_train)  [승인된 교정판; 원안 d/(2n) 단독은 NC·마지널이 구성상 FAIL]",
            P3="NC 템플릿은 n_train 무관 blind(≈0.5) 유지",
            success_criteria="(i) CI배제 ≥7/10 AND (ii) Δ_qe중앙 ≥0.05 AND (iii) AUC_q ≥0.75",
            dof=DOF, ncfloor_pop=float(nc_pop),
            note="본 블록은 실행 전 고정. 결과와 무관하게 그대로 보고한다."),
        params=dict(seed=SEED, v_normal=V_NORMAL, n_train_grid=N_TRAIN_GRID,
                    n_train_det=N_TRAIN_DET, n_eval=N_EVAL, train_seeds=TRAIN_SEEDS,
                    restarts=RESTARTS, reps=REPS, boot=BOOT, smooth_fixed=SMOOTH_FIXED,
                    smooth_sens_n=SMOOTH_SENS_N,
                    attack_kl=ex["kl_to_normal"], attack_source="verification.exact_slsqp"))

    # ---- 템플릿 적합 (F-1/F-2 공용) ----
    tasks = [(nt, ts) for nt in N_TRAIN_GRID for ts in range(TRAIN_SEEDS)]
    with ProcessPoolExecutor(max_workers=JOBS) as ex_:
        res = list(ex_.map(_tpl_one, tasks))
    res.sort(key=lambda r: (r[0], r[1]))                       # R5: 인덱스 정렬
    T = {}
    for r in res:
        T.setdefault(r[0], []).append(r)
    print("[tpl] %.0fs" % (time.time() - t0), flush=True)

    # ---- F-1: 훈련오차 곡선 ----
    f1 = []
    for nt in N_TRAIN_GRID:
        rows = T[nt]
        rec = dict(n_train=nt, per_ctx=rows[0][8]["per_ctx"], default_smooth=rows[0][8]["default_smooth"])
        for k in ("quantum", "empirical", "classical", "marginal"):
            v = sorted(r[6][k] for r in rows)
            rec[k] = dict(median=float(np.median(v)), min=v[0], max=v[-1], vals=v,
                          pred_d_over_2n=DOF[k] / (2.0 * nt))
        bad = sum(1 for r in rows if r[6]["quantum"] > 3 * np.median([x[6]["quantum"] for x in rows]))
        rec["quantum_fit_diag"] = dict(n_bad_fits=int(bad),
                                       ce_spread_median=float(np.median([r[8]["ce_spread"] for r in rows])),
                                       note="bad = 참값대비 KL 이 시드 중앙값의 3배 초과. 최적화 실패 vs 표본한계 구분용.")
        if nt in SMOOTH_SENS_N:
            rec["smooth_sensitivity"] = {k: float(np.median([r[7][k] for r in rows]))
                                         for k in rows[0][7]}
        f1.append(rec)
        print("  [F-1] n_train=%-6d q=%.4e emp=%.4e nc=%.4e (%.0fs)"
              % (nt, rec["quantum"]["median"], rec["empirical"]["median"],
                 rec["classical"]["median"], time.time() - t0), flush=True)
    fits = {k: fit_floor_plus_c_over_n([r["n_train"] for r in f1], [r[k]["median"] for r in f1])
            for k in ("quantum", "empirical", "classical", "marginal")}
    for k, v in fits.items():
        v["c_pred_d_over_2"] = DOF[k] / 2.0
        v["c_ratio"] = v["c"] / (DOF[k] / 2.0)
    out["F1_training_error"] = dict(rows=f1, fits=fits,
                                    P2_verdict={k: ("PASS" if 0.6 <= v["c_ratio"] <= 1.6 and v["r2"] > 0.9 else "FAIL")
                                                for k, v in fits.items()})

    # ---- F-2: 탐지 창 ----
    cells = []
    for nt in N_TRAIN_DET:
        for Ne in N_EVAL:
            rows = []
            for si, r in enumerate(T[nt]):
                models = dict(quantum=np.array(r[2]), empirical=np.array(r[3]),
                              classical=np.array(r[4]))
                raw = B.raw_scores_23(e_n, e_atk, models, Ne, REPS, SEED + 5000 + 1000 * si)
                raw = {k: (np.asarray(v[0]), np.asarray(v[1])) for k, v in raw.items()}
                aucs = {k: auc_pair(*v) for k, v in raw.items()}
                bqe = paired_boot(raw, "quantum", "empirical", BOOT,
                                  np.random.default_rng(SEED + 7000 + 1000 * si))
                bqn = paired_boot(raw, "quantum", "classical", BOOT,
                                  np.random.default_rng(SEED + 8000 + 1000 * si))
                rows.append(dict(seed=si, auc=aucs, boot_qe=bqe, boot_qn=bqn))
            rows.sort(key=lambda z: z["seed"])
            keys = list(rows[0]["auc"].keys())
            med = {k: float(np.median([r["auc"][k] for r in rows])) for k in keys}
            dqe_med = float(np.median([r["boot_qe"]["delta"] for r in rows]))
            n_excl = int(sum(r["boot_qe"]["excludes_zero"] for r in rows))
            crit = dict(i_ci=bool(n_excl >= 7), ii_effect=bool(dqe_med >= 0.05),
                        iii_detect=bool(med["quantum"] >= 0.75))
            cells.append(dict(n_train=nt, n_eval=Ne, auc_median=med,
                              auc_range={k: [min(r["auc"][k] for r in rows),
                                             max(r["auc"][k] for r in rows)] for k in keys},
                              delta_qe=dict(median=dqe_med, n_seeds_ci_excl_zero=n_excl,
                                            ci_median=[float(np.median([r["boot_qe"]["ci_lo"] for r in rows])),
                                                       float(np.median([r["boot_qe"]["ci_hi"] for r in rows]))],
                                            vals=sorted(r["boot_qe"]["delta"] for r in rows)),
                              delta_qn=dict(median=float(np.median([r["boot_qn"]["delta"] for r in rows])),
                                            n_seeds_ci_excl_zero=int(sum(r["boot_qn"]["excludes_zero"] for r in rows))),
                              criteria=crit, window=bool(all(crit.values()))))
            print("  [F-2] ntr=%-6d Ne=%-5d q=%.3f emp=%.3f nc=%.3f | Δqe=%+.4f (%d/10) | 창=%s (%.0fs)"
                  % (nt, Ne, med["quantum"], med["empirical"], med["classical"],
                     dqe_med, n_excl, cells[-1]["window"], time.time() - t0), flush=True)
    out["F2_detection"] = dict(cells=cells,
                               window_points=[dict(n_train=c["n_train"], n_eval=c["n_eval"]) for c in cells if c["window"]],
                               verdict=("창 실재" if any(c["window"] for c in cells) else "창 부재"))

    # ---- P1 / P3 판정 ----
    best = max(cells, key=lambda c: c["delta_qe"]["median"])
    out["P1_verdict"] = dict(
        predicted_window=[500, 1500],
        observed_max_delta_at=dict(n_train=best["n_train"], n_eval=best["n_eval"],
                                   delta=best["delta_qe"]["median"],
                                   n_seeds_ci_excl_zero=best["delta_qe"]["n_seeds_ci_excl_zero"]),
        verdict=("PASS" if any(c["window"] and 400 <= c["n_train"] <= 1600 for c in cells) else "FAIL"))
    ncs = [c["auc_median"]["classical"] for c in cells]
    out["P3_verdict"] = dict(nc_auc_min=min(ncs), nc_auc_max=max(ncs),
                             max_abs_dev=max(abs(x - 0.5) for x in ncs),
                             verdict=("PASS" if max(abs(x - 0.5) for x in ncs) < 0.15 else "FAIL"))

    # ---- F-3: 운용 지도 ----
    out["F3_map"] = [dict(n_train=c["n_train"], n_eval=c["n_eval"],
                          auc_q=c["auc_median"]["quantum"], auc_emp=c["auc_median"]["empirical"],
                          auc_nc=c["auc_median"]["classical"],
                          detect_q=bool(c["auc_median"]["quantum"] >= 0.9),
                          detect_emp=bool(c["auc_median"]["empirical"] >= 0.9),
                          detect_nc=bool(c["auc_median"]["classical"] >= 0.9)) for c in cells]

    out["limits"] = ("공격점 1개(exact_slsqp). 훈련시드 10벌·reps=100. 부트스트랩은 시드별 페어드라 "
                     "시드 간 변동은 CI 밖(‘CI배제 k/10’으로 병기). 평활 민감도는 n_train ∈ {245,490,1000}. "
                     "★성공해도 서술은 ‘적은 교정 데이터에서의 표본효율’이지 탐지력 우위·supremacy 아님. R6 준수.")
    out["elapsed_s"] = time.time() - t0
    return out


if __name__ == "__main__":
    r = main()
    p = sys.argv[1] if len(sys.argv) > 1 else os.path.join(REPO, "results2/gate_s7/s7_phaseF.json")
    json.dump(r, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nwrote", p, "| %.0fs" % r["elapsed_s"])
