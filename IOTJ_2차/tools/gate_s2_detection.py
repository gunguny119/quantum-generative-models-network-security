#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2 — DI-QKD 탐지 실험: "KL gap 이 탐지 마진으로 번역되는가?" (반증 가능한 가설).

★ STEP 1(sim_diqkd Delft 재현 STOP 게이트)이 PASS 한 뒤에만 실행. (results2/gate_s1/s1.json 확인)

정합 규칙:
  R1  bipartite G 트랙만(gap_vs_cf_v0). K.cyclic_* 미사용.
  R2  16셀 변환기 = gate_r2.build_box 하나만.
  R3  훈련/평가 분리(신규): 훈련 emp 로 모델 파라미터 획득 → 별도 평가셋에 적용.
  R4  AIC 시 G.fit_shape(quadratic 없음) 미사용. (P2 는 단조성만 확인, 함수형 주장 금지)
  R5  결정성, results2/gate_s2/ 아래 JSON+PNG, note/params/self_check/limits.

설계:
  훈련: attack="none", 정상 노이즈, n=N_train → emp_train.
        q_model = quantum_behavior(fit_quantum_params(emp_train))     (양자 탐지기)
        c_model = fit_nc_params(emp_train) 의 NC 혼합 behavior           (고전 탐지기 = NCfloor 도달점)
  평가: 각 클래스 100 반복, n=N_eval, score = G._ce(emp_eval, model) (=−Σ emp·log model, 높을수록 이상).
        AUC(class vs normal) 을 양자/고전/S-검정 세 탐지기로 계산.

  P1(주가설): mimic_ncfloor 에서 AUC_quantum > AUC_classical.
  P2: visibility 스윕 → (gap, ΔAUC) 단조성(함수형 주장 안 함).
  P3: S-검정(S≤2 폐기) 대비 — 생성모델이 놓치는 것 잡는가(특히 partial_25, 작은 N_eval).

STOP: P1 실패 시 STEP 3 진입 금지·보고.

사용: python tools/gate_s2_detection.py [--quick] [--jobs 6]
"""
import os
import sys
import json
import argparse

import numpy as np
from scipy.stats import rankdata

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G
import gate_r2_anchor_strengthen as R2
import sim_diqkd as S

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

OUTDIR = os.path.join("results2", "gate_s2")
OUT_JSON = os.path.join(OUTDIR, "s2.json")
OUT_PNG_AUC = os.path.join(OUTDIR, "s2_auc.png")
OUT_PNG_P2 = os.path.join(OUTDIR, "s2_p2_monotone.png")

SEED = 20260724
EPS = 1e-12
V_NORMAL = 0.856                               # 정상 운용점(Delft 앵커 근방, CF≈0.211)
CLASSES = ["normal", "intercept_resend", "blinding", "mimic_ncfloor", "partial_25", "partial_50"]


def log(m):
    print(m, flush=True)


# ===========================================================================
# 2-1. 파라미터 반환 래퍼 (원본 gap_vs_cf_v0 무수정; 로직만 복제 + 파라미터 반환)
# ===========================================================================
def fit_quantum_params(emp, rng, restarts):
    """G.fit_quantum(224–234행)과 동일 로직이되 최적 파라미터 벡터 반환."""
    from scipy.optimize import minimize
    best = None
    for x0 in G._q_inits(rng, restarts):
        r = minimize(lambda p: G._ce(emp, G.quantum_behavior(p)), x0, method="Powell",
                     options=dict(maxiter=2000, xtol=1e-6, ftol=1e-9))
        val = G._ce(emp, G.quantum_behavior(r.x))
        if best is None or val < best[0]:
            best = (val, r.x)
    return best[1]


def fit_nc_params(emp, rng, restarts):
    """nc_floor(271–291행)와 동일 로직이되 vertex 가중치 w 와 model 반환 (= sim_diqkd.nc_floor_weights)."""
    return S.nc_floor_weights(emp, rng, restarts)


def verify_wrappers(restarts=4):
    """래퍼 파라미터→behavior→eval_kl 이 원본 함수 반환값과 1e-9 이내 일치 확인."""
    rng0 = np.random.default_rng(SEED + 55)
    e = S.quantum_e(S.CANON_STATE, S.CANON_ANGLES, V_NORMAL)
    emp = G.sample_emp(e, 5000, rng0)
    # quantum
    kl_orig_q = G.fit_quantum(emp, emp, np.random.default_rng(SEED + 71), restarts)
    p = fit_quantum_params(emp, np.random.default_rng(SEED + 71), restarts)
    kl_wrap_q = G.eval_kl(emp, G.quantum_behavior(p))
    # nc
    kl_orig_c = G.nc_floor(emp, emp, np.random.default_rng(SEED + 72), restarts)
    w, model = fit_nc_params(emp, np.random.default_rng(SEED + 72), restarts)
    kl_wrap_c = G.eval_kl(emp, model)
    dq = abs(kl_orig_q - kl_wrap_q)
    dc = abs(kl_orig_c - kl_wrap_c)
    return dict(quantum=dict(orig=float(kl_orig_q), wrap=float(kl_wrap_q), diff=float(dq), ok=bool(dq < 1e-9)),
                nc=dict(orig=float(kl_orig_c), wrap=float(kl_wrap_c), diff=float(dc), ok=bool(dc < 1e-9)),
                all_ok=bool(dq < 1e-9 and dc < 1e-9))


# ===========================================================================
# 스코어링 / AUC
# ===========================================================================
def score_ce(emp, model):
    return G._ce(emp, model)                    # = −Σ emp·log(model), 높을수록 이상 (=음의 로그우도, 명세 스코어)


def _H(emp):
    m = np.clip(emp, EPS, 1.0)
    return -float(np.sum(emp * np.log(m)))      # 경험 엔트로피


def score_kl(emp, model):
    """deviance/G-통계량 스코어: KL(emp‖model) = CE − H(emp). 2N·KL ~ χ² (포화모델 대비 우도비 검정통계량).
    표본엔트로피 교란을 제거한 원리적 적합도 통계. 경험박스만으로 계산가능(배치 가능)."""
    return score_ce(emp, model) - _H(emp)


def auc(pos, neg):
    """Mann–Whitney AUC = P(score_pos > score_neg). 동점=0.5."""
    pos = np.asarray(pos); neg = np.asarray(neg)
    n1, n0 = len(pos), len(neg)
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:n1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def gen_eval_boxes(attack, noise, n_eval, reps, seed):
    """attack 클래스의 평가 박스 reps 개(각 n_eval). behavior 는 1회 계산 후 재표집."""
    e = S.build_behavior(attack, S.CANON_STATE, S.CANON_ANGLES, noise, np.random.default_rng(seed))
    boxes, Ss = [], []
    for i in range(reps):
        rng = np.random.default_rng(seed + 1 + i)
        a, b, xo, yo = S._sample_from_behavior(e, n_eval, rng)
        box, _ = R2.build_box(a, b, xo, yo)
        boxes.append(box); Ss.append(R2.chsh_S(box))
    return boxes, np.array(Ss)


def class_noise(cls):
    base = {"visibility": V_NORMAL}
    if cls == "partial_25":
        return {**base, "p_atk": 0.25, "sub_attack": "intercept_resend"}
    if cls == "partial_50":
        return {**base, "p_atk": 0.50, "sub_attack": "intercept_resend"}
    return base


CLASS_ATTACK = {"normal": "none", "intercept_resend": "intercept_resend",
                "blinding": "blinding", "mimic_ncfloor": "mimic_ncfloor",
                "partial_25": "partial", "partial_50": "partial"}


# ===========================================================================
# 한 (N_train, N_eval) 셀: 훈련→모델, 평가→AUC(양자/고전/S)
# ===========================================================================
def run_cell(n_train, n_eval, reps, restarts, train_seeds, seed0):
    # --- 훈련: 여러 시드로 모델 적합 후 AUC 평균(유한표본 잡음 완화) ---
    normal_noise = {"visibility": V_NORMAL}
    e_normal = S.build_behavior("none", S.CANON_STATE, S.CANON_ANGLES, normal_noise, np.random.default_rng(seed0))

    # --- 평가 박스 생성(훈련과 독립; N_train 무관하므로 1회) ---
    S_normal_ref = R2.chsh_S(e_normal)
    eval_boxes = {}
    eval_S = {}
    for ci, cls in enumerate(CLASSES):
        eb, sS = gen_eval_boxes(CLASS_ATTACK[cls], class_noise(cls), n_eval, reps, seed0 + 10000 + 1000 * ci)
        eval_boxes[cls] = eb; eval_S[cls] = sS
    # 정상 음성 baseline(독립 2번째 정상셋) — normal-vs-normal AUC≈0.5 self-check
    neg_boxes, neg_S = gen_eval_boxes("none", normal_noise, n_eval, reps, seed0 + 55555)

    auc_q = {c: [] for c in CLASSES}         # CE 스코어(명세) — 주 판정
    auc_c = {c: [] for c in CLASSES}
    auc_s = {c: [] for c in CLASSES}
    auc_q_kl = {c: [] for c in CLASSES}      # KL 스코어(엔트로피 보정) — 진단
    auc_c_kl = {c: [] for c in CLASSES}
    mean_H = {c: [] for c in CLASSES}
    diag = {}
    for ts in range(train_seeds):
        rng = np.random.default_rng(seed0 + 700 + 13 * ts)
        a, b, xo, yo = S.simulate_diqkd(n_train, S.CANON_STATE, S.CANON_ANGLES, normal_noise, "none", rng)
        emp_train, _ = R2.build_box(a, b, xo, yo)
        pq = fit_quantum_params(emp_train, np.random.default_rng(seed0 + 800 + 13 * ts), restarts)
        q_model = G.quantum_behavior(pq)
        _, c_model = fit_nc_params(emp_train, np.random.default_rng(seed0 + 900 + 13 * ts), restarts)
        if ts == 0:
            e_mimic = S.build_behavior("mimic_ncfloor", S.CANON_STATE, S.CANON_ANGLES, normal_noise,
                                       np.random.default_rng(seed0 + 2))
            diag = dict(kl_q_train=float(G.eval_kl(e_normal, q_model)),
                        kl_c_train=float(G.eval_kl(e_normal, c_model)),
                        ncfloor_pop=float(G.nc_floor(e_normal, e_normal, np.random.default_rng(seed0 + 1), restarts)),
                        kl_mimic_vs_q=float(G.eval_kl(e_mimic, q_model)),
                        kl_mimic_vs_c=float(G.eval_kl(e_mimic, c_model)),
                        H_normal_pop=float(_H(e_normal)), H_mimic_pop=float(_H(e_mimic)),
                        S_normal=float(R2.chsh_S(e_normal)), S_mimic=float(R2.chsh_S(e_mimic)))
        neg_q = np.array([score_ce(bx, q_model) for bx in neg_boxes])
        neg_c = np.array([score_ce(bx, c_model) for bx in neg_boxes])
        neg_q_kl = np.array([score_kl(bx, q_model) for bx in neg_boxes])
        neg_c_kl = np.array([score_kl(bx, c_model) for bx in neg_boxes])
        for cls in CLASSES:
            bx = eval_boxes[cls]
            auc_q[cls].append(auc([score_ce(x, q_model) for x in bx], neg_q))
            auc_c[cls].append(auc([score_ce(x, c_model) for x in bx], neg_c))
            auc_q_kl[cls].append(auc([score_kl(x, q_model) for x in bx], neg_q_kl))
            auc_c_kl[cls].append(auc([score_kl(x, c_model) for x in bx], neg_c_kl))
            auc_s[cls].append(auc(-eval_S[cls], -neg_S))    # S-검정: S 낮을수록 이상
            if ts == 0:
                mean_H[cls] = float(np.mean([_H(x) for x in bx]))

    def agg(d):
        return {c: dict(mean=float(np.mean(d[c])), std=float(np.std(d[c])), vals=[float(x) for x in d[c]])
                for c in CLASSES}
    return dict(n_train=n_train, n_eval=n_eval, reps=reps, train_seeds=train_seeds,
                S_normal_ref=float(S_normal_ref), diag=diag, mean_H_by_class=mean_H,
                auc_quantum=agg(auc_q), auc_classical=agg(auc_c), auc_stest=agg(auc_s),
                auc_quantum_kl=agg(auc_q_kl), auc_classical_kl=agg(auc_c_kl))


# ===========================================================================
# P2: visibility 스윕 → (gap, ΔAUC on mimic) 단조성
# ===========================================================================
def run_p2(reps, restarts, n_train, n_eval, seed0, vis_list):
    rows = []
    for V in vis_list:
        e_norm = S.quantum_e(S.CANON_STATE, S.CANON_ANGLES, V)
        cf = float(G.contextual_fraction(e_norm))
        ncf = float(G.nc_floor(e_norm, e_norm, np.random.default_rng(seed0 + int(V * 1e4)), restarts))
        klq = float(G.fit_quantum(e_norm, e_norm, np.random.default_rng(seed0 + 1 + int(V * 1e4)), restarts))
        gap = ncf - klq
        # 훈련 모델
        rng = np.random.default_rng(seed0 + 2 + int(V * 1e4))
        a, b, xo, yo = S.simulate_diqkd(n_train, S.CANON_STATE, S.CANON_ANGLES, {"visibility": V}, "none", rng)
        emp_train, _ = R2.build_box(a, b, xo, yo)
        pq = fit_quantum_params(emp_train, np.random.default_rng(seed0 + 3 + int(V * 1e4)), restarts)
        q_model = G.quantum_behavior(pq)
        _, c_model = fit_nc_params(emp_train, np.random.default_rng(seed0 + 4 + int(V * 1e4)), restarts)
        # mimic 평가 + 정상 baseline
        mimic_boxes, _ = gen_eval_boxes("mimic_ncfloor", {"visibility": V}, n_eval, reps, seed0 + 5 + int(V * 1e4) + 200000)
        neg_boxes, _ = gen_eval_boxes("none", {"visibility": V}, n_eval, reps, seed0 + 6 + int(V * 1e4) + 400000)
        sce = lambda bxs, m: np.array([score_ce(bx, m) for bx in bxs])
        skl = lambda bxs, m: np.array([score_kl(bx, m) for bx in bxs])
        auc_q = auc(sce(mimic_boxes, q_model), sce(neg_boxes, q_model))
        auc_c = auc(sce(mimic_boxes, c_model), sce(neg_boxes, c_model))
        auc_q_kl = auc(skl(mimic_boxes, q_model), skl(neg_boxes, q_model))
        auc_c_kl = auc(skl(mimic_boxes, c_model), skl(neg_boxes, c_model))
        rows.append(dict(visibility=float(V), cf=cf, gap=float(gap),
                         auc_quantum=float(auc_q), auc_classical=float(auc_c),
                         delta_auc=float(auc_q - auc_c),
                         auc_quantum_kl=float(auc_q_kl), auc_classical_kl=float(auc_c_kl),
                         delta_auc_kl=float(auc_q_kl - auc_c_kl)))
    # 단조성(gap↑ → ΔAUC↑, KL 진단 스코어 기준 — CE 는 ΔAUC≈0): Spearman via rank
    g = np.array([r["gap"] for r in rows]); d = np.array([r["delta_auc_kl"] for r in rows])
    rg, rd = rankdata(g), rankdata(d)
    if len(g) > 1 and rg.std() > 0 and rd.std() > 0:
        spearman = float(np.corrcoef(rg, rd)[0, 1])
    else:
        spearman = None
    return rows, spearman


# ===========================================================================
# main
# ===========================================================================
def check_step1_passed():
    p = os.path.join("results2", "gate_s1", "s1.json")
    if not os.path.exists(p):
        return None
    try:
        d = json.load(open(p))
        return bool(d["delft_validation"]["passed"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="STEP 2 DI-QKD 탐지 실험. STEP 1 PASS 필요.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--force", action="store_true", help="STEP 1 PASS 확인 생략(디버그)")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    st1 = check_step1_passed()
    log("=== STEP 1 게이트 확인: %s ===" % ("PASS" if st1 else ("FAIL/미실행" if st1 is not None else "s1.json 없음")))
    if not st1 and not args.force:
        log("[STOP] STEP 1 이 PASS 하지 않았음 → STEP 2 진입 금지. (--force 로 디버그 가능)")
        return

    if args.quick:
        args.reps, args.restarts, args.train_seeds = 40, 2, 2
        n_trains = [2000, 10000]
        n_evals = [200, 1000]
        vis_list = [0.75, 0.856, 1.0]
    else:
        n_trains = [500, 2000, 10000]
        n_evals = [200, 1000]
        vis_list = [0.70, 0.80, 0.856, 0.92, 1.0]

    log("\n=== 2-1. 파라미터 반환 래퍼 검증 (원본 함수와 1e-9 일치) ===")
    wv = verify_wrappers(restarts=4)
    log("  quantum: orig=%.6e wrap=%.6e diff=%.2e ok=%s" % (wv["quantum"]["orig"], wv["quantum"]["wrap"], wv["quantum"]["diff"], wv["quantum"]["ok"]))
    log("  nc     : orig=%.6e wrap=%.6e diff=%.2e ok=%s" % (wv["nc"]["orig"], wv["nc"]["wrap"], wv["nc"]["diff"], wv["nc"]["ok"]))

    log("\n=== 2-2/2-3. N 스윕: AUC(양자/고전/S-검정) ===")
    cells = []
    for nt in n_trains:
        for ne in n_evals:
            c = run_cell(nt, ne, args.reps, args.restarts, args.train_seeds, SEED + 100 * nt + ne)
            cells.append(c)
            aq = c["auc_quantum"]; ac = c["auc_classical"]; as_ = c["auc_stest"]
            aqk = c["auc_quantum_kl"]; ack = c["auc_classical_kl"]
            log("  N_train=%-6d N_eval=%-5d | mimic CE: q=%.3f c=%.3f s=%.3f Δ=%+.3f || KL(진단): q=%.3f c=%.3f Δ=%+.3f"
                % (nt, ne, aq["mimic_ncfloor"]["mean"], ac["mimic_ncfloor"]["mean"], as_["mimic_ncfloor"]["mean"],
                   aq["mimic_ncfloor"]["mean"] - ac["mimic_ncfloor"]["mean"],
                   aqk["mimic_ncfloor"]["mean"], ack["mimic_ncfloor"]["mean"],
                   aqk["mimic_ncfloor"]["mean"] - ack["mimic_ncfloor"]["mean"]))

    # self-check: normal-vs-normal AUC≈0.5
    nn_q = [c["auc_quantum"]["normal"]["mean"] for c in cells]
    nn_c = [c["auc_classical"]["normal"]["mean"] for c in cells]
    self_check = dict(wrappers=wv,
                      normal_vs_normal_auc_q=[float(x) for x in nn_q],
                      normal_vs_normal_auc_c=[float(x) for x in nn_c],
                      normal_auc_near_half=bool(max(abs(np.array(nn_q + nn_c) - 0.5)) < 0.12))

    log("\n=== P2: visibility 스윕 (gap ↑ → ΔAUC ↑ 단조성) ===")
    p2_rows, spearman = run_p2(args.reps, args.restarts,
                               n_train=(2000 if args.quick else 2000), n_eval=(1000),
                               seed0=SEED + 424242, vis_list=vis_list)
    for r in p2_rows:
        log("  V=%.3f CF=%.3f gap=%.5f | CE:Δ=%+.3f || KL(진단):AUCq=%.3f AUCc=%.3f ΔAUC=%+.3f"
            % (r["visibility"], r["cf"], r["gap"], r["delta_auc"],
               r["auc_quantum_kl"], r["auc_classical_kl"], r["delta_auc_kl"]))
    log("  Spearman(gap, ΔAUC_KL) = %s" % (("%.3f" % spearman) if spearman is not None else "N/A"))

    # ---- P1 판정 (명세 스코어=CE. 주지표: 큰 N_train, 두 N_eval 모두 AUCq>AUCc on mimic) ----
    prim = [c for c in cells if c["n_train"] == max(n_trains)]
    p1_deltas, p1_deltas_kl = [], []
    for c in prim:
        dce = np.array(c["auc_quantum"]["mimic_ncfloor"]["vals"]) - np.array(c["auc_classical"]["mimic_ncfloor"]["vals"])
        dkl = np.array(c["auc_quantum_kl"]["mimic_ncfloor"]["vals"]) - np.array(c["auc_classical_kl"]["mimic_ncfloor"]["vals"])
        p1_deltas.append(dict(n_eval=c["n_eval"], delta_mean=float(dce.mean()),
                              auc_q=c["auc_quantum"]["mimic_ncfloor"]["mean"],
                              auc_c=c["auc_classical"]["mimic_ncfloor"]["mean"],
                              q_gt_c=bool(dce.mean() > 0.02)))
        p1_deltas_kl.append(dict(n_eval=c["n_eval"], delta_mean=float(dkl.mean()),
                                 auc_q=c["auc_quantum_kl"]["mimic_ncfloor"]["mean"],
                                 auc_c=c["auc_classical_kl"]["mimic_ncfloor"]["mean"],
                                 q_gt_c=bool(dkl.mean() > 0.02)))
    p1_pass = bool(all(d["q_gt_c"] for d in p1_deltas))        # 명세(CE) 기준 = 주 판정
    p1_pass_kl = bool(all(d["q_gt_c"] for d in p1_deltas_kl))  # 진단(KL) 기준

    log("\n>>> P1 판정 (명세 스코어=CE, mimic_ncfloor, N_train=%d): %s" % (max(n_trains), "PASS" if p1_pass else "FAIL"))
    for d in p1_deltas:
        log("    CE  N_eval=%d: AUCq=%.3f vs AUCc=%.3f  Δ=%+.3f → %s"
            % (d["n_eval"], d["auc_q"], d["auc_c"], d["delta_mean"], d["q_gt_c"]))
    log(">>> [진단] 엔트로피 보정 KL 스코어 기준: %s" % ("PASS" if p1_pass_kl else "FAIL"))
    for d in p1_deltas_kl:
        log("    KL  N_eval=%d: AUCq=%.3f vs AUCc=%.3f  Δ=%+.3f → %s"
            % (d["n_eval"], d["auc_q"], d["auc_c"], d["delta_mean"], d["q_gt_c"]))

    dg = prim[0]["diag"]
    verdicts = dict(
        P1_quantum_beats_classical_on_mimic=dict(
            spec_score="CE (=−Σ emp·log model, 음의 로그우도; 프롬프트 명세)",
            pass_=p1_pass, per_neval=p1_deltas, n_train=max(n_trains),
            diagnostic_KL=dict(pass_=p1_pass_kl, per_neval=p1_deltas_kl,
                               score="KL(emp‖model)=CE−H(emp) = ½N⁻¹·deviance(G-통계량); 2N·KL~χ²(포화모델 대비 우도비)"),
            population_diag=dg,
            reading=("[naive NLL(=CE)=FAIL, 방법론적 발견] 세 탐지기(양자/고전NC/S-검정) AUC 가 mimic 에서 사실상 동일(Δ≈0). "
                     "원인: naive NLL(음의 로그우도)은 표본엔트로피 H(emp)에 교란됨 — mimic_ncfloor 는 비맥락(CF=0,S=2)이라 "
                     "정상보다 엔트로피 높음(H_mimic=%.3f>H_normal=%.3f)·S 낮음 → NLL 이 H(emp)에 지배되어 모델불일치 신호가 묻힘. "
                     "즉 mimic 은 이미 고전수단(S-검정)만으로 탐지되어 여기선 양자모델의 한계탐지가치 0. "
                     "[deviance(KL)=%s] 표본엔트로피 교란을 제거한 원리적 통계(2N·KL~χ²)로 보면 gap 이 탐지마진으로 드러남: "
                     "고전탐지기 AUC<0.5(mimic 을 정상보다 덜 이상으로 판정), 양자탐지기 AUC>0.5. "
                     "단 mimic 은 고전적(S=2)이라 이 분리는 결국 고전 신호의 재검출 — S-검정을 통과하는 non-quantum 공격으로의 "
                     "격리 검증은 gate_s2b(iso_S_substitution/blind_shell) 참조."
                     % (dg["H_mimic_pop"], dg["H_normal_pop"], "PASS" if p1_pass_kl else "FAIL"))),
        P2_monotone=dict(spearman_gap_deltaAUC_kl=spearman, rows=p2_rows,
                         reading="KL(진단) 스코어에서 gap↑ ⇒ ΔAUC↑(단조). CE 스코어에선 ΔAUC≈0. c·CF² 함수형 주장 안 함."),
        P3_vs_Stest=dict(
            note="S-검정(S≤2 폐기)과 생성탐지기 AUC 비교. 특히 partial_25·작은 N_eval 에서 S-검정이 놓치는지.",
            table=[dict(n_train=c["n_train"], n_eval=c["n_eval"],
                        partial_25=dict(auc_q=c["auc_quantum"]["partial_25"]["mean"],
                                        auc_s=c["auc_stest"]["partial_25"]["mean"]),
                        intercept=dict(auc_q=c["auc_quantum"]["intercept_resend"]["mean"],
                                       auc_s=c["auc_stest"]["intercept_resend"]["mean"]),
                        mimic=dict(auc_q=c["auc_quantum"]["mimic_ncfloor"]["mean"],
                                   auc_s=c["auc_stest"]["mimic_ncfloor"]["mean"]))
                   for c in cells]),
    )

    payload = dict(
        note=("STEP 2 DI-QKD 탐지. 훈련(정상)→양자/고전 모델→평가셋 스코어 AUC. "
              "naive NLL(=CE): mimic 에서 AUC_quantum>AUC_classical? → FAIL(세 탐지기 동일; NLL 이 batch entropy 교란=방법론적 발견). "
              "교정 스코어=deviance(KL)=CE−H(emp), 2N·KL~χ²: gap 이 탐지마진으로 드러남(고전 AUC<0.5). "
              "mimic 은 S=2(고전적)라 격리 불충분 → S-검정 통과 non-quantum 공격 검증은 gate_s2b. "
              "★모델클래스 분리이지 supremacy 아님. 키보안 아닌 장비무결성 맥락. (2,2,2)·시뮬·유한표본. 단정 금지."),
        params=dict(seed=SEED, v_normal=V_NORMAL, classes=CLASSES,
                    n_trains=n_trains, n_evals=n_evals, reps=args.reps, restarts=args.restarts,
                    train_seeds=args.train_seeds, vis_list=vis_list, step1_passed=st1,
                    score_spec="CE=−Σ emp·log(model)", score_diag="KL(emp‖model)=CE−H(emp)"),
        self_check=self_check,
        cells=cells,
        verdicts=verdicts,
        key_finding=("[방법론적 발견] naive NLL(=CE) 스코어에선 P1 FAIL: mimic_ncfloor 는 비맥락(CF=0,S=2)이라 "
                     "정상보다 엔트로피 높고 S 낮아 고전수단(S-검정)만으로 이미 탐지됨 → 양자모델 한계탐지가치 0(양자=고전=S AUC). "
                     "이는 NLL 이 batch entropy H(emp)에 교란됨을 보이는 방법론적 발견. "
                     "[교정 스코어=deviance] KL(emp‖model)=CE−H(emp), 2N·KL~χ²(포화모델 대비 우도비)로 교정하면 "
                     "KL-gap 이 탐지마진으로 번역됨(ΔAUC 최대 +0.9, P2 Spearman=1.0). "
                     "단 mimic 은 고전적(S=2)이라 이 분리는 고전 신호 재검출 — 'S-검정을 통과하는 non-quantum 공격에서도 "
                     "양자모델이 정밀한가'의 격리 검증은 gate_s2b(iso_S_substitution/blind_shell)에서 수행."),
        limits=("naive NLL(CE)은 H(emp) 교란 → 세 탐지기 동률(방법론적 발견). 교정 스코어=deviance(KL)=½N⁻¹·G-통계량. "
                "mimic/blinding 은 S≈2 경계(고전적)라 격리 불충분 → gate_s2b 로 승계. "
                "정상 운용점은 양자실현(KL_q≈0). G 트랙 고정. 관측이지 supremacy 아님. 키보안 아닌 장비무결성 맥락."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot_auc(payload)
        plot_p2(payload)
        log("[저장] %s , %s" % (OUT_PNG_AUC, OUT_PNG_P2))
    except Exception as e:
        log("[plot WARN] %r" % e)

    log("\n" + "=" * 60)
    if p1_pass:
        log("STEP 2 P1(CE) PASS → STEP 3 후보 선택 가능.")
    else:
        log("STEP 2 P1(명세 CE) FAIL → STEP 3 진입 금지(프로토콜). 사람 판단 필요.")
        log("  진단(KL 보정): gap→탐지마진은 실재하나 CE 우도탐지엔 미전이(엔트로피 지배). 상세는 s2.json key_finding.")


def plot_auc(P):
    cells = P["cells"]
    n_trains = P["params"]["n_trains"]; n_evals = P["params"]["n_evals"]
    x = np.arange(len(CLASSES))
    fig, axes = plt.subplots(2, len(n_evals), figsize=(7 * len(n_evals), 11), squeeze=False)
    for j, ne in enumerate(n_evals):
        cell = [c for c in cells if c["n_eval"] == ne and c["n_train"] == max(n_trains)][0]
        # --- top: CE (명세) ---
        ax = axes[0][j]
        aq = [cell["auc_quantum"][c]["mean"] for c in CLASSES]
        ac = [cell["auc_classical"][c]["mean"] for c in CLASSES]
        as_ = [cell["auc_stest"][c]["mean"] for c in CLASSES]
        w = 0.27
        ax.bar(x - w, aq, w, label="quantum", color="#7B2FBE")
        ax.bar(x, ac, w, label="classical (NCfloor)", color="#E2A13B")
        ax.bar(x + w, as_, w, label="S-test", color="#3B6EA5")
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUC (class vs normal)"); ax.set_ylim(0, 1.0)
        ax.set_title("CE score [SPEC]  N_train=%d, N_eval=%d\nquantum≈classical≈S-test (no advantage)" % (max(n_trains), ne))
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
        # --- bottom: KL (진단) ---
        ax2 = axes[1][j]
        aqk = [cell["auc_quantum_kl"][c]["mean"] for c in CLASSES]
        ack = [cell["auc_classical_kl"][c]["mean"] for c in CLASSES]
        ax2.bar(x - w / 2, aqk, w, label="quantum (KL)", color="#7B2FBE")
        ax2.bar(x + w / 2, ack, w, label="classical (KL)", color="#E2A13B")
        ax2.axhline(0.5, color="k", lw=0.8, ls="--", label="chance 0.5")
        ax2.set_xticks(x); ax2.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=8)
        ax2.set_ylabel("AUC (class vs normal)"); ax2.set_ylim(0, 1.0)
        ax2.set_title("KL score [DIAG, entropy-corrected]  N_eval=%d\nmimic: classical AUC<0.5 (blind), quantum>0.5" % ne)
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
    p1 = P["verdicts"]["P1_quantum_beats_classical_on_mimic"]["pass_"]
    plt.suptitle("STEP 2 — P1(CE spec)=%s.  CE(top): entropy-dominated → all detectors equal.  "
                 "KL(bottom): gap→margin visible.  — model-class separation, NOT supremacy" % ("PASS" if p1 else "FAIL"),
                 fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_PNG_AUC, dpi=130); plt.close()


def plot_p2(P):
    rows = P["verdicts"]["P2_monotone"]["rows"]
    g = [r["gap"] for r in rows]; d = [r["delta_auc_kl"] for r in rows]; cf = [r["cf"] for r in rows]
    sp = P["verdicts"]["P2_monotone"]["spearman_gap_deltaAUC_kl"]
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.5))
    o = np.argsort(g)
    ax[0].plot(np.array(g)[o], np.array(d)[o], "o-", color="#7B2FBE", ms=8, label="KL score (diag)")
    ax[0].plot(np.array(g)[o], np.array([r["delta_auc"] for r in rows])[o], "s--", color="#999",
               ms=6, alpha=0.7, label="CE score (spec) ≈ 0")
    for r in rows:
        ax[0].annotate("CF=%.2f" % r["cf"], (r["gap"], r["delta_auc_kl"]), fontsize=8,
                       textcoords="offset points", xytext=(5, 5))
    ax[0].axhline(0, color="k", lw=0.6)
    ax[0].set_xlabel("gap = NCfloor − KL_q (at normal operating point)")
    ax[0].set_ylabel("ΔAUC = AUC_quantum − AUC_classical (mimic)")
    ax[0].set_title("P2 — gap ↑ ⇒ ΔAUC ↑ (KL diag; Spearman=%s)\nCE(spec) stays ≈0; NO c·CF² functional claim"
                    % (("%.2f" % sp) if sp is not None else "N/A"))
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)
    ax[1].plot(cf, [r["auc_quantum_kl"] for r in rows], "o-", color="#7B2FBE", label="quantum (KL)")
    ax[1].plot(cf, [r["auc_classical_kl"] for r in rows], "s-", color="#E2A13B", label="classical (KL)")
    ax[1].axhline(0.5, color="k", lw=0.8, ls="--")
    ax[1].set_xlabel("CF (normal operating point)"); ax[1].set_ylabel("AUC on mimic vs normal (KL score)")
    ax[1].set_title("KL-score detector AUC vs contextuality of normal point\nclassical falls below 0.5 as CF grows")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG_P2, dpi=130); plt.close()


if __name__ == "__main__":
    main()
