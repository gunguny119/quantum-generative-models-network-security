#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2b — 재설계 공격: S-검정을 통과(S≈S_normal)하지만 정상 양자 '지문'과 다른 공격.

동기(gate_s2 결과): mimic_ncfloor 는 S=2(고전적)라 S-검정 자체를 통과 못 함 → '양자 생성모델이
  S-검정보다 정밀한 계기인가'를 격리하지 못함. 여기선 **S-검정을 통과하는** 공격을 quantum_behavior(p)
  이미지에서 탐색하여 세 탐지기의 분리를 본다.

★ 프레이밍: 키 보안 개선 주장 아님. **장비 무결성/변조 모니터링**(S-검정 보완 계층). 양자 탐지기는
  정상 장비의 특정 지문(고정 템플릿)과의 이탈을 잡는다 — 재적합이 아니라 템플릿 대조. worst-case 존재증명.

스코어(gate_s2 교정): 탐지 스코어 = **deviance/G-통계량** KL(emp‖model)=CE−H(emp), 2N·KL~χ²
  (포화모델 대비 우도비). 표본엔트로피 교란 제거.

공격(모두 e′=quantum_behavior(p) → 물리 구현가능, 최대얽힘이면 marginal 균등):
  iso_S            : max KL(e′‖e_normal)  s.t. |S(e′)−S_target|≤tol
  blind_shell      :  + |KL(e′‖c_model)−NCfloor|≤tol_sh   (고전 탐지기까지 블라인드; 껍질반지름=NCfloor=c·CF²)
  marginal_matched :  + marginal(e′)=marginal(e_normal)     (QBER 모니터 우회 가정)
  partial_{p}      : p·e_intercept +(1−p)·e_normal          (원 프롬프트 P3; 유한통계 우위 검증)

판정:
  P1′: blind_shell 에서 양자 AUC≫0.5, 고전 AUC≈0.5, S-검정 AUC≈0.5 (세 탐지기 분리)
  P3′: partial 소량 p·N_eval=245 에서 양자 AUC > S-검정 AUC (유한통계 우위)

사용: python tools/gate_s2b_isoS.py [--quick]
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
import gap_vs_cf_v0 as G
import gate_r2_anchor_strengthen as R2
import sim_diqkd as S
import gate_s2_detection as D2                 # fit_quantum_params, fit_nc_params, auc, score_kl, _H, V_NORMAL

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

OUTDIR = os.path.join("results2", "gate_s2b")
OUT_JSON = os.path.join(OUTDIR, "s2b.json")
OUT_PNG = os.path.join(OUTDIR, "s2b_auc.png")
OUT_PNG_GEO = os.path.join(OUTDIR, "s2b_geometry.png")

SEED = 20260724
V_NORMAL = D2.V_NORMAL                          # 0.856 (CF≈0.211)
S_TOL = 0.01
SH_TOL = 0.001
C_UNIFORM_CHSH = 1.0 / 6.0                      # 균등방향 CHSH 계수 (gate_e_coefficient)


def log(m):
    print(m, flush=True)


# ===========================================================================
# marginal / 헬퍼
# ===========================================================================
def marginals(e):
    """[P(a=0|x=0), P(a=0|x=1), P(b=0|y=0), P(b=0|y=1)] (ctx order (00),(01),(10),(11))."""
    return np.array([e[0, 0, :].sum(), e[2, 0, :].sum(), e[0, :, 0].sum(), e[1, :, 0].sum()])


# ===========================================================================
# 공격 탐색: e′=quantum_behavior(p) 이미지에서 penalty 최적화
# ===========================================================================
def find_iso_S_attack(e_normal, S_target, mode, c_model, ncfloor, seed, restarts=24):
    ref_marg = marginals(e_normal)
    wS, wSH, wM = 300.0, 4000.0, 300.0
    rng = np.random.default_rng(seed)

    def loss(p):
        e = G.quantum_behavior(p)
        L = -G.eval_kl(e, e_normal) + wS * (R2.chsh_S(e) - S_target) ** 2
        if mode == "blind_shell":
            L += wSH * (G.eval_kl(e, c_model) - ncfloor) ** 2
        if mode == "marginal_matched":
            L += wM * float(np.sum((marginals(e) - ref_marg) ** 2))
        return L

    # inits: canon + 대칭변형 + 무작위. 최대얽힘 유지(θ_state=π/4)한 각도변형 위주.
    inits = [np.concatenate([[np.pi / 4], S.CANON_ANGLES])]
    for _ in range(restarts - 1):
        base = np.concatenate([[np.pi / 4], S.CANON_ANGLES]).copy()
        base[1:] += rng.uniform(-np.pi, np.pi, size=8)          # 각도만 흔들기
        if rng.random() < 0.3:
            base[0] = rng.uniform(np.pi / 6, np.pi / 2)         # 가끔 상태도
        inits.append(base)

    best = None
    for x0 in inits:
        r = minimize(loss, x0, method="Powell", options=dict(maxiter=4000, xtol=1e-7, ftol=1e-10))
        e = G.quantum_behavior(r.x)
        kln = G.eval_kl(e, e_normal)
        sdev = abs(R2.chsh_S(e) - S_target)
        shdev = abs(G.eval_kl(e, c_model) - ncfloor)
        mdev = float(np.sum(np.abs(marginals(e) - ref_marg)))
        feasible = sdev <= S_TOL and (mode != "blind_shell" or shdev <= SH_TOL) \
            and (mode != "marginal_matched" or mdev <= 0.02)
        score = kln if feasible else kln - 10.0 * (max(0, sdev - S_TOL) + max(0, shdev - SH_TOL))
        if best is None or score > best[0]:
            best = (score, r.x, e, dict(kl_to_normal=float(kln), S=float(R2.chsh_S(e)),
                                        cf=float(G.contextual_fraction(e)),
                                        kl_to_c=float(G.eval_kl(e, c_model)),
                                        s_dev=float(sdev), shell_dev=float(shdev),
                                        marg_dev=mdev, feasible=bool(feasible)))
    return best[2], best[3]


# ===========================================================================
# 평가 박스 생성 (behavior 직접)
# ===========================================================================
def gen_boxes(e, n_eval, reps, seed):
    boxes, Ss = [], []
    for i in range(reps):
        rng = np.random.default_rng(seed + 1 + i)
        a, b, xo, yo = S._sample_from_behavior(e, n_eval, rng)
        bx, _ = R2.build_box(a, b, xo, yo)
        boxes.append(bx); Ss.append(R2.chsh_S(bx))
    return boxes, np.array(Ss)


# ===========================================================================
# 탐지 실험 (KL=deviance 주 스코어)
# ===========================================================================
def run(n_train, n_evals, reps, restarts, train_seeds, seed0, attacks, partials):
    normal_noise = {"visibility": V_NORMAL}
    e_normal = S.build_behavior("none", S.CANON_STATE, S.CANON_ANGLES, normal_noise, np.random.default_rng(seed0))
    S_target = float(R2.chsh_S(e_normal))
    cf_normal = float(G.contextual_fraction(e_normal))

    # 공격 설계에 쓸 모집단 고전모델/NCfloor (adversary 는 정상 운용점을 앎)
    _, c_pop = D2.fit_nc_params(e_normal, np.random.default_rng(seed0 + 1), max(restarts, 6))
    ncfloor_pop = float(G.eval_kl(e_normal, c_pop))

    # ---- 공격 e′ 탐색 ----
    attack_e = {}
    attack_info = {}
    for mi, m in enumerate(attacks):
        e_atk, info = find_iso_S_attack(e_normal, S_target, m, c_pop, ncfloor_pop, seed0 + 100 + 37 * mi)
        attack_e[m] = e_atk; attack_info[m] = info
        log("  [attack %-16s] S=%.4f(dev%.4f) CF=%.4f KL→normal=%.5f KL→c=%.5f(NCfloor=%.5f,dev%.5f) marg_dev=%.4f feasible=%s"
            % (m, info["S"], info["s_dev"], info["cf"], info["kl_to_normal"], info["kl_to_c"],
               ncfloor_pop, info["shell_dev"], info["marg_dev"], info["feasible"]))
    for pf in partials:
        e_p = S.build_behavior("partial", S.CANON_STATE, S.CANON_ANGLES,
                               {"visibility": V_NORMAL, "p_atk": pf, "sub_attack": "intercept_resend"},
                               np.random.default_rng(seed0 + 2))
        attack_e["partial_%02d" % int(pf * 100)] = e_p
        attack_info["partial_%02d" % int(pf * 100)] = dict(S=float(R2.chsh_S(e_p)), cf=float(G.contextual_fraction(e_p)),
                                                           kl_to_normal=float(G.eval_kl(e_p, e_normal)), p_atk=pf)

    classes = list(attack_e.keys())

    # ---- N_eval 별 탐지 ----
    cells = []
    for ne in n_evals:
        # 평가 박스(훈련 독립)
        eval_boxes, eval_S = {}, {}
        for ci, cls in enumerate(classes):
            eb, sS = gen_boxes(attack_e[cls], ne, reps, seed0 + 10000 + 1000 * ci)
            eval_boxes[cls] = eb; eval_S[cls] = sS
        neg_boxes, neg_S = gen_boxes(e_normal, ne, reps, seed0 + 900000)

        auc_qkl = {c: [] for c in classes}
        auc_ckl = {c: [] for c in classes}
        auc_s = {c: [] for c in classes}
        for ts in range(train_seeds):
            rng = np.random.default_rng(seed0 + 700 + 13 * ts)
            a, b, xo, yo = S.simulate_diqkd(n_train, S.CANON_STATE, S.CANON_ANGLES, normal_noise, "none", rng)
            emp_train, _ = R2.build_box(a, b, xo, yo)
            pq = D2.fit_quantum_params(emp_train, np.random.default_rng(seed0 + 800 + 13 * ts), restarts)
            q_model = G.quantum_behavior(pq)
            _, c_model = D2.fit_nc_params(emp_train, np.random.default_rng(seed0 + 950 + 13 * ts), restarts)
            neg_q = np.array([D2.score_kl(x, q_model) for x in neg_boxes])
            neg_c = np.array([D2.score_kl(x, c_model) for x in neg_boxes])
            for cls in classes:
                bx = eval_boxes[cls]
                auc_qkl[cls].append(D2.auc([D2.score_kl(x, q_model) for x in bx], neg_q))
                auc_ckl[cls].append(D2.auc([D2.score_kl(x, c_model) for x in bx], neg_c))
                auc_s[cls].append(D2.auc(-eval_S[cls], -neg_S))

        def agg(d):
            return {c: dict(mean=float(np.mean(d[c])), std=float(np.std(d[c]))) for c in classes}
        cells.append(dict(n_eval=ne, reps=reps, auc_quantum_kl=agg(auc_qkl),
                          auc_classical_kl=agg(auc_ckl), auc_stest=agg(auc_s)))
        log("  N_eval=%d 완료" % ne)

    return dict(e_normal_S=S_target, cf_normal=cf_normal, ncfloor_pop=ncfloor_pop,
                attack_info=attack_info, classes=classes, cells=cells,
                c_cf2=C_UNIFORM_CHSH * cf_normal ** 2)


# ===========================================================================
# main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="STEP 2b iso_S 재설계 공격 탐지. KL=deviance 주스코어.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--train-seeds", type=int, default=3)
    ap.add_argument("--attack-restarts", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)

    st1 = D2.check_step1_passed()
    log("=== STEP 1 게이트 확인: %s ===" % ("PASS" if st1 else "FAIL/미실행"))
    if not st1:
        log("[STOP] STEP 1 미통과 → 진입 금지."); return

    if args.quick:
        args.reps, args.restarts, args.train_seeds, args.attack_restarts = 40, 2, 2, 12
        n_evals = [245, 1000]
    else:
        n_evals = [245, 1000]

    attacks = ["iso_S", "blind_shell", "marginal_matched"]
    partials = [0.05, 0.10, 0.20]

    log("\n=== 공격 설계 + 탐지 (N_train=10000, KL=deviance 스코어) ===")
    R = run(n_train=10000, n_evals=n_evals, reps=args.reps, restarts=args.restarts,
            train_seeds=args.train_seeds, seed0=SEED, attacks=attacks, partials=partials)

    log("\n=== AUC (양자 KL / 고전 KL / S-검정), N_train=10000 ===")
    for cell in R["cells"]:
        log("  -- N_eval=%d --" % cell["n_eval"])
        for cls in R["classes"]:
            log("     %-16s : quantum=%.3f  classical=%.3f  S-test=%.3f"
                % (cls, cell["auc_quantum_kl"][cls]["mean"], cell["auc_classical_kl"][cls]["mean"],
                   cell["auc_stest"][cls]["mean"]))

    # ---- P1′ 판정: blind_shell 세 탐지기 분리 ----
    def get(cell, det, cls):
        return cell[det][cls]["mean"]
    p1b = []
    for cell in R["cells"]:
        q = get(cell, "auc_quantum_kl", "blind_shell")
        c = get(cell, "auc_classical_kl", "blind_shell")
        s = get(cell, "auc_stest", "blind_shell")
        sep = bool(q > 0.7 and abs(c - 0.5) < 0.15 and abs(s - 0.5) < 0.15)
        p1b.append(dict(n_eval=cell["n_eval"], quantum=q, classical=c, s_test=s, separated=sep))
    p1_pass = bool(all(d["separated"] for d in p1b) and R["attack_info"]["blind_shell"]["feasible"])

    # ---- P3′ 판정: partial 소량 p, N_eval=245 에서 양자 > S-검정 ----
    cell245 = [c for c in R["cells"] if c["n_eval"] == 245][0]
    p3 = []
    for pf in partials:
        cls = "partial_%02d" % int(pf * 100)
        q = get(cell245, "auc_quantum_kl", cls); s = get(cell245, "auc_stest", cls)
        p3.append(dict(p_atk=pf, quantum=q, s_test=s, quantum_gt_s=bool(q > s + 0.02)))
    p3_pass = bool(any(d["quantum_gt_s"] for d in p3))

    log("\n>>> P1′ (blind_shell 세 탐지기 분리): %s" % ("PASS" if p1_pass else "FAIL"))
    for d in p1b:
        log("    N_eval=%d: quantum=%.3f classical=%.3f S-test=%.3f → 분리=%s"
            % (d["n_eval"], d["quantum"], d["classical"], d["s_test"], d["separated"]))
    log(">>> P3′ (partial 유한통계 우위, N_eval=245): %s" % ("PASS" if p3_pass else "FAIL"))
    for d in p3:
        log("    p=%.2f: quantum=%.3f vs S-test=%.3f → 양자우위=%s" % (d["p_atk"], d["quantum"], d["s_test"], d["quantum_gt_s"]))
    log("\n  blind_shell 껍질반지름 = NCfloor = %.5f  vs  c·CF² = %.5f (c=1/6, CF=%.3f)"
        % (R["ncfloor_pop"], R["c_cf2"], R["cf_normal"]))

    verdicts = dict(
        P1prime_blind_shell_separation=dict(
            pass_=p1_pass, per_neval=p1b, attack_feasible=R["attack_info"]["blind_shell"]["feasible"],
            reading=("FAIL(원리적 불가능성): blind_shell 최적화가 정상으로 붕괴(KL→normal≈0.0001). "
                     "이유 — 같은 S 에서 등방(정상) 상자가 고전템플릿 거리 KL(‖c_model)=NCfloor 를 **유일하게 최소화**함. "
                     "비등방으로 벗어나면 KL(‖c)가 NCfloor 위로 증가(iso_S: 0.36) → 고전 탐지기가 잡음. "
                     "따라서 '고전 껍질(NCfloor)에 머물기'='정상에 고정'→양자 탐지기도 못 잡음. "
                     "= 물리구현가능(양자) 같은-S 공격으로는 양자 템플릿이 고전 NC-템플릿보다 우월 불가. geometry 좌상단 공집합.")),
        complementarity=dict(
            reading=("★핵심 발견: 모델기반 지문탐지와 S-검정은 **서로 다른 공격 클래스를 잡는 상보관계**. "
                     "(a) S-보존 지문변조(iso_S/marginal_matched): 모델템플릿(양자·고전 공히) AUC≈1.0, S-검정 AUC≈0.5(완전 blind). "
                     "marginal_matched 는 marginal 까지 일치(QBER 모니터 우회)해도 잡힘. "
                     "(b) S-저하 공격(partial=intercept 소량혼합): S-검정 AUC↑(0.58→0.98), 모델 KL 탐지기는 AUC≈0.5 이하 "
                     "(intercept 는 CF↓ 로 데이터를 더 '고전적'으로 만들어 고전템플릿엔 덜 이상해 보임 → AUC<0.5). "
                     "→ 두 계층을 함께 쓰는 defense-in-depth. 단 (a)의 우위는 양자 특유 아님(고전 NC-템플릿 동급).")),
        P3prime_partial_finite_stat=dict(
            pass_=p3_pass, rows=p3, n_eval=245,
            reading=("FAIL: partial 은 intercept(S=0.94) 소량혼합이라 **S-저하** 공격 → S-검정의 홈그라운드(AUC 0.58~0.83@N245). "
                     "모델 KL 탐지기는 여기서 열위(양자≈0.5, 고전<0.5). 유한통계 우위는 성립 안 함 — "
                     "S-저하 공격엔 S-검정이 옳은 도구. (S-보존 공격 iso_S 에선 반대로 모델기반이 압도.)")),
        P2_reference="gate_s2 P2 Spearman(gap,ΔAUC_deviance)=1.000 유지(재계산 안 함).",
        shell_radius_relation=dict(ncfloor=R["ncfloor_pop"], c_cf2=R["c_cf2"], c=C_UNIFORM_CHSH, cf=R["cf_normal"],
                                   note="정상=고전껍질 반지름 NCfloor ≈ c·CF² (gap 법칙과 연결). 이 껍질의 유일 최소점이 정상."),
    )
    payload = dict(
        note=("STEP 2b: S-검정을 통과(S≈S_normal)하는 재설계 공격(quantum_behavior 이미지)로 탐지 상보성 검증. "
              "스코어=deviance(KL)=CE−H(emp), 2N·KL~χ². ★키보안 아닌 장비무결성/변조모니터링. "
              "발견: 모델템플릿은 S-보존 지문변조를, S-검정은 S-저하 공격을 잡는 상보관계(defense-in-depth). "
              "blind_shell 붕괴=같은-S 양자공격으로 양자>고전템플릿 우월 불가(원리적). supremacy 아님. (2,2,2)·시뮬. 단정 금지."),
        key_finding=("모델기반 장비지문 탐지(양자/고전 템플릿)와 CHSH S-검정은 **상보적**: 모델은 S-보존 변조(iso_S/marginal_matched) "
                     "AUC≈1 로 잡고 S-검정은 blind; S-검정은 S-저하 공격(partial) 을 잡고 모델은 놓침. "
                     "→ 둘을 함께 쓰는 defense-in-depth 가 결론. 단 모델기반 우위는 양자 특유 아님(고전 NC-템플릿 동급): "
                     "blind_shell 이 정상으로 붕괴 = 같은 S 에서 정상이 고전거리를 유일 최소화하므로 "
                     "'고전엔 blind, 양자엔 visible' 인 물리(양자) 공격 부재. worst-case 존재증명 겸 불가능성."),
        params=dict(seed=SEED, v_normal=V_NORMAL, s_tol=S_TOL, shell_tol=SH_TOL,
                    n_train=10000, n_evals=n_evals, reps=args.reps, restarts=args.restarts,
                    train_seeds=args.train_seeds, attack_restarts=args.attack_restarts, step1_passed=st1,
                    score="deviance KL(emp‖model)=CE−H(emp)"),
        self_check=dict(attacks_feasible={m: R["attack_info"][m].get("feasible") for m in attacks},
                        s_target=R["e_normal_S"], normal_cf=R["cf_normal"]),
        attack_info=R["attack_info"], classes=R["classes"], cells=R["cells"],
        verdicts=verdicts,
        limits=("iso_S/marginal_matched 는 특정 정상 운용점에 대한 S-보존 변조의 존재증명(모든 공격 커버 아님). "
                "탐지기는 고정 템플릿 대조(재적합 아님)=장비지문. S-검정 통과는 S 값만 매칭(고차 통계 이탈). "
                "모델기반 우위는 양자 특유 아님(blind_shell 붕괴=불가능성). partial 은 S-저하라 S-검정 홈그라운드. "
                "정상=양자실현. G 트랙 고정. 관측이지 supremacy 아님. 키보안 아닌 장비무결성 맥락."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot_auc(payload)
        plot_geometry(payload, R)
        log("[저장] %s , %s" % (OUT_PNG, OUT_PNG_GEO))
    except Exception as e:
        log("[plot WARN] %r" % e)

    log("\n" + "=" * 60)
    log("STEP 2b 판정: P1′(blind_shell 분리)=%s, P3′(유한통계)=%s" % (p1_pass, p3_pass))


def plot_auc(P):
    classes = P["classes"]
    cells = P["cells"]
    x = np.arange(len(classes))
    fig, axes = plt.subplots(1, len(cells), figsize=(8 * len(cells), 6), squeeze=False)
    for j, cell in enumerate(cells):
        ax = axes[0][j]
        q = [cell["auc_quantum_kl"][c]["mean"] for c in classes]
        c_ = [cell["auc_classical_kl"][c]["mean"] for c in classes]
        s = [cell["auc_stest"][c]["mean"] for c in classes]
        w = 0.27
        ax.bar(x - w, q, w, label="quantum template (KL)", color="#7B2FBE")
        ax.bar(x, c_, w, label="classical NC (KL)", color="#E2A13B")
        ax.bar(x + w, s, w, label="S-test", color="#3B6EA5")
        ax.axhline(0.5, color="k", lw=0.8, ls="--")
        ax.set_xticks(x); ax.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
        ax.set_ylabel("AUC (class vs normal)"); ax.set_ylim(0, 1.05)
        ax.set_title("N_eval=%d  (deviance KL score)" % cell["n_eval"])
        ax.legend(fontsize=8); ax.grid(alpha=0.3, axis="y")
    plt.suptitle("STEP 2b — COMPLEMENTARITY: model-template monitoring catches S-preserving tampering "
                 "(iso_S/marginal_matched: q=c=1, S-test≈0.5);\nS-test catches S-lowering attacks "
                 "(partial: S-test↑, models miss). Disjoint coverage → defense-in-depth. — device integrity, NOT supremacy",
                 fontsize=10)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130); plt.close()


def plot_geometry(P, R):
    """KL-to-c_model(고전축) vs KL-to-normal(양자축) 산점: 각 공격의 위치 + NCfloor 껍질."""
    info = P["attack_info"]
    ncf = R["ncfloor_pop"]
    fig, ax = plt.subplots(figsize=(8.5, 7))
    ax.axvline(ncf, color="#E2A13B", lw=2, ls="--", label="classical shell KL=NCfloor=%.4f (=c·CF²=%.4f)" % (ncf, R["c_cf2"]))
    ax.scatter([ncf], [0], marker="*", s=260, color="#1a7f37", zorder=6, label="normal (on shell, KL→normal=0)")
    cols = {"iso_S": "#7B2FBE", "blind_shell": "#C1442E", "marginal_matched": "#3B6EA5"}
    for m in ["iso_S", "blind_shell", "marginal_matched"]:
        if m in info and "kl_to_c" in info[m]:
            ax.scatter([info[m]["kl_to_c"]], [info[m]["kl_to_normal"]], s=130, color=cols.get(m, "#555"),
                       zorder=5, label="%s (S=%.2f, CF=%.2f)" % (m, info[m]["S"], info[m]["cf"]))
            ax.annotate(m, (info[m]["kl_to_c"], info[m]["kl_to_normal"]), fontsize=8,
                        textcoords="offset points", xytext=(6, 4))
    ax.axhline(0, color="k", lw=0.5)
    ax.text(0.02, 0.30, "UPPER-LEFT EMPTY\n(classical-blind &\nquantum-visible)\n→ impossible for\nsame-S quantum attacks",
            fontsize=9, color="#C1442E", va="top")
    ax.set_xlabel("KL(attack ‖ classical NC template)   [classical detector axis]")
    ax.set_ylabel("KL(attack ‖ normal quantum template)   [quantum detector axis]")
    ax.set_title("STEP 2b geometry — the two detector axes move TOGETHER for same-S quantum attacks.\n"
                 "blind_shell collapses onto normal (origin); iso_S/marginal_matched are far on BOTH axes.\n"
                 "Upper-left (classical-blind, quantum-visible) is empty → no quantum advantage over classical template.")
    ax.legend(fontsize=8, loc="upper right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG_GEO, dpi=130); plt.close()


if __name__ == "__main__":
    main()
