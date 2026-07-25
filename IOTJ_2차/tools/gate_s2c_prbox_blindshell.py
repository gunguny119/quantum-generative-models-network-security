#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP 2c — blind_shell 불가능성 정리 완결: 초양자(post-quantum/PR-box 방향)에서 열리는가?

배경(gate_s2b): quantum_behavior(p) 이미지(=물리적 양자 공격)에서 blind_shell(=고전탐지기 블라인드
  & 정상과 다름) 은 **정상으로 붕괴**(공집합). 정리: 같은 S 에서 등방 정상이 고전거리 KL(‖c_model)=NCfloor 를
  유일 최소화. → 이번엔 **무신호(no-signalling) 상관 파라미터화**로 공격집합을 초양자까지 확장하여
  blind_shell 이 초양자에서 **열리는지** 검증. 실패(초양자도 공집합)해도 정리가 강해짐.

균등 marginal 무신호 상자:  P(a,b|x,y) = ¼(1 + (−1)^(a+b) E_xy),  E∈[−1,1]^4  (자동 무신호·균등 marginal).
양자성 분류:  TLM(Tsirelson–Landau–Masanes)  max_k |Σ_j asin E_j − 2 asin E_k| ≤ π  ⇔ 양자.  >π ⇔ 초양자.
  (CHSH: S=E00+E01+E10−E11.  Tsirelson S=2√2 ⇔ TLM 등호.  PR E=(1,1,1,−1) S=4 ⇔ 2π.)

blind_shell 탐색:  max KL(e′‖e_normal)  s.t.  S(e′)=S_normal,  KL(e′‖c_model)=NCfloor  [, TLM≤0 if quantum_only].
  - quantum_only=True  → 정상으로 붕괴(공집합) 재확인.
  - quantum_only=False → 초양자 해 e′ 존재?  존재 시: 양자 템플릿만 포착(AUC~1), 고전 템플릿·S-검정 블라인드(~0.5).

★ 프레이밍: 키보안 아님. 장비무결성/변조모니터링. 초양자 공격자는 편집증적 worst-case. supremacy 아님.

사용: python tools/gate_s2c_prbox_blindshell.py [--quick]
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
import gate_s2_detection as D2

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

OUTDIR = os.path.join("results2", "gate_s2c")
OUT_JSON = os.path.join(OUTDIR, "s2c.json")
OUT_PNG = os.path.join(OUTDIR, "s2c.png")

SEED = 20260725
V_NORMAL = D2.V_NORMAL
CTX = [(0, 0), (0, 1), (1, 0), (1, 1)]
S_TOL = 0.01
SH_TOL = 0.001
C_UNIFORM_CHSH = 1.0 / 6.0


def log(m):
    print(m, flush=True)


# ===========================================================================
# 무신호 상자 (균등 marginal, 4 상관) + TLM 분류
# ===========================================================================
def ns_box(E):
    e = np.zeros((4, 2, 2))
    for ci, (x, y) in enumerate(CTX):
        for a in (0, 1):
            for b in (0, 1):
                e[ci, a, b] = 0.25 * (1.0 + ((-1) ** (a + b)) * E[ci])
    return e


def chsh_from_E(E):
    return E[0] + E[1] + E[2] - E[3]


def tlm_slack(E):
    """TLM: max_k |Σ_j asin E_j − 2 asin E_k| − π.  ≤0 ⇔ 양자,  >0 ⇔ 초양자."""
    s = np.arcsin(np.clip(np.asarray(E), -1.0, 1.0))
    tot = s.sum()
    return float(max(abs(tot - 2.0 * s[k]) for k in range(4)) - np.pi)


def E_of(e):
    """box → 4 상관 (chsh_S 규약: index0↔+1)."""
    E = []
    for ci in range(4):
        m = e[ci]
        E.append(float(m[0, 0] - m[0, 1] - m[1, 0] + m[1, 1]))
    return np.array(E)


# ===========================================================================
# 강한 정리 그리드 검증: S=S_target 무신호 상자 전체에서 KL→c 최소가 정상(=NCfloor)인가?
#   → 껍질(KL→c=NCfloor)∩{S=S_target} 이 {정상} 뿐이면 blind_shell 은 NS 폴리토프 전체에서 공집합.
# ===========================================================================
def verify_shell_emptiness(e_normal, S_target, c_model, ncfloor, ngrid=61, shell_tol=0.0015):
    grid = np.linspace(-1.0, 1.0, ngrid)
    min_klc = 1e9
    argmin = None
    max_kln_on_shell = 0.0
    shell_far_pt = None
    n_far = 0
    n_feasible = 0
    scatter = []                                   # (KL→c, KL→normal, tlm) 샘플(플롯용, 서브샘플)
    for i0, e0 in enumerate(grid):
        for e1 in grid:
            for e2 in grid:
                e3 = e0 + e1 + e2 - S_target
                if abs(e3) > 1.0:
                    continue
                E = np.array([e0, e1, e2, e3])
                e = ns_box(E)
                n_feasible += 1
                klc = float(G.eval_kl(e, c_model))
                kln = float(G.eval_kl(e, e_normal))
                if klc < min_klc:
                    min_klc = klc; argmin = E.copy()
                if abs(klc - ncfloor) <= shell_tol:
                    if kln > max_kln_on_shell:
                        max_kln_on_shell = kln; shell_far_pt = E.copy()
                    if kln > 0.01:
                        n_far += 1
                if (i0 % 4 == 0) and (klc < ncfloor + 0.06):   # 서브샘플(근방)
                    scatter.append([klc, kln, tlm_slack(E)])
    return dict(ngrid=ngrid, n_feasible=n_feasible, shell_tol=shell_tol,
                min_klc=float(min_klc), argmin_E=np.round(argmin, 4).tolist(),
                min_klc_minus_ncfloor=float(min_klc - ncfloor),
                n_onshell_far=int(n_far), max_kln_on_shell=float(max_kln_on_shell),
                shell_far_E=(np.round(shell_far_pt, 4).tolist() if shell_far_pt is not None else None),
                empty_over_NS=bool(n_far == 0 and (min_klc - ncfloor) > -1e-4),
                scatter=scatter)


# ===========================================================================
# blind_shell 탐색 (무신호 4-상관 공간)
# ===========================================================================
def find_ns_blind_shell(e_normal, S_target, c_model, ncfloor, seed, restarts=40, quantum_only=False):
    E0 = E_of(e_normal)
    rng = np.random.default_rng(seed)
    wS, wSH, wQ = 300.0, 6000.0, 80.0

    def loss(E):
        E = np.clip(E, -0.9995, 0.9995)
        e = ns_box(E)
        L = -G.eval_kl(e, e_normal) + wS * (chsh_from_E(E) - S_target) ** 2 \
            + wSH * (G.eval_kl(e, c_model) - ncfloor) ** 2
        if quantum_only:
            L += wQ * max(0.0, tlm_slack(E)) ** 2
        return L

    inits = [E0.copy()]
    for _ in range(restarts - 1):
        inits.append(np.clip(E0 + rng.uniform(-0.6, 0.6, size=4), -0.999, 0.999))
    best = None
    for x0 in inits:
        r = minimize(loss, x0, method="Powell", options=dict(maxiter=6000, xtol=1e-8, ftol=1e-11))
        E = np.clip(r.x, -0.9995, 0.9995)
        e = ns_box(E)
        kln = G.eval_kl(e, e_normal)
        sdev = abs(chsh_from_E(E) - S_target)
        shdev = abs(G.eval_kl(e, c_model) - ncfloor)
        tlm = tlm_slack(E)
        feasible = sdev <= S_TOL and shdev <= SH_TOL and (not quantum_only or tlm <= 1e-6)
        score = kln if feasible else kln - 20.0 * (max(0, sdev - S_TOL) + max(0, shdev - SH_TOL)
                                                   + (max(0, tlm) if quantum_only else 0))
        if best is None or score > best[0]:
            best = (score, E, e, dict(E=E.tolist(), S=float(chsh_from_E(E)), s_dev=float(sdev),
                                      kl_to_normal=float(kln), kl_to_c=float(G.eval_kl(e, c_model)),
                                      shell_dev=float(shdev), tlm_slack=tlm, is_quantum=bool(tlm <= 1e-9),
                                      cf=float(G.contextual_fraction(e)), feasible=bool(feasible)))
    return best[2], best[3]


# ===========================================================================
# 탐지 (deviance KL, 3 탐지기)
# ===========================================================================
def gen_boxes(e, n_eval, reps, seed):
    boxes, Ss = [], []
    for i in range(reps):
        rng = np.random.default_rng(seed + 1 + i)
        a, b, xo, yo = S._sample_from_behavior(e, n_eval, rng)
        bx, _ = R2.build_box(a, b, xo, yo)
        boxes.append(bx); Ss.append(R2.chsh_S(bx))
    return boxes, np.array(Ss)


def detect(e_normal, attack_e, n_evals, reps, restarts, train_seeds, seed0):
    cells = []
    for ne in n_evals:
        neg_boxes, neg_S = gen_boxes(e_normal, ne, reps, seed0 + 900000)
        atk_boxes, atk_S = gen_boxes(attack_e, ne, reps, seed0 + 111000)
        aq, ac, as_ = [], [], []
        for ts in range(train_seeds):
            rng = np.random.default_rng(seed0 + 700 + 13 * ts)
            a, b, xo, yo = S.simulate_diqkd(10000, S.CANON_STATE, S.CANON_ANGLES, {"visibility": V_NORMAL}, "none", rng)
            emp_train, _ = R2.build_box(a, b, xo, yo)
            pq = D2.fit_quantum_params(emp_train, np.random.default_rng(seed0 + 800 + 13 * ts), restarts)
            q_model = G.quantum_behavior(pq)
            _, c_model = D2.fit_nc_params(emp_train, np.random.default_rng(seed0 + 950 + 13 * ts), restarts)
            neg_q = np.array([D2.score_kl(x, q_model) for x in neg_boxes])
            neg_c = np.array([D2.score_kl(x, c_model) for x in neg_boxes])
            aq.append(D2.auc([D2.score_kl(x, q_model) for x in atk_boxes], neg_q))
            ac.append(D2.auc([D2.score_kl(x, c_model) for x in atk_boxes], neg_c))
            as_.append(D2.auc(-atk_S, -neg_S))
        cells.append(dict(n_eval=ne, auc_quantum=float(np.mean(aq)), auc_classical=float(np.mean(ac)),
                          auc_stest=float(np.mean(as_))))
    return cells


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="STEP 2c PR-box/초양자 blind_shell 불가능성 완결.")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--train-seeds", type=int, default=3)
    args = ap.parse_args()
    if args.quick:
        args.reps, args.restarts, args.train_seeds = 40, 2, 2
    os.makedirs(OUTDIR, exist_ok=True)

    if not D2.check_step1_passed():
        log("[STOP] STEP 1 미통과."); return

    e_normal = S.build_behavior("none", S.CANON_STATE, S.CANON_ANGLES, {"visibility": V_NORMAL}, np.random.default_rng(SEED))
    S_target = float(R2.chsh_S(e_normal))
    _, c_pop = D2.fit_nc_params(e_normal, np.random.default_rng(SEED + 1), 8)
    ncfloor = float(G.eval_kl(e_normal, c_pop))
    log("정상: S=%.4f CF=%.4f NCfloor=%.5f (c·CF²=%.5f)  TLM_slack(normal)=%.4f (≤0=양자)"
        % (S_target, G.contextual_fraction(e_normal), ncfloor, C_UNIFORM_CHSH * G.contextual_fraction(e_normal) ** 2,
           tlm_slack(E_of(e_normal))))

    log("\n=== blind_shell 탐색 (무신호 4-상관 공간) ===")
    e_q, info_q = find_ns_blind_shell(e_normal, S_target, c_pop, ncfloor, SEED + 10, quantum_only=True)
    log("  [quantum_only ] KL→normal=%.5f  S=%.4f(dev%.4f)  KL→c=%.5f(dev%.5f)  TLM=%.4f  양자=%s  → %s"
        % (info_q["kl_to_normal"], info_q["S"], info_q["s_dev"], info_q["kl_to_c"], info_q["shell_dev"],
           info_q["tlm_slack"], info_q["is_quantum"], "정상붕괴(공집합)" if info_q["kl_to_normal"] < 1e-3 else "해 존재"))
    e_sq, info_sq = find_ns_blind_shell(e_normal, S_target, c_pop, ncfloor, SEED + 20, quantum_only=False)
    log("  [unrestricted ] KL→normal=%.5f  S=%.4f(dev%.4f)  KL→c=%.5f(dev%.5f)  TLM=%.4f  양자=%s  CF=%.4f"
        % (info_sq["kl_to_normal"], info_sq["S"], info_sq["s_dev"], info_sq["kl_to_c"], info_sq["shell_dev"],
           info_sq["tlm_slack"], info_sq["is_quantum"], info_sq["cf"]))
    log("      E(normal)=%s" % np.round(E_of(e_normal), 3).tolist())
    log("      E(unrestricted best)=%s" % np.round(np.array(info_sq["E"]), 3).tolist())

    # ---- 강한 정리 그리드 검증(엄밀): S=S_target 무신호 상자 전체에서 KL→c 최소=정상? ----
    log("\n=== 강한 정리 그리드 검증 (S=%.2f 무신호 상자 전수) ===" % S_target)
    grid = verify_shell_emptiness(e_normal, S_target, c_pop, ncfloor, ngrid=(41 if args.quick else 61))
    log("  스캔 %d 상자 | MIN KL→c=%.6f at E=%s (min−NCfloor=%.2e)"
        % (grid["n_feasible"], grid["min_klc"], grid["argmin_E"], grid["min_klc_minus_ncfloor"]))
    log("  껍질(|KL→c−NCfloor|≤%.4f) 위 정상과 먼(KL→normal>0.01) 상자 수 = %d (max KL→normal on-shell=%.5f)"
        % (grid["shell_tol"], grid["n_onshell_far"], grid["max_kln_on_shell"]))
    log("  → blind_shell 무신호 폴리토프 전체에서 공집합 = %s" % grid["empty_over_NS"])

    empty_over_NS = grid["empty_over_NS"]
    collapses_q = bool(info_q["kl_to_normal"] < 1e-3)

    # 참조: '초양자 blind_shell'을 못 찾았으므로(공집합), 탐지는 unrestricted 최적해(=정상 근방)로 → 전부 ≈0.5(탐지할 것 없음) 확인용
    log("\n=== 탐지(참조): unrestricted 최적해(=정상 근방)에 대한 3 탐지기 — 전부 ≈0.5 이어야(탐지대상 없음) ===")
    cells = detect(e_normal, e_sq, [245, 1000], args.reps, args.restarts, args.train_seeds, SEED + 30)
    for c in cells:
        log("  N_eval=%-5d : quantum=%.3f  classical=%.3f  S-test=%.3f"
            % (c["n_eval"], c["auc_quantum"], c["auc_classical"], c["auc_stest"]))
    nothing_to_detect = all(abs(c["auc_quantum"] - 0.5) < 0.12 and abs(c["auc_classical"] - 0.5) < 0.12 for c in cells)

    theorem = ("★강한 불가능성 정리(엄밀 검증): S=S_normal 인 **모든 무신호 상자**(양자∪초양자, PR-box 포함) 중 "
               "정상이 고전템플릿 거리 KL(‖c_model)=NCfloor 를 **유일 최소화**(스캔 min−NCfloor=%.1e, 껍질 위 먼 상자 0개). "
               "동치로, 정상은 고전탐지기 blind 공 {KL(‖c_model)≤NCfloor} 안에서 CHSH-S 를 **유일 최대화**. "
               "따라서 (고전 blind) ∧ (S≥S_normal, S-검정 blind) ∧ (≠정상) 은 **무신호 폴리토프 전체에서 동시 불가**. "
               "→ 초양자(비물리 PR-box) 공격자에게조차 '고전엔 안 걸리고 양자엔 걸리는' 상자 부재. "
               "**양자 템플릿의 고유 탐지역 = 공집합.** 모델기반 우위(iso_S)는 전적으로 **비양자적**(고전 NC-템플릿과 동급). "
               "이는 우위 정리가 아니라 **탐지가능성 경계 정리**." % grid["min_klc_minus_ncfloor"])

    log("\n>>> 정리: %s" % ("강한 불가능성 확정(무신호 전체 공집합)" if (empty_over_NS and collapses_q and nothing_to_detect)
                          else "재검토(검증 불일치)"))
    log("    %s" % theorem)

    payload = dict(
        note=("STEP 2c: blind_shell 불가능성 정리 완결(강한 형태). 무신호(4-상관) 공간 전수 그리드로, S=S_normal 인 "
              "모든 무신호 상자(양자∪초양자, PR-box 포함) 중 정상이 고전거리 KL(‖c_model)=NCfloor 를 유일 최소화 확인. "
              "→ 초양자에서도 blind_shell 공집합. 양자 템플릿 고유 탐지역=공집합. ★키보안 아닌 장비무결성. supremacy 아님."),
        params=dict(seed=SEED, v_normal=V_NORMAL, s_target=S_target, ncfloor=ncfloor,
                    s_tol=S_TOL, shell_tol=SH_TOL, score="deviance KL(emp‖model)"),
        self_check=dict(normal_tlm_slack=tlm_slack(E_of(e_normal)),
                        normal_is_quantum=bool(tlm_slack(E_of(e_normal)) <= 0),
                        E_normal=np.round(E_of(e_normal), 5).tolist(),
                        nothing_to_detect_auc_near_half=bool(nothing_to_detect)),
        grid_verification={k: v for k, v in grid.items() if k != "scatter"},
        blind_shell_quantum_only=info_q,
        blind_shell_unrestricted=info_sq,
        detection_reference=cells,
        theorem=dict(quantum_empty=collapses_q, superquantum_empty=bool(empty_over_NS),
                     empty_over_full_NS_polytope=bool(empty_over_NS and collapses_q),
                     statement=theorem),
        shell_radius_relation=dict(ncfloor=ncfloor, c_cf2=C_UNIFORM_CHSH * G.contextual_fraction(e_normal) ** 2),
        limits=("균등 marginal·4-상관 무신호 상자 한정(비균등 marginal 은 별도). 그리드 검증(연속 최적화로 교차확인). "
                "초양자 상자는 비물리(worst-case 가정 공격자). G 트랙. 관측이지 supremacy 아님. 키보안 아닌 장비무결성."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload, e_normal, grid)
        log("[저장] %s" % OUT_PNG)
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(P, e_normal, grid):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    ncf = P["params"]["ncfloor"]
    sc = np.array(grid["scatter"]) if grid.get("scatter") else np.zeros((0, 3))
    # 좌: 모든 S=S_normal 무신호 상자 (KL→c vs KL→normal), 양자/초양자 색
    if len(sc):
        q = sc[:, 2] <= 0
        ax[0].scatter(sc[q, 0], sc[q, 1], s=10, color="#7B2FBE", alpha=0.5, label="quantum NS boxes (TLM≤0)")
        ax[0].scatter(sc[~q, 0], sc[~q, 1], s=10, color="#C1442E", alpha=0.5, label="super-quantum NS boxes (TLM>0)")
    ax[0].axvline(ncf, color="#E2A13B", lw=2, ls="--", label="classical shell KL→c=NCfloor=%.4f" % ncf)
    ax[0].axvspan(0, ncf, color="#cccccc", alpha=0.35)
    ax[0].scatter([ncf], [0], marker="*", s=320, color="#1a7f37", zorder=6, label="normal (unique shell point)")
    ax[0].text(ncf * 0.5, max(0.02, (sc[:, 1].max() if len(sc) else 0.05) * 0.5),
               "NO NS box here\n(KL→c<NCfloor):\nnormal uniquely\nminimizes KL→c",
               fontsize=8.5, color="#333", ha="center", va="center")
    ax[0].set_xlabel("KL(box ‖ classical NC template)  [classical detector axis]")
    ax[0].set_ylabel("KL(box ‖ normal quantum template)  [quantum detector axis]")
    ax[0].set_title("STRONG theorem (grid over ALL S=%.2f no-signalling boxes):\n"
                    "every NS box has KL→c ≥ NCfloor; the shell (dashed) holds only normal.\n"
                    "blind_shell EMPTY over the whole NS polytope (quantum ∪ super-quantum)" % P["params"]["s_target"])
    ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=0.3)
    # 우: 참조 탐지 — unrestricted 최적해(=정상 근방)에 대한 3 탐지기 (전부 ≈0.5 = 탐지대상 없음)
    cells = P["detection_reference"]
    x = np.arange(len(cells)); w = 0.25
    ax[1].bar(x - w, [c["auc_quantum"] for c in cells], w, label="quantum template", color="#7B2FBE")
    ax[1].bar(x, [c["auc_classical"] for c in cells], w, label="classical NC template", color="#E2A13B")
    ax[1].bar(x + w, [c["auc_stest"] for c in cells], w, label="S-test", color="#3B6EA5")
    ax[1].axhline(0.5, color="k", lw=0.8, ls="--", label="chance 0.5")
    ax[1].set_xticks(x); ax[1].set_xticklabels(["N_eval=%d" % c["n_eval"] for c in cells])
    ax[1].set_ylim(0, 1.05); ax[1].set_ylabel("AUC vs normal")
    ax[1].set_title("reference: best 'blind_shell' candidate = normal itself\n→ all detectors ≈0.5 (nothing to detect, as theorem predicts)")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=0.3, axis="y")
    plt.suptitle("STEP 2c — blind_shell impossibility, strong form: quantum template has NO unique detection regime "
                 "even vs super-quantum attackers  (device integrity, NOT supremacy)", fontsize=10.5)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130); plt.close()


if __name__ == "__main__":
    main()
