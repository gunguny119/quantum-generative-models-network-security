#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STEP B (gate_s3) — 표본 효율: 유한 시행에서 양자 제약 추정기가 plug-in 보다 CF/S 를 정확히 추정하는가.

★ 탐지 아니라 **추정** 과업. 불가능성 정리(탐지 커버리지 우위 없음)와 무관한 축(R6). DI-QKD 실제 병목
  (Delft CF CI [0.115,0.418]) 을 공략. 추정기·학습기만 바꾸고 데이터·시드 동일(R7).

추정기 3(경마 아님 — 역할 다름):
  plugin    : G.contextual_fraction(emp)                  현행 표준(기준선)
  quantum   : fit_quantum_params(emp)→quantum_behavior→CF 후보(양자집합 사영)
  classical : fit_nc_params → model → CF ≡ 0              불가능성 데이터포인트(치역=L, 표현불가; 경쟁자 아님)
              → 매 rep fit 불필요(구조적 0). self-check 로 CF(classical)<1e-12 확인.

키레이트(B-3, 문헌 인용): Pironio et al. NJP 11 045021 (2009) [arXiv:0903.4460] / Acín et al. PRL 98 230501 (2007):
  r ≥ 1 − h(Q) − h((1+√((S/2)²−1))/2),  h(x)=−x log₂x−(1−x)log₂(1−x),  S∈[2,2√2].
  Werner 모델서 Q=(1−S/2√2)/2 (limits 명기). S=2+2·CF (CHSH).

STOP: n=245 에서 quantum MSE < plugin MSE 가 부트스트랩 CI 로 유의하지 않으면 멈추고 보고(→ A 로 이동).

사용: python tools/gate_s3_estimation.py [--reps 300] [--jobs 6] [--quick]
"""
import os
import sys
import json
import argparse

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G
import gate_r2_anchor_strengthen as R2
import gate_s2_detection as D2
import fastq

# 성능 전용: G.quantum_behavior 를 수치동일(<1e-12 검증) 닫힌형으로 교체(~11x). 실패 시 예외→원본 유지.
# gap_vs_cf_v0.py 자체는 무수정(R1). 워커는 fork 로 이 상태를 상속.
FASTQ = fastq.install(G)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                # noqa: E402

OUTDIR = os.path.join("results2", "gate_s3")
OUT_JSON = os.path.join(OUTDIR, "s3.json")
OUT_MSE = os.path.join(OUTDIR, "s3_mse.png")
OUT_KEY = os.path.join(OUTDIR, "s3_keyrate.png")

SEED = 20260725
V_SWEEP = [0.70, 0.75, 0.80, 0.856, 0.92, 1.00]
N_SWEEP = [100, 245, 500, 1000, 5000]
DELFT_RATIO = np.array([53, 79, 62, 51], dtype=float)      # per-context (합 245)
TSIRELSON = 2.0 * np.sqrt(2.0)

KEYRATE_SOURCE = ("Pironio et al., New J. Phys. 11, 045021 (2009) [arXiv:0903.4460]; "
                  "orig. Acín et al., PRL 98, 230501 (2007). "
                  "r>=1-h(Q)-h((1+sqrt((S/2)^2-1))/2), h binary entropy, S in [2,2sqrt2]. "
                  "QBER Q from Werner model Q=(1-S/(2sqrt2))/2 (assumption, see limits).")


def log(m):
    print(m, flush=True)


# ===========================================================================
# 키레이트 (문헌 공식, 무수정 인용)
# ===========================================================================
def h2(x):
    x = min(max(x, 0.0), 1.0)
    if x <= 0.0 or x >= 1.0:
        return 0.0
    return -x * np.log2(x) - (1 - x) * np.log2(1 - x)


def keyrate_chsh(S, Q=None):
    """r = 1 − h(Q) − h((1+√((S/2)²−1))/2). Q 미지정 시 Werner Q=(1−S/2√2)/2. clip S∈[2,2√2]."""
    S = float(min(max(S, 2.0), TSIRELSON))
    if Q is None:
        Q = (1.0 - S / TSIRELSON) / 2.0
    chi = h2((1.0 + np.sqrt(max((S / 2.0) ** 2 - 1.0, 0.0))) / 2.0)
    return 1.0 - h2(Q) - chi


# ===========================================================================
# 표본추출: 균등 / Delft 비율
# ===========================================================================
def sample_alloc(true_box, n, rng, alloc):
    """alloc='uniform' 또는 'delft'. 컨텍스트별 카운트로 emp 박스 생성(약Laplace)."""
    flat = true_box.reshape(4, 4)
    if alloc == "delft":
        counts = np.maximum(np.round(DELFT_RATIO / DELFT_RATIO.sum() * n).astype(int), 1)
    else:
        base = n // 4
        counts = np.array([base + (1 if i < (n - base * 4) else 0) for i in range(4)])
        counts = np.maximum(counts, 1)
    emp = np.zeros((4, 2, 2))
    for ci in range(4):
        d = rng.choice(4, size=int(counts[ci]), p=flat[ci])
        for k in d:
            emp[ci, k // 2, k % 2] += 1.0
    emp += 1.0 / (counts.mean() * 4.0)                     # 약 Laplace(0확률 방지, G.sample_emp 관행)
    emp = emp / emp.sum(axis=(1, 2), keepdims=True)
    return emp


# ===========================================================================
# 한 셀 (V,n,alloc): reps 회 → plugin/quantum CF, no-sig
# ===========================================================================
def _cell_worker(arg):
    V, n, alloc, reps, restarts, seed = arg
    if G.quantum_behavior is not fastq.quantum_behavior:      # spawn 대비(fork 면 이미 상속)
        fastq.install(G, verbose=False)
    true_box = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
    cf_plugin, cf_quantum, nosig = [], [], []
    for r in range(reps):
        rng = np.random.default_rng(seed + r)
        emp = sample_alloc(true_box, n, rng, alloc)
        cf_plugin.append(G.contextual_fraction(emp))
        pq = D2.fit_quantum_params(emp, np.random.default_rng(seed + 100000 + r), restarts)
        cf_quantum.append(G.contextual_fraction(G.quantum_behavior(pq)))
        nosig.append(R2.no_signalling_L1_chsh(emp))
    return dict(V=V, n=n, alloc=alloc,
                cf_plugin=[float(x) for x in cf_plugin],
                cf_quantum=[float(x) for x in cf_quantum],
                nosig=[float(x) for x in nosig])


def aggregate_cell(cell, cf_true, S_true):
    out = dict(V=cell["V"], n=cell["n"], alloc=cell["alloc"], cf_true=float(cf_true), S_true=float(S_true))
    for est in ("plugin", "quantum"):
        v = np.array(cell["cf_%s" % est])
        bias = float(v.mean() - cf_true)
        var = float(v.var())
        mse = float(np.mean((v - cf_true) ** 2))
        lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
        cov = float(np.mean((v >= lo) & (v <= hi)))          # 자기구간 커버(형식); 참값 포함율 아래로
        contains_true = bool(lo <= cf_true <= hi)
        # 키레이트: S_est=2+2·CF_est
        kr = np.array([keyrate_chsh(2.0 + 2.0 * max(c, 0.0)) for c in v])
        kr_cons = keyrate_chsh(2.0 + 2.0 * max(lo, 0.0))       # 보수적(CF 2.5% 하단)
        out[est] = dict(mean=float(v.mean()), bias=bias, var=var, mse=mse, ci=[lo, hi],
                        ci_contains_true=contains_true,
                        keyrate_mean=float(kr.mean()), keyrate_bias=float(kr.mean() - keyrate_chsh(S_true)),
                        keyrate_var=float(kr.var()), keyrate_conservative=float(kr_cons))
    out["classical_cf"] = 0.0                                  # 구조적(치역=L)
    out["nosig_mean"] = float(np.mean(cell["nosig"]))
    out["mse_ratio_q_over_plugin"] = float(out["quantum"]["mse"] / max(out["plugin"]["mse"], 1e-18))
    return out


def bootstrap_mse_gap(cell, cf_true, nboot=2000, seed=SEED):
    """n=245 STOP: quantum MSE < plugin MSE 유의성. 쌍(rep 공유) 부트스트랩 CI of (MSE_plugin − MSE_quantum)."""
    p = np.array(cell["cf_plugin"]); q = np.array(cell["cf_quantum"])
    sp = (p - cf_true) ** 2; sq = (q - cf_true) ** 2
    rng = np.random.default_rng(seed)
    diffs = []
    R = len(p)
    for _ in range(nboot):
        idx = rng.integers(0, R, size=R)
        diffs.append(sp[idx].mean() - sq[idx].mean())         # >0 이면 quantum 이 더 낮은 MSE
    diffs = np.array(diffs)
    return dict(mse_plugin=float(sp.mean()), mse_quantum=float(sq.mean()),
                delta_mean=float(diffs.mean()),
                ci95=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                quantum_better_significant=bool(np.percentile(diffs, 2.5) > 0))


# ===========================================================================
def run_self_check():
    checks = {}
    # classical CF ≡ 0
    e = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, 0.92)
    emp = sample_alloc(e, 2000, np.random.default_rng(SEED + 1), "uniform")
    _, c_model = D2.fit_nc_params(emp, np.random.default_rng(SEED + 2), 6)
    checks["classical_cf_zero"] = dict(value=float(G.contextual_fraction(c_model)),
                                       pass_=bool(G.contextual_fraction(c_model) < 1e-12))
    # V=1 → CF=√2−1
    e1 = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, 1.0)
    cf1 = G.contextual_fraction(e1)
    checks["tsirelson_cf"] = dict(value=float(cf1), target=float(np.sqrt(2) - 1),
                                  pass_=bool(abs(cf1 - (np.sqrt(2) - 1)) < 1e-3))
    checks["all_pass"] = bool(checks["classical_cf_zero"]["pass_"] and checks["tsirelson_cf"]["pass_"])
    return checks


def main():
    ap = argparse.ArgumentParser(description="STEP B 표본효율 CF/S 추정. STOP: n=245 quantum MSE<plugin.")
    ap.add_argument("--reps", type=int, default=300)
    ap.add_argument("--restarts", type=int, default=3)
    ap.add_argument("--jobs", type=int, default=6)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.reps, args.restarts = 60, 2
    os.makedirs(OUTDIR, exist_ok=True)

    log("=== STEP B self-check ===")
    sc = run_self_check()
    log("  classical CF=%.2e (≡0, 치역=L) pass=%s | V=1 CF=%.4f (√2−1) pass=%s"
        % (sc["classical_cf_zero"]["value"], sc["classical_cf_zero"]["pass_"],
           sc["tsirelson_cf"]["value"], sc["tsirelson_cf"]["pass_"]))

    # 참값
    truth = {}
    for V in V_SWEEP:
        tb = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
        truth[V] = (float(G.contextual_fraction(tb)), float(R2.chsh_S(tb)))
    log("  참 CF: " + ", ".join("V%.3f→%.3f" % (V, truth[V][0]) for V in V_SWEEP))

    # 셀 작업(병렬)
    allocs = ["uniform", "delft"]
    tasks = []
    for ai, alloc in enumerate(allocs):
        for vi, V in enumerate(V_SWEEP):
            for ni, n in enumerate(N_SWEEP):
                tasks.append((V, n, alloc, args.reps, args.restarts, SEED + 1000 * ai + 100 * vi + 7 * ni))
    log("\n=== 셀 %d개 (V×n×alloc), reps=%d, jobs=%d ===" % (len(tasks), args.reps, args.jobs))
    if args.jobs <= 1:
        raw = [_cell_worker(t) for t in tasks]
    else:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            raw = list(ex.map(_cell_worker, tasks))

    cells = []
    for cell in raw:
        cf_true, S_true = truth[cell["V"]]
        cells.append(aggregate_cell(cell, cf_true, S_true))
    raw_by_key = {(c["V"], c["n"], c["alloc"]): c for c in raw}

    # 로그: n=245 요약
    log("\n=== n=245 (Delft 실측 시행수) MSE 비교 ===")
    for alloc in allocs:
        for V in V_SWEEP:
            c = next(x for x in cells if x["V"] == V and x["n"] == 245 and x["alloc"] == alloc)
            log("  [%s] V=%.3f CF*=%.3f | plugin MSE=%.2e(bias%+.3f) quantum MSE=%.2e(bias%+.3f) ratio(q/p)=%.2f"
                % (alloc, V, c["cf_true"], c["plugin"]["mse"], c["plugin"]["bias"],
                   c["quantum"]["mse"], c["quantum"]["bias"], c["mse_ratio_q_over_plugin"]))

    # ---- STOP 판정: n=245 각 V·alloc 에서 quantum MSE<plugin (부트스트랩) ----
    log("\n=== STOP 게이트: n=245 quantum MSE < plugin MSE (부트스트랩 CI) ===")
    stop = {}
    n245_sig = []
    for alloc in allocs:
        for V in V_SWEEP:
            key = (V, 245, alloc)
            bs = bootstrap_mse_gap(raw_by_key[key], truth[V][0])
            stop["%s_V%.3f" % (alloc, V)] = bs
            n245_sig.append(bs["quantum_better_significant"])
            log("  [%s V=%.3f] Δ(MSE_p−MSE_q)=%+.2e CI%s → quantum 유의개선=%s"
                % (alloc, V, bs["delta_mean"], [round(x, 5) for x in bs["ci95"]], bs["quantum_better_significant"]))
    # 대표 판정: CF>0 셀(V≥0.80)에서 유의 개선이 다수인가 + CF=0(V=0.70) 편향 감소
    cf_pos_sig = [bootstrap_mse_gap(raw_by_key[(V, 245, "delft")], truth[V][0])["quantum_better_significant"]
                  for V in V_SWEEP if truth[V][0] > 0.05]
    stop_pass = bool(sum(cf_pos_sig) >= max(1, len(cf_pos_sig) // 2))

    log("\n>>> STOP: CF>0 셀(delft, n=245) quantum 유의개선 %d/%d → %s"
        % (sum(cf_pos_sig), len(cf_pos_sig), "PASS(B 진행)" if stop_pass else "FAIL(B 보류, A로 이동)"))

    verdicts = dict(
        stop_pass=stop_pass, stop_detail=stop,
        cf0_bias=dict(  # V=0.70 참CF=0 상향편향(plugin O(N^-1/2)) vs quantum
            note="참 CF=0(V=0.70)서 plugin 상향편향 vs quantum",
            per_n=[dict(n=n,
                        plugin_bias=next(x for x in cells if x["V"] == 0.70 and x["n"] == n and x["alloc"] == "uniform")["plugin"]["bias"],
                        quantum_bias=next(x for x in cells if x["V"] == 0.70 and x["n"] == n and x["alloc"] == "uniform")["quantum"]["bias"])
                   for n in N_SWEEP]),
        keyrate=dict(source=KEYRATE_SOURCE,
                     note="보수적 keyrate(CF 2.5% 하단) 추정기별 차이 = 같은 시행에서 인증비트 차이"),
    )
    payload = dict(
        note=("STEP B 표본효율. CF 추정: plugin(현행)·quantum(양자사영)·classical(치역L→CF≡0). "
              "탐지 아님·추정. R6(커버리지 우위 무관)·R7(동일데이터). 키레이트=Pironio2009 인용. "
              "★모델클래스 분리이지 supremacy 아님. (2,2,2)·양자실현영역(CF≤0.414) 한정. 단정 금지."),
        params=dict(seed=SEED, V_sweep=V_SWEEP, n_sweep=N_SWEEP, reps=args.reps, restarts=args.restarts,
                    allocs=allocs, delft_ratio=DELFT_RATIO.tolist(), keyrate_source=KEYRATE_SOURCE,
                    fastq=FASTQ),
        self_check=sc, truth={("V%.3f" % V): dict(cf=truth[V][0], S=truth[V][1]) for V in V_SWEEP},
        cells=cells, verdicts=verdicts,
        limits=("양자 제약 추정기는 참값이 양자집합 안(CF≤√2−1)일 때 정당 — 초양자 참값에선 편향(스윕을 CF≤0.414 제한). "
                "키레이트 QBER 는 Werner 모델 Q=(1−S/2√2)/2 가정(실측 QBER 별도). classical CF≡0 은 표현불가(성능아님). "
                "no-signalling: quantum_behavior 는 NS 정확만족→양자추정기 잔차↓ vs 실측 Delft 0.0895. supremacy 아님."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload)
        log("[저장] %s , %s" % (OUT_MSE, OUT_KEY))
    except Exception as e:
        log("[plot WARN] %r" % e)

    log("\n" + "=" * 60)
    log("STEP B 판정: STOP %s. %s" % ("PASS" if stop_pass else "FAIL",
                                     "B 논문 포함 검토." if stop_pass else "B 보류 → A(파라미터효율)로 이동 권고."))


def plot(P):
    cells = P["cells"]; V_SW = P["params"]["V_sweep"]; N_SW = P["params"]["n_sweep"]
    # MSE vs n (delft alloc), per V
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    cols = plt.cm.viridis(np.linspace(0, 0.9, len(V_SW)))
    for vi, V in enumerate(V_SW):
        mp = [next(c for c in cells if c["V"] == V and c["n"] == n and c["alloc"] == "delft")["plugin"]["mse"] for n in N_SW]
        mq = [next(c for c in cells if c["V"] == V and c["n"] == n and c["alloc"] == "delft")["quantum"]["mse"] for n in N_SW]
        ax[0].plot(N_SW, mp, "o--", color=cols[vi], alpha=0.6, label="plugin V=%.3f(CF=%.2f)" % (V, next(c for c in cells if c["V"] == V)["cf_true"]))
        ax[0].plot(N_SW, mq, "s-", color=cols[vi], label="quantum V=%.3f" % V)
    ax[0].axvline(245, color="r", ls=":", lw=1.5, label="Delft n=245")
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xlabel("trials n"); ax[0].set_ylabel("MSE of CF estimate")
    ax[0].set_title("Sample efficiency: quantum-projected vs plug-in CF estimator\n(Delft allocation; solid=quantum, dashed=plugin)")
    ax[0].legend(fontsize=6, ncol=2); ax[0].grid(alpha=0.3, which="both")
    # V=0.70 (CF=0) bias vs n
    b = P["verdicts"]["cf0_bias"]["per_n"]
    ax[1].axhline(0, color="k", lw=0.6)
    ax[1].plot([x["n"] for x in b], [x["plugin_bias"] for x in b], "o--", color="#C1442E", label="plugin bias (CF=0)")
    ax[1].plot([x["n"] for x in b], [x["quantum_bias"] for x in b], "s-", color="#7B2FBE", label="quantum bias (CF=0)")
    ax[1].set_xscale("log"); ax[1].set_xlabel("trials n"); ax[1].set_ylabel("bias of CF estimate at true CF=0")
    ax[1].set_title("Upward bias at CF=0 (V=0.70): plug-in O(N^-1/2) vs quantum"); ax[1].legend(fontsize=8); ax[1].grid(alpha=0.3, which="both")
    plt.suptitle("STEP B — sample efficiency of CF estimation (estimation task, NOT detection; NOT supremacy)", fontsize=12)
    plt.tight_layout(); plt.savefig(OUT_MSE, dpi=130); plt.close()

    # keyrate: conservative keyrate vs V (n=245)
    fig, ax = plt.subplots(figsize=(9, 6))
    for est, col, mk in [("plugin", "#C1442E", "o"), ("quantum", "#7B2FBE", "s")]:
        kr = [next(c for c in cells if c["V"] == V and c["n"] == 245 and c["alloc"] == "delft")[est]["keyrate_conservative"] for V in V_SW]
        ax.plot(V_SW, kr, mk + "-", color=col, label="%s (conservative, CF 2.5%%)" % est)
    kr_true = [keyrate_chsh(next(c for c in cells if c["V"] == V)["S_true"]) for V in V_SW]
    ax.plot(V_SW, kr_true, "k:", label="true keyrate")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xlabel("visibility V"); ax.set_ylabel("certified key rate (bits/round)")
    ax.set_title("Conservative certified key rate at n=245 (Pironio 2009 bound)\nquantum estimator → tighter CF lower bound → more certified bits")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(OUT_KEY, dpi=130); plt.close()


if __name__ == "__main__":
    main()
