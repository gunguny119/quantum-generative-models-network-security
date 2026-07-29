"""Gate S3b — 무신호 사영 대조군: "양자 제약인가, 물리성 제약 일반인가".

추정기 사다리: plugin(원시 p̂) → NS 사영(무신호 폴리토프 최근접, 비음수 제약 포함) →
quantum(양자 적합). 셋 다 같은 CF-LP 에 입력하고 **같은 표본**을 공유한다.

★ tools/ 무수정. gate_s3_estimation 의 표본 추출기·시드 체계·집계 의미를 그대로 재사용한다.
  NS 사영은 난수를 쓰지 않으므로 s3 의 RNG 스트림을 교란하지 않는다 → s3 비트 재현 가능.
"""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cvxpy as cp

R2DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R2DIR, "tools"))
import gap_vs_cf_v0 as G                 # noqa: E402
import gate_s3_estimation as S3          # noqa: E402
import gate_s2_detection as D2           # noqa: E402
import gate_r2_anchor_strengthen as R2   # noqa: E402
import fastq                             # noqa: E402

V_SWEEP = S3.V_SWEEP if hasattr(S3, "V_SWEEP") else [0.7, 0.75, 0.8, 0.856, 0.92, 1.0]
N_SWEEP = S3.N_SWEEP if hasattr(S3, "N_SWEEP") else [100, 245, 500, 1000, 5000]
SEED = S3.SEED
OUTDIR = os.path.join(R2DIR, "results2", "gate_s3b")
OUT = os.path.join(OUTDIR, "s3b.json")
PNG = os.path.join(OUTDIR, "s3b_bias_vs_n.png")

# 사영 솔버 설정 (A-3(a) 항등성 기준 1e-8 을 위해 기본값보다 조임)
CLARABEL_KW = dict(solver="CLARABEL", tol_gap_abs=1e-12, tol_gap_rel=1e-12, tol_feas=1e-12)


# ---------------------------------------------------------------------------
# A-1: 무신호 사영 (비음수를 제약에 포함 — 사후 클리핑 금지)
#   min ‖q − p̂‖²  s.t.  q ≥ 0, 컨텍스트별 Σ=1, A주변 y무관, B주변 x무관
#   CTX 순서는 gap_vs_cf_v0 관례: 0=(A0,B0) 1=(A0,B1) 2=(A1,B0) 3=(A1,B1)
# ---------------------------------------------------------------------------
def ns_project(phat):
    q = cp.Variable((4, 4), nonneg=True)
    cons = [cp.sum(q[i, :]) == 1 for i in range(4)]
    mA = lambda i: q[i, 0] + q[i, 1]          # p(a=0|x,y)
    mB = lambda i: q[i, 0] + q[i, 2]          # p(b=0|x,y)
    cons += [mA(0) == mA(1), mA(2) == mA(3)]  # A 주변이 y 에 무관
    cons += [mB(0) == mB(2), mB(1) == mB(3)]  # B 주변이 x 에 무관
    prob = cp.Problem(cp.Minimize(cp.sum_squares(q - phat.reshape(4, 4))), cons)
    prob.solve(**CLARABEL_KW)
    return np.asarray(q.value).reshape(4, 2, 2), prob.status


def cell_worker(arg):
    V, n, alloc, reps, restarts, seed = arg
    if G.quantum_behavior is not fastq.quantum_behavior:
        fastq.install(G, verbose=False)
    true_box = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
    cf_plugin, cf_ns, cf_quantum, nosig = [], [], [], []
    ns_resid, ns_mincell, ns_dist, plug_dist, ns_status = [], [], [], [], []
    t0 = time.time()
    for r in range(reps):
        # --- s3 _cell_worker(111-127) 와 동일한 RNG 소비 순서 ---
        rng = np.random.default_rng(seed + r)
        emp = S3.sample_alloc(true_box, n, rng, alloc)
        cf_plugin.append(G.contextual_fraction(emp))
        # --- 신규 칸(난수 미사용) ---
        q, st = ns_project(emp)
        cf_ns.append(G.contextual_fraction(q))
        ns_resid.append(R2.no_signalling_L1_chsh(q))
        ns_mincell.append(float(q.min()))
        ns_dist.append(float(np.linalg.norm(q - true_box)))
        plug_dist.append(float(np.linalg.norm(emp - true_box)))
        ns_status.append(st)
        # --- s3 와 동일 ---
        pq = D2.fit_quantum_params(emp, np.random.default_rng(seed + 100000 + r), restarts)
        cf_quantum.append(G.contextual_fraction(G.quantum_behavior(pq)))
        nosig.append(R2.no_signalling_L1_chsh(emp))
    return dict(V=V, n=n, alloc=alloc,
                cf_plugin=[float(x) for x in cf_plugin],
                cf_ns=[float(x) for x in cf_ns],
                cf_quantum=[float(x) for x in cf_quantum],
                nosig=[float(x) for x in nosig],
                ns_resid=ns_resid, ns_mincell=ns_mincell,
                ns_dist=ns_dist, plug_dist=plug_dist,
                ns_status_counts={s: ns_status.count(s) for s in set(ns_status)},
                elapsed_s=time.time() - t0)


# ---------------------------------------------------------------------------
def agg_est(v, cf_true, S_true):
    v = np.asarray(v, float)
    lo, hi = float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))
    kr = np.array([S3.keyrate_chsh(2.0 + 2.0 * max(c, 0.0)) for c in v])
    return dict(mean=float(v.mean()), bias=float(v.mean() - cf_true), var=float(v.var()),
                mse=float(np.mean((v - cf_true) ** 2)), ci=[lo, hi],
                ci_contains_true=bool(lo <= cf_true <= hi),
                keyrate_mean=float(kr.mean()),
                keyrate_bias=float(kr.mean() - S3.keyrate_chsh(S_true)),
                keyrate_var=float(kr.var()),
                keyrate_conservative=float(S3.keyrate_chsh(2.0 + 2.0 * max(lo, 0.0))))


def paired_boot(a, b, cf_true, rng, B=2000):
    """MSE 차의 페어드 부트스트랩. >0 이면 b 가 더 낮은 MSE."""
    sa = (np.asarray(a, float) - cf_true) ** 2
    sb = (np.asarray(b, float) - cf_true) ** 2
    R = sa.size
    idx = rng.integers(0, R, size=(B, R))
    d = sa[idx].mean(axis=1) - sb[idx].mean(axis=1)
    return dict(delta_mean=float(d.mean()),
                ci95=[float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))],
                significant=bool(np.percentile(d, 2.5) > 0 or np.percentile(d, 97.5) < 0),
                mse_a=float(sa.mean()), mse_b=float(sb.mean()))


def anchors():
    out = {}
    # (a) 이미 무신호인 행동 → 사영 항등
    ids = []
    for V in (0.70, 0.856, 1.0):
        tb = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
        q, _ = ns_project(tb)
        ids.append(float(np.linalg.norm(q - tb)))
    _, cmod = D2.fit_nc_params(
        S3.sample_alloc(D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, 0.92), 2000,
                        np.random.default_rng(SEED + 1), "uniform"),
        np.random.default_rng(SEED + 2), 6)
    qc, _ = ns_project(cmod)
    ids.append(float(np.linalg.norm(qc - cmod)))
    out["a_identity"] = dict(values=ids, max=max(ids), threshold=1e-8, pass_=bool(max(ids) < 1e-8),
                             note="원 프롬프트 기준 1e-10 은 QP 솔버 정밀도로 달성 불가 — 1e-8 로 완화(사전 고지).")
    # (a') classical_cf ≡ 0 self-check 유지
    out["a_classical_cf_zero"] = dict(value=float(G.contextual_fraction(cmod)),
                                      pass_=bool(G.contextual_fraction(cmod) < 1e-12))
    # 결정성
    emp = S3.sample_alloc(D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, 0.7), 245,
                          np.random.default_rng(SEED), "delft")
    q1, _ = ns_project(emp)
    q2, _ = ns_project(emp)
    out["determinism"] = dict(max_abs_diff=float(np.abs(q1 - q2).max()),
                              pass_=bool(np.abs(q1 - q2).max() < 1e-12))
    return out


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    reps, restarts, jobs = 300, 3, 14
    print("=== Phase A 앵커 ===", flush=True)
    anc = anchors()
    print("  (a) 항등성 max = %.3e (기준 1e-8) → %s" % (anc["a_identity"]["max"], anc["a_identity"]["pass_"]), flush=True)
    print("  classical_cf = %.2e → %s" % (anc["a_classical_cf_zero"]["value"], anc["a_classical_cf_zero"]["pass_"]), flush=True)
    print("  결정성 %.2e → %s" % (anc["determinism"]["max_abs_diff"], anc["determinism"]["pass_"]), flush=True)
    if not (anc["a_identity"]["pass_"] and anc["determinism"]["pass_"]):
        print("!! 앵커 실패 — 중단", flush=True)
        json.dump(dict(anchors=anc, aborted=True), open(OUT, "w"), ensure_ascii=False, indent=1)
        return

    truth = {}
    for V in V_SWEEP:
        tb = D2.S.quantum_e(D2.S.CANON_STATE, D2.S.CANON_ANGLES, V)
        truth[V] = (float(G.contextual_fraction(tb)), float(R2.chsh_S(tb)))

    tasks = []
    for ai, alloc in enumerate(["uniform", "delft"]):
        for vi, V in enumerate(V_SWEEP):
            for ni, n in enumerate(N_SWEEP):
                tasks.append((V, n, alloc, reps, restarts,
                              SEED + 1000 * ai + 100 * vi + 7 * ni))   # s3 main() 과 동일
    print("\n=== Phase B: %d 셀 × %d rep, %d 워커 ===" % (len(tasks), reps, jobs), flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        raw = list(ex.map(cell_worker, tasks))
    print("셀 계산 완료 %.0fs" % (time.time() - t0), flush=True)

    # --- s3 비트 재현 대조 (0-3) ---
    s3 = json.load(open(os.path.join(R2DIR, "results2/gate_s3/s3.json")))
    s3map = {(c["V"], c["n"], c["alloc"]): c for c in s3["cells"]}
    rep_diffs = []
    for cell in raw:
        cf_true, S_true = truth[cell["V"]]
        ref = s3map[(cell["V"], cell["n"], cell["alloc"])]
        for est, key in (("plugin", "cf_plugin"), ("quantum", "cf_quantum")):
            mine = agg_est(cell[key], cf_true, S_true)
            rep_diffs.append(dict(V=cell["V"], n=cell["n"], alloc=cell["alloc"], est=est,
                                  d_mse=abs(mine["mse"] - ref[est]["mse"]),
                                  d_bias=abs(mine["bias"] - ref[est]["bias"])))
    max_dmse = max(r["d_mse"] for r in rep_diffs)
    max_dbias = max(r["d_bias"] for r in rep_diffs)
    repro = dict(max_abs_diff_mse=max_dmse, max_abs_diff_bias=max_dbias,
                 threshold=1e-12, pass_=bool(max_dmse < 1e-12 and max_dbias < 1e-12),
                 worst=sorted(rep_diffs, key=lambda r: -r["d_mse"])[:5])
    print("\n=== 0-3 s3 재현: max|Δmse| = %.3e, max|Δbias| = %.3e → %s ==="
          % (max_dmse, max_dbias, repro["pass_"]), flush=True)

    # --- 집계 + 페어드 비교 ---
    rng = np.random.default_rng(SEED + 4242)
    cells = []
    for cell in raw:
        cf_true, S_true = truth[cell["V"]]
        rec = dict(V=cell["V"], n=cell["n"], alloc=cell["alloc"],
                   cf_true=cf_true, S_true=S_true,
                   plugin=agg_est(cell["cf_plugin"], cf_true, S_true),
                   ns=agg_est(cell["cf_ns"], cf_true, S_true),
                   quantum=agg_est(cell["cf_quantum"], cf_true, S_true),
                   classical_cf=0.0, nosig_mean=float(np.mean(cell["nosig"])),
                   ns_diag=dict(resid_max=float(np.max(cell["ns_resid"])),
                                mincell_min=float(np.min(cell["ns_mincell"])),
                                neg_cell_reps=int(np.sum(np.array(cell["ns_mincell"]) < -1e-9)),
                                closer_frac=float(np.mean(np.array(cell["ns_dist"])
                                                          <= np.array(cell["plug_dist"]) + 1e-12)),
                                status=cell["ns_status_counts"]),
                   delta1_plugin_minus_ns=paired_boot(cell["cf_plugin"], cell["cf_ns"], cf_true, rng),
                   delta2_ns_minus_quantum=paired_boot(cell["cf_ns"], cell["cf_quantum"], cf_true, rng),
                   elapsed_s=cell["elapsed_s"])
        cells.append(rec)

    hl = [c for c in cells if c["V"] == 0.7 and c["n"] == 245 and c["alloc"] == "delft"][0]
    verdicts = dict(
        headline_cell=dict(V=0.7, n=245, alloc="delft"),
        PC1=dict(ns_ci=hl["ns"]["ci"], ns_ci_contains_zero=hl["ns"]["ci_contains_true"],
                 plugin_ci_contains_zero=hl["plugin"]["ci_contains_true"],
                 quantum_ci_contains_zero=hl["quantum"]["ci_contains_true"],
                 reading=("배제 → 억제는 양자 제약 특유 쪽 / 포함 → 물리성 제약만으로 해소")),
        PC2=dict(delta1=hl["delta1_plugin_minus_ns"], delta2=hl["delta2_ns_minus_quantum"]),
        PC3=dict(abs_bias=dict(plugin=abs(hl["plugin"]["bias"]), ns=abs(hl["ns"]["bias"]),
                               quantum=abs(hl["quantum"]["bias"])),
                 predicted_order="plugin > NS > quantum",
                 observed_order=" > ".join(
                     k for k, _ in sorted([("plugin", abs(hl["plugin"]["bias"])),
                                           ("NS", abs(hl["ns"]["bias"])),
                                           ("quantum", abs(hl["quantum"]["bias"]))],
                                          key=lambda kv: -kv[1]))))

    json.dump(dict(
        note=("Gate S3b 무신호 사영 대조군. plugin(12자유도) → NS 사영(8자유도) → "
              "quantum(9-파라 과잉매개화, 차원 ≤8). 셋 다 같은 CF-LP·같은 표본. "
              "목적: s3 의 편향 억제가 양자 제약 특유인지 물리성 제약 일반인지 격리."),
        params=dict(seed=SEED, V_sweep=V_SWEEP, n_sweep=N_SWEEP, allocs=["uniform", "delft"],
                    reps=reps, restarts=restarts, jobs=jobs,
                    seed_rule="SEED + 1000*alloc_idx + 100*V_idx + 7*n_idx (gate_s3_estimation.main 과 동일)",
                    projection=("cvxpy CLARABEL QP, 비음수를 제약에 포함(사후 클리핑 없음), "
                                "tol_gap_abs/rel=tol_feas=1e-12"),
                    corrections_applied=[
                        "헤드라인 셀 기준값을 delft 로 통일 (plugin bias 0.147958, quantum bias 0.039391)",
                        "A-3(a) 항등성 기준 1e-10 → 1e-8 (QP 솔버 정밀도 한계)",
                        "자유도 표: quantum 은 9-파라 과잉매개화이며 차원은 ≤8",
                    ]),
        preregistration=dict(
            PC1="헤드라인 셀에서 NS 사영의 95% 구간이 참값 0 을 배제하는가 (이진).",
            PC2="Δ1 = MSE(plugin)−MSE(NS), Δ2 = MSE(NS)−MSE(quantum). Δ2 의 CI95 가 0 배제 시 양자 증분 유의.",
            PC3="|bias| 순서 plugin > NS > quantum 예측. 깨지면 그대로 보고.",
            note="판정선 사후 변경 금지. Δ2 비유의 시 서술은 '양자 제약은 무신호 제약을 넘어서는 측정 가능한 이득을 주지 않는다'."),
        anchors=anc, s3_reproduction=repro, verdicts=verdicts, cells=cells,
        raw=[{k: v for k, v in c.items() if k.startswith("cf_") or k in
              ("V", "n", "alloc", "nosig", "ns_resid", "ns_mincell")} for c in raw],
        limits=("(2,2,2) 한정 · Werner 가정 · 양자 추정기는 참값이 양자집합 안일 때만 정당 · "
                "NS 사영은 폴리토프이나 양자집합은 곡면이라 두 제약의 기하가 다름 · "
                "Phase C(NPA 사영) 생략 → **앤자츠(9파라)와 양자집합을 분리하지 않았다** · "
                "CF-LP 는 신호가 있는 입력에도 형식적으로 돌지만 그 해석은 plugin 칸에 한함."),
    ), open(OUT, "w"), ensure_ascii=False, indent=1)
    print("[저장] %s" % OUT, flush=True)
    try:
        plot(cells)
    except Exception as ex:
        print("[plot WARN] %r" % ex, flush=True)


def plot(cells):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, alloc in zip(axes, ["uniform", "delft"]):
        for est, c, mk in (("plugin", "#d1495b", "o"), ("ns", "#2a9d8f", "s"), ("quantum", "#264653", "^")):
            ns_ = [next(x for x in cells if x["V"] == 0.7 and x["n"] == n and x["alloc"] == alloc)[est]["bias"]
                   for n in N_SWEEP]
            ax.plot(N_SWEEP, ns_, mk + "-", color=c, label={"ns": "NS 사영"}.get(est, est))
        ax.set_xscale("log"); ax.axhline(0, color="k", lw=.8, ls=":")
        ax.set_xlabel("n"); ax.set_ylabel("bias (참 CF = 0)")
        ax.set_title("V=0.70, alloc=%s" % alloc); ax.grid(alpha=.3); ax.legend(fontsize=8)
    fig.suptitle("Gate S3b: CF bias vs n — plugin / NS 사영 / quantum")
    fig.tight_layout(); fig.savefig(PNG, dpi=150)
    print("[저장] %s" % PNG, flush=True)


if __name__ == "__main__":
    main()
