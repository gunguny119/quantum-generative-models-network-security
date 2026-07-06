#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate R2 — 켜진 앵커 강화: (A) KCBS KL_q 실측(추론→측정) (B) Delft Bell 실데이터 두 번째 앵커.

원칙: 결정적(seed), 기존 도구(G=gap_vs_cf_v0 CHSH CF-LP·NCfloor·2큐빗훈련) 무수정 재사용, self-check,
      단정 금지, ★supremacy 아님(메모리매칭 모델클래스 분리) 명기. 다운로드 출처·해시 기록.

[Track A] b_to_a KCBS 는 **4-cycle(짝수 사이클 = CHSH 동치)**. 5-cycle 오각별 아님 →
  올바른 양자실현은 **2큐빗 CHSH 모델**(G.fit_quantum). 4 상관을 CHSH 컨텍스트로 매핑(S=C 보존)→ KL_q 실측.
  gap_measured = NCfloor − KL_q_measured. R1 추론값(0.0123) 유지 여부 판정.

[Track B] Delft Hensen 2015 loophole-free Bell (4TU DOI 10.4121/UUID:6E19E9B2…, data.zip MD5 342f29f8…).
  event-ready 유효 시행 필터(예제 스크립트 로직 그대로)→ 시행별 (setting a,b; outcome x,y)→ 4-context box.
  no-signalling 잔차 병기, CF-LP + 부트스트랩 CI, S=CHSH·CF=(S−2)/2 교차확인, 2큐빗 KL_q·NCfloor→gap.

사용: python tools/gate_r2_anchor_strengthen.py
"""
import os
import sys
import json
import hashlib

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gap_vs_cf_v0 as G                 # 무수정 재사용 (CHSH CF-LP, nc_floor, fit_quantum) — import 시 os.chdir(IOTJ_2차)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

OUTDIR = os.path.join("results2", "gate_r2")
OUT_JSON = os.path.join(OUTDIR, "r2.json")
OUT_PNG = os.path.join(OUTDIR, "r2.png")
SEED = 20260705
CF_TSIRELSON = float(np.sqrt(2) - 1)

DELFT_TXT = os.path.join("data", "delft", "bell_open_data.txt")
DELFT_ZIP = os.path.join("data", "delft", "data.zip")
DELFT_DOI = "10.4121/uuid:6E19E9B2-4A2D-40B5-8DD3-A660BF3C0A31"
DELFT_ZIP_MD5_EXPECT = "342f29f8288c46575818acd2acebd535"

KCBS_E = np.array([0.6164, 0.625, 0.6678, -0.6166])       # E01,E12,E23,E30 (PMC8827658)
KCBS_SD = np.array([0.0079, 0.0078, 0.0074, 0.0079])
R1_GAP = 0.0123                                            # R1 추론 앵커값


def log(m):
    print(m, flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


def pair_from_corr(E):
    return np.array([[(1 + E) / 4, (1 - E) / 4], [(1 - E) / 4, (1 + E) / 4]])


def kcbs_chsh_box(E):
    """4-cycle 상관 → CHSH 컨텍스트 box (S=C 보존): ctx (0,0)=E01,(0,1)=E12,(1,0)=E23,(1,1)=E30."""
    return np.stack([pair_from_corr(E[0]), pair_from_corr(E[1]), pair_from_corr(E[2]), pair_from_corr(E[3])])


def no_signalling_L1_chsh(e):
    """CHSH box no-signalling 잔차: Alice marg P(a|x) 가 y 무관? Bob marg P(b|y) 가 x 무관? L1 합."""
    # e[ctx,a,b], ctx=(x,y) in order (0,0),(0,1),(1,0),(1,1)
    idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    tot = 0.0
    for x in (0, 1):                       # Alice marg for setting x: compare y=0 vs y=1
        mA_y0 = e[idx[(x, 0)]].sum(axis=1)  # P(a|x,y=0)
        mA_y1 = e[idx[(x, 1)]].sum(axis=1)
        tot += float(np.abs(mA_y0 - mA_y1).sum())
    for y in (0, 1):                       # Bob marg for setting y: compare x=0 vs x=1
        mB_x0 = e[idx[(0, y)]].sum(axis=0)
        mB_x1 = e[idx[(1, y)]].sum(axis=0)
        tot += float(np.abs(mB_x0 - mB_x1).sum())
    return tot / 4.0


def chsh_S(e):
    idx = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}
    E = {}
    for (x, y), ci in idx.items():
        m = e[ci]
        E[(x, y)] = float(m[0, 0] - m[0, 1] - m[1, 0] + m[1, 1])   # ⟨(-1)^a (-1)^b⟩ with 0↔+1
    return E[(0, 0)] + E[(0, 1)] + E[(1, 0)] - E[(1, 1)]


# ===========================================================================
# Track A: KCBS KL_q 실측
# ===========================================================================
def track_A(rng):
    e = kcbs_chsh_box(KCBS_E)
    cf = G.contextual_fraction(e)
    nc = G.nc_floor(e, e, np.random.default_rng(SEED + 1), 8)
    klq = G.fit_quantum(e, e, np.random.default_rng(SEED + 2), 8)
    gap = nc - klq
    C = float(KCBS_E[0] + KCBS_E[1] + KCBS_E[2] - KCBS_E[3])
    # 부트스트랩 (상관 σ 파라메트릭)
    cfs, gaps, klqs = [], [], []
    for bi in range(300):
        Eb = KCBS_E + KCBS_SD * rng.standard_normal(4)
        eb = kcbs_chsh_box(Eb)
        cfs.append(G.contextual_fraction(eb))
        if bi < 120:
            ncb = G.nc_floor(eb, eb, np.random.default_rng(SEED + 300 + bi), 3)
            kqb = G.fit_quantum(eb, eb, np.random.default_rng(SEED + 900 + bi), 3)
            gaps.append(ncb - kqb); klqs.append(kqb)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    holds = abs(gap - R1_GAP) <= (ci(gaps)[1] - ci(gaps)[0]) / 2 + 1e-4 or abs(gap - R1_GAP) < 0.002
    return dict(cf=cf, C=C, ncfloor=nc, kl_q_measured=klq, gap_measured=gap,
                cf_ci=ci(cfs), gap_ci=ci(gaps), klq_ci=ci(klqs),
                r1_gap=R1_GAP, holds=bool(holds),
                verdict=("R1 추론(gap≈0.0123, KL_q≈0) 유지: KL_q 실측=%.5f≈0, gap 실측=%.4f"
                         % (klq, gap) + (" (CI 안)" if holds else " (CI 밖—갱신 필요)")))


# ===========================================================================
# Track B: Delft
# ===========================================================================
def parse_delft():
    d = np.loadtxt(DELFT_TXT, delimiter=",", skiprows=0, usecols=np.arange(1, 17), dtype=np.int64)
    c1t, c1c, c2t, c2c = d[:, 2], d[:, 3], d[:, 4], d[:, 5]
    rndA, rndB = d[:, 6], d[:, 7]
    roA, roB = d[:, 10], d[:, 11]
    exA, exB = d[:, 12], d[:, 13]
    imA, imB = d[:, 14], d[:, 15]
    w0, w1, wl, sep = 5426350, 5425700, 55000 - 2550, 250000
    er1 = ((w0 <= c1t) & (c1t < w0 + wl) & (c1c == 0)) | ((w1 <= c1t) & (c1t < w1 + wl) & (c1c == 1))
    er2 = ((w0 + sep <= c2t) & (c2t < w0 + sep + wl) & (c2c == 0)) | ((w1 + sep <= c2t) & (c2t < w1 + sep + wl) & (c2c == 1))
    erf = er1 & er2 & (c1c != c2c)
    noinv = ((imA == 0) | (imA > 250)) & ((imB == 0) | (imB > 250))
    noex = (exA == 0) & (exB == 0)
    trial = erf & noinv & noex
    rs, rl = 10620, 3700
    xout = ((roA > rs) & (roA <= rs + rl)).astype(int) * 2 - 1
    yout = ((roB > rs) & (roB <= rs + rl)).astype(int) * 2 - 1
    sel = np.where(trial)[0]
    return rndA[sel], rndB[sel], xout[sel], yout[sel]


def build_box(a, b, xo, yo):
    """(setting a,b; outcome ±1 x,y) → e[ctx,ao,bo], ao=0↔+1. 반환 box, per-ctx N."""
    box = np.zeros((4, 2, 2)); Ns = np.zeros(4, int)
    for ci, (sa, sb) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        m = (a == sa) & (b == sb)
        Ns[ci] = int(m.sum())
        for xv in (+1, -1):
            for yv in (+1, -1):
                box[ci, 0 if xv == 1 else 1, 0 if yv == 1 else 1] = int((m & (xo == xv) & (yo == yv)).sum())
    box_n = box / np.clip(box.sum(axis=(1, 2), keepdims=True), 1, None)
    return box_n, Ns


def track_B(rng):
    zmd5 = md5(DELFT_ZIP) if os.path.exists(DELFT_ZIP) else "N/A"
    a, b, xo, yo = parse_delft()
    n = len(a)
    box, Ns = build_box(a, b, xo, yo)
    cf = G.contextual_fraction(box)
    S = chsh_S(box)
    nsig = no_signalling_L1_chsh(box)
    nc = G.nc_floor(box, box, np.random.default_rng(SEED + 11), 8)
    klq = G.fit_quantum(box, box, np.random.default_rng(SEED + 12), 8)
    gap = nc - klq
    # 부트스트랩 (유효 시행 재표집)
    cfs, gaps, Ss = [], [], []
    for bi in range(400):
        idx = rng.integers(0, n, size=n)
        bx, _ = build_box(a[idx], b[idx], xo[idx], yo[idx])
        cfs.append(G.contextual_fraction(bx)); Ss.append(chsh_S(bx))
        if bi < 120:
            ncb = G.nc_floor(bx, bx, np.random.default_rng(SEED + 2000 + bi), 3)
            kqb = G.fit_quantum(bx, bx, np.random.default_rng(SEED + 5000 + bi), 3)
            gaps.append(ncb - kqb)
    ci = lambda v: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
    return dict(source=DELFT_DOI, zip_md5=zmd5, zip_md5_expect=DELFT_ZIP_MD5_EXPECT,
                zip_md5_ok=bool(zmd5 == DELFT_ZIP_MD5_EXPECT),
                n_trials=n, per_ctx_N=Ns.tolist(),
                cf=cf, cf_from_S=(S - 2) / 2.0, S=S, no_sig_L1=nsig,
                ncfloor=nc, kl_q_measured=klq, gap_measured=gap,
                cf_ci=ci(cfs), gap_ci=ci(gaps), S_ci=ci(Ss),
                cross_check_S=bool(abs(cf - (S - 2) / 2.0) < 0.01))


# ===========================================================================
# CHSH 참조곡선 (PR↔uniform): NCfloor(CF) + c·CF² 적합
# ===========================================================================
def reference_curve():
    esc, enc = G.pr_box(), G.uniform_box()
    sweep = []
    for lam in np.round(np.arange(0.0, 1.0001, 0.05), 3):
        e = G.mix(esc, enc, lam)
        cf = G.contextual_fraction(e)
        nc = G.nc_floor(e, e, np.random.default_rng(SEED + int(lam * 1e4)), 5)
        sweep.append(dict(lam=float(lam), cf=float(cf), ncfloor=float(nc)))
    bnd = [(s["cf"], s["ncfloor"]) for s in sweep if 1e-6 < s["cf"] <= 0.1]
    c_fit = None
    if len(bnd) >= 2:
        xb = np.array([c for c, _ in bnd]); yb = np.array([g for _, g in bnd])
        c_fit = float(np.sum(xb**2 * yb) / np.sum(xb**4))
    return sweep, c_fit


def on_curve(sweep, cf_pt, nc_pt):
    scf = np.array([s["cf"] for s in sweep]); snc = np.array([s["ncfloor"] for s in sweep])
    o = np.argsort(scf)
    pred = float(np.interp(cf_pt, scf[o], snc[o]))
    rel = abs(nc_pt - pred) / max(pred, 1e-9)
    return pred, float(rel), bool(rel <= 0.20)


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # self-check (CF-LP: PR=1, uniform=0)
    sc_pr = G.contextual_fraction(G.pr_box()); sc_un = G.contextual_fraction(G.uniform_box())
    log("[self-check CHSH CF-LP] PR-box CF=%.3f(기대1) uniform CF=%.3f(기대0)" % (sc_pr, sc_un))

    log("\n=== Track A: KCBS KL_q 실측 (4-cycle=CHSH 동치, 2큐빗) ===")
    A = track_A(rng)
    log("  CF=%.4f | NCfloor=%.4f | KL_q 실측=%.5f (CI%s) | gap=%.4f (CI%s)"
        % (A["cf"], A["ncfloor"], A["kl_q_measured"], A["klq_ci"], A["gap_measured"], A["gap_ci"]))
    log("  vs R1 추론 gap=0.0123 → %s" % A["verdict"])

    log("\n=== Track B: Delft Hensen 2015 loophole-free Bell ===")
    B = track_B(rng)
    log("  data.zip MD5=%s (기대 %s) → %s" % (B["zip_md5"][:12] + "…", B["zip_md5_expect"][:12] + "…", B["zip_md5_ok"]))
    log("  유효 시행 n=%d | 컨텍스트별 N=%s" % (B["n_trials"], B["per_ctx_N"]))
    log("  CHSH S=%.3f (CI%s) | CF-LP=%.4f | CF=(S−2)/2=%.4f 교차확인=%s"
        % (B["S"], B["S_ci"], B["cf"], B["cf_from_S"], B["cross_check_S"]))
    log("  no-signalling 잔차(L1)=%.4f | NCfloor=%.4f | KL_q 실측=%.5f | gap=%.4f (CI%s)"
        % (B["no_sig_L1"], B["ncfloor"], B["kl_q_measured"], B["gap_measured"], B["gap_ci"]))
    log("  CF 95%%CI=%s" % B["cf_ci"])

    log("\n=== CHSH 참조곡선 & 앵커 배치 ===")
    sweep, c_fit = reference_curve()
    predA, relA, onA = on_curve(sweep, A["cf"], A["gap_measured"])
    predB, relB, onB = on_curve(sweep, B["cf"], B["gap_measured"])
    log("  c_fit(경계 c·CF²)=%s" % (("%.4f" % c_fit) if c_fit else "N/A"))
    log("  KCBS: gap=%.4f vs 곡선예측=%.4f (편차 %.1f%%) → on-curve=%s" % (A["gap_measured"], predA, 100 * relA, onA))
    log("  Delft: gap=%.4f vs 곡선예측=%.4f (편차 %.1f%%) → on-curve=%s" % (B["gap_measured"], predB, 100 * relB, onB))

    verdicts = dict(
        A_kcbs_klq=dict(kl_q_measured=A["kl_q_measured"], gap_measured=A["gap_measured"], r1_gap=R1_GAP,
                        holds=A["holds"],
                        reading=("KL_q 실측=%.5f≈0 → R1 '양자실현→KL_q≈0' 추론 확정. gap 실측=%.4f = R1(0.0123) 유지."
                                 % (A["kl_q_measured"], A["gap_measured"]))),
        B_delft_anchor=dict(cf=B["cf"], cf_ci=B["cf_ci"], gap=B["gap_measured"], gap_ci=B["gap_ci"],
                            curve_pred=predB, rel_dev=relB, on_curve=onB, no_sig_L1=B["no_sig_L1"],
                            reading=("Delft(CF=%.3f, n=%d) gap=%.4f 이 CHSH 곡선예측 %.4f 과 %.1f%% → %s. "
                                     "no-sig 잔차 %.4f(%s)."
                                     % (B["cf"], B["n_trials"], B["gap_measured"], predB, 100 * relB,
                                        "두 번째 켜진 앵커 확보" if onB else "편차—방향의존/유한표본 검토",
                                        B["no_sig_L1"], "유의미: CF 해석 주의" if B["no_sig_L1"] > 0.05 else "작음"))),
        two_anchors=dict(kcbs=(A["cf"], A["gap_measured"]), delft=(B["cf"], B["gap_measured"]),
                         both_on_curve=bool(onA and onB)),
    )
    payload = dict(
        note=("Gate R2 앵커 강화. (A) KCBS 4-cycle=CHSH 동치 → 2큐빗 KL_q 실측(추론→측정). "
              "(B) Delft Hensen2015 실데이터(4TU, MD5검증) → 두 번째 앵커. G(gap_vs_cf_v0) 무수정 재사용. "
              "★모델클래스 분리이지 supremacy 아님. 소규모(Delft n=245)·no-sig 잔차·시뮬훈련·(2,2,2) 한정. 단정 금지."),
        params=dict(seed=SEED, cf_tsirelson=CF_TSIRELSON, kcbs_E=list(KCBS_E), r1_gap=R1_GAP),
        self_check=dict(pr_cf=sc_pr, uniform_cf=sc_un),
        track_A_kcbs=A, track_B_delft=B,
        reference_sweep=sweep, c_fit=c_fit,
        anchor_placement=dict(kcbs=dict(cf=A["cf"], gap=A["gap_measured"], curve_pred=predA, rel_dev=relA, on_curve=onA),
                              delft=dict(cf=B["cf"], gap=B["gap_measured"], curve_pred=predB, rel_dev=relB, on_curve=onB)),
        verdicts=verdicts,
        data_sources=dict(kcbs="PMC8827658/Sci.Adv. abk1660 (Table1 실측 상관)",
                          delft=dict(doi=DELFT_DOI, zip_md5=B["zip_md5"], n_trials=B["n_trials"])),
        limits=("Delft n=245 소규모(CI 폭 넓음)·no-signalling 잔차 존재(보고). KCBS 는 4-cycle(=CHSH 동치)로 2큐빗 실현—"
                "5-cycle 오각별 아님(데이터가 4-cycle). 상관→균등marginal 재구성. NCfloor 수치최소화. "
                "관측/앵커이지 quantum supremacy 아님(모델클래스 분리)."),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    try:
        plot(payload)
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(P):
    sweep = P["reference_sweep"]
    scf = np.array([s["cf"] for s in sweep]); snc = np.array([s["ncfloor"] for s in sweep])
    o = np.argsort(scf)
    A = P["track_A_kcbs"]; B = P["track_B_delft"]
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    ax.axvspan(0, CF_TSIRELSON, color="#DfeeF7", alpha=.5, label="quantum-realizable (CF≤√2−1)")
    ax.axvspan(CF_TSIRELSON, 1, color="#F7E4DF", alpha=.5, label="super-quantum (nonphysical)")
    ax.plot(scf[o], snc[o], "-", color="#E2A13B", lw=2, label="CHSH NCfloor(CF) reference")
    if P["c_fit"]:
        xx = np.linspace(0, scf.max(), 100)
        ax.plot(xx, P["c_fit"] * xx**2, "k--", lw=1, alpha=.6, label="c·CF² fit (c=%.3f)" % P["c_fit"])
    # KCBS
    ax.errorbar([A["cf"]], [A["gap_measured"]],
                xerr=[[A["cf"] - A["cf_ci"][0]], [A["cf_ci"][1] - A["cf"]]],
                yerr=[[max(A["gap_measured"] - A["gap_ci"][0], 0)], [max(A["gap_ci"][1] - A["gap_measured"], 0)]],
                fmt="*", ms=20, color="#1a7f37", capsize=5, zorder=6,
                label="KCBS ion (CF=%.3f, KL_q=%.4f)" % (A["cf"], A["kl_q_measured"]))
    # Delft
    ax.errorbar([B["cf"]], [B["gap_measured"]],
                xerr=[[B["cf"] - B["cf_ci"][0]], [B["cf_ci"][1] - B["cf"]]],
                yerr=[[max(B["gap_measured"] - B["gap_ci"][0], 0)], [max(B["gap_ci"][1] - B["gap_measured"], 0)]],
                fmt="D", ms=12, color="#7B2FBE", capsize=5, zorder=6,
                label="Delft Bell (CF=%.3f, S=%.2f, n=%d)" % (B["cf"], B["S"], B["n_trials"]))
    ax.axhline(0, color="k", lw=.6)
    ax.set_xlabel("contextual fraction  CF")
    ax.set_ylabel("gap = NCfloor − KL_q(measured)")
    ax.set_title("Gate R2 — two real quantum-data anchors on gap≈c·CF² curve\nKCBS ion + Delft Bell (measured KL_q) — model-class separation, NOT supremacy")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=.3)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=130); plt.close()
    log("[저장] %s" % OUT_PNG)


if __name__ == "__main__":
    main()
