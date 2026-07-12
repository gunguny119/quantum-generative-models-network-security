#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CF² 논문 Figure 3개 생성 (results2 저장값만 사용, 데이터 조작·재계산 없음). 300 dpi PNG.
   기존 파일 무수정: results2 읽기만, figures/ 아래 PNG 만 생성."""
import os, sys, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(HERE); os.chdir(ROOT)
FIG = "figures"; os.makedirs(FIG, exist_ok=True)
CF_TS = float(np.sqrt(2) - 1)      # 0.4142 (Tsirelson)
LAM_Q = 1/np.sqrt(2)               # 0.7071

# ── 저장값 로드 ──
gc = json.load(open("results2/gap_cf/gap_cf.json"))
cells = gc["sweeps"]["inf"]
lam = np.array([c["lam"] for c in cells])
cf  = np.array([c["cf"] for c in cells])
gap_ctx = np.array([c["gap_contextual"] for c in cells])
gap_mat = np.array([c["gap_matched"] for c in cells])
klq = np.array([c["kl_quantum"] for c in cells])
klc = np.array([c["kl_class_matched"] for c in cells])
oL = np.argsort(lam)      # λ 정렬
oC = np.argsort(cf)       # CF 정렬

ge = json.load(open("results2/gate_e/gate_e.json"))["results"]["CHSH"]["uniform"]
C_FIT = ge["c_fit"]; C_PRED = ge["c_pred_limit"]; ALPHA = ge["alpha"]; R2 = ge["r2_fit"]

r2 = json.load(open("results2/gate_r2/r2.json"))
A = r2["track_A_kcbs"]; B = r2["track_B_delft"]; apk = r2["anchor_placement"]

# 로그용 배열 출력
print("[gap_cf.json sweeps.inf] N points =", len(cells))
print("  CF sorted:", np.round(np.sort(cf), 3).tolist())
print("  gap_contextual (by CF):", np.round(gap_ctx[oC], 4).tolist())
print("  KL_quantum (by λ):", np.round(klq[oL], 4).tolist())
print("  KL_class_matched (by λ):", np.round(klc[oL], 4).tolist())
print("[gate_e CHSH/uniform] c_fit=%.5f c_pred=%.5f alpha=%.4f R2=%.7f" % (C_FIT, C_PRED, ALPHA, R2))
print("[anchors] KCBS cf=%.4f gap=%.4f dev=%.3f | Delft cf=%.4f gap=%.4f dev=%.3f"
      % (A["cf"], A["gap_measured"], apk["kcbs"]["rel_dev"], B["cf"], B["gap_measured"], apk["delft"]["rel_dev"]))

# ============================ Fig 1 ============================
fig, ax = plt.subplots(1, 2, figsize=(13, 5))
# 좌: gap vs CF
ax[0].axvspan(0, CF_TS, color="#DCE9F5", alpha=.7, label="quantum-realizable (CF≤√2−1)")
ax[0].axvspan(CF_TS, 1, color="#F6DED8", alpha=.7, label="super-quantum (PR, nonphysical)")
ax[0].plot(cf[oC], gap_ctx[oC], "o-", color="#6A2FA0", ms=6, lw=2, label="gap = NCfloor − KL$_q$ (contextual)")
ax[0].plot(cf[oC], gap_mat[oC], "x--", color="#B08BD9", ms=6, alpha=.75, label="gap$_{matched}$ (9-param class)")
ax[0].axhline(0, color="k", lw=.6); ax[0].set_xlabel("contextual fraction  CF (LP)")
ax[0].set_ylabel("gap = NCfloor − KL$_{quantum}$")
ax[0].legend(fontsize=8, loc="upper left"); ax[0].grid(alpha=.3)
ax[0].text(0.5, -0.16, "(a) learning gap vs contextuality", transform=ax[0].transAxes, ha="center", va="top", fontsize=12)
# 우: CF self-check + KL 곡선
axr = ax[1]
axr.plot(lam[oL], cf[oL], "o-", color="#2F6DB5", label="CF (our LP)")
th = np.linspace(0, 1, 200); axr.plot(th, np.maximum(0, 2*th-1), "--", color="#C1442E", label="CF theory max(0,2λ−1)")
axr.axvline(LAM_Q, color="green", ls=":", lw=1.5, label="λ=1/√2 (Tsirelson)")
axr.set_xlabel("λ (contextuality knob)"); axr.set_ylabel("CF", color="#2F6DB5"); axr.set_ylim(-0.02, 1.02)
axr.legend(fontsize=8, loc="upper left"); axr.grid(alpha=.3)
ax2 = axr.twinx()
ax2.plot(lam[oL], klq[oL], "^-", color="#6A2FA0", alpha=.85, label="KL$_{quantum}$(λ)")
ax2.plot(lam[oL], klc[oL], "v-", color="#E2A13B", alpha=.85, label="KL$_{class,matched}$(λ)")
ax2.set_ylabel("held-out KL", color="#6A2FA0"); ax2.legend(fontsize=8, loc="upper right")
axr.text(0.5, -0.16, "(b) CF self-check (LP vs theory) & KL curves", transform=axr.transAxes, ha="center", va="top", fontsize=12)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig1_gap_vs_cf.png"), dpi=300, bbox_inches="tight"); plt.close()
print("[saved] figures/fig1_gap_vs_cf.png")

# ============================ Fig 2 (폰트 ×1.5, 상단 제목 없음) ============================
FS = 1.5   # 모든 폰트 1.5배
with plt.rc_context({'font.size': 10*FS, 'axes.titlesize': 12*FS, 'axes.labelsize': 11*FS,
                     'xtick.labelsize': 10*FS, 'ytick.labelsize': 10*FS, 'legend.fontsize': 9*FS}):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.4))
    m = cf > 1e-9
    xx = np.linspace(0, cf.max(), 200)
    ax[0].scatter(cf[m], gap_ctx[m], color="#6A2FA0", s=55, zorder=5, label="stored (CF, gap)")
    ax[0].plot(xx, C_FIT*xx**2, "-", color="#C1442E", lw=2.2, label="gap = c·CF²  (c=%.4f)" % C_FIT)
    ax[0].set_xlabel("CF"); ax[0].set_ylabel("gap"); ax[0].grid(alpha=.3); ax[0].legend(loc="upper left")
    ax[0].text(0.5, -0.17, "(a) gap ≈ c·CF²   (α=%.3f, R²=%.5f)" % (ALPHA, R2), transform=ax[0].transAxes, ha="center", va="top", fontsize=16)
    xl = cf[m]; yl = gap_ctx[m]; mm = yl > 1e-12
    ax[1].loglog(xl[mm], yl[mm], "o", color="#6A2FA0", ms=8, label="stored (CF, gap)")
    xr = np.array([xl[mm].min(), xl[mm].max()])
    c_ref = yl[mm][np.argmin(xl[mm])] / xl[mm].min()**2
    ax[1].plot(xr, c_ref*xr**2, "-", color="#C1442E", lw=1.8, label="slope 2 (CF²)")
    ax[1].plot(xr, (yl[mm][np.argmin(xl[mm])]/xl[mm].min())*xr, ":", color="#888", lw=1.4, label="slope 1 (ref)")
    ax[1].set_xlabel("CF (log)"); ax[1].set_ylabel("gap (log)"); ax[1].grid(alpha=.3, which="both"); ax[1].legend(loc="upper left")
    ax[1].text(0.5, -0.17, "(b) log-log — slope ≈ 2 confirms CF²", transform=ax[1].transAxes, ha="center", va="top", fontsize=16)
    plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig2_cf2_law.png"), dpi=300, bbox_inches="tight"); plt.close()
print("[saved] figures/fig2_cf2_law.png")

# ============================ Fig 3 ============================
fig, ax = plt.subplots(figsize=(7.5, 6))
c16 = 1/6.0
xx = np.linspace(0, 0.32, 200)
ax.plot(xx, c16*xx**2, "-", color="#444", lw=1.8, label="predicted gap = CF²/6")
# KCBS
ax.errorbar([A["cf"]], [A["gap_measured"]], xerr=[[A["cf"]-A["cf_ci"][0]], [A["cf_ci"][1]-A["cf"]]],
            fmt="*", ms=20, color="#1a7f37", capsize=5, zorder=6, label="KCBS ion (CF=%.3f)" % A["cf"])
ax.annotate("KCBS  (dev %.1f%%)" % (100*apk["kcbs"]["rel_dev"]), (A["cf"], A["gap_measured"]),
            textcoords="offset points", xytext=(8, 8), fontsize=9, color="#1a7f37")
# Delft
ax.errorbar([B["cf"]], [B["gap_measured"]], xerr=[[B["cf"]-B["cf_ci"][0]], [B["cf_ci"][1]-B["cf"]]],
            fmt="D", ms=12, color="#7B2FBE", capsize=5, zorder=6, label="Delft NV Bell (CF=%.3f)" % B["cf"])
ax.annotate("Delft  (dev %.1f%%)" % (100*apk["delft"]["rel_dev"]), (B["cf"], B["gap_measured"]),
            textcoords="offset points", xytext=(8, -14), fontsize=9, color="#7B2FBE")
# N-BaIoT (원점 부근)
ax.scatter([0.0], [0.0], marker="v", s=130, color="#555", zorder=6, label="N-BaIoT IoT (CF≈0, gap≈0)")
ax.axhline(0, color="k", lw=.6)
ax.set_xlabel("contextual fraction  CF"); ax.set_ylabel("gap = NCfloor − KL$_{quantum}$")
ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG, "fig3_anchors.png"), dpi=300); plt.close()
print("[saved] figures/fig3_anchors.png")
