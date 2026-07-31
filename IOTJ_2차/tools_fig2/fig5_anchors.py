"""Fig 5 — measured quantum anchors on the gap-CF curve, plus the off anchor.
sources: results2/gate_r2/r2.json (anchors), results2/gate_r3/r3.json (N-BaIoT)"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style2 import plt, COL1, COL2, ROOT, TSIRELSON, save, legend_figure

NAME = "fig5_anchors"
R2F = os.path.join(ROOT, "results2/gate_r2/r2.json")
R3F = os.path.join(ROOT, "results2/gate_r3/r3.json")
# ── 제외/표시 규칙 (하드코딩) ───────────────────────────────────────
C_UNIFORM = 1.0 / 6.0     # gate_e: CHSH 균등방향 경계극한 계수
# ────────────────────────────────────────────────────────────────────

r2 = json.load(open(R2F)); r3 = json.load(open(R3F))
rs = r2["reference_sweep"]
cfs = np.array([r["cf"] for r in rs]); ncs = np.array([r["ncfloor"] for r in rs])
o = np.argsort(cfs)
A = r2["anchor_placement"]
print("[filter] r2 anchors used: 2/2 (kcbs, delft). reference_sweep rows: %d" % len(rs))
cells = r3["cells"]
da = [c for c in cells if c["definition"] == "a"]
db = [c for c in cells if c["definition"] == "b"]
print("[filter] r3 cells used: %d/%d  (definition a: %d, b: %d); 0 excluded"
      % (len(cells), len(cells), len(da), len(db)))
print("[note] rel_dev in r2.json is measured against the NUMERICAL reference sweep "
      "(local c = NC/CF^2 drifts 0.171 -> 0.288), not against CF^2/6.")

fig, ax = plt.subplots(figsize=(COL1, 2.75))
xx = np.linspace(0, 0.45, 300)
ax.plot(cfs[o], ncs[o], "-", color="#1b3a5c", lw=1.0,
        label="numerical NC floor\n(reference sweep)")
ax.plot(xx, C_UNIFORM * xx ** 2, "--", color="#9a9a9a", lw=0.7,
        label=r"$\mathrm{CF}^2/6$  (boundary limit)")

style = {"kcbs": ("KCBS ion", "o", "#c2453f"), "delft": ("Delft NV", "s", "#2a8c7a")}
for key, (lab, mk, col) in style.items():
    src = r2["track_A_kcbs"] if key == "kcbs" else r2["track_B_delft"]
    cf, gap, ci = A[key]["cf"], A[key]["gap"], src["cf_ci"]
    ax.errorbar([cf], [gap], xerr=[[cf - ci[0]], [ci[1] - cf]], fmt=mk, color=col,
                ms=5.0, mew=0.8, elinewidth=0.8, capsize=1.8, label=lab, zorder=5)
    ax.annotate("%.1f%%" % (100 * A[key]["rel_dev"]), xy=(cf, gap),
                xytext=(4.5, -7.5), textcoords="offset points", fontsize=6, color=col)
    print("   %-10s CF=%.4f CI=[%.4f, %.4f] gap=%.6f curve_pred=%.6f rel_dev=%.2f%%"
          % (lab, cf, ci[0], ci[1], gap, A[key]["curve_pred"], 100 * A[key]["rel_dev"]))

ax.axvline(TSIRELSON, color="k", ls=(0, (1, 2)), lw=0.6)
ax.text(TSIRELSON - 0.008, 0.0455, "Tsirelson", fontsize=5.5, rotation=90, va="top", ha="right")
ax.plot([c["cf"] for c in cells], [c["gap_upper"] for c in cells], "v", color="#5a5a5a",
        ms=3.0, mfc="none", mew=0.7, label="N-BaIoT (upper bnd)")
ax.set_xlabel("contextual fraction  CF")
ax.set_ylabel("learning gap  (nats)")
ax.set_xlim(-0.015, 0.452); ax.set_ylim(-0.0022, 0.049)
ax.grid(alpha=0.3)

# ── 원점 확대 인셋 ──────────────────────────────────────────────────
axi = ax.inset_axes([0.405, 0.625, 0.315, 0.325])
axi.plot([c["cf"] for c in da], [c["gap_upper"] for c in da], "v", color="#5a5a5a",
         ms=2.6, mfc="none", mew=0.7, label="def a (%d)" % len(da))
axi.plot([c["cf"] for c in db], [c["gap_upper"] for c in db], "P", color="#b07d2b",
         ms=2.6, mew=0.6, label="def b (%d)" % len(db))
axi.set_xlim(-4e-6, 1.0e-4); axi.set_ylim(-2e-10, 5e-9)
axi.tick_params(labelsize=4.4, width=0.5, pad=1.2)
axi.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))
axi.xaxis.get_offset_text().set_fontsize(4.2)
axi.yaxis.get_offset_text().set_fontsize(4.2)
axi.grid(alpha=0.25)
for sp in axi.spines.values():
    sp.set_linewidth(0.5)

fig.tight_layout()
save(fig, NAME)

legend_figure([
    dict(label="numerical NC floor (reference sweep)", color="#1b3a5c", marker="None", ls="-"),
    dict(label=r"$\mathrm{CF}^2/6$  (boundary limit)", color="#9a9a9a", marker="None", ls="--", lw=0.7),
    dict(label="KCBS ion", color="#c2453f", marker="o", ls="none", ms=5.0),
    dict(label="Delft NV", color="#2a8c7a", marker="s", ls="none", ms=5.0),
    dict(label="N-BaIoT, gap upper bound", color="#5a5a5a", marker="v", ls="none", mfc="white"),
    dict(label="inset: definition a (%d cells)" % len(da), color="#5a5a5a", marker="v", ls="none", mfc="white", ms=2.8),
    dict(label="inset: definition b (%d cells)" % len(db), color="#b07d2b", marker="P", ls="none", ms=2.8),
], NAME + "_legend", ncol=3,
   note="horizontal bars: 95% CI on CF.  percentages: deviation from the numerical sweep.")
