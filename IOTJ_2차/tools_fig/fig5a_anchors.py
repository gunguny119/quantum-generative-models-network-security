"""Fig 5a — two measured quantum anchors sit on the predicted curve.
source: results2/gate_r2/r2.json.  메시지 하나: 실측 2점이 곡선 위에 있다(3.7%, 5.3%)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style import plt, COL1, ROOT, TSIRELSON, save, legend_figure

NAME = "fig5a_anchors"
SRC = os.path.join(ROOT, "results2/gate_r2/r2.json")
C_UNIFORM = 1.0 / 6.0
r2 = json.load(open(SRC))
rs = r2["reference_sweep"]
cfs = np.array([r["cf"] for r in rs]); ncs = np.array([r["ncfloor"] for r in rs])
o = np.argsort(cfs); A = r2["anchor_placement"]
print("[filter] anchors 2/2; reference_sweep %d rows; N-BaIoT moved to fig5b" % len(rs))

fig, ax = plt.subplots(figsize=(COL1, 2.4))
xx = np.linspace(0, 0.45, 300)
ax.plot(cfs[o], ncs[o], "-", color="#1b3a5c", lw=1.1)
ax.plot(xx, C_UNIFORM * xx ** 2, "--", color="#9a9a9a", lw=0.7)
for key, mk, col, src in (("kcbs", "o", "#c2453f", "track_A_kcbs"),
                          ("delft", "s", "#2a8c7a", "track_B_delft")):
    cf, gap, ci = A[key]["cf"], A[key]["gap"], r2[src]["cf_ci"]
    ax.errorbar([cf], [gap], xerr=[[cf - ci[0]], [ci[1] - cf]], fmt=mk, color=col,
                ms=5.5, mew=0.8, elinewidth=0.9, capsize=2.0, zorder=5)
    ax.annotate("%.1f%%" % (100 * A[key]["rel_dev"]), xy=(cf, gap), xytext=(5, -8),
                textcoords="offset points", fontsize=6.5, color=col)
    print("   %-6s CF=%.4f  gap=%.6f  pred=%.6f  dev=%.2f%%"
          % (key, cf, gap, A[key]["curve_pred"], 100 * A[key]["rel_dev"]))
ax.axvline(TSIRELSON, color="k", ls=(0, (1, 2)), lw=0.6)
ax.set_xlabel("contextual fraction  CF"); ax.set_ylabel("learning gap  (nats)")
ax.set_xlim(-0.012, 0.45); ax.set_ylim(-0.0018, 0.048); ax.grid(alpha=0.3)
fig.tight_layout(); save(fig, NAME)
legend_figure([
    dict(label="predicted curve (numerical NC floor)", color="#1b3a5c", marker="None", ls="-", lw=1.1),
    dict(label=r"$\mathrm{CF}^2/6$  (boundary limit)", color="#9a9a9a", marker="None", ls="--", lw=0.7),
    dict(label="KCBS ion", color="#c2453f", marker="o", ls="none", ms=5.0),
    dict(label="Delft NV", color="#2a8c7a", marker="s", ls="none", ms=5.0),
], NAME + "_legend", ncol=2,
   note="horizontal bars: 95% CI on CF.  percentages: deviation from the predicted curve.  dotted line: Tsirelson CF = 0.4142.")
