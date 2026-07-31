"""Fig 4a — the CF^2 law (quadratic directions only).  source: results2/gate_e/gate_e.json
메시지 하나: 매끄러운 facet 방향 4개에서 NC floor 가 c_pred*CF^2 예측을 따른다(기울기 2)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style2 import plt, COL1, ROOT, save, legend_figure

NAME = "fig4a_cf2_law"
SRC = os.path.join(ROOT, "results2/gate_e/gate_e.json")
QUAD = [("CHSH",  "uniform",         "CHSH / uniform",      "o", "-",  "#1b3a5c"),
        ("CHSH",  "det_vertex_0101", "CHSH / vertex 0101",  "s", "--", "#c2453f"),
        ("KCBS5", "uniform",         "KCBS / uniform",      "^", ":",  "#2a8c7a"),
        ("KCBS5", "det_vertex_all0", "KCBS / vertex all0",  "D", "-.", "#b07d2b")]
d = json.load(open(SRC))
print("[filter] quadratic directions only: %d used, 2 linear directions moved to fig4b" % len(QUAD))

fig, ax = plt.subplots(figsize=(COL1, 2.5))
for scn, dn, lab, mk, ls, col in QUAD:
    r = d["results"][scn][dn]
    cf = np.array([p["cf"] for p in r["points"]]); nc = np.array([p["ncfloor"] for p in r["points"]])
    o = np.argsort(cf)
    xx = np.linspace(cf.min(), cf.max(), 200)
    ax.plot(xx, r["c_pred_limit"] * xx ** 2, ls, color=col, lw=0.8, zorder=1)
    ax.plot(cf[o], nc[o], mk, color=col, ms=3.4, mew=0.9, lw=0, zorder=3)
    print("   %-20s alpha=%.4f  R2=%.5f  c_pred=%.4f" % (lab, r["alpha"], r["r2_fit"], r["c_pred_limit"]))
c2 = float(np.median([d["results"][s][x]["c_fit"] for s, x, *_ in QUAD]))
xr = np.array([9e-4, 1.15e-1])
ax.plot(xr, c2 * xr ** 2, "-", color="#c8c8c8", lw=0.7, zorder=0)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("contextual fraction  CF"); ax.set_ylabel("NC floor  (nats)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); save(fig, NAME)
legend_figure([dict(label=l, color=c, marker=m, ls=ls, lw=0.8) for _, _, l, m, ls, c in QUAD]
              + [dict(label="slope 2", color="#c8c8c8", marker="None", ls="-", lw=0.7)],
              NAME + "_legend", ncol=3,
              note=r"lines: information-geometry prediction $c_{\rm pred}\,\mathrm{CF}^2$;  markers: measured.  $\alpha$ = 2.003-2.008,  $R^2 \geq 0.99997$")
