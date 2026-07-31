"""Fig 4 — the CF^2 law and its linear-regime exception.  source: results2/gate_e/gate_e.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style2 import plt, COL2, ROOT, save, legend_figure

NAME = "fig4_cf2_law"
SRC = os.path.join(ROOT, "results2/gate_e/gate_e.json")
# ── 방향 표기: 색 + 마커 + 선종류 3중 구분 (그레이스케일 안전) ────────
DIRS = [
    ("CHSH",  "uniform",            "CHSH / uniform",         "o", "-",  "#1b3a5c", False),
    ("CHSH",  "det_vertex_0101",    "CHSH / det vertex 0101", "s", "--", "#c2453f", False),
    ("KCBS5", "uniform",            "KCBS / uniform",         "^", ":",  "#2a8c7a", False),
    ("KCBS5", "det_vertex_all0",    "KCBS / det vertex all0", "D", "-.", "#b07d2b", False),
    ("CHSH",  "det_vertex_0000",    "CHSH / det vertex 0000", "x", "-",  "#8a8a8a", True),
    ("CHSH",  "local_perfect_corr", "CHSH / local perf corr", "+", "--", "#8a8a8a", True),
]
d = json.load(open(SRC))
print("[filter] 6/6 directions used, 0 excluded "
      "(gate_e.json contains no V=1.000 or generic_23 rows; those rules are vacuous here)")

fig = plt.figure(figsize=(COL2, 2.95))
axL = fig.add_axes([0.075, 0.135, 0.395, 0.825])
axR = fig.add_axes([0.578, 0.135, 0.395, 0.825])
axi = axL.inset_axes([0.525, 0.045, 0.445, 0.355])

tab, handles = [], []
for scn, dn, lab, mk, ls, col, is_lin in DIRS:
    r = d["results"][scn][dn]
    cf = np.array([p["cf"] for p in r["points"]])
    nc = np.array([p["ncfloor"] for p in r["points"]])
    o = np.argsort(cf)
    for ax, ms in ((axL, 3.2), (axR, 3.2)):
        h, = ax.plot(cf[o], nc[o], mk, color=col, ms=ms, mew=0.9, lw=0)
        if ax is axL:
            handles.append((h, lab))
    if not is_lin:
        xx = np.linspace(cf.min(), cf.max(), 200)
        for ax in (axL, axR, axi):
            ax.plot(xx, r["c_pred_limit"] * xx ** 2, ls, color=col, lw=0.7, alpha=0.9)
        axi.plot(cf[o], nc[o], mk, color=col, ms=2.3, mew=0.8, lw=0)
    tab.append((lab, r["alpha"], r["r2_fit"]))

# ── (a) 선형축 + quadratic 확대 인셋 ────────────────────────────────
axL.set_xlabel("contextual fraction  CF"); axL.set_ylabel("NC floor  (nats)")
axL.grid(alpha=0.3)
axi.set_xlim(0, 0.104); axi.set_ylim(0, 0.0021)
axi.tick_params(labelsize=4.6, width=0.5, pad=1.2)
axi.grid(alpha=0.25)
for sp in axi.spines.values():
    sp.set_linewidth(0.5)

# ── (b) log-log + 실측 스케일에 정박한 기울기 기준선 ────────────────
c_quad = float(np.median([d["results"][s_][d_]["c_fit"]
                          for s_, d_, _, _, _, _, lin in DIRS if not lin]))
lin_pts = [q for s_, d_, _, _, _, _, lin in DIRS if lin
           for q in d["results"][s_][d_]["points"]]
c_lin = float(np.median([q["ncfloor"] / q["cf"] for q in lin_pts]))
xr = np.array([9e-4, 1.15e-1])
for pw, lab, y0 in ((2, "slope 2", c_quad), (1, "slope 1", c_lin)):
    axR.plot(xr, y0 * xr ** pw, "-", color="#c8c8c8", lw=0.6, zorder=0)
    axR.text(xr[-1] * 1.06, y0 * xr[-1] ** pw, lab, fontsize=5.4, color="#888888", va="center")
print("   [ref] slope-2 anchor c=%.4f (median c_fit of 4 quadratic dirs); "
      "slope-1 anchor c=%.4f (median NC/CF of 2 linear dirs)" % (c_quad, c_lin))
axR.set_xscale("log"); axR.set_yscale("log")
axR.set_xlabel("contextual fraction  CF"); axR.set_ylabel("NC floor  (nats)")
axR.grid(alpha=0.3, which="both")


# ── 공통 범례 (하단) ────────────────────────────────────────────────
save(fig, NAME)

legend_figure([dict(label=lab, color=col, marker=mk, ls=ls, lw=0.7)
               for _, _, lab, mk, ls, col, _ in DIRS]
              + [dict(label=r"$c_{\rm pred}\,\mathrm{CF}^2$ prediction (quadratic dirs)",
                      color="#555555", marker="None", ls="-", lw=0.7),
                 dict(label="slope 1 / slope 2 reference", color="#c8c8c8", marker="None", ls="-", lw=0.7)],
              NAME + "_legend", ncol=3,
              note="inset (left panel): quadratic directions, points vs prediction curve")
for lab, a, r2 in tab:
    print("   %-24s alpha=%.4f  R2=%.5f" % (lab, a, r2))
