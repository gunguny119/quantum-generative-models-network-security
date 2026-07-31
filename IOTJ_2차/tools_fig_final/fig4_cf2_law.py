"""Fig 4 (final) — the CF^2 law (left) and where it breaks (right).
source: results2/gate_e/gate_e.json.  좌우 두 패널, 공유 y축으로 두 기울기를 직접 비교한다."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style3 import plt, COL2, ROOT, save, panel_legend

NAME = "fig4_cf2_law"
SRC = os.path.join(ROOT, "results2/gate_e/gate_e.json")
QUAD = [("CHSH",  "uniform",         "CHSH / uniform",     "o", "-",  "#1b3a5c"),
        ("CHSH",  "det_vertex_0101", "CHSH / vertex 0101", "s", "--", "#c2453f"),
        ("KCBS5", "uniform",         "KCBS / uniform",     "^", ":",  "#2a8c7a"),
        ("KCBS5", "det_vertex_all0", "KCBS / vertex all0", "D", "-.", "#b07d2b")]
LIN = [("CHSH", "det_vertex_0000",    "CHSH / vertex 0000",  "x", "#7a2b2b"),
       ("CHSH", "local_perfect_corr", "CHSH / perfect corr", "+", "#b07d2b")]
d = json.load(open(SRC))
print("[filter] left panel: %d quadratic directions;  right panel: %d linear directions" % (len(QUAD), len(LIN)))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL2, 2.9), sharey=True)
xr = np.array([9e-4, 1.15e-1])

# ── 좌: 법칙이 성립하는 방향 ────────────────────────────────────────
for scn, dn, lab, mk, ls, col in QUAD:
    r = d["results"][scn][dn]
    cf = np.array([p["cf"] for p in r["points"]]); nc = np.array([p["ncfloor"] for p in r["points"]])
    o = np.argsort(cf)
    xx = np.linspace(cf.min(), cf.max(), 200)
    axL.plot(xx, r["c_pred_limit"] * xx ** 2, ls, color=col, lw=0.8, zorder=1)
    axL.plot(cf[o], nc[o], mk, color=col, ms=3.4, mew=0.9, lw=0, zorder=3)
    print("   L %-20s alpha=%.4f  R2=%.5f" % (lab, r["alpha"], r["r2_fit"]))
c2 = float(np.median([d["results"][s][x]["c_fit"] for s, x, *_ in QUAD]))
axL.plot(xr, c2 * xr ** 2, "-", color="#c8c8c8", lw=0.7, zorder=0)

# ── 우: 법칙이 깨지는 방향 (좌측 4방향은 회색 띠로 대비) ────────────
curves = []
for s_, x_, *_ in QUAD:
    pts = d["results"][s_][x_]["points"]
    cfq = np.array([q["cf"] for q in pts]); ncq = np.array([q["ncfloor"] for q in pts])
    m = (cfq > 0) & (ncq > 0); o_ = np.argsort(cfq[m])
    curves.append((np.log(cfq[m][o_]), np.log(ncq[m][o_])))
lo_cf = max(c[0][0] for c in curves); hi_cf = min(c[0][-1] for c in curves)
grid = np.linspace(lo_cf, hi_cf, 120)
vals = np.array([np.interp(grid, lx, ly) for lx, ly in curves])
axR.fill_between(np.exp(grid), np.exp(vals.min(axis=0)), np.exp(vals.max(axis=0)),
                 color="#9a9a9a", alpha=0.35, lw=0, zorder=0)
for scn, dn, lab, mk, col in LIN:
    r = d["results"][scn][dn]
    cf = np.array([p["cf"] for p in r["points"]]); nc = np.array([p["ncfloor"] for p in r["points"]])
    o = np.argsort(cf)
    axR.plot(cf[o], nc[o], mk, color=col, ms=4.0, mew=1.1, lw=0, zorder=3)
    print("   R %-20s alpha=%.4f  R2(forced CF^2)=%.5f" % (lab, r["alpha"], r["r2_fit"]))
lin_pts = [p for s, x, *_ in LIN for p in d["results"][s][x]["points"]]
c1 = float(np.median([p["ncfloor"] / p["cf"] for p in lin_pts]))
axR.plot(xr, c1 * xr, "-", color="#c8c8c8", lw=0.7, zorder=1)

for ax in (axL, axR):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("contextual fraction  CF"); ax.grid(alpha=0.3, which="both")
axL.set_ylabel("NC floor  (nats)")
fig.tight_layout()
fig.subplots_adjust(wspace=0.42)   # tight_layout 뒤에 적용해야 유지된다 (좌우 레전드 분리)

panel_legend(axL,
    [dict(label=l, color=c, marker=m, ls=ls, lw=0.8) for _, _, l, m, ls, c in QUAD]
    + [dict(label="slope 2", color="#c8c8c8", marker="None", ls="-", lw=0.7)],
    ncol=2, y=-0.24)

panel_legend(axR,
    [dict(label=l, color=c, marker=m, ls="none", ms=4.0) for _, _, l, m, c in LIN]
    + [dict(label="slope 1", color="#c8c8c8", marker="None", ls="-", lw=0.7),
       dict(label="quadratic range", color="#9a9a9a", patch=True, alpha=0.35)],
    ncol=2, y=-0.24)

save(fig, NAME)
