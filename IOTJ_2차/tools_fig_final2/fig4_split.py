"""Fig 4 (split) — tools_fig_final/fig4_cf2_law.py 의 좌/우 패널을 두 장으로 분리한다.
  fig4a_cf2_law        : CF^2 법칙이 성립하는 4방향 (좌 패널)
  fig4b_linear_regime  : 법칙이 깨지는 2방향 (우 패널)
source: results2/gate_e/gate_e.json.

합본은 sharey=True 로 두 기울기를 직접 비교했다. 분리해도 그 비교가 살아 있도록
**두 그림의 x·y 범위를 합집합으로 맞춰 강제한다** (각자 autoscale 하면 비교가 깨진다)."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style4 import plt, COL1, ROOT, save, panel_legend

SRC = os.path.join(ROOT, "results2/gate_e/gate_e.json")
QUAD = [("CHSH",  "uniform",         "CHSH / uniform",     "o", "-",  "#1b3a5c"),
        ("CHSH",  "det_vertex_0101", "CHSH / vertex 0101", "s", "--", "#c2453f"),
        ("KCBS5", "uniform",         "KCBS / uniform",     "^", ":",  "#2a8c7a"),
        ("KCBS5", "det_vertex_all0", "KCBS / vertex all0", "D", "-.", "#b07d2b")]
LIN = [("CHSH", "det_vertex_0000",    "CHSH / vertex 0000",  "x", "#7a2b2b"),
       ("CHSH", "local_perfect_corr", "CHSH / perfect corr", "+", "#b07d2b")]
d = json.load(open(SRC))
print("[filter] panel a: %d quadratic directions;  panel b: %d linear directions" % (len(QUAD), len(LIN)))

xr = np.array([9e-4, 1.15e-1])

# ── (a) 법칙이 성립하는 방향 ────────────────────────────────────────
figA, axL = plt.subplots(figsize=(COL1, 2.7))
for scn, dn, lab, mk, ls, col in QUAD:
    r = d["results"][scn][dn]
    cf = np.array([p["cf"] for p in r["points"]]); nc = np.array([p["ncfloor"] for p in r["points"]])
    o = np.argsort(cf)
    xx = np.linspace(cf.min(), cf.max(), 200)
    axL.plot(xx, r["c_pred_limit"] * xx ** 2, ls, color=col, lw=0.8, zorder=1)
    axL.plot(cf[o], nc[o], mk, color=col, ms=3.4, mew=0.9, lw=0, zorder=3)
    print("   a %-20s alpha=%.4f  R2=%.5f" % (lab, r["alpha"], r["r2_fit"]))
c2 = float(np.median([d["results"][s][x]["c_fit"] for s, x, *_ in QUAD]))
axL.plot(xr, c2 * xr ** 2, "-", color="#c8c8c8", lw=0.7, zorder=0)

# ── (b) 법칙이 깨지는 방향 (좌측 4방향은 회색 띠로 대비) ────────────
figB, axR = plt.subplots(figsize=(COL1, 2.7))
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
    print("   b %-20s alpha=%.4f  R2(forced CF^2)=%.5f" % (lab, r["alpha"], r["r2_fit"]))
lin_pts = [p for s, x, *_ in LIN for p in d["results"][s][x]["points"]]
c1 = float(np.median([p["ncfloor"] / p["cf"] for p in lin_pts]))
axR.plot(xr, c1 * xr, "-", color="#c8c8c8", lw=0.7, zorder=1)

for ax in (axL, axR):
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("contextual fraction  CF")
    ax.set_ylabel("NC floor  (nats)")
    ax.grid(alpha=0.3, which="both")

# ── 합본의 sharey 를 대신하는 범위 강제 ──────────────────────────────
ylim = (min(axL.get_ylim()[0], axR.get_ylim()[0]), max(axL.get_ylim()[1], axR.get_ylim()[1]))
xlim = (min(axL.get_xlim()[0], axR.get_xlim()[0]), max(axL.get_xlim()[1], axR.get_xlim()[1]))
for ax in (axL, axR):
    ax.set_xlim(xlim); ax.set_ylim(ylim)
print("   [shared range] x=[%.3e, %.3e]  y=[%.3e, %.3e] (두 그림에 동일 적용)"
      % (xlim[0], xlim[1], ylim[0], ylim[1]))

figA.tight_layout()
panel_legend(axL,
    [dict(label=l, color=c, marker=m, ls=ls, lw=0.8) for _, _, l, m, ls, c in QUAD]
    + [dict(label="slope 2", color="#c8c8c8", marker="None", ls="-", lw=0.7)],
    ncol=2, y=-0.24)
save(figA, "fig4a_cf2_law")

figB.tight_layout()
panel_legend(axR,
    [dict(label=l, color=c, marker=m, ls="none", ms=4.0) for _, _, l, m, c in LIN]
    + [dict(label="slope 1", color="#c8c8c8", marker="None", ls="-", lw=0.7),
       dict(label="quadratic range", color="#9a9a9a", patch=True, alpha=0.35)],
    ncol=2, y=-0.24)
save(figB, "fig4b_linear_regime")
