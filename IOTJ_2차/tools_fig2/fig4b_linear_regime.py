"""Fig 4b — where the CF^2 law breaks (boundary-anchored directions).
source: results2/gate_e/gate_e.json.  메시지 하나: 이 두 방향은 기울기 1(선형)이다."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style2 import plt, COL1, ROOT, save, legend_figure

NAME = "fig4b_linear_regime"
SRC = os.path.join(ROOT, "results2/gate_e/gate_e.json")
LIN = [("CHSH", "det_vertex_0000",    "CHSH / vertex 0000",  "x", "-",  "#7a2b2b"),
       ("CHSH", "local_perfect_corr", "CHSH / perfect corr", "+", "--", "#b07d2b")]
QUAD = [("CHSH", "uniform"), ("CHSH", "det_vertex_0101"), ("KCBS5", "uniform"), ("KCBS5", "det_vertex_all0")]
d = json.load(open(SRC))
print("[filter] linear directions only: %d used; the 4 quadratic ones are drawn as a grey band" % len(LIN))

fig, ax = plt.subplots(figsize=(COL1, 2.5))
# 대비용: quadratic 4방향을 개별 계열이 아니라 회색 띠 하나로.
# 방향별 CF 격자가 다르므로 공통 로그격자에 log-log 보간한 뒤 min/max 를 취한다.
curves = []
for s_, x_ in QUAD:
    pts = d["results"][s_][x_]["points"]
    cfq = np.array([q["cf"] for q in pts]); ncq = np.array([q["ncfloor"] for q in pts])
    m = (cfq > 0) & (ncq > 0)
    o_ = np.argsort(cfq[m])
    curves.append((np.log(cfq[m][o_]), np.log(ncq[m][o_])))
lo_cf = max(c[0][0] for c in curves); hi_cf = min(c[0][-1] for c in curves)
grid = np.linspace(lo_cf, hi_cf, 120)
vals = np.array([np.interp(grid, lx, ly) for lx, ly in curves])
ax.fill_between(np.exp(grid), np.exp(vals.min(axis=0)), np.exp(vals.max(axis=0)),
                color="#9a9a9a", alpha=0.35, lw=0, zorder=0)
print("   [band] quadratic range over CF in [%.4f, %.4f] (common overlap of the 4 directions)"
      % (np.exp(lo_cf), np.exp(hi_cf)))

for scn, dn, lab, mk, ls, col in LIN:
    r = d["results"][scn][dn]
    cf = np.array([p["cf"] for p in r["points"]]); nc = np.array([p["ncfloor"] for p in r["points"]])
    o = np.argsort(cf)
    ax.plot(cf[o], nc[o], mk, color=col, ms=4.0, mew=1.1, lw=0, zorder=3)
    print("   %-20s alpha=%.4f  R2(CF^2 fit)=%.5f  qmin=%.2e" % (lab, r["alpha"], r["r2_fit"], r["qmin_ref"]))
lin_pts = [p for s, x, *_ in LIN for p in d["results"][s][x]["points"]]
c1 = float(np.median([p["ncfloor"] / p["cf"] for p in lin_pts]))
xr = np.array([9e-4, 1.15e-1])
ax.plot(xr, c1 * xr ** 1, "-", color="#c8c8c8", lw=0.7, zorder=1)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("contextual fraction  CF"); ax.set_ylabel("NC floor  (nats)")
ax.grid(alpha=0.3, which="both")
fig.tight_layout(); save(fig, NAME)
legend_figure([dict(label=l, color=c, marker=m, ls="none", ms=4.0) for _, _, l, m, ls, c in LIN]
              + [dict(label="slope 1", color="#c8c8c8", marker="None", ls="-", lw=0.7),
                 dict(label="quadratic directions (range)", color="#9a9a9a", patch=True, alpha=0.30)],
              NAME + "_legend", ncol=2,
              note=r"$\alpha$ = 1.002 / 1.007.  forcing a $c\,\mathrm{CF}^2$ fit gives $R^2$ = 0.784 / 0.792 — the quadratic law does not hold here.")
