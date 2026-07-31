"""Fig 5b — the real-IoT anchor is OFF: N-BaIoT gap upper bounds vs the quantum anchors.
source: results2/gate_r3/r3.json (+ r2.json for the anchor band).
메시지 하나: 실 IoT 텔레메트리는 CF~0 이라 gap 상한이 양자 앵커보다 수십 자릿수 아래다."""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style2 import plt, COL1, ROOT, save, legend_figure

NAME = "fig5b_iot_off_anchor"
r3 = json.load(open(os.path.join(ROOT, "results2/gate_r3/r3.json")))
r2 = json.load(open(os.path.join(ROOT, "results2/gate_r2/r2.json")))
cells = r3["cells"]
da = [c for c in cells if c["definition"] == "a"]
db = [c for c in cells if c["definition"] == "b"]
print("[filter] r3 cells %d/%d used (definition a: %d, b: %d)" % (len(cells), len(cells), len(da), len(db)))
gaps = [r2["anchor_placement"][k]["gap"] for k in ("kcbs", "delft")]
FLOOR = 1e-34

fig, ax = plt.subplots(figsize=(COL1, 2.4))
ax.axhspan(min(gaps), max(gaps), color="#c2453f", alpha=0.30, lw=0)
rng = np.random.default_rng(0)
for grp, xc, mk, col, lab in ((da, 0, "v", "#5a5a5a", "definition a"),
                              (db, 1, "P", "#b07d2b", "definition b")):
    y = np.array([max(c["gap_upper"], FLOOR) for c in grp])
    x = xc + (np.linspace(-0.22, 0.22, len(y)) if len(y) > 1 else np.zeros(1))
    ax.plot(x, y, mk, color=col, ms=3.6, mfc="none" if mk == "v" else col, mew=0.9, lw=0)
    print("   %-14s n=%d  gap_upper max=%.3e  min=%.3e" % (lab, len(y), max(y), min(y)))
ax.set_yscale("log")
ax.set_xticks([0, 1]); ax.set_xticklabels(["definition a\n(%d cells)" % len(da),
                                           "definition b\n(%d cells)" % len(db)], fontsize=7)
ax.set_xlim(-0.6, 1.6)
ax.set_ylim(FLOOR, 0.09)
ax.set_ylabel("gap upper bound  (nats)")
ax.grid(alpha=0.3, axis="y", which="major")
fig.tight_layout(); save(fig, NAME)
print("   anchor band: %.6f - %.6f nats;  max IoT bound %.3e  ->  %.0f orders of magnitude below"
      % (min(gaps), max(gaps), r3["verdict"]["gap_upper_max"],
         np.log10(min(gaps) / r3["verdict"]["gap_upper_max"])))
legend_figure([
    dict(label="N-BaIoT, definition a", color="#5a5a5a", marker="v", ls="none", mfc="white"),
    dict(label="N-BaIoT, definition b", color="#b07d2b", marker="P", ls="none"),
    dict(label="quantum anchors (KCBS, Delft)", color="#c2453f", patch=True, alpha=0.30),
], NAME + "_legend", ncol=3,
   note="all 18 cells have CF $\\approx$ 0 (CI upper $\\leq 1.5\\times10^{-4}$); values below $10^{-34}$ are numerical zeros and are clipped to the axis floor.")
