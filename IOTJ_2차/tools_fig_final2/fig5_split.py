"""Fig 5 (split) — tools_fig_final/fig5_anchors.py 의 좌/우 패널을 두 장으로 분리한다.
  fig5a_anchors          : 실측 앵커가 예측곡선 위에 (좌 패널)
  fig5b_iot_off_anchor   : 실 IoT 는 꺼진 앵커 (우 패널)
sources: results2/gate_r2/r2.json, results2/gate_r3/r3.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style4 import plt, COL1, ROOT, TSIRELSON, save, panel_legend

C_UNIFORM = 1.0 / 6.0
FLOOR = 1e-12    # def-a 값(~1e-33~1e-31)은 수치적 0 → 이 높이로 클리핑
YBOT = 1.2e-13   # 축 바닥은 FLOOR 보다 더 아래 → 클리핑된 마커가 경계선에 묻히지 않는다
r2 = json.load(open(os.path.join(ROOT, "results2/gate_r2/r2.json")))
r3 = json.load(open(os.path.join(ROOT, "results2/gate_r3/r3.json")))
rs = r2["reference_sweep"]
cfs = np.array([r["cf"] for r in rs]); ncs = np.array([r["ncfloor"] for r in rs])
o = np.argsort(cfs); A = r2["anchor_placement"]
cells = r3["cells"]
da = [c for c in cells if c["definition"] == "a"]
db = [c for c in cells if c["definition"] == "b"]
gaps = [A[k]["gap"] for k in ("kcbs", "delft")]
print("[filter] panel a: 2 anchors + %d reference rows;  panel b: %d cells (a:%d, b:%d)"
      % (len(rs), len(cells), len(da), len(db)))

# ── (a) 실측 앵커가 예측곡선 위에 ───────────────────────────────────
figA, axL = plt.subplots(figsize=(COL1, 2.7))
xx = np.linspace(0, 0.45, 300)
axL.plot(cfs[o], ncs[o], "-", color="#1b3a5c", lw=1.1)
axL.plot(xx, C_UNIFORM * xx ** 2, "--", color="#9a9a9a", lw=0.7)
for key, mk, col, src in (("kcbs", "o", "#c2453f", "track_A_kcbs"),
                          ("delft", "s", "#2a8c7a", "track_B_delft")):
    cf, gap, ci = A[key]["cf"], A[key]["gap"], r2[src]["cf_ci"]
    axL.errorbar([cf], [gap], xerr=[[cf - ci[0]], [ci[1] - cf]], fmt=mk, color=col,
                 ms=5.5, mew=0.8, elinewidth=0.9, capsize=2.0, zorder=5)
    axL.annotate("%.1f%%" % (100 * A[key]["rel_dev"]), xy=(cf, gap), xytext=(5, -8),
                 textcoords="offset points", fontsize=8.5, color=col)
    print("   a %-6s CF=%.4f gap=%.6f dev=%.2f%%" % (key, cf, gap, 100 * A[key]["rel_dev"]))
axL.axvline(TSIRELSON, color="k", ls=(0, (1, 2)), lw=0.6)
axL.set_xlabel("contextual fraction  CF"); axL.set_ylabel("learning gap  (nats)")
axL.set_xlim(-0.012, 0.45); axL.set_ylim(-0.0018, 0.048); axL.grid(alpha=0.3)
figA.tight_layout()
panel_legend(axL, [
    dict(label="predicted curve (NC floor)", color="#1b3a5c", marker="None", ls="-", lw=1.1),
    dict(label=r"$\mathrm{CF}^2/6$ (boundary limit)", color="#9a9a9a", marker="None", ls="--", lw=0.7),
    dict(label="KCBS ion", color="#c2453f", marker="o", ls="none", ms=5.0),
    dict(label="Delft NV", color="#2a8c7a", marker="s", ls="none", ms=5.0)],
    ncol=2, y=-0.22)
save(figA, "fig5a_anchors")

# ── (b) 실 IoT 는 꺼진 앵커 ─────────────────────────────────────────
figB, axR = plt.subplots(figsize=(COL1, 2.7))
axR.axhspan(min(gaps), max(gaps), color="#c2453f", alpha=0.30, lw=0)
for grp, xc, mk, col, lab in ((da, 0, "v", "#4a4a4a", "definition a"),
                              (db, 1, "P", "#b07d2b", "definition b")):
    y = np.array([max(c["gap_upper"], FLOOR) for c in grp])
    x = xc + (np.linspace(-0.33, 0.33, len(y)) if len(y) > 1 else np.zeros(1))
    axR.plot(x, y, mk, color=col, ms=4.6, mfc="none" if mk == "v" else col, mew=1.0, lw=0)
    print("   b %-13s n=%d  gap_upper max=%.3e" % (lab, len(y), max(y)))
axR.set_yscale("log")
axR.set_xticks([0, 1]); axR.set_xticklabels(["definition a\n(%d)" % len(da),
                                             "definition b\n(%d)" % len(db)], fontsize=9)
axR.set_xlim(-0.62, 1.62); axR.set_ylim(YBOT, 0.05)
axR.set_yticks([1e-12, 1e-10, 1e-8, 1e-6, 1e-4, 1e-2])
axR.set_ylabel("gap upper bound  (nats)")
axR.grid(alpha=0.3, axis="y", which="major")
print("   -> max IoT bound %.3e is %.0f orders of magnitude below the anchor band"
      % (r3["verdict"]["gap_upper_max"], np.log10(min(gaps) / r3["verdict"]["gap_upper_max"])))
figB.tight_layout()
panel_legend(axR, [
    dict(label="N-BaIoT, def. a", color="#5a5a5a", marker="v", ls="none", mfc="white"),
    dict(label="N-BaIoT, def. b", color="#b07d2b", marker="P", ls="none"),
    dict(label="quantum anchor band", color="#c2453f", patch=True, alpha=0.30)],
    ncol=2, y=-0.22)
save(figB, "fig5b_iot_off_anchor")
