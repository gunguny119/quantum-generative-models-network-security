"""Fig 7 — certified min-entropy: CHSH-S alone vs full statistics.
source: results2/gate_s8/s8_phaseB1.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style3 import plt, COL1, COL2, ROOT, TSIRELSON, save, panel_legend

NAME = "fig7_certification_loss"
SRC = os.path.join(ROOT, "results2/gate_s8/s8_phaseB1.json")
# ── 제외 규칙 (하드코딩) ─────────────────────────────────────────────
BOX, X_STAR, LEVEL, V_MAX = "CANON_23", 0, "2", 0.96
DROP_INACCURATE = True          # status 에 optimal_inaccurate 포함 행 제외
DELFT_CF = 0.210567
THEORY_ENDPOINTS = [(0.0, 0.0), (TSIRELSON, 0.0)]
# ────────────────────────────────────────────────────────────────────

d = json.load(open(SRC))
allr = d["results"]
sel = [r for r in allr if r["box"] == BOX and r["x_star"] == X_STAR and r["level"] == LEVEL]
kept, dropped = [], []
for r in sorted(sel, key=lambda x: x["V"]):
    why = []
    if r["V"] > V_MAX:
        why.append("V=%.3f > %.2f" % (r["V"], V_MAX))
    if DROP_INACCURATE and any("inaccurate" in s for s in (r["status_S"], r["status_full"])):
        why.append("status %s/%s" % (r["status_S"], r["status_full"]))
    (dropped if why else kept).append((r, "; ".join(why)))
print("[filter] box=%s x_star=%d level=%s V<=%.2f, drop_inaccurate=%s"
      % (BOX, X_STAR, LEVEL, V_MAX, DROP_INACCURATE))
print("[filter] %d/%d rows kept from the %d matching rows; %d total excluded"
      % (len(kept), len(sel), len(sel), len(allr) - len(kept)))
for r, why in dropped:
    print("   EXCLUDED V=%.3f  (%s)" % (r["V"], why))
assert not any(abs(r["V"] - 1.0) < 1e-9 for r, _ in kept), "V=1.000 leaked through the filter"
print("   [check] V=1.000 present in kept rows? False")

cf = np.array([r["CF"] for r, _ in kept])
hS = np.array([r["h_S"] for r, _ in kept])
hF = np.array([r["h_full"] for r, _ in kept])
dH = np.array([r["dH"] for r, _ in kept])
rel = 100 * dH / hS
o = np.argsort(cf)
imax = int(np.argmax(dH))
brk = (cf[o][max(imax - 1, 0)], cf[o][min(imax + 1, len(cf) - 1)])
print("   dH max is at CF=%.6f; bracketed by neighbours [%.6f, %.6f] (grid cannot localise it)"
      % (cf[imax], brk[0], brk[1]))

fig, (axT, axB) = plt.subplots(1, 2, figsize=(COL2, 3.2),
                               gridspec_kw=dict(width_ratios=[1.0, 1.0], wspace=0.34))

# ── 상단: h_S vs h_full ─────────────────────────────────────────────
axT.fill_between(cf[o], hS[o], hF[o], color="#c2453f", alpha=0.16, lw=0,
                 label=r"accounting gap $\Delta H$")
axT.plot(cf[o], hS[o], "o-", color="#1b3a5c", label=r"CHSH-$S$ only  $H^{S}$")
axT.plot(cf[o], hF[o], "s--", color="#c2453f", label=r"full statistics  $H^{\rm full}$")
axT.set_xlabel("contextual fraction  CF")
axT.set_ylabel(r"certified $H_{\min}$  (bits)")
axT.set_xlim(-0.02, 0.435)
axT.grid(alpha=0.3)

# ── 하단: dH ────────────────────────────────────────────────────────
axB.axvspan(brk[0], brk[1], color="#b07d2b", alpha=0.13, lw=0)
axB.text(np.mean(brk), 0.0034, "maximum\nbracketed", fontsize=7.5, ha="center",
         va="bottom", color="#7a5a1a")
axB.plot(cf[o], dH[o], "o-", color="#c2453f", label="measured  (NPA level 2)")
tx = [p[0] for p in THEORY_ENDPOINTS]; ty = [p[1] for p in THEORY_ENDPOINTS]
axB.plot(tx, ty, "s", mfc="none", mec="#333333", mew=0.9, ms=4.2, label="theoretical endpoints")
axB.axvline(DELFT_CF, color="#777777", ls=(0, (1, 2)), lw=0.6)
i245 = int(np.argmin(np.abs(cf - DELFT_CF)))
axB.annotate("Delft operating point\n%+.4f bits (%.1f%%)" % (dH[i245], rel[i245]),
             xy=(DELFT_CF, dH[i245]), xytext=(0.0, 0.0252), fontsize=7.8,
             arrowprops=dict(arrowstyle="->", lw=0.5))
axB.set_xlabel("contextual fraction  CF")
axB.set_ylabel(r"$\Delta H = H^{\rm full} - H^{S}$  (bits)")
axB.set_xlim(-0.02, 0.435); axB.set_ylim(-0.0022, 0.0335)
axB.grid(alpha=0.3)

axB2 = axB.twinx()
axB2.plot(cf[o], rel[o], alpha=0)
axB2.set_ylim(axB.get_ylim()[0] / hS[i245] * 100 * dH[i245] / dH[i245], None)
axB2.set_ylim(-0.0022 / 0.019395 * 9.31, 0.0335 / 0.019395 * 9.31)
axB2.set_ylabel(r"relative gap $\Delta H / H^{S}$  (%)", fontsize=9)
axB2.tick_params(labelsize=8)

panel_legend(axT, [
    dict(label=r"CHSH-$S$ only  $H^{S}$", color="#1b3a5c", marker="o", ls="-"),
    dict(label=r"full statistics  $H^{\rm full}$", color="#c2453f", marker="s", ls="--"),
    dict(label=r"gap $\Delta H$", color="#c2453f", patch=True, alpha=0.16)],
    ncol=1, y=-0.22)

panel_legend(axB, [
    dict(label="measured (NPA level 2)", color="#c2453f", marker="o", ls="-"),
    dict(label="theoretical endpoints", color="#333333", marker="s", ls="none", mfc="white", ms=4.2),
    dict(label="maximum bracketed", color="#b07d2b", patch=True, alpha=0.13)],
    ncol=1, y=-0.22)

save(fig, NAME)
