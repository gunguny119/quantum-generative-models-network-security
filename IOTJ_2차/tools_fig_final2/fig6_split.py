"""Fig 6 (split) — tools_fig_final/fig6_estimators.py 의 좌/우 패널을 두 장으로 분리한다.
  fig6a_bias_vs_n  : V=0.70, alloc=delft, bias vs n (좌 패널)
  fig6b_boundary   : V=1.00 경계 병리 (우 패널)
source: results2/gate_s3b/s3b.json

합본은 두 패널이 같은 세 추정기를 쓰므로 **레전드 하나**를 그림 아래에 붙였다.
분리하면 그 공유가 성립하지 않으므로 **각 그림에 같은 레전드를 따로 붙인다.**"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style4 import plt, COL1, ROOT, save, panel_legend

SRC = os.path.join(ROOT, "results2/gate_s3b/s3b.json")
# ── 제외 규칙 (하드코딩, 합본과 동일) ────────────────────────────────
MAIN_V, MAIN_ALLOC = 0.70, "delft"      # 주 패널
BOUND_V = 1.00                          # 경계 병리 패널 전용
N_GRID = [100, 245, 500, 1000, 5000]
# ────────────────────────────────────────────────────────────────────
EST = [("plugin", "plug-in", "o", "-", "#c2453f"),
       ("ns", "projected (NS)", "s", "--", "#2a8c7a"),
       ("quantum", "quantum", "^", ":", "#1b3a5c")]

LEGEND = [dict(label=lab, color=col, marker=mk, ls=ls, ms=4.0)
          for key, lab, mk, ls, col in EST] \
    + [dict(label="filled: excludes true CF",
            color="#444444", marker="o", ls="none", ms=4.0),
       dict(label="open: contains true CF",
            color="#444444", marker="o", ls="none", ms=4.0, mfc="white")]

d = json.load(open(SRC))
C = d["cells"]
print("[filter] total cells %d" % len(C))
main = [c for c in C if c["V"] == MAIN_V and c["alloc"] == MAIN_ALLOC]
print("[filter] panel a: V=%.2f alloc=%s -> %d cells used, %d excluded"
      % (MAIN_V, MAIN_ALLOC, len(main), len(C) - len(main)))
bnd = [c for c in C if c["V"] == BOUND_V]
print("[filter] panel b: V=%.2f -> %d cells (both allocs)" % (BOUND_V, len(bnd)))
print("[note] s3b stores `ci` as the 2.5/97.5 percentiles of the CF estimate; "
      "plotted here as ci - cf_true so the error bar lives on the bias axis.")

# ── (a) V=0.70 : bias vs n ──────────────────────────────────────────
figA, axL = plt.subplots(figsize=(COL1, 2.7))
for key, lab, mk, ls, col in EST:
    xs, ys, lo, hi, filled = [], [], [], [], []
    for n in N_GRID:
        c = next(x for x in main if x["n"] == n)
        e = c[key]; t = c["cf_true"]
        xs.append(n); ys.append(e["bias"])
        lo.append(e["bias"] - (e["ci"][0] - t)); hi.append((e["ci"][1] - t) - e["bias"])
        filled.append(not e["ci_contains_true"])
    axL.plot(xs, ys, ls, color=col, lw=1.0, zorder=2)
    axL.errorbar(xs, ys, yerr=[lo, hi], fmt="none", ecolor=col,
                 elinewidth=0.7, capsize=1.6, zorder=2)
    for x, y, f in zip(xs, ys, filled):
        axL.plot([x], [y], mk, color=col, ms=4.0, mew=0.9,
                 mfc=col if f else "white", zorder=3)
axL.axhline(0.0, color="k", lw=0.8)
axL.axvline(245, color="#777777", ls=(0, (1, 2)), lw=0.6)
axL.text(262, 0.418, "Delft trial count", fontsize=7.6, rotation=90, va="top")
p245 = next(x for x in main if x["n"] == 245)
axL.annotate("plug-in  %+.3f\nprojected %+.3f" % (p245["plugin"]["bias"], p245["ns"]["bias"]),
             xy=(245, p245["plugin"]["bias"]), xytext=(430, 0.238), fontsize=7.8,
             arrowprops=dict(arrowstyle="->", lw=0.5))
axL.set_xscale("log")
axL.set_xticks(N_GRID); axL.set_xticklabels([str(n) for n in N_GRID])
axL.minorticks_off()
axL.set_xlabel("samples  n"); axL.set_ylabel(r"bias of $\widehat{\mathrm{CF}}$  (true CF $=0$)")
axL.grid(alpha=0.3)
figA.tight_layout()
panel_legend(axL, LEGEND, ncol=2, y=-0.22)
save(figA, "fig6a_bias_vs_n")

# ── (b) V=1.00 : 경계 병리 ──────────────────────────────────────────
figB, axR = plt.subplots(figsize=(COL1, 2.7))
allocs = sorted({c["alloc"] for c in bnd})
xpos, xlab = [], []
for gi, al in enumerate(allocs):
    c = next(x for x in bnd if x["alloc"] == al and x["n"] == 245)
    t = c["cf_true"]
    for ei, (key, lab, mk, ls, col) in enumerate(EST):
        e = c[key]
        x = gi * 4 + ei
        axR.errorbar([x], [e["bias"]],
                     yerr=[[e["bias"] - (e["ci"][0] - t)], [(e["ci"][1] - t) - e["bias"]]],
                     fmt=mk, color=col, ms=4.0, mew=0.9,
                     mfc=col if not e["ci_contains_true"] else "white",
                     elinewidth=0.7, capsize=1.8)
        xpos.append(x); xlab.append(lab.split(" ")[0])
        if key == "quantum":
            axR.annotate("%+.3f" % e["bias"], xy=(x, e["bias"]), xytext=(5, -1),
                         textcoords="offset points", fontsize=7.6, color=col, va="center")
    axR.annotate(al, xy=(gi * 4 + 1, 0.965), xycoords=("data", "axes fraction"),
                 fontsize=8, ha="center", va="top")
axR.axhline(0.0, color="k", lw=0.8)
axR.set_xticks(xpos); axR.set_xticklabels(xlab, rotation=90, fontsize=7.6)
axR.set_xlim(-0.9, max(xpos) + 1.5)
axR.set_ylabel("bias  (true CF $= 0.4142$)")
axR.grid(alpha=0.3, axis="y")
figB.tight_layout()
panel_legend(axR, LEGEND, ncol=2, y=-0.40)   # 세로 회전된 눈금 라벨이 차지하는 높이만큼 더 내린다
save(figB, "fig6b_boundary")

for n in N_GRID:
    c = next(x for x in main if x["n"] == n)
    print("   n=%-5d plugin %+.6f(%s)  ns %+.6f(%s)  quantum %+.6f(%s)"
          % (n, c["plugin"]["bias"], "0-excl" if not c["plugin"]["ci_contains_true"] else "0-incl",
             c["ns"]["bias"], "0-excl" if not c["ns"]["ci_contains_true"] else "0-incl",
             c["quantum"]["bias"], "0-excl" if not c["quantum"]["ci_contains_true"] else "0-incl"))
