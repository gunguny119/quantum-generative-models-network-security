"""공통 스타일 — 각 그림 스크립트가 import 한다(지시서 스타일 블록과 동일 값)."""
import os
import numpy as np
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt   # noqa: E402,F401

MM = 1 / 25.4
COL1 = 88 * MM
COL2 = 181 * MM

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
    "lines.markersize": 3.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.01,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGDIR = os.path.join(ROOT, "figures")
TSIRELSON = 0.41421356237309515


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (name, ext)))
    print("[saved] figures/%s.{pdf,png}" % name)


# ---------------------------------------------------------------------------
# 레전드를 본 그림에서 분리해 별도 파일로 저장한다.
#   entries: [{label, color, marker, ls, mfc, ms, lw, patch, alpha}, ...]
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Patch         # noqa: E402


def legend_figure(entries, name, ncol=None, note=None, width=None):
    handles = []
    for e in entries:
        if e.get("patch"):
            handles.append(Patch(facecolor=e["color"], alpha=e.get("alpha", 1.0),
                                 edgecolor="none"))
        else:
            handles.append(Line2D([], [], marker=e.get("marker", "o"),
                                  linestyle=e.get("ls", "-"), color=e["color"],
                                  markerfacecolor=e.get("mfc", e["color"]),
                                  markeredgecolor=e.get("mec", e["color"]),
                                  markersize=e.get("ms", 3.8), markeredgewidth=0.9,
                                  lw=e.get("lw", 1.0)))
    ncol = ncol or len(handles)
    nrow = int(np.ceil(len(handles) / ncol))
    h = 0.16 + 0.155 * nrow + (0.16 if note else 0.0)
    fig = plt.figure(figsize=(width or COL2, h))
    fig.legend(handles, [e["label"] for e in entries], loc="upper center", ncol=ncol,
               frameon=False, handlelength=2.0, columnspacing=1.5, handletextpad=0.5,
               fontsize=7, borderaxespad=0.0)
    if note:
        fig.text(0.5, 0.10, note, ha="center", va="bottom", fontsize=6, color="#444444")
    save(fig, name)
