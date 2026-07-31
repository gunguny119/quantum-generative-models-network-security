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
    "font.size": 10, "axes.labelsize": 10, "axes.titlesize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 9,
    "axes.linewidth": 0.6, "grid.linewidth": 0.4, "lines.linewidth": 1.0,
    "lines.markersize": 3.5, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "legend.frameon": False, "figure.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.005,
    "pdf.fonttype": 42, "ps.fonttype": 42,
})

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
FIGDIR = os.path.join(ROOT, "figures_final")
TSIRELSON = 0.41421356237309515


_LAST = {"fig": None, "name": None}


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (name, ext)))
    _LAST["fig"], _LAST["name"] = fig, name
    print("[saved] figures_final/%s.{pdf,png}" % name)


# ---------------------------------------------------------------------------
# 레전드를 본 그림에서 분리해 별도 파일로 저장한다.
#   entries: [{label, color, marker, ls, mfc, ms, lw, patch, alpha}, ...]
# ---------------------------------------------------------------------------
from matplotlib.lines import Line2D          # noqa: E402
from matplotlib.patches import Patch         # noqa: E402


def legend_figure(entries, name, ncol=None, note=None, width=None):
    """figures2 판: 별도 파일을 만들지 않고 **직전에 저장한 그림의 하단**에 레전드를 붙여
    같은 이름으로 다시 저장한다. savefig.bbox='tight' 이므로 축 밖 레전드도 잘리지 않는다."""
    fig, base = _LAST["fig"], _LAST["name"]
    if fig is None:
        raise RuntimeError("save(fig, name) 가 먼저 호출되어야 한다")
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
    y0 = -0.004
    fig.legend(handles, [e["label"] for e in entries], loc="upper center", ncol=ncol,
               frameon=False, handlelength=2.0, columnspacing=1.5, handletextpad=0.5,
               fontsize=10, borderaxespad=0.0, bbox_to_anchor=(0.5, y0))
    if note:
        fig.text(0.5, y0 - 0.045 * nrow - 0.030, note, ha="center", va="top",
                 fontsize=6, color="#444444", transform=fig.transFigure)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIGDIR, "%s.%s" % (base, ext)))
    print("[saved] figures_final/%s.{pdf,png}  (bottom legend attached)" % base)


def panel_legend(ax, entries, ncol=None, y=-0.22, note=None, fontsize=9.5):
    """**해당 패널 바로 아래**에 그 패널의 레전드만 붙인다(패널 간 혼동 방지).
    y 는 축 좌표계 기준 위치(음수 = 축 아래). savefig bbox='tight' 라 잘리지 않는다."""
    handles = []
    for e in entries:
        if e.get("patch"):
            handles.append(Patch(facecolor=e["color"], alpha=e.get("alpha", 1.0), edgecolor="none"))
        else:
            handles.append(Line2D([], [], marker=e.get("marker", "o"),
                                  linestyle=e.get("ls", "-"), color=e["color"],
                                  markerfacecolor=e.get("mfc", e["color"]),
                                  markeredgecolor=e.get("mec", e["color"]),
                                  markersize=e.get("ms", 3.6), markeredgewidth=0.9,
                                  lw=e.get("lw", 1.0)))
    ncol = ncol or 1
    leg = ax.legend(handles, [e["label"] for e in entries], loc="upper center",
                    bbox_to_anchor=(0.5, y), ncol=ncol, frameon=False,
                    handlelength=1.4, columnspacing=0.7, handletextpad=0.35,
                    fontsize=fontsize, borderaxespad=0.0)
    if note:
        nrow = int(np.ceil(len(entries) / ncol))
        ax.text(0.5, y - 0.125 * nrow - 0.06, note, transform=ax.transAxes,
                ha="center", va="top", fontsize=5.8, color="#444444")
    return leg
