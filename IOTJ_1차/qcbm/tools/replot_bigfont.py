#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
폰트만 키워 figures_paper/ 그림을 다시 생성하는 래퍼 (비파괴).

- 기존 소스/데이터/결과(results2/*) 무수정. 원본 PNG/JSON 덮어쓰기 금지:
  * Figure.savefig 를 가로채 타깃 그림만 docs/figures_paper/<F-name>.png 로 저장,
    그 외 저장은 스크래치로 흘려보냄(원본 results2 PNG 보존).
  * builtins.open 의 쓰기 모드 중 results2/ 아래 경로는 스크래치로 redirect(원본 JSON/npz 보존).
- 폰트: rcParams 확대 + 코드에 박힌 fontsize=/size= 인자를 FACTOR 배로 monkeypatch.
- F1~F5: 기존 diagnose/analyze 스크립트를 그대로 재실행(재학습 없음, 저장된 npz/csv만 사용).
- F6: 16 restart ckpt(loss_hist/gnorm_hist)로 학습곡선 직접 작도.

실행: python3 tools/replot_bigfont.py
"""
import os
import sys
import builtins
import runpy

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes

FACTOR = 1.25  # "조금 더 크게"

HERE = os.path.dirname(os.path.abspath(__file__))      # .../qcbm/tools
ROOT = os.path.dirname(HERE)                            # .../qcbm
RES2 = os.path.join(ROOT, "results2")
DEST = os.path.join(ROOT, "docs", "figures_paper")
SCRATCH = os.path.join(ROOT, "results2", "_replot_scratch")
os.makedirs(DEST, exist_ok=True)
os.makedirs(SCRATCH, exist_ok=True)

# 원본 results2 PNG basename -> figures_paper 의 F-이름
FIGMAP = {
    "cross_dataset_diagnosis.png": "F1_joint_gain_cross_dataset.png",
    "attack_complexity.png":       "F2_complexity_vs_quantum_boundary.png",
    "param_efficiency_bot.png":    "F3a_param_efficiency_bot.png",
    "param_eff_unsw.png":          "F3b_param_efficiency_unsw.png",
    "tv_auc_dissociation.png":     "F4_tv_auc_dissociation.png",
    "unsup_selection.png":         "F4b_unsup_selection.png",
    "bot_efficiency_robustness.png": "F5a_restart_dist_bot.png",
    # param_eff_unsw.png 는 F3b 와 F5b 공유 — savefig 시 F3b 로 저장 후 아래에서 F5b 로 복사
}

# ----------------------------------------------------------------------------
# 1) 폰트 확대: rcParams (인자 없이 그려지는 요소들)
# ----------------------------------------------------------------------------
_BASE = {
    "font.size": 10, "axes.titlesize": 12, "axes.labelsize": 10,
    "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 10,
    "figure.titlesize": 14,
}
for k, v in _BASE.items():
    matplotlib.rcParams[k] = v * FACTOR


# ----------------------------------------------------------------------------
# 2) 코드에 박힌 fontsize=/size=/labelsize= 인자도 FACTOR 배로 스케일
# ----------------------------------------------------------------------------
def _scale_kw(kwargs, keys):
    for key in keys:
        v = kwargs.get(key)
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            kwargs[key] = v * FACTOR
    return kwargs


def _wrap(orig, keys):
    def inner(*args, **kwargs):
        return orig(*args, **_scale_kw(kwargs, keys))
    return inner


Axes.set_title       = _wrap(Axes.set_title,       ["fontsize", "size"])
Axes.set_xlabel      = _wrap(Axes.set_xlabel,      ["fontsize", "size"])
Axes.set_ylabel      = _wrap(Axes.set_ylabel,      ["fontsize", "size"])
Axes.text            = _wrap(Axes.text,            ["fontsize", "size"])
Axes.annotate        = _wrap(Axes.annotate,        ["fontsize", "size"])
Axes.legend          = _wrap(Axes.legend,          ["fontsize"])
Axes.set_xticklabels = _wrap(Axes.set_xticklabels, ["fontsize", "size"])
Axes.set_yticklabels = _wrap(Axes.set_yticklabels, ["fontsize", "size"])
Axes.tick_params     = _wrap(Axes.tick_params,     ["labelsize"])
Axes.bar_label       = _wrap(Axes.bar_label,       ["fontsize", "size"])
Figure.suptitle      = _wrap(Figure.suptitle,      ["fontsize", "size"])
Figure.legend        = _wrap(Figure.legend,        ["fontsize"])
Figure.text          = _wrap(Figure.text,          ["fontsize", "size"])
plt.suptitle         = _wrap(plt.suptitle,         ["fontsize", "size"])
plt.figtext          = _wrap(plt.figtext,          ["fontsize", "size"])


# ----------------------------------------------------------------------------
# 3) savefig redirect — 원본 results2 PNG 보존
# ----------------------------------------------------------------------------
_orig_savefig = Figure.savefig


def _savefig(self, fname, *args, **kwargs):
    try:
        base = os.path.basename(str(fname))
    except Exception:
        base = ""
    if base in FIGMAP:
        out = os.path.join(DEST, FIGMAP[base])
        print("  [savefig] %s -> figures_paper/%s" % (base, FIGMAP[base]))
        return _orig_savefig(self, out, *args, **kwargs)
    # 타깃 외 그림은 원본 보호를 위해 스크래치로
    return _orig_savefig(self, os.path.join(SCRATCH, base or "untitled.png"), *args, **kwargs)


Figure.savefig = _savefig


# ----------------------------------------------------------------------------
# 4) 쓰기용 open redirect — results2/ 아래 원본 JSON/npz 보존
# ----------------------------------------------------------------------------
_orig_open = builtins.open
_WMODES = ("w", "a", "x", "+")


def _open(file, mode="r", *args, **kwargs):
    try:
        is_write = isinstance(mode, str) and any(m in mode for m in _WMODES)
        p = os.path.abspath(str(file)) if isinstance(file, (str, bytes, os.PathLike)) else ""
    except Exception:
        is_write, p = False, ""
    if is_write and p.startswith(os.path.abspath(RES2) + os.sep) \
            and not p.startswith(os.path.abspath(DEST) + os.sep) \
            and not p.startswith(os.path.abspath(SCRATCH) + os.sep):
        redirected = os.path.join(SCRATCH, os.path.basename(p))
        return _orig_open(redirected, mode, *args, **kwargs)
    return _orig_open(file, mode, *args, **kwargs)


builtins.open = _open


# ----------------------------------------------------------------------------
# 5) F1~F5: 기존 스크립트 재실행 (재학습 없음)
# ----------------------------------------------------------------------------
SCRIPTS = [
    "diagnose_cross_dataset.py",          # F1
    "diagnose_attack_complexity.py",      # F2
    "diagnose_param_efficiency_bot.py",   # F3a
    "diagnose_param_efficiency_unsw.py",  # F3b (+ F5b 공유)
    "analyze_tv_auc_dissociation.py",     # F4
    "explore_unsup_selection.py",         # F4b
    "verify_bot_efficiency_robustness.py",  # F5a
]

if HERE not in sys.path:
    sys.path.insert(0, HERE)

for s in SCRIPTS:
    path = os.path.join(HERE, s)
    print("\n=== replot: %s ===" % s)
    try:
        runpy.run_path(path, run_name="__main__")
    except SystemExit:
        pass
    except Exception as e:
        print("  [WARN] %s 실패: %r" % (s, e))
    finally:
        plt.close("all")

# F3b 원본(param_eff_unsw.png)은 효율(좌)+restart분포(우) 2패널 — F5b 로도 복사
import shutil
src_unsw = os.path.join(DEST, "F3b_param_efficiency_unsw.png")
if os.path.exists(src_unsw):
    shutil.copy2(src_unsw, os.path.join(DEST, "F5b_restart_dist_unsw.png"))
    print("\n[copy] F3b -> F5b (UNSW 공유 패널)")


# ----------------------------------------------------------------------------
# 6) F6: 학습곡선 (16 restart ckpt 의 loss_hist/gnorm_hist)
# ----------------------------------------------------------------------------
import glob
import numpy as np


def plot_train(ckpt_dir, out_name, title):
    files = sorted(glob.glob(os.path.join(ckpt_dir, "restart_*.npz")))
    if not files:
        print("  [F6] no ckpt in %s" % ckpt_dir)
        return
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    for f in files:
        z = np.load(f, allow_pickle=True)
        lh = np.asarray(z["loss_hist"], dtype=float)
        gh = np.asarray(z["gnorm_hist"], dtype=float)
        ep = np.arange(1, len(lh) + 1)
        ax[0].plot(ep, lh, alpha=0.6, lw=1.3)
        ax[1].plot(ep, gh, alpha=0.6, lw=1.3)
    ax[0].set_yscale("log"); ax[1].set_yscale("log")
    ax[0].set_xlabel("epoch"); ax[0].set_ylabel("MMD$^2$ (loss)")
    ax[0].set_title("MMD$^2$ loss (16 restarts)"); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("epoch"); ax[1].set_ylabel("gradient norm")
    ax[1].set_title("gradient norm — barren plateau 없음 (16 restarts)"); ax[1].grid(alpha=.3)
    plt.suptitle(title)
    plt.tight_layout()
    _orig_savefig(fig, os.path.join(DEST, out_name), dpi=125)
    plt.close(fig)
    print("  [F6] %s (%d restarts)" % (out_name, len(files)))


print("\n=== replot: F6 학습곡선 ===")
plot_train(os.path.join(RES2, "bot_ckpt"),  "F6a_train_curve_bot.png",
           "CIC Bot QCBM (216p) 학습곡선 — 정상 수렴")
plot_train(os.path.join(RES2, "unsw_ckpt"), "F6b_train_curve_unsw.png",
           "UNSW QCBM (180p) 학습곡선 — 정상 수렴")

print("\n완료. figures_paper/ 에 폰트 확대본 저장. results2 원본 무수정.")
