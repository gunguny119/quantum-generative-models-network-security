"""figures_final2 용 스타일.

tools_fig_final/_style3 을 **그대로 재사용**하고 출력 폴더만 figures_final2 로 바꾼다.
폰트·선폭·레전드 규칙을 한 곳(_style3)에만 두기 위해 복사하지 않는다 —
_style3 을 고치면 이 쪽 그림에도 그대로 반영된다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "tools_fig_final"))

import _style3 as S                                   # noqa: E402

# save() 는 호출 시점에 모듈 전역 FIGDIR 을 읽으므로 여기서 바꿔치기하면 된다.
S.FIGDIR = os.path.join(S.ROOT, "figures_final2")

from _style3 import (plt, COL1, COL2, ROOT, TSIRELSON,   # noqa: E402,F401
                     save, panel_legend, legend_figure)

FIGDIR = S.FIGDIR
