"""Gate S8 Phase B-1: 격차 곡선 (V 6점 × 박스 2종 × 추측목표 3종 × 레벨 2종)."""
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
sys.path.insert(0, os.path.dirname(__file__))
import npa_guess as N            # noqa: E402
import gate_s5_multifacet as M   # noqa: E402

CH = [(0, 0, +1), (0, 1, +1), (1, 0, +1), (1, 1, -1)]
V_GRID = [0.75, 0.80, 0.856, 0.92, 0.96, 1.00]
GENERIC_23 = np.array([0.7853981633974483, 0.0, 0.0, 1.5707963267948966, 0.0, 0.7, 0.9,
                       0.7853981633974483, 0.0, -0.7853981633974483, 0.0, 1.1, 2.0])
BOXES = {"CANON_23": M.CANON_23, "generic_23": GENERIC_23}
OUT = os.path.join(os.path.dirname(__file__), "..", "results2", "gate_s8", "s8_phaseB1.json")


def to_p(e):
    p = np.zeros((3, 3, 2, 2))
    for ci, (x, y) in enumerate(M.CTX9):
        p[x, y] = e[ci]
    return p


def job(arg):
    box, V, xs, lv = arg
    e = M.werner_23(BOXES[box], V)
    p, S = to_p(e), M.chsh_S_sub(e)
    g = N.GuessSDP(3, 3, lv, xs)
    t = time.time()
    rS = g.solve(S_obs=S, mode="S", chsh_idx=CH)
    rF = g.solve(p_obs=p, mode="full")
    dh = None
    if rS["h_min"] is not None and rF["h_min"] is not None:
        dh = rF["h_min"] - rS["h_min"]
    return dict(box=box, V=V, x_star=xs, level=lv, S=float(S),
                CF=float(M.contextual_fraction_23(e)),
                pironio_222=float(N.pironio_hmin(S)),
                h_S=rS["h_min"], h_full=rF["h_min"], dH=dh,
                status_S=rS["status"], status_full=rF["status"],
                p_guess_S=rS["p_guess"], p_guess_full=rF["p_guess"],
                lam_S=rS["lam"], lam_full=rF["lam"],
                n_mon=rS["n_mon"], elapsed_s=time.time() - t)


def main():
    jobs = [(b, V, xs, lv) for b in BOXES for V in V_GRID for xs in (0, 2) for lv in ("1+AB", "2")]
    print("총 %d 작업" % len(jobs), flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    res = []
    with ProcessPoolExecutor(max_workers=14) as ex:
        for i, r in enumerate(ex.map(job, jobs), 1):
            res.append(r)
            print("[%2d/%d] %-11s V=%.3f x*=A%d lv=%-5s  H^S=%s H^full=%s dH=%s (%.0fs)"
                  % (i, len(jobs), r["box"], r["V"], r["x_star"], r["level"],
                     "%.6f" % r["h_S"] if r["h_S"] is not None else "None",
                     "%.6f" % r["h_full"] if r["h_full"] is not None else "None",
                     "%+.6f" % r["dH"] if r["dH"] is not None else "None",
                     r["elapsed_s"]), flush=True)
    json.dump(dict(
        note=("Gate S8 Phase B-1 격차 곡선. NPA 브랜치 SDP(추측확률). ★x*=A2 는 CHSH 4컨텍스트 "
              "밖이라 S-단독 제약이 A2 를 전혀 묶지 못해 H^S≡0 이 된다 — 이때의 ΔH 는 "
              "Bancal 류 '전체통계 이득'이 아니라 '미측정 설정' 인공물이다. 논문 인용은 x*=A0."),
        params=dict(V_grid=V_GRID, boxes=list(BOXES), x_star=[0, 2],
                    levels=["1+AB", "2"], solver="SCS", eps=1e-8, max_iters=200000,
                    generic_23=GENERIC_23.tolist(), canon_23=M.CANON_23.tolist()),
        results=res,
        limits=("점추정·i.i.d.·점근 인증 비교. 유한키 조합가능 보안 아님. 성숙한 CHSH "
                "프로토콜의 마틴게일 단측한계는 이 병리를 피한다 — 표적은 CF-LP/plugin 관행이다. "
                "CANON_23 은 A2≡A0·B2≡B0 로 (2,2,2)에 퇴화(s4.json limits 참조). "
                "SCS 수치해이며 Tsirelson 경계(V=1.0)에서 optimal_inaccurate 발생 가능."),
    ), open(OUT, "w"), ensure_ascii=False, indent=1)
    print("[저장] %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
