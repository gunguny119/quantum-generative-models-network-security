"""Gate S8 Phase B-3: 공격 장부 효과. CANON_23·x*=A0·레벨 1+AB/2·SCS+CLARABEL 교차."""
import json
import os
import sys

import numpy as np

R2 = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(R2, "tools"))
sys.path.insert(0, os.path.dirname(__file__))
import npa_guess as N            # noqa: E402
import gate_s5_multifacet as M   # noqa: E402

CH = [(0, 0, +1), (0, 1, +1), (1, 0, +1), (1, 1, -1)]
OUT = os.path.join(R2, "results2", "gate_s8", "s8_phaseB3.json")


def to_p(e):
    p = np.zeros((3, 3, 2, 2))
    for ci, (x, y) in enumerate(M.CTX9):
        p[x, y] = e[ci]
    return p


def load_behaviors():
    d5 = json.load(open(os.path.join(R2, "results2/gate_s5/s5.json")))
    d7 = json.load(open(os.path.join(R2, "results2/gate_s7/s7_phase1.json")))
    out = [dict(name="e_n", src="werner_23(CANON_23, 0.856)", tau=None,
                e=M.werner_23(M.CANON_23, 0.856), kl=0.0)]
    ex = d5["verification"]["exact_slsqp"]
    pe = np.array(ex["p"])
    out.append(dict(name="e_prime", src="s5.json:verification.exact_slsqp.p", tau=None,
                    e=M.werner_23(pe[:13], pe[13]), kl=ex["kl_to_normal"]))
    for row in d7["phase1C"]["rows"]:
        if abs(row["tau"] - 1e-3) < 1e-12 or abs(row["tau"] - 1e-2) < 1e-12:
            pa = np.array(row["p"])
            out.append(dict(name="tau_%g" % row["tau"], tau=row["tau"],
                            src="s7_phase1.json:phase1C.rows[tau=%g].p" % row["tau"],
                            e=M.werner_23(pa[:13], pa[13]), kl=row["kl"]))
    return out, float(d7["phase1C"]["tau_break"])


def main():
    behs, tau_break = load_behaviors()
    rows = []
    for b in behs:
        e = b["e"]
        p, S = to_p(e), M.chsh_S_sub(e)
        rec = dict(name=b["name"], src=b["src"], tau=b["tau"], kl_to_normal=b["kl"],
                   S_sub=float(S), CF=float(M.contextual_fraction_23(e)),
                   raw36=p.tolist(), levels={})
        for lv in ("1+AB", "2"):
            g = N.GuessSDP(3, 3, lv, 0)
            cell = {}
            for sv in ("SCS", "CLARABEL"):
                kw = dict(solver=sv)
                if sv == "SCS":
                    kw.update(eps=1e-8, max_iters=200000)
                try:
                    rS = g.solve(S_obs=S, mode="S", chsh_idx=CH, **kw)
                    rF = g.solve(p_obs=p, mode="full", **kw)
                    cell[sv] = dict(h_S=rS["h_min"], h_full=rF["h_min"],
                                    dH=(rF["h_min"] - rS["h_min"]),
                                    status_S=rS["status"], status_full=rF["status"],
                                    p_guess_S=rS["p_guess"], p_guess_full=rF["p_guess"],
                                    lam_full=rF["lam"])
                except Exception as ex:
                    cell[sv] = dict(error=repr(ex))
            if "error" not in cell["SCS"] and "error" not in cell["CLARABEL"]:
                cell["solver_agree_hS"] = abs(cell["SCS"]["h_S"] - cell["CLARABEL"]["h_S"])
                cell["solver_agree_hfull"] = abs(cell["SCS"]["h_full"] - cell["CLARABEL"]["h_full"])
            rec["levels"][lv] = cell
        rows.append(rec)
        print("[%-9s] S=%.12f CF=%.8f" % (b["name"], S, rec["CF"]), flush=True)
        for lv in ("1+AB", "2"):
            c = rec["levels"][lv]["SCS"]
            print("    lv %-5s H^S=%.8f H^full=%.8f dH=%+.8f  (솔버차 %.1e/%.1e)"
                  % (lv, c["h_S"], c["h_full"], c["dH"],
                     rec["levels"][lv].get("solver_agree_hS", float("nan")),
                     rec["levels"][lv].get("solver_agree_hfull", float("nan"))), flush=True)

    base = [r for r in rows if r["name"] == "e_n"][0]
    ledger = []
    for r in rows:
        ent = dict(name=r["name"], tau=r["tau"], kl_to_normal=r["kl_to_normal"],
                   dS_from_e_n=r["S_sub"] - base["S_sub"])
        for lv in ("1+AB", "2"):
            c, cb = r["levels"][lv]["SCS"], base["levels"][lv]["SCS"]
            ent["Delta_S_" + lv] = c["h_S"] - cb["h_S"]
            ent["Delta_attack_" + lv] = c["h_full"] - cb["h_full"]
            ent["level_monotone_ok"] = bool(
                r["levels"]["2"]["SCS"]["h_full"] >= r["levels"]["1+AB"]["SCS"]["h_full"] - 1e-9)
        ledger.append(ent)

    json.dump(dict(
        note=("Gate S8 Phase B-3 공격 장부 효과. CANON_23·x*=A0 한정(x*=A2 는 H^S≡0 이라 "
              "S-대조 불성립). 레벨 2 가 인용값, 레벨 1+AB 는 수렴 증빙용. "
              "Δ_attack 은 부호 미지로 사전등록됨."),
        preregistration=dict(
            PB3_1="e′ 는 S_sub 동일 → Δ_S = 0 강제. 검증이지 발견 아님.",
            PB3_2="Δ_attack 부호 미지. 판정막대 ΔH(V=0.856)=0.019395: >0.0194 침식우위 / ≈0.0194 동급 / <0.005 중립.",
            PB3_3="τ=1e-3, 1e-2 는 tau_break=%g 초과 → 사각지대 밖. 비교 기준선으로만 서술." % tau_break,
            judgement_bar=0.019395, tau_break=tau_break),
        behaviors=rows, ledger=ledger,
        limits=("점추정·i.i.d.·점근 인증 비교. 유한키 조합가능 보안 아님. 마틴게일 단측한계를 쓰는 "
                "성숙 CHSH 프로토콜은 이 병리를 피한다 — 표적은 CF-LP/plugin 관행이다. "
                "정렬(CANON_23, A2≡A0·B2≡B0) 기하 한정 · x*=A0 한정 · V≤0.96 한정. "
                "CANON_23 은 (2,2,2)로 퇴화(s4.json limits). NPA 레벨 2 상계이며 레벨 3 미확인."),
    ), open(OUT, "w"), ensure_ascii=False, indent=1)
    print("[저장] %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
