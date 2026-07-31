"""Fig 3 — learning gap vs contextual fraction.  source: results2/gap_cf/gap_cf.json"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from _style3 import plt, COL2, ROOT, TSIRELSON, save, panel_legend

NAME = "fig3_gap_vs_cf"
SRC = os.path.join(ROOT, "results2/gap_cf/gap_cf.json")
# ── 제외 규칙 (하드코딩) ─────────────────────────────────────────────
SWEEP_KEY = "inf"          # 모집단 스윕만 사용 (N=5000 유한표본 행 제외)
# ────────────────────────────────────────────────────────────────────

d = json.load(open(SRC))
rows = d["sweeps"][SWEEP_KEY]
n_excl = len(d["sweeps"]["5000"])
print("[filter] sweeps='%s' %d rows used; excluded %d rows (sweeps['5000'])"
      % (SWEEP_KEY, len(rows), n_excl))
cf = np.array([r["cf"] for r in rows])
lam = np.array([r["lam"] for r in rows])
ncf = np.array([r["kl_nc_floor"] for r in rows])
klq = np.array([r["kl_quantum"] for r in rows])
gctx = np.array([r["gap_contextual"] for r in rows])
gmat = np.array([r["gap_matched"] for r in rows])
sd_ctx = np.array([np.std(r["gap_ctx_seedwise"]) for r in rows])
sd_mat = np.array([np.std(r["gap_seedwise"]) for r in rows])
n_cf0 = int((cf == 0).sum())
print("[note] %d rows have CF=0 (lam<=%.2f); they overplot at x=0 on the CF axis."
      % (n_cf0, lam[cf == 0].max()))

fig, (axL, axR) = plt.subplots(1, 2, figsize=(COL2, 3.05))

# ── 좌: KL vs CF ────────────────────────────────────────────────────
o = np.argsort(cf)
axL.plot(cf[o], ncf[o], "o-", color="#1b3a5c", label=r"NC floor  $\mathrm{KL}(e\,\|\,\mathrm{NC})$")
axL.plot(cf[o], klq[o], "s--", color="#c2453f", label=r"quantum  $\mathrm{KL}(e\,\|\,Q)$")
axL.plot(cf[o], gctx[o], "^:", color="#2a8c7a", label="gap (contextual)")
axL.fill_between(cf[o], (gctx - sd_ctx)[o], (gctx + sd_ctx)[o], color="#2a8c7a", alpha=0.20, lw=0)
axL.plot(cf[o], gmat[o], "D-.", color="#8a8a8a", lw=0.7, ms=2.6,
         label="parameter-matched\n(expressivity confound)")
axL.fill_between(cf[o], (gmat - sd_mat)[o], (gmat + sd_mat)[o], color="#8a8a8a", alpha=0.18, lw=0)
axL.set_ylim(-0.014, 0.335)
axL.set_xlim(-0.035, 1.035)
axL.axvline(TSIRELSON, color="k", ls=(0, (1, 2)), lw=0.6)
axL.set_xlabel("contextual fraction  CF")
axL.set_ylabel("KL divergence  (nats)")
axL.grid(alpha=0.3)

# ── 우: CF self-check ───────────────────────────────────────────────
sc = [r for r in d["self_check_cf"] if str(r["N"]) == "inf"]
sl = np.array([r["lam"] for r in sc]); sp = np.array([r["cf_lp"] for r in sc])
os_ = np.argsort(sl)
xx = np.linspace(0, 1, 400)
axR.plot(xx, np.maximum(0.0, 2 * xx - 1), "-", color="#1b3a5c",
         label=r"closed form  $\max(0,\,2\lambda-1)$")
axR.plot(sl[os_], sp[os_], "o", mfc="none", mec="#c2453f", mew=0.8, label="LP solution")
axR.axvline(0.5, color="k", ls=(0, (1, 2)), lw=0.6)
axR.set_xlabel(r"noise knob  $\lambda$")
axR.set_ylabel("contextual fraction  CF")
axR.set_xlim(-0.02, 1.02); axR.set_ylim(-0.03, 1.03)
axR.grid(alpha=0.3)

fig.tight_layout()

panel_legend(axL, [
    dict(label=r"NC floor  $\mathrm{KL}(e\,\|\,\mathrm{NC})$", color="#1b3a5c", marker="o", ls="-"),
    dict(label=r"quantum  $\mathrm{KL}(e\,\|\,Q)$", color="#c2453f", marker="s", ls="--"),
    dict(label="gap (contextual)", color="#2a8c7a", marker="^", ls=":"),
    dict(label="parameter-matched (confound)", color="#8a8a8a", marker="D", ls="-.", ms=2.8, lw=0.7)],
    ncol=2, y=-0.22)

panel_legend(axR, [
    dict(label=r"closed form  $\max(0,\,2\lambda-1)$", color="#1b3a5c", marker="None", ls="-"),
    dict(label="LP solution", color="#c2453f", marker="o", ls="none", mfc="white")],
    ncol=2, y=-0.22)

save(fig, NAME)
