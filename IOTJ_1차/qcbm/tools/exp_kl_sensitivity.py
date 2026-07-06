#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
잔여 병목 분리 (i) — KL 과민성 진단. n10 identity-block이 KL 0.14에서 멈춘 게 forward KL 의 희소상태 1/p 험함 때문인가.

loss=KL(−Σ q·log(p+ε)) 유지(안 바꿈). 두 축으로 진단:
  축1 (시작점 확산): identity-block pert 스윕 → 초기 분포 확산(max p, 엔트로피) + 학습 후 best KL. pert↑ → KL↓ 인가?
  축2 (KL 과민성 직접): 학습 후 최종 p 에서 상태별 KL 기여 q_s·log((q_s+ε)/(p_s+ε)) 정렬 → 상위 k 지배율,
       ε 민감도(1e-10~1e-4), 상위 기여 상태 성격(q vs p).
※ pert↑ 는 확산+gradient 동시 변경(격리 아님, 명시). 이 ansatz·완화법·KL 한정. 단정 금지. 합성만. 기존 코드 무수정(import).

사용: python tools/exp_kl_sensitivity.py --smoke
"""
import os
import sys
import json
import time
import argparse

import numpy as np

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

from pennylane import numpy as pnp        # noqa: E402
import exp_strong_bp_mitigation as SBP    # noqa: E402  train_idblock, make_idb, synth_low, EPS
import diagnose_order_decomposition as OD  # noqa: E402  kl_div

OUTDIR = os.path.join("results2", "kl_sensitivity")
EPS = SBP.EPS                              # 1e-10 (학습 cost 와 동일)
NQ = 10
L = 6


def log(m):
    print(m, flush=True)


def entropy(p):
    p = np.asarray(p, float); m = p > 0
    return float(-np.sum(p[m] * np.log(p[m])))


def idblock_init_p(nq, layers, seed, pert):
    """train_idblock 과 동일 init 규약으로 초기 분포 p_init 계산(학습 전)."""
    half = layers // 2
    shp = (half, nq, 3)
    rng = np.random.default_rng(2000 + seed)
    wA0 = rng.uniform(0, 2 * np.pi, shp)
    wB0 = wA0 + rng.normal(0, pert, shp)
    circ = SBP.make_idb(nq)
    p = np.array(circ(pnp.array(wA0), pnp.array(wB0)), float)
    p = np.clip(p, 0, None); p /= p.sum()
    return p


def kl_contributions(p_true, p_model, eps):
    """상태별 forward-KL 기여 c_s = q_s·log((q_s+eps)/(p_s+eps)), q_s>0 만. 반환 정렬된 |c| 누적비율 등."""
    q = np.asarray(p_true, float); p = np.asarray(p_model, float)
    mask = q > 0
    c = q[mask] * np.log((q[mask] + eps) / (p[mask] + eps))
    idx = np.argsort(-np.abs(c))
    c_sorted = c[idx]
    total = float(np.sum(c))                 # ≈ KL(q||p)
    cum = np.cumsum(c_sorted) / total if total != 0 else np.zeros_like(c_sorted)
    # 상위 기여 상태 성격
    states = np.where(mask)[0][idx]
    top = [dict(state=int(states[i]), q=float(q[states[i]]), p=float(p[states[i]]), contrib=float(c_sorted[i]))
           for i in range(min(8, len(c_sorted)))]
    return dict(total_KL=total, n_support=int(mask.sum()),
                topk_cum=[float(x) for x in cum[:20]], top_states=top)


def eps_sensitivity(p_true, p_model):
    q = np.asarray(p_true, float); p = np.asarray(p_model, float); m = q > 0
    out = {}
    for e in [1e-10, 1e-8, 1e-6, 1e-4]:
        out["%.0e" % e] = float(np.sum(q[m] * np.log((q[m]) / (p[m] + e))))
    return out


def run(perts, seeds, epochs, tag="run"):
    os.makedirs(OUTDIR, exist_ok=True)
    p_true = SBP.synth_low(NQ)
    Hq = entropy(p_true)
    log("n=%d L=%d | H(p_true)=%.4f, support=%d/%d | EPS=%.0e" % (NQ, L, Hq, int((p_true > 0).sum()), 1 << NQ, EPS))

    # ── 축1: pert 스윕 ──
    axis1 = {}
    best_overall = None      # (KL, p_final, pert)
    for pert in perts:
        p_init = idblock_init_p(NQ, L, seed=0, pert=pert)
        init_maxp = float(np.max(p_init)); init_H = entropy(p_init)
        kls = []; pf_best = None; t0 = time.time()
        for s in range(seeds):
            p, _ = SBP.train_idblock(p_true, NQ, L, epochs, seed=s, pert=pert)
            k = OD.kl_div(p_true, p, 1e-12)
            kls.append(k)
            if pf_best is None or k < pf_best[0]:
                pf_best = (k, p)
        a = np.array(kls)
        axis1["%.1f" % pert] = dict(init_maxp=init_maxp, init_entropy=init_H,
                                    best_KL=float(a.min()), mean_KL=float(a.mean()), spread=float(a.max() - a.min()))
        if best_overall is None or pf_best[0] < best_overall[0]:
            best_overall = (pf_best[0], pf_best[1], pert)
        log("  pert=%.1f | 초기 max p=%.4f H=%.3f | 학습후 best KL=%.4f (spread %.4f) (%.0fs)"
            % (pert, init_maxp, init_H, a.min(), a.max() - a.min(), time.time() - t0))

    # ── 축2: 최종 p 의 KL 과민성 ──
    best_KL, best_p, best_pert = best_overall
    contrib = kl_contributions(p_true, best_p, EPS)
    epssens = eps_sensitivity(p_true, best_p)
    log("\n[축2] best p (pert=%.1f, KL=%.4f): support=%d, 상위5 누적=%.1f%%, 상위10=%.1f%%"
        % (best_pert, best_KL, contrib["n_support"], 100 * contrib["topk_cum"][4], 100 * contrib["topk_cum"][9]))
    log("      ε 민감도 KL: " + " ".join("%s=%.4f" % (k, v) for k, v in epssens.items()))
    log("      상위 기여 상태(q,p): " + " ".join("s%d(q%.3f,p%.1e)" % (t["state"], t["q"], t["p"]) for t in contrib["top_states"][:5]))

    sc = dict(repro_pert01=axis1.get("0.1", {}).get("best_KL"),
              init_maxp_monotone=bool(all(axis1["%.1f" % perts[i]]["init_maxp"] >= axis1["%.1f" % perts[i + 1]]["init_maxp"] - 1e-6
                                          for i in range(len(perts) - 1))))
    payload = dict(
        note=("(i) KL 과민성 진단. loss=KL 유지. 축1: identity-block pert 스윕(초기확산+best KL). 축2: 최종 p 상태별 KL기여·ε민감도. "
              "pert↑는 확산+gradient 동시변경(격리 아님). 이 ansatz·완화법·KL 한정. 단정 금지."),
        config=dict(nq=NQ, L=L, perts=perts, seeds=seeds, epochs=epochs, EPS=EPS),
        axis1=axis1, axis2=dict(best_pert=best_pert, best_KL=best_KL, **contrib, eps_sensitivity=epssens),
        self_check=sc)
    outjson = os.path.join(OUTDIR, "kl_sensitivity%s.json" % ("" if tag == "run" else "_" + tag))
    with open(outjson, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s | self-check: pert0.1 재현=%s, 초기maxp 단조=%s"
        % (outjson, sc["repro_pert01"], sc["init_maxp_monotone"]))
    try:
        plot(axis1, contrib, epssens, perts, tag)
    except Exception as e:
        log("[plot WARN] %r" % e)
    return payload


def plot(axis1, contrib, epssens, perts, tag):
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.5))
    ps = [float(k) for k in axis1]
    ax[0].plot(ps, [axis1["%.1f" % p]["best_KL"] for p in ps], "o-", color="#7B2FBE", label="best KL")
    ax0b = ax[0].twinx()
    ax0b.plot(ps, [axis1["%.1f" % p]["init_maxp"] for p in ps], "s--", color="#9AA0A6", label="init max p")
    ax[0].set_xlabel("pert (identity-block)"); ax[0].set_ylabel("best KL", color="#7B2FBE"); ax0b.set_ylabel("init max p", color="#9AA0A6")
    ax[0].set_title("축1: 시작확산(pert) vs KL"); ax[0].grid(alpha=.3)
    ax[1].plot(range(1, len(contrib["topk_cum"]) + 1), [100 * c for c in contrib["topk_cum"]], "o-", color="#C1442E")
    ax[1].axhline(80, color="gray", ls=":"); ax[1].set_xlabel("top-k 상태"); ax[1].set_ylabel("누적 KL 기여 %")
    ax[1].set_title("축2: 상위 상태 KL 지배율"); ax[1].grid(alpha=.3)
    es = list(epssens.items())
    ax[2].plot(range(len(es)), [v for _, v in es], "o-", color="#3B6EA5")
    ax[2].set_xticks(range(len(es))); ax[2].set_xticklabels([k for k, _ in es])
    ax[2].set_xlabel("ε"); ax[2].set_ylabel("KL(q||p)"); ax[2].set_title("축2: ε 민감도"); ax[2].grid(alpha=.3)
    plt.suptitle("Residual bottleneck (i): KL over-sensitivity diagnosis (loss=KL kept, identity-block, n=10)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTDIR, "kl_sensitivity%s.png" % ("" if tag == "run" else "_" + tag)), dpi=120); plt.close()
    log("[저장] kl_sensitivity png")


def main():
    ap = argparse.ArgumentParser(description="KL 과민성 진단(loss=KL 유지). 합성. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--perts", type=str, default="0.1,0.3,0.5,0.8")
    ap.add_argument("--seeds", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=300)
    args = ap.parse_args()
    if args.smoke:
        run([0.1, 0.5], seeds=1, epochs=100, tag="smoke"); return
    perts = [float(x) for x in args.perts.split(",") if x.strip()]
    run(perts, args.seeds, args.epochs, tag="run")


if __name__ == "__main__":
    main()
