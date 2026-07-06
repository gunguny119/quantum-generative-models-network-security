#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Drebin binary static feature 고차 상관 차수 측정 — flow(2차 73~99% 지배) 대비 정말 고차 지배인가.

순수 측정(학습 없음, QCBM 없음). 기존 flow 차수분해 인프라 재사용(import만, 무수정):
  diagnose_order_decomposition.decompose(p, nq, kmax=4) — 2^nq 상태 경험분포 p 의 1/2/3/4/≥5차 KL 기여.
Drebin 을 같은 잣대·같은 규모(k∈{10,12} = flow UNSW 10 / Bot·HOIC 12)로 측정해 직접 대비한다.

부분집합 선정 3방식(강건성): (a)고빈도 p(1)최대 (b)고분산 p(1-p)최대 (c)카테고리 다양(라운드로빈).
지표: KL1=KL(p‖1차 maxent) 중 c2(2차)/ge3(≥3차) 비중. Drebin ge3 비중이 flow(19~27%, HOIC 1%)보다 큰가.

★ 측정값만. "malware 고차다 / 양자 유리다" 단정 금지. 우위·QCBM 성능 언급 없음.

사용: python tools/diagnose_drebin_order.py [--smoke] [--splits 0,1,2] [--ks 10,12]
                                          [--methods highfreq,highvar,category] [--no-plot]
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

import diagnose_order_decomposition as OD  # noqa: E402  decompose, self_check, kl_div, bit_matrix

DDIR = os.path.join("data", "malware", "CADE", "data")
OUTDIR = os.path.join("results2", "drebin_order")
OUT_JSON = os.path.join(OUTDIR, "drebin_order.json")
OUT_PNG = os.path.join(OUTDIR, "drebin_order.png")
FLOW_JSON = os.path.join("results2", "order_decomposition.json")
EPS = 1e-12
METHODS = ["highfreq", "highvar", "category"]


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 로드
# ---------------------------------------------------------------------------
def load_split(split):
    """split 상위 npz 로드. X = vstack(X_train, X_test) (측정 전용 pooling; 학습 아님→leakage 무관).
    feature명(열 1:1)도 로드."""
    npz = os.path.join(DDIR, "drebin_new_%d.npz" % split)
    with np.load(npz) as z:
        X = np.vstack([z["X_train"], z["X_test"]]).astype(np.int8)
    ftxt = os.path.join(DDIR, "drebin_new_%d" % split, "drebin_new%d_train_selected_features.txt" % split)
    names = [l.strip() for l in open(ftxt, errors="replace")]
    assert len(names) == X.shape[1], "feature명 %d != 열 %d (split %d)" % (len(names), X.shape[1], split)
    cats = [n.split("::")[0] if "::" in n else "?" for n in names]
    return X, names, cats


# ---------------------------------------------------------------------------
# 부분집합 선정 3방식
# ---------------------------------------------------------------------------
def select_subset(X, cats, method, k):
    """method 별 k 개 feature 열 인덱스. 전부 비상수 가정(Drebin 확인). 반환 인덱스는 재현 위해 정렬 안 함(선정 순서)."""
    mean = X.mean(0)
    var = mean * (1.0 - mean)
    nonconst = np.where((mean > 0) & (mean < 1))[0]
    if method == "highfreq":
        cand = nonconst[np.argsort(-mean[nonconst])]
        return cand[:k].tolist()
    if method == "highvar":
        cand = nonconst[np.argsort(-var[nonconst])]
        return cand[:k].tolist()
    if method == "category":
        # 카테고리별 고분산 순 큐 → 라운드로빈으로 다양성 확보
        from collections import defaultdict, OrderedDict
        bycat = defaultdict(list)
        for i in nonconst[np.argsort(-var[nonconst])]:
            bycat[cats[i]].append(int(i))
        # 등장 feature 많은 카테고리 순으로 라운드로빈(안정적 순서)
        catorder = sorted(bycat.keys(), key=lambda c: -len(bycat[c]))
        queues = OrderedDict((c, list(bycat[c])) for c in catorder)
        picked = []
        while len(picked) < k and any(queues.values()):
            for c in catorder:
                if queues[c]:
                    picked.append(queues[c].pop(0))
                    if len(picked) >= k:
                        break
        return picked[:k]
    raise ValueError("unknown method %s" % method)


def emp_dist(Xsub):
    """이진 부분집합 → 2^k 경험분포 (정수코드 bincount)."""
    Xsub = np.asarray(Xsub, dtype=np.int64)
    k = Xsub.shape[1]
    code = np.zeros(len(Xsub), dtype=np.int64)
    for j in range(k):
        code += (Xsub[:, j] << j)
    p = np.bincount(code, minlength=1 << k).astype(np.float64)
    return p / p.sum(), int((p > 0).sum())


# ---------------------------------------------------------------------------
# 고차(≥3차) 집중 조합: 경험적 connected 3-body 상관 상위 triple
# ---------------------------------------------------------------------------
def top_triples(Xsub, names_sub, cats_sub, topn=8):
    """s_i = 1-2*bit. connected 3-body: ⟨sisjsk⟩ - (⟨si⟩⟨sjsk⟩+...) + 2⟨si⟩⟨sj⟩⟨sk⟩.
    순수 경험 계산(가벼움). 상위 |값| triple 반환."""
    import itertools
    s = (1 - 2 * np.asarray(Xsub, dtype=np.float64))          # (N, k) ±1
    k = s.shape[1]
    m1 = s.mean(0)                                            # ⟨si⟩
    m2 = (s.T @ s) / len(s)                                   # ⟨sisj⟩
    out = []
    for (i, j, l) in itertools.combinations(range(k), 3):
        m3 = float(np.mean(s[:, i] * s[:, j] * s[:, l]))
        conn = (m3
                - (m1[i] * m2[j, l] + m1[j] * m2[i, l] + m1[l] * m2[i, j])
                + 2 * m1[i] * m1[j] * m1[l])
        out.append((abs(conn), conn, (i, j, l)))
    out.sort(reverse=True)
    res = []
    for _, conn, (i, j, l) in out[:topn]:
        res.append(dict(triple=[int(i), int(j), int(l)], connected_3body=float(conn),
                        features=[names_sub[i], names_sub[j], names_sub[l]],
                        cats=[cats_sub[i], cats_sub[j], cats_sub[l]]))
    return res


# ---------------------------------------------------------------------------
# 한 (split, method, k) 측정
# ---------------------------------------------------------------------------
def measure(X, names, cats, method, k, kmax=4, fine_max_k=10, max_iter=20000, tol=1e-8):
    """headline 2차 vs ≥3차 비중은 KL1(1차 maxent)·KL2(2차 maxent)만으로 계산(저차 fit=값싸고 수렴 견고).
    ≥3차 내부(3/4/≥5차) 세부 분해는 k≤fine_max_k 에서만 OD.decompose(kmax=4)로 수행 —
    12bit는 3·4차 maxent(298·793 params)가 희소 4096상태서 비수렴·수 분 소요 → headline만, 세부는 생략(명시)."""
    idx = select_subset(X, cats, method, k)
    Xsub = X[:, idx]
    p, occ = emp_dist(Xsub)
    bm = OD.bit_matrix(k)
    q1, i1 = OD.maxent_upto(p, k, bm, 1, max_iter=max_iter, tol=tol)
    q2, i2 = OD.maxent_upto(p, k, bm, 2, max_iter=max_iter, tol=tol)
    KL1 = OD.kl_div(p, q1, EPS)
    KL2 = OD.kl_div(p, q2, EPS)  # = ≥3차 잔여 (KL(p‖2차 maxent))
    rec = dict(
        method=method, k=k, subset=idx, occ=occ, nstates=1 << k, N=int(len(X)),
        KL_1=KL1, KL_2=KL2, H=OD.entropy(p),
        share_2nd=((KL1 - KL2) / KL1 if KL1 > 0 else None),
        share_ge3=(KL2 / KL1 if KL1 > 0 else None),
        headline_converged=bool(i1["converged"] and i2["converged"]),
        cats_selected=[cats[i] for i in idx], fine=None,
    )
    if k <= fine_max_k:
        dec = OD.decompose(p, k, kmax=kmax, eps=EPS, max_iter=max_iter, tol=tol)
        ct = dec["contrib"]
        rec["fine"] = dict(
            c2=ct["c2"], c3=ct["c3"], c4=ct["c4"], resid5=ct["resid5"],
            monotonic_ok=dec["monotonic_ok"],
            per_k_conv={kk: v["converged"] for kk, v in dec["per_k"].items()})
    return rec, idx


# ---------------------------------------------------------------------------
# self-check: (i) 합성 도구검증  (ii) flow(UNSW) 재현
# ---------------------------------------------------------------------------
def self_check():
    out = {}
    log("  (i) OD.self_check 합성(parity/ising)...")
    sc = OD.self_check(1e-12)
    out["synthetic"] = dict(parity3_c3=sc["parity3"]["c3"], parity4_c4=sc["parity4"]["c4"],
                            ising2_c3=sc["ising2"]["c3"], all_monotonic=sc["all_monotonic"])
    # (ii) flow UNSW 재빌드 후 재현 (32MB CSV, 가벼움). Bot/HOIC 은 350MB CSV라 저장 json 인용.
    flow_prev = {}
    try:
        fj = json.load(open(FLOW_JSON))
        for nm, r in fj["datasets"].items():
            tot = r["KL_1"]
            flow_prev[nm] = dict(share_2nd=r["c2"] / tot, share_ge3=r["ge3_sum"] / tot,
                                 c2=r["c2"], ge3=r["ge3_sum"], KL1=tot, bits=r["bits"])
    except Exception as e:
        log("  [WARN] flow json 로드 실패: %r" % e)
    repro = None
    try:
        import train_qcbm_unsw_resumable as TUNSW
        log("  (ii) flow UNSW 재빌드→OD.decompose 재현 중...")
        d = TUNSW.build_unsw()
        p = np.asarray(d["q_train"], float); p = p / p.sum()
        dec = OD.decompose(p, TUNSW.NQ, kmax=4, eps=1e-12)
        ct = dec["contrib"]; KL1 = dec["KL"]["1"]
        ge3 = ct["c3"] + (ct["c4"] or 0.0) + ct["resid5"]
        cur = dict(share_2nd=ct["c2"] / KL1, share_ge3=ge3 / KL1, KL1=KL1)
        prev = flow_prev.get("UNSW", {})
        repro = dict(current=cur, stored=dict(share_2nd=prev.get("share_2nd"), share_ge3=prev.get("share_ge3"),
                                              KL1=prev.get("KL1")),
                     ge3_share_abs_diff=(abs(cur["share_ge3"] - prev["share_ge3"]) if prev else None))
    except Exception as e:
        repro = dict(error=repr(e), note="UNSW 재빌드 실패(CSV 부재 등) → 저장 json 인용만")
        log("  [WARN] UNSW 재현 실패: %r" % e)
    out["flow_reproduce_unsw"] = repro
    out["flow_stored"] = flow_prev
    return out


# ---------------------------------------------------------------------------
def run(splits, ks, methods, do_plot=True, kmax=4, do_triples=True):
    os.makedirs(OUTDIR, exist_ok=True)
    log("=== self-check (도구 유효성 + flow 재현) ===")
    sc = self_check()
    syn = sc["synthetic"]
    log("  합성 parity3_c3=%.3f parity4_c4=%.3f ising2_c3=%.1e mono=%s"
        % (syn["parity3_c3"], syn["parity4_c4"], syn["ising2_c3"], syn["all_monotonic"]))
    rp = sc["flow_reproduce_unsw"]
    if rp and "current" in rp:
        log("  flow UNSW 재현: ge3비중 현재=%.3f 저장=%.3f (Δ=%.1e)"
            % (rp["current"]["share_ge3"], rp["stored"]["share_ge3"], rp["ge3_share_abs_diff"]))

    # ── Drebin 측정 ──
    results = {}           # results[method][k] = [split별 rec]
    best_for_triples = None  # (share_ge3, X, idx, names, cats)
    for split in splits:
        X, names, cats = load_split(split)
        log("\n=== Drebin split %d | pooled N=%d, %d features ===" % (split, len(X), X.shape[1]))
        for method in methods:
            for k in ks:
                t0 = time.time()
                rec, idx = measure(X, names, cats, method, k, kmax=kmax)
                rec["split"] = split
                results.setdefault(method, {}).setdefault(str(k), []).append(rec)
                fine_tag = ("fine mono=%s" % rec["fine"]["monotonic_ok"]) if rec["fine"] else "headline만(12bit)"
                log("  %-9s k=%2d | occ=%4d/%d | KL1=%.4f | 2차=%.1f%% ≥3차=%.1f%% | conv=%s %s (%.1fs)"
                    % (method, k, rec["occ"], rec["nstates"], rec["KL_1"],
                       100 * rec["share_2nd"], 100 * rec["share_ge3"],
                       rec["headline_converged"], fine_tag, time.time() - t0))
                if do_triples and (best_for_triples is None or rec["share_ge3"] > best_for_triples[0]):
                    best_for_triples = (rec["share_ge3"], X, idx, names, cats, method, k, split)

    # ── 방식별 집계(split 평균/스프레드) ──
    summary = {}
    for method in results:
        summary[method] = {}
        for k in results[method]:
            recs = results[method][k]
            s2 = np.array([r["share_2nd"] for r in recs])
            s3 = np.array([r["share_ge3"] for r in recs])
            summary[method][k] = dict(
                n_splits=len(recs),
                share_2nd_mean=float(s2.mean()), share_2nd_min=float(s2.min()), share_2nd_max=float(s2.max()),
                share_ge3_mean=float(s3.mean()), share_ge3_min=float(s3.min()), share_ge3_max=float(s3.max()),
                KL2_mean=float(np.mean([r["KL_2"] for r in recs])),
                KL1_mean=float(np.mean([r["KL_1"] for r in recs])),
                all_headline_converged=bool(all(r["headline_converged"] for r in recs)),
                fine_available=bool(recs[0]["fine"] is not None))

    # ── 고차 집중 조합 ──
    triples = None
    if do_triples and best_for_triples is not None:
        sh, X, idx, names, cats, method, k, split = best_for_triples
        ns = [names[i] for i in idx]; cs = [cats[i] for i in idx]
        triples = dict(from_subset=dict(method=method, k=k, split=split, share_ge3=sh),
                       top_triples=top_triples(X[:, idx], ns, cs, topn=8))
        log("\n=== 고차 집중 조합 (ge3 최대 부분집합: %s k=%d split=%d, ≥3차=%.1f%%) ===" % (method, k, split, 100 * sh))
        for t in triples["top_triples"][:5]:
            log("  |conn3|=%.3f  %s" % (abs(t["connected_3body"]), " + ".join(t["cats"])))

    # ── flow 대비표 ──
    flow_cmp = sc["flow_stored"]

    payload = dict(
        note=("Drebin binary static feature 차수 측정. 기존 OD.decompose(kmax=4) 재사용(무수정). "
              "지표=KL1 중 2차(c2)/≥3차(ge3) 비중. flow(UNSW ge3 27%, Bot 19%, HOIC 1%)와 같은 잣대·규모(k=10,12) 대비. "
              "선정 3방식(고빈도/고분산/카테고리)·split{0,1,2} 강건성. 측정값만, 단정 금지, 우위·QCBM 언급 없음."),
        params=dict(splits=splits, ks=ks, methods=methods, kmax=kmax, eps=EPS,
                    pooling="X_train+X_test (측정전용, 학습아님)", units="nats/share"),
        self_check=sc, per_split=results, summary=summary,
        flow_comparison=flow_cmp, high_order_triples=triples)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if do_plot and summary:
        try:
            plot(summary, flow_cmp, ks, methods)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def plot(summary, flow_cmp, ks, methods):
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # 왼쪽: Drebin 방식별(k=최대) 2차 vs ≥3차 누적 + flow
    ksel = str(max(int(x) for x in ks))
    labels, s2s, s3s = [], [], []
    for m in methods:
        if ksel in summary.get(m, {}):
            labels.append("%s\n(k%s)" % (m, ksel))
            s2s.append(100 * summary[m][ksel]["share_2nd_mean"])
            s3s.append(100 * summary[m][ksel]["share_ge3_mean"])
    for nm in ["UNSW", "Bot", "HOIC"]:
        if nm in flow_cmp:
            labels.append("flow\n%s" % nm)
            s2s.append(100 * flow_cmp[nm]["share_2nd"])
            s3s.append(100 * flow_cmp[nm]["share_ge3"])
    x = np.arange(len(labels))
    ax[0].bar(x, s2s, label="2nd-order", color="#4C9F70")
    ax[0].bar(x, s3s, bottom=s2s, label=">=3rd-order", color="#C1442E")
    for xi, v in enumerate(s3s):
        ax[0].text(xi, s2s[xi] + v, "%.0f%%" % v, ha="center", va="bottom", fontsize=9)
    ax[0].set_xticks(x); ax[0].set_xticklabels(labels, fontsize=8)
    ax[0].set_ylabel("share of total correlation (KL1) [%]")
    ax[0].set_title("Drebin (k=%s) vs flow: 2nd vs >=3rd-order share" % ksel)
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")
    # 오른쪽: 방식×k별 ≥3차 비중 (split min-max 범위)
    w = 0.8 / max(1, len(ks))
    for ki, k in enumerate(ks):
        xs = np.arange(len(methods))
        vals = [100 * summary[m][str(k)]["share_ge3_mean"] if str(k) in summary.get(m, {}) else 0 for m in methods]
        lo = [100 * summary[m][str(k)]["share_ge3_min"] if str(k) in summary.get(m, {}) else 0 for m in methods]
        hi = [100 * summary[m][str(k)]["share_ge3_max"] if str(k) in summary.get(m, {}) else 0 for m in methods]
        err = [np.array(vals) - np.array(lo), np.array(hi) - np.array(vals)]
        ax[1].bar(xs + ki * w, vals, w, yerr=err, capsize=3, label="k=%d" % k)
    ax[1].set_xticks(np.arange(len(methods)) + w * (len(ks) - 1) / 2)
    ax[1].set_xticklabels(methods, fontsize=9)
    ax[1].axhline(27, color="gray", ls="--", lw=1, label="flow UNSW 27%")
    ax[1].set_ylabel(">=3rd-order share [%]")
    ax[1].set_title("Drebin >=3rd-order share by method x k (split min-max)")
    ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Drebin higher-order correlation: is malware static feature more high-order than flow?")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="Drebin 고차 상관 차수 측정(OD 재사용). 학습 없음. 단정 금지.")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--splits", type=str, default="0,1,2")
    ap.add_argument("--ks", type=str, default="10,12")
    ap.add_argument("--methods", type=str, default=",".join(METHODS))
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--no-triples", action="store_true")
    args = ap.parse_args()
    if args.smoke:
        run([0], [10], ["highvar"], do_plot=False, do_triples=False)
        return
    splits = [int(x) for x in args.splits.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    run(splits, ks, methods, do_plot=not args.no_plot, do_triples=not args.no_triples)


if __name__ == "__main__":
    main()
