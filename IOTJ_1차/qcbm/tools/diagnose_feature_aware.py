#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
feature-인지 분해 — joint gain / 고차 상관에서 "feature 내부 비트 상관" 오염을 제거하고 재측정.

순수 측정(학습 없음). 경험적 분포 + 기존 평가/maxent 인프라 재사용(import만). 단일 프로세스.

배경: 현 marginal_dist 는 비트 단위 독립이라, joint−marginal gain 과 bit-level 차수분해의 "고차"에
(a) feature 내부 비트상관, (b) feature 간 상관이 섞임. feature 경계를 인지해 (a)/(b)를 분리한다.

정보기하(정확): bit_marg ⊂ feat_marg ⊂ joint (nested maxent) →
  KL(joint||bit_marg) = KL(joint||feat_marg) + KL(feat_marg||bit_marg)
  within(A)=KL(feat_marg||bit_marg)=feature 내부, cross(B)=KL(joint||feat_marg)=순수 feature 간.

사용: python tools/diagnose_feature_aware.py [--dataset {unsw,bot,hoic,all}] [--kmax 4]
                                            [--max-iter N] [--no-plot]
"""
import os
import sys
import json
import argparse
import itertools

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

import experiment2 as E                   # noqa: E402  empirical_dist, auc_score
import train_qcbm_iat_b2 as TB2           # noqa: E402  marginal_dist(bit_marg), build_b2(HOIC)
import train_qcbm_bot_resumable as TBOT   # noqa: E402  build_bot()
import train_qcbm_unsw_resumable as TUNSW  # noqa: E402  build_unsw()
import diagnose_order_decomposition as OD  # noqa: E402  maxent_upto, kl_div, bit_matrix

OUT = "results2"
OUT_JSON = os.path.join(OUT, "feature_aware_decomposition.json")
OUT_PNG = os.path.join(OUT, "feature_aware_decomposition.png")
EPS = 1e-12

# 비트→feature 매핑 (직전 조사 확정). 값은 비트 인덱스(i=0=LSB).
FEAT_HOIC = {"proto": [10, 11], "syn": [9], "psh": [8], "size": [4, 5, 6, 7],
             "fwd_iat": [2, 3], "flow_iat": [0, 1]}
FEAT_UNSW = {"proto": [8, 9], "size": [4, 5, 6, 7], "fwd_iat": [2, 3], "flow_iat": [0, 1]}

DATASETS = {
    "unsw": dict(name="UNSW", build=TUNSW.build_unsw, nq=TUNSW.NQ, feats=FEAT_UNSW),
    "bot":  dict(name="Bot",  build=TBOT.build_bot,   nq=TBOT.NQ, feats=FEAT_HOIC),
    "hoic": dict(name="HOIC", build=TB2.build_b2,     nq=TB2.NQ, feats=FEAT_HOIC),
}
ORDER = ["unsw", "bot", "hoic"]


def log(msg):
    print(msg, flush=True)


# ---------------------------------------------------------------------------
# feature_marginal: 각 feature 내부 joint 유지, feature끼리 독립
# ---------------------------------------------------------------------------
def feature_marginal(states, nq, feat_bits):
    """benign 상태 배열 states 로부터 feature-단위 marginal(외적) 분포(길이 2^nq) 추정."""
    states = np.asarray(states, dtype=np.int64)
    N = 1 << nq
    bm = OD.bit_matrix(nq)                                  # (N, nq)
    q = np.ones(N, dtype=np.float64)
    for bits in feat_bits.values():
        bits = list(bits)
        # 표본의 feature 부분상태
        sub_s = np.zeros(len(states), dtype=np.int64)
        sub_all = np.zeros(N, dtype=np.int64)
        for j, b in enumerate(bits):
            sub_s += (((states >> b) & 1) << j)
            sub_all += (bm[:, b] << j)
        pf = np.bincount(sub_s, minlength=(1 << len(bits))).astype(np.float64)
        pf /= pf.sum()
        q *= pf[sub_all]
    return q / q.sum()


def auc_of(q, st_te, st_atk):
    s = np.concatenate([-np.log(np.asarray(q)[st_te] + EPS), -np.log(np.asarray(q)[st_atk] + EPS)])
    lab = np.concatenate([np.zeros(len(st_te), int), np.ones(len(st_atk), int)])
    return float(E.auc_score(s, lab))


# ---------------------------------------------------------------------------
# 부분집합 → Φ (spin 곱), feature 내부/간 분류
# ---------------------------------------------------------------------------
def feat_of_bit(nq, feat_bits):
    fo = {}
    for fi, (name, bits) in enumerate(feat_bits.items()):
        for b in bits:
            fo[b] = fi
    return [fo[b] for b in range(nq)]


def phi_from_subsets(bm, subsets):
    s = (1 - 2 * bm).astype(np.float64)
    if not subsets:
        return np.zeros((bm.shape[0], 0))
    cols = [np.prod(s[:, list(S)], axis=1) for S in subsets]
    return np.stack(cols, axis=1)


def build_constraint_subsets(nq, feat_bits, kmax):
    """C1=order1, 이후 각 차수 k: within_k(같은 feature) 먼저, cross_k(2+ feature) 다음으로 누적."""
    fo = feat_of_bit(nq, feat_bits)
    order1 = [(i,) for i in range(nq)]
    seq = [("order1", list(order1))]
    cur = list(order1)
    for k in range(2, kmax + 1):
        within, cross = [], []
        for S in itertools.combinations(range(nq), k):
            if len({fo[i] for i in S}) == 1:
                within.append(S)
            else:
                cross.append(S)
        cur = cur + within
        seq.append(("within%d" % k, list(cur)))
        cur = cur + cross
        seq.append(("cross%d" % k, list(cur)))
    return seq


# ---------------------------------------------------------------------------
# self-check
# ---------------------------------------------------------------------------
def self_check():
    out = {}
    nq = 4
    rng = np.random.default_rng(0)
    bm = OD.bit_matrix(nq)
    # 표본: 임의 joint 에서 추출(상태 인덱스 샘플)
    p = rng.random(1 << nq); p /= p.sum()
    states = rng.choice(1 << nq, size=20000, p=p)
    joint = E.empirical_dist(states, 1 << nq)
    bit_marg = TB2.marginal_dist(states, nq)

    # (i) feature 1개 = 전체 비트 -> feat_marg == joint
    fm_all = feature_marginal(states, nq, {"all": [0, 1, 2, 3]})
    out["i_one_feature_eq_joint_KL"] = OD.kl_div(joint, fm_all, EPS)
    assert out["i_one_feature_eq_joint_KL"] < 1e-9

    # (ii) feature = 각 1비트 -> feat_marg == bit_marg
    fm_bits = feature_marginal(states, nq, {str(i): [i] for i in range(nq)})
    out["ii_singlebit_eq_bitmarg_KL"] = OD.kl_div(fm_bits, bit_marg, EPS)
    assert out["ii_singlebit_eq_bitmarg_KL"] < 1e-9

    # (iii) 독립 두 feature-블록 -> cross B ≈ 0
    pa = rng.random(4); pa /= pa.sum()
    pb = rng.random(4); pb /= pb.sum()
    bm2 = OD.bit_matrix(nq)
    suba = bm2[:, 0] + 2 * bm2[:, 1]
    subb = bm2[:, 2] + 2 * bm2[:, 3]
    pind = pa[suba] * pb[subb]; pind /= pind.sum()
    st_ind = rng.choice(1 << nq, size=40000, p=pind)
    feats2 = {"A": [0, 1], "B": [2, 3]}
    jt = E.empirical_dist(st_ind, 1 << nq)
    fm = feature_marginal(st_ind, nq, feats2)
    out["iii_independent_features_crossB_KL"] = OD.kl_div(jt, fm, EPS)
    assert out["iii_independent_features_crossB_KL"] < 5e-3   # 표본오차 허용

    # (iv) 단조성: 상관 있는 두 feature 에서 nested 제약열 KL 단조
    pcorr = rng.random(1 << nq); pcorr /= pcorr.sum()
    st_c = rng.choice(1 << nq, size=40000, p=pcorr)
    jt_c = E.empirical_dist(st_c, 1 << nq)
    seq = build_constraint_subsets(nq, feats2, kmax=4)
    kls = []
    for _, subs in seq:
        q, _info = OD.maxent_upto(jt_c, nq, bm2, kmax=4, Phi=phi_from_subsets(bm2, subs))
        kls.append(OD.kl_div(jt_c, q, EPS))
    mono = all(kls[i] >= kls[i + 1] - 1e-7 for i in range(len(kls) - 1))
    out["iv_monotonic"] = bool(mono)
    assert mono
    return out


# ---------------------------------------------------------------------------
def run(keys, kmax=4, max_iter=20000, do_plot=True):
    log("=== self-check ===")
    sc = self_check()
    log("  (i) 1feature=joint KL=%.2e | (ii) 1bit=bitmarg KL=%.2e | (iii) indep cross=%.2e | (iv) 단조=%s"
        % (sc["i_one_feature_eq_joint_KL"], sc["ii_singlebit_eq_bitmarg_KL"],
           sc["iii_independent_features_crossB_KL"], sc["iv_monotonic"]))

    results = {}
    for key in keys:
        cfg = DATASETS[key]
        nm, nq, feats = cfg["name"], cfg["nq"], cfg["feats"]
        log("\n=== %s (%d bit, %d features) ===" % (nm, nq, len(feats)))
        d = cfg["build"]()
        st_tr, st_te, st_atk = d["st_tr"], d["st_te"], d["st_atk"]
        N = 1 << nq
        bm = OD.bit_matrix(nq)

        joint = np.asarray(d["q_train"], float)
        bit_marg = TB2.marginal_dist(st_tr, nq)
        feat_marg = feature_marginal(st_tr, nq, feats)

        # goal2: AUC 분해
        auc_bit = auc_of(bit_marg, st_te, st_atk)
        auc_feat = auc_of(feat_marg, st_te, st_atk)
        auc_joint = auc_of(joint, st_te, st_atk)
        contamination = auc_feat - auc_bit
        pure_cross = auc_joint - auc_feat

        # goal3a: 정보량 within/cross (정확)
        total = OD.kl_div(joint, bit_marg, EPS)
        A = OD.kl_div(feat_marg, bit_marg, EPS)
        B = OD.kl_div(joint, feat_marg, EPS)
        pyth_resid = A + B - total

        log("  AUC bit=%.4f feat=%.4f joint=%.4f | contamination=%.4f pure_cross=%.4f"
            % (auc_bit, auc_feat, auc_joint, contamination, pure_cross))
        log("  KL total=%.4f within(A)=%.4f cross(B)=%.4f | B/total=%.3f | Pythagoras잔차=%.2e"
            % (total, A, B, (B / total if total > 0 else float("nan")), pyth_resid))

        # goal3b: 차수×{within/cross} maxent 분해
        seq = build_constraint_subsets(nq, feats, kmax)
        kls, per_fit = [], []
        for label, subs in seq:
            Phi = phi_from_subsets(bm, subs)
            q, info = OD.maxent_upto(joint, nq, bm, kmax=kmax, Phi=Phi, max_iter=max_iter, verbose=False)
            klv = OD.kl_div(joint, q, EPS)
            kls.append((label, klv))
            per_fit.append(dict(step=label, nparams=int(Phi.shape[1]), KL=klv,
                                converged=info["converged"], iters=info["iters"],
                                max_violation=info["max_violation"]))
            log("    %-9s D=%4d KL=%.4f conv=%s viol=%.2e" % (label, Phi.shape[1], klv, info["converged"], info["max_violation"]))
        klmap = dict(kls)
        # 단계 차이 = 기여
        contrib = {}
        labels = [lab for lab, _ in kls]
        for a, b in zip(labels[:-1], labels[1:]):
            contrib[b] = klmap[a] - klmap[b]      # b 단계가 추가한 제약의 기여
        resid_top = klmap[labels[-1]]             # ≥(kmax+1)차 잔여 (max feature=4bit이라 전부 cross)
        # ≥3차 분해
        order2_residual = klmap.get("cross2")     # order≤2 (all) 후 잔여 = ≥3차 합(bit-level)
        within_ge3 = sum(contrib.get("within%d" % k, 0.0) for k in range(3, kmax + 1))
        cross_ge3 = sum(contrib.get("cross%d" % k, 0.0) for k in range(3, kmax + 1)) + resid_top
        ge3_total = within_ge3 + cross_ge3
        ge3_cross_share = (cross_ge3 / ge3_total) if ge3_total > 0 else None

        monotonic = all(klmap[a] >= klmap[b] - 1e-7 for a, b in zip(labels[:-1], labels[1:]))
        neg = {k: (v < -1e-7) for k, v in contrib.items()}
        log("  ≥3차: within=%.4f cross=%.4f (resid_top=%.4f) | ≥3차 cross 비중=%s | 단조=%s"
            % (within_ge3, cross_ge3, resid_top,
               None if ge3_cross_share is None else round(ge3_cross_share, 3), monotonic))

        results[nm] = dict(
            bits=nq, nfeatures=len(feats),
            auc=dict(bit_marg=auc_bit, feat_marg=auc_feat, joint=auc_joint),
            contamination=contamination, pure_cross=pure_cross,
            KL_total=total, within_A=A, cross_B=B,
            B_over_total=(B / total if total > 0 else None), pythagoras_residual=pyth_resid,
            order_contrib=contrib, resid_top=resid_top,
            ge3_within=within_ge3, ge3_cross=cross_ge3, ge3_total=ge3_total,
            ge3_cross_share=ge3_cross_share,
            order2_residual_bitlevel=order2_residual,
            monotonic=bool(monotonic), negative_contrib=neg, per_fit=per_fit,
        )

    payload = dict(
        note=("feat_marg=각 feature 내부 joint 유지·feature끼리 독립. within(A)=KL(feat_marg||bit_marg)=feature 내부, "
              "cross(B)=KL(joint||feat_marg)=순수 feature 간. AUC 분해 contamination/pure_cross. "
              "차수×{within/cross} maxent로 ≥3차 중 feature간 비중. 학습 없음. 단정 금지."),
        dissociation_recheck="확인불가: analyze_tv_auc_dissociation 는 QCBM 16 restart fit-vs-AUC 상관 측정 → "
                             "feature_marginal(단일 분포)로 재정의하려면 새 해리 정의 필요(금지). goal1-3만 수행.",
        params=dict(kmax=kmax, max_iter=max_iter, eps=EPS),
        self_check=sc, datasets=results,
    )
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if do_plot and results:
        plot(results)
    return payload


def plot(results):
    names = list(results.keys())
    x = np.arange(len(names))
    fig, ax = plt.subplots(1, 2, figsize=(15, 6))
    # 왼쪽: AUC 3종
    w = 0.26
    ax[0].bar(x - w, [results[n]["auc"]["bit_marg"] for n in names], w, label="bit_marginal", color="#9AA0A6")
    ax[0].bar(x,     [results[n]["auc"]["feat_marg"] for n in names], w, label="feature_marginal", color="#4C9F70")
    ax[0].bar(x + w, [results[n]["auc"]["joint"] for n in names], w, label="joint", color="#C1442E")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names); ax[0].set_ylabel("detection AUC")
    ax[0].set_title("AUC: bit_marg vs feature_marg vs joint\n(contamination=feat-bit, pure_cross=joint-feat)")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")
    # 오른쪽: ≥3차 within vs cross
    gw = [results[n]["ge3_within"] for n in names]
    gc = [results[n]["ge3_cross"] for n in names]
    ax[1].bar(x, gw, label="within-feature (>=3rd)", color="#3B6EA5")
    ax[1].bar(x, gc, bottom=gw, label="cross-feature (>=3rd)", color="#E2A13B")
    for xi, n in enumerate(names):
        sh = results[n]["ge3_cross_share"]
        if sh is not None:
            ax[1].text(xi, gw[xi] + gc[xi], "cross %.0f%%" % (100 * sh), ha="center", va="bottom", fontsize=10)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names); ax[1].set_ylabel(">=3rd-order KL (nats)")
    ax[1].set_title(">=3rd-order correlation: within- vs cross-feature")
    ax[1].legend(fontsize=9); ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Feature-aware decomposition: is joint-gain / higher-order genuinely cross-feature?")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="feature-인지 분해(joint gain·고차의 within/cross 분리). 학습 없음.")
    ap.add_argument("--dataset", default="all", choices=["all"] + ORDER, help="기본 all (UNSW 먼저)")
    ap.add_argument("--kmax", type=int, default=4)
    ap.add_argument("--max-iter", type=int, default=20000)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    keys = ORDER if args.dataset == "all" else [args.dataset]
    run(keys, kmax=args.kmax, max_iter=args.max_iter, do_plot=not args.no_plot)


if __name__ == "__main__":
    main()
