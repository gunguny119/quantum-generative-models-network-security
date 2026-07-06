#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
게이트1 — IAT 시퀀스에 진짜 contextuality가 존재하는가 (정적=0 반례 확인). 소규모 개념검증.

측정 전용(새 학습 없음), 결정적(seed 고정), 무설치(struct·numpy·scipy만). 단정 금지.
근거: arXiv:2507.11604(sequential empirical model, k) + Abramsky-Brandenburger sheaf contextuality(전역절편 LP·contextual fraction).
개념·알고리즘 출처=그들, IAT/트래픽 적용=우리.

게이트0(선행): samples/*.pcap 에서 패킷 타임스탬프→IAT 시퀀스 복원(struct 직접 파싱, dpkt 불요). PASS.

정의 T-b(우선, 시간창=context, 삼각형 3-cycle 커버):
- IAT 시퀀스를 d빈(분위) 이산화. 변수 V={0,1,2}(연속 3위치의 IAT 빈).
- 컨텍스트(겹치는 창): C01=(X_t,X_{t+1})[lag1], C12=(X_{t+1},X_{t+2})[lag1], C02=(X_t,X_{t+2})[lag2].
  각 컨텍스트를 시퀀스 lag 풀에서 **독립 추정** → 단일 결합을 강요 안 함(전역절편 실현가능성이 진짜 질문).
- no-signalling 위반: 공유 변수의 marginal 이 컨텍스트 간 얼마나 불일치(L1). 크면 signalling(프레임 부적합).
- contextuality: 전역절편(global joint) LP 실현가능성 → contextual fraction CF∈[0,1]. CF=1−max(비접촉 질량).
정적 대조(핵심): 같은 IAT 를 셔플(순서 파괴=iid 정적) → 동일 CF/no-sig. 가설 "시퀀스 CF>0 & 정적 CF≈0".

★ 소규모 개념검증(일반화 아님). CF는 LP 상한/근사. no-sig 위반 반드시 병기. 관측이지 인과 아님. 우위 주장 없음.

사용: python tools/gate1_iat_contextuality.py [--bins 3,4] [--maxpk 40000] [--smoke]
"""
import os
import sys
import json
import struct
import argparse
import itertools

import numpy as np
from scipy.optimize import linprog

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IOTJ_DIR = os.path.dirname(QCBM_DIR)
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

OUTDIR = os.path.join("results2", "gate1_ctx")
OUT_JSON = os.path.join(OUTDIR, "gate1.json")
OUT_PNG = os.path.join(OUTDIR, "gate1.png")
SEED = 31337

PCAPS = {
    "Tinba(attack)":       os.path.join(IOTJ_DIR, "samples", "Tinba.pcap"),
    "Facetime(benign)":    os.path.join(IOTJ_DIR, "samples", "Facetime.pcap"),
    "aim_chat(benign)":    os.path.join(IOTJ_DIR, "samples", "iscx", "aim_chat_3a.pcap"),
    "email(benign)":       os.path.join(IOTJ_DIR, "samples", "iscx", "email1a.pcap"),
    "facebook(benign)":    os.path.join(IOTJ_DIR, "samples", "iscx", "facebook_audio1a.pcap"),
}


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 게이트0: pcap 타임스탬프 → IAT (struct 직접 파싱)
# ---------------------------------------------------------------------------
def read_iat(path, maxpk=40000):
    with open(path, "rb") as f:
        gh = f.read(24)
        if len(gh) < 24:
            return None
        magic = struct.unpack("<I", gh[:4])[0]
        if magic in (0xa1b2c3d4, 0xa1b23c4d):
            endi, nano = "<", (magic == 0xa1b23c4d)
        elif magic in (0xd4c3b2a1, 0x4d3cb2a1):
            endi, nano = ">", (magic == 0x4d3cb2a1)
        else:
            return None
        ts = []
        while len(ts) < maxpk:
            rh = f.read(16)
            if len(rh) < 16:
                break
            s, us, incl, orig = struct.unpack(endi + "IIII", rh)
            ts.append(s + us * (1e-9 if nano else 1e-6))
            f.read(incl)
    t = np.array(ts, float)
    iat = np.diff(t)
    return iat[iat >= 0]


def discretize(iat, d):
    """분위(equal-frequency) d빈. 반환 정수 시퀀스."""
    x = iat.copy()
    # log 스케일(헤비테일) 후 분위
    x = np.log10(x + 1e-9)
    qs = np.quantile(x, np.linspace(0, 1, d + 1))
    qs[0] = -np.inf; qs[-1] = np.inf
    # 중복 경계 방지
    qs = np.unique(qs)
    if len(qs) - 1 < 2:
        return None
    return np.clip(np.digitize(x, qs[1:-1]), 0, len(qs) - 2), len(qs) - 1


# ---------------------------------------------------------------------------
# empirical model: 3 컨텍스트(삼각형) 의 pairwise 분포 (독립 추정)
# ---------------------------------------------------------------------------
def pair_dist(seq, lag, d):
    a = seq[:-lag]; b = seq[lag:]
    M = np.zeros((d, d))
    for x, y in zip(a, b):
        M[x, y] += 1
    s = M.sum()
    return M / s if s > 0 else M


def build_model(seq, d):
    # 변수 역할: var0=X_t, var1=X_{t+1}, var2=X_{t+2}
    P01 = pair_dist(seq, 1, d)          # (var0,var1)
    P12 = pair_dist(seq, 1, d)          # (var1,var2)  동일 lag1 풀(정상성 가정 시 동일)
    P02 = pair_dist(seq, 2, d)          # (var0,var2)  lag2
    ctx = [((0, 1), P01), ((1, 2), P12), ((0, 2), P02)]
    return ctx


def no_signalling_violation(ctx, d):
    """공유 변수 marginal 의 컨텍스트 간 L1 불일치 합."""
    marg = {v: [] for v in range(3)}
    for (vs, M) in ctx:
        marg[vs[0]].append(M.sum(1))       # 첫 변수 marginal
        marg[vs[1]].append(M.sum(0))       # 둘째 변수 marginal
    tot = 0.0; cnt = 0
    for v in range(3):
        ms = marg[v]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                tot += float(np.abs(ms[i] - ms[j]).sum()); cnt += 1
    return tot / cnt if cnt else 0.0


# ---------------------------------------------------------------------------
# contextual fraction: NCF = max Σ b_g  s.t. ∀(C,o) Σ_{g|C=o} b_g ≤ P_C(o), b≥0
# ---------------------------------------------------------------------------
def contextual_fraction(ctx, d):
    assigns = list(itertools.product(range(d), repeat=3))     # 전역 결정론적 배정
    G = len(assigns)
    A_rows = []; b_ub = []
    for (vs, M) in ctx:
        for a in range(d):
            for bb in range(d):
                row = np.zeros(G)
                for gi, g in enumerate(assigns):
                    if g[vs[0]] == a and g[vs[1]] == bb:
                        row[gi] = 1.0
                A_rows.append(row); b_ub.append(M[a, bb])
    A_ub = np.array(A_rows); b_ub = np.array(b_ub)
    c = -np.ones(G)                                            # maximize Σ b
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=[(0, None)] * G, method="highs")
    if not res.success:
        return None, None
    ncf = float(-res.fun)
    ncf = min(max(ncf, 0.0), 1.0)
    return float(1.0 - ncf), ncf


# ---------------------------------------------------------------------------
def analyze_seq(seq, d, tag):
    ctx = build_model(seq, d)
    nsv = no_signalling_violation(ctx, d)
    cf, ncf = contextual_fraction(ctx, d)
    return dict(tag=tag, d=int(d), cf=cf, ncf=ncf, no_sig_L1=nsv, n=int(len(seq)))


def run(bins, maxpk, do_smoke=False):
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)
    results = {}
    pcaps = list(PCAPS.items())
    if do_smoke:
        pcaps = pcaps[:2]; bins = [3]
    for name, path in pcaps:
        if not os.path.exists(path):
            log("  [skip] %s 없음" % path); continue
        iat = read_iat(path, maxpk)
        if iat is None or len(iat) < 100:
            log("  [skip] %s IAT 부족" % name); continue
        per_bin = {}
        for d in bins:
            disc = discretize(iat, d)
            if disc is None:
                per_bin[str(d)] = dict(error="이산화 실패(빈 붕괴)"); continue
            seq, dd = disc
            # 시퀀스(순서 유지) vs 정적(셔플=iid)
            seq_shuf = seq.copy(); rng.shuffle(seq_shuf)
            r_seq = analyze_seq(seq, dd, "sequence")
            r_static = analyze_seq(seq_shuf, dd, "static_shuffle")
            per_bin[str(d)] = dict(sequence=r_seq, static=r_static, d_eff=int(dd))
            log("  %-18s d=%d | 시퀀스 CF=%.4f no-sig=%.4f | 정적 CF=%.4f no-sig=%.4f | n=%d"
                % (name, dd, r_seq["cf"], r_seq["no_sig_L1"], r_static["cf"], r_static["no_sig_L1"], r_seq["n"]))
        results[name] = per_bin

    payload = dict(
        note=("게이트1 IAT 시퀀스 contextuality. T-b(시간창=context, 삼각형 3-cycle) CF(LP)+no-sig 위반. "
              "정적 대조=IAT 셔플(iid). 가설: 시퀀스 CF>0 & 정적 CF≈0. CF≈0이면 동역학도 non-contextual(B사망→A강화). "
              "no-sig 위반 지배면 signalling(프레임 부적합). 소규모 개념검증·CF는 LP상한·관측정렬·우위주장 없음. 단정 금지."),
        params=dict(bins=bins, maxpk=maxpk, seed=SEED, pcaps=list(PCAPS.keys()),
                    contexts="C01[lag1],C12[lag1],C02[lag2] over vars(0,1,2)"),
        gate0="PASS (struct 파싱으로 pcap 패킷 타임스탬프→IAT 복원, dpkt 불요)",
        results=results)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    if not do_smoke:
        try:
            plot(results, bins)
        except Exception as e:
            log("[plot WARN] %r" % e)
    return payload


def plot(results, bins):
    names = list(results.keys())
    d0 = str(bins[0])
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.5))
    x = np.arange(len(names))
    cf_seq = [results[n].get(d0, {}).get("sequence", {}).get("cf", np.nan) for n in names]
    cf_sta = [results[n].get(d0, {}).get("static", {}).get("cf", np.nan) for n in names]
    ns_seq = [results[n].get(d0, {}).get("sequence", {}).get("no_sig_L1", np.nan) for n in names]
    w = 0.35
    ax[0].bar(x - w / 2, cf_seq, w, label="sequence CF", color="#7B2FBE")
    ax[0].bar(x + w / 2, cf_sta, w, label="static(shuffle) CF", color="#9AA0A6")
    ax[0].set_xticks(x); ax[0].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax[0].set_ylabel("contextual fraction CF"); ax[0].set_title("Gate1: sequence vs static CF (d=%s)" % d0)
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")
    ax[1].bar(x, ns_seq, color="#C1442E")
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, rotation=25, ha="right", fontsize=7)
    ax[1].set_ylabel("no-signalling violation (L1)"); ax[1].set_title("sequence no-sig violation (signalling?)")
    ax[1].grid(alpha=.3, axis="y")
    plt.suptitle("Gate1 IAT contextuality: does dynamics create genuine (no-signalling) contextuality?")
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=125); plt.close()
    log("[저장] %s" % OUT_PNG)


def main():
    ap = argparse.ArgumentParser(description="게이트1 IAT 시퀀스 contextuality(CF-LP). 측정 전용. 단정 금지.")
    ap.add_argument("--bins", type=str, default="3,4")
    ap.add_argument("--maxpk", type=int, default=40000)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    bins = [int(x) for x in args.bins.split(",") if x.strip()]
    run(bins, args.maxpk, do_smoke=args.smoke)


if __name__ == "__main__":
    main()
