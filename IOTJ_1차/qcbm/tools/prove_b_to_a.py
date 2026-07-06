#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
B→A 확보 — contextuality 있는 실제 데이터에서 (양자)우위 실증 (문헌+실데이터 2층).

메인결론 "실보안데이터 CF=0 → 양자우위 없음"의 대우 "CF>0 → 우위"를 실세계서 확립 시도.
★인공생성 금지(도구 self-check 제외). 실제 존재하는 데이터/결과만 증거.

측정 전용·결정적·무설치(numpy·scipy)·self-check(CF-LP)·단정 금지.

층1 — 문헌 기반 B→A: contextuality→보안우위(키레이트/난수/인증) 실증 논문 표(LIT). 실데이터/no-signalling 명기.
층2 — 실제 공개데이터 재확인: 후보 게이트 (a)공개다운로드 (b)CF>0(no-sig 병기) (c)양자>고전 KL/AUC (d)contextual 제거시 우위소멸.
  ★핵심 관찰: 실제 contextual 데이터(Bell/KCBS)는 **양자물리 측정**이라 (c)/(d)의 "생성모델 성능축"과 도메인 불일치 → 표기.
  실데이터 CF>0 확인: PMC8827658(loophole-free KCBS, 원자이온) 의 **실측 상관** 으로 empirical model 구성 후 CF-LP(=gate1 재사용 일반화).

사용: python tools/prove_b_to_a.py
"""
import os
import sys
import json
import itertools

import numpy as np
from scipy.optimize import linprog

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
for p in (QCBM_DIR, TOOLS_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(QCBM_DIR)

OUTDIR = os.path.join("results2", "b_to_a")
OUT_JSON = os.path.join(OUTDIR, "b_to_a.json")


def log(m):
    print(m, flush=True)


# ---------------------------------------------------------------------------
# 일반 CF-LP (n변수, 각 d값, 임의 컨텍스트 커버) — gate1 삼각형 버전 일반화
# ---------------------------------------------------------------------------
def contextual_fraction(nvar, d, contexts):
    """contexts=[(vars_tuple, dist_ndarray over d^len(vars))]. 반환 (CF, NCF)."""
    assigns = list(itertools.product(range(d), repeat=nvar))
    G = len(assigns)
    A_rows = []; b_ub = []
    for vs, M in contexts:
        M = np.asarray(M, float)
        it = np.ndindex(*M.shape)
        for o in it:
            row = np.zeros(G)
            for gi, g in enumerate(assigns):
                if all(g[vs[t]] == o[t] for t in range(len(vs))):
                    row[gi] = 1.0
            A_rows.append(row); b_ub.append(float(M[o]))
    res = linprog(-np.ones(G), A_ub=np.array(A_rows), b_ub=np.array(b_ub),
                  bounds=[(0, None)] * G, method="highs")
    if not res.success:
        return None, None
    ncf = min(max(float(-res.fun), 0.0), 1.0)
    return float(1.0 - ncf), ncf


def no_signalling(nvar, contexts):
    """공유변수 marginal 컨텍스트 간 L1 불일치 평균."""
    marg = {v: [] for v in range(nvar)}
    for vs, M in contexts:
        M = np.asarray(M, float)
        for t, v in enumerate(vs):
            axes = tuple(a for a in range(M.ndim) if a != t)
            marg[v].append(M.sum(axis=axes))
    tot = 0.0; cnt = 0
    for v in range(nvar):
        ms = marg[v]
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                m = min(len(ms[i]), len(ms[j]))
                tot += float(np.abs(ms[i][:m] - ms[j][:m]).sum()); cnt += 1
    return tot / cnt if cnt else 0.0


def pair_from_corr(E):
    """이분(±1) 상관 E, 균등 marginal 가정 → 2x2 결합 P(a,b), a,b∈{0,1}(=+1,-1)."""
    # ab=+1 (a==b): (1+E)/4 각각;  ab=-1 (a!=b): (1-E)/4 각각
    return np.array([[(1 + E) / 4, (1 - E) / 4],
                     [(1 - E) / 4, (1 + E) / 4]])


# ---------------------------------------------------------------------------
# self-check: 좌절 삼각형(=1), 정합(=0)
# ---------------------------------------------------------------------------
def self_check():
    anti = np.array([[0.0, 0.5], [0.5, 0.0]])
    cf_f, _ = contextual_fraction(3, 2, [((0, 1), anti), ((1, 2), anti), ((0, 2), anti)])
    corr = np.array([[0.5, 0.0], [0.0, 0.5]])
    cf_c, _ = contextual_fraction(3, 2, [((0, 1), corr), ((1, 2), corr), ((0, 2), corr)])
    return dict(frustrated_CF=cf_f, consistent_CF=cf_c,
                valid=bool(cf_f > 0.5 and cf_c < 1e-6))


# ---------------------------------------------------------------------------
# 실데이터 CF>0: PMC8827658 loophole-free KCBS(원자이온) 실측 상관 (Table 1)
# ---------------------------------------------------------------------------
def real_kcbs():
    # 실측 상관 (Science Adv abk1660 / PMC8827658, Table1): 4-cycle chained contextuality
    E = {"01": 0.6164, "12": 0.625, "23": 0.6678, "30": -0.6166}
    C = E["01"] + E["12"] + E["23"] - E["30"]          # ≤2 비접촉, 측정 2.526
    contexts = [((0, 1), pair_from_corr(E["01"])),
                ((1, 2), pair_from_corr(E["12"])),
                ((2, 3), pair_from_corr(E["23"])),
                ((3, 0), pair_from_corr(E["30"]))]
    cf, ncf = contextual_fraction(4, 2, contexts)
    nsv = no_signalling(4, contexts)
    return dict(source="PMC8827658 / Sci.Adv. abk1660 (loophole-free KCBS, 2 ion species)",
                correlations=E, C_value=float(C), noncontextual_bound=2.0,
                CF=cf, NCF=ncf, no_sig_L1=float(nsv),
                note="실측 상관→균등marginal 재구성 empirical model. C=2.526>2=contextual. no-sig≈0(state-indep).")


# ---------------------------------------------------------------------------
# 층1 문헌표 (실제 논문, 웹조사)
# ---------------------------------------------------------------------------
LIT = [
    dict(ref="arXiv:1512.02256 (Troupe & Farinholt 2015)", system="단일 큐빗 prepare&measure QKD(이론 프로토콜)",
         ctx_measure="Spekkens 준비/측정 contextuality(weak-value POVM)", advantage="키레이트 ≥ BB84 + detector 공격 면역",
         real_data="이론(실험 미포함)", no_signalling="prepare&measure(비국소성 아님)"),
    dict(ref="Nature Sci.Rep. srep01627 (2013)", system="실험(광자/이온계) 단일계 contextuality",
         ctx_measure="KS/KCBS 부등식 위반", advantage="인증된 양자난수(min-entropy 하한)",
         real_data="예 (실험 데이터)", no_signalling="단일계 state-(in)dependent contextuality"),
    dict(ref="Phys.Rev.Applied 13,034077 (2020)", system="실험 단일계",
         ctx_measure="contextuality 부등식", advantage="Randomness Expansion(보안 난수 확장)",
         real_data="예", no_signalling="contextuality(단일계)"),
    dict(ref="Sci.Adv. abk1660 / PMC8827658 (2022)", system="실험: 두 원자이온종 (loophole-free)",
         ctx_measure="KCBS(주기형) contextuality, C=2.526>2", advantage="상태독립 contextuality 실증(난수/보안 기반)",
         real_data="예 (본 도구서 CF>0 재확인)", no_signalling="loophole-free·state-independent(무신호 유사)"),
    dict(ref="arXiv:2601.08392 (on-chip semi-DI QRNG)", system="실험: on-chip 광자계",
         ctx_measure="contextuality(semi-DI)", advantage="genuine randomness 21.7 bit/s, min-entropy 0.077/round",
         real_data="예", no_signalling="semi-device-independent"),
]

# ---------------------------------------------------------------------------
# 층2 후보 게이트표 (실제 공개성/도메인 적합성)
# ---------------------------------------------------------------------------
CANDIDATES = [
    dict(cand="1) loophole-free Bell test raw (NIST-2015)",
         gate_a_public="공개(NIST 저장소+AWS, Zenodo 15655461/15655509 분석). 타임태그 3×8B 정수 raw",
         gate_b_CF=">0 원리적(CHSH 위반, 비국소성)",
         gate_c_perf="부적합: 물리 측정데이터(양자상태), 생성모델 KL/AUC 과제 아님",
         gate_d_causal="부적합(동)", verdict="탈락: 도메인 불일치(c/d 축)"),
    dict(cand="2) KCBS contextuality (2-species ion; ETH TIQI public / Sci.Adv.)",
         gate_a_public="공개(논문 Table1 실측 상관; ETH tiqi.ethz.ch 공개 raw)",
         gate_b_CF=">0 (본 도구 CF-LP로 실측 상관서 확인)",
         gate_c_perf="부적합: 물리 측정데이터, 보안 생성/탐지 데이터셋 아님",
         gate_d_causal="부적합", verdict="CF>0 실증(b통과), 그러나 c/d 축 도메인 불일치"),
    dict(cand="3) 공개 QRNG 비트스트림(NIST beacon 등)",
         gate_a_public="공개(비트스트림)", gate_b_CF="contextuality 잔존 미보장(후처리된 비트=구조 소거)",
         gate_c_perf="부적합", gate_d_causal="부적합", verdict="탈락: CF 검증 불가(후처리)+도메인"),
    dict(cand="4) 공개 QKD sifted-key/QBER",
         gate_a_public="일부 공개/요청필요", gate_b_CF="키/QBER는 CF 직접측정 대상 아님",
         gate_c_perf="부적합", gate_d_causal="부적합", verdict="탈락: CF 측정대상 아님+도메인"),
    dict(cand="5) IBM Q 등 공개 contextuality 시나리오",
         gate_a_public="재현 가능하나 계정/실행 필요(비-정적 다운로드)", gate_b_CF=">0 가능",
         gate_c_perf="부적합(측정 시나리오)", gate_d_causal="부적합", verdict="탈락: 도메인 불일치(+실행 필요)"),
]


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    log("=== B→A 확보: 문헌+실데이터 2층 ===\n")
    sc = self_check()
    log("[self-check CF-LP] 좌절삼각형 CF=%.3f, 정합 CF=%.3f → valid=%s" %
        (sc["frustrated_CF"], sc["consistent_CF"], sc["valid"]))

    log("\n--- 층1 문헌표 (contextuality→보안우위, 실데이터/무신호 명기) ---")
    for e in LIT:
        log("  %s | %s | 우위=%s | 실데이터=%s" % (e["ref"], e["ctx_measure"], e["advantage"], e["real_data"]))

    log("\n--- 층2 실데이터 CF>0 (실측 KCBS 상관) ---")
    rk = real_kcbs()
    log("  %s" % rk["source"])
    log("  C=%.3f (>2 비접촉한계) | **CF=%.4f** (>0) | no-sig=%.4f (state-indep)" %
        (rk["C_value"], rk["CF"], rk["no_sig_L1"]))
    log("  → 실제 contextual 데이터서 CF>0 확인 (도구가 실세계 contextuality 감지).")

    log("\n--- 층2 후보 게이트표 (공개성/도메인 적합) ---")
    for c in CANDIDATES:
        log("  %s → %s" % (c["cand"], c["verdict"]))

    verdict = dict(
        layer1=("확보(원리축): 실제 양자실험서 contextuality(KCBS/Bell 위반)→보안우위(인증난수·키레이트≥BB84·"
                "detector면역·randomness expansion)가 문헌에 확립. 단 우위축=**DI-난수/키레이트**(양자 측정/암호), "
                "우리 프로젝트의 **생성모델 KL/AUC 축과 다름**."),
        layer2=("실제 contextual 공개데이터(KCBS/Bell) 존재·다운로드 가능(a통과), CF>0 실측 확인(b통과). "
                "그러나 이들은 **양자물리 측정데이터**라 (c)양자>고전 생성/탐지 성능·(d)contextual 제거 인과 축과 "
                "**도메인 불일치** → 우리 축에서의 실증은 미완(실 contextual 보안-탐지 데이터셋 부재)."),
        synthesis=("B→A는 **원리로서 확립**(문헌 인용)되나 **classical 보안-데이터 생성모델링 축으로 전이 안 됨**: "
                   "contextuality는 양자 측정 시나리오의 성질이고, classical 보안데이터는 non-contextual(gate1 CF=0). "
                   "따라서 대우 'CF>0→우위'는 양자물리/암호에서 참이나, malware/traffic 같은 classical 데이터엔 "
                   "적용 대상(CF>0 데이터)이 없어 QCBM 무우위 결론과 정합. 단정 금지·이 세팅 한정·우위주장 없음."),
        precedent_check=("1512.02256은 정확히 우리 생성모델 주장을 replicate 아님(QKD 키레이트 축). "
                         "우리 축(고차 분포 생성 KL)의 contextual-우위 실증은 문헌서 미발견 → 우리가 채울 자리이나, "
                         "실 contextual 보안-탐지 데이터 부재가 병목."),
    )
    payload = dict(
        note=("B→A 2층 실증. 층1 문헌(실 contextuality→보안우위), 층2 실데이터 CF>0(KCBS 실측)+후보 게이트. "
              "인공데이터 증거 아님(self-check만). 측정전용·무설치·단정금지·이 세팅 한정·우위주장 없음."),
        self_check=sc, real_data_CF=rk, literature=LIT, candidates=CANDIDATES, verdict=verdict)
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    return payload


if __name__ == "__main__":
    main()
