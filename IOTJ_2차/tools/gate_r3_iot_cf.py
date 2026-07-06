#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate R3 — 실제 IoT 데이터셋 CF 실측 (꺼진 앵커의 IoT 정박, IoTJ 스코프용).

목적: 진짜 IoT 텔레메트리에서 **CF≈0 실측 + CF²법칙으로 잔여 gap 상한 정량화**.
      ★발견 목적 아님(정적 결합분포는 이론상 CF=0 자명) — **정박·정량화** 목적(IoTJ 서술용).
원칙: 결정적(seed), 기존 도구(cyclic CF-LP = kink_origin_v1.K, contextuality_v0 정의) 무수정 재사용,
      self-check, 단정 금지, 출처·MD5 기록. ★supremacy 아님(모델클래스 분리).

데이터: N-BaIoT (UCI 442). TON_IoT/BoT-IoT 는 CloudStor(AARNET) 폐지로 접근 불가 → 확보분(N-BaIoT 다기기)으로 진행(정직).
속도: 기기별 ≤50만 행 샘플(seed), 필요 열만, multiprocessing Pool(cpu), 부트스트랩 500.

측정(1차 방법론):
 - 이진 feature: 연속→중앙값 이진화(1차 방식). k=8 부분집합 5종: highvar / targeted(고차 conn3) / random×3.
 - 정의 2종: (a) feature-block cyclic context(k-cycle), (b) 시간창 cyclic context(행순서=시간대용, 3-cycle lag1,1,2).
 - CF-LP(K.cyclic_cf) + 부트스트랩 500 95%CI + no-signalling 잔차. CF²법칙: gap_upper = c·CF_upper² (c=1/8 gate_e cyclic, 경계근사 주의).

사용: python tools/gate_r3_iot_cf.py [--devices 3] [--rows 500000] [--k 8] [--boot 500] [--jobs 16] [--smoke]
"""
import os
import sys
import json
import time
import zipfile
import hashlib
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import kink_origin_v1 as K                 # cyclic CF-LP·setup (무수정; import 시 os.chdir(IOTJ_2차))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt            # noqa: E402

OUTDIR = os.path.join("results2", "gate_r3")
OUT_JSON = os.path.join(OUTDIR, "r3.json")
OUT_PNG = os.path.join(OUTDIR, "r3.png")
PROC = os.path.join("data", "nbaiot", "proc")
ZIP = os.path.join("data", "nbaiot", "nbaiot.zip")
SEED = 20260705
C_GATE_E = 0.125                            # gate_e cyclic(KCBS) 경계 c=1/8 (경계근사)
EPS = 1e-12


def log(m):
    print(m, flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


# ===========================================================================
# 데이터 로드/이진화 (기기별) — zip 에서 benign + gafgyt + mirai 일부 concat, ≤rows 샘플
# ===========================================================================
def list_devices(zf):
    devs = {}
    for n in zf.namelist():
        if n.endswith(".csv"):
            top = n.split("/")[0]
            devs.setdefault(top, []).append(n)
    return devs


def load_device(zf, members, rows, seed):
    """benign + 공격 일부 concat → 중앙값 이진화 int8 (Nrows, F). 상수열 제거."""
    rng = np.random.default_rng(seed)
    ben = [m for m in members if "benign" in m.lower()]
    gaf = [m for m in members if "gafgyt" in m.lower()][:1]
    mir = [m for m in members if "mirai" in m.lower()][:1]
    pick = ben[:1] + gaf + mir
    if not pick:
        pick = members[:2]
    per = max(rows // len(pick), 1)
    frames = []
    for m in pick:
        with zf.open(m) as f:
            df = pd.read_csv(f)
        if len(df) > per:
            df = df.iloc[rng.choice(len(df), per, replace=False)]
        frames.append(df)
    D = pd.concat(frames, ignore_index=True)
    D = D.select_dtypes(include=[np.number])
    X = D.to_numpy(dtype=np.float64)
    med = np.median(X, axis=0)
    Xb = (X > med).astype(np.int8)
    keep = np.where((Xb.mean(0) > 0.001) & (Xb.mean(0) < 0.999))[0]
    return Xb[:, keep], list(D.columns[keep]), int(len(D)), pick


# ===========================================================================
# 부분집합 선정
# ===========================================================================
def conn3_sum(Xs):
    s = (1 - 2 * Xs.astype(np.float64))
    k = s.shape[1]
    m1 = s.mean(0); m2 = (s.T @ s) / len(s)
    tot = 0.0
    for i, j, l in itertools.combinations(range(k), 3):
        m3 = float(np.mean(s[:, i] * s[:, j] * s[:, l]))
        tot += abs(m3 - (m1[i] * m2[j, l] + m1[j] * m2[i, l] + m1[l] * m2[i, j]) + 2 * m1[i] * m1[j] * m1[l])
    return tot


def select_subset(Xb, method, k, seed):
    mean = Xb.mean(0); var = mean * (1 - mean)
    nonc = np.where((mean > 0) & (mean < 1))[0]
    if method == "highvar":
        return nonc[np.argsort(-var[nonc])][:k].tolist()
    if method.startswith("random"):
        rng = np.random.default_rng(SEED + hash(method) % 9999 + seed)
        return sorted(int(i) for i in rng.choice(nonc, min(k, len(nonc)), replace=False))
    if method == "targeted":
        pool = nonc[np.argsort(-var[nonc])][:40].tolist()
        Xs = Xb[:min(len(Xb), 20000)]
        chosen = pool[:2]
        while len(chosen) < k:
            best = None
            for c in pool:
                if c in chosen:
                    continue
                v = conn3_sum(Xs[:, chosen + [c]])
                if best is None or v > best[0]:
                    best = (v, c)
            chosen.append(best[1])
        return chosen
    raise ValueError(method)


# ===========================================================================
# empirical model + CF (cyclic)
# ===========================================================================
def feature_cycle_model(Xsub):
    """k feature → k-cycle: e[i]=P(f_i, f_{i+1}) 2x2."""
    k = Xsub.shape[1]
    e = np.zeros((k, 2, 2))
    for i in range(k):
        j = (i + 1) % k
        a = Xsub[:, i]; b = Xsub[:, j]
        for av in (0, 1):
            for bv in (0, 1):
                e[i, av, bv] = np.count_nonzero((a == av) & (b == bv))
        e[i] /= max(e[i].sum(), 1)
    return e


def time_cycle_model(col):
    """한 이진 feature 의 행순서(시간대용) 3-cycle: (X_t,X_{t+1}),(X_{t+1},X_{t+2}),(X_t,X_{t+2})."""
    def pair(a, b):
        m = np.zeros((2, 2))
        for av in (0, 1):
            for bv in (0, 1):
                m[av, bv] = np.count_nonzero((a == av) & (b == bv))
        return m / max(m.sum(), 1)
    x0, x1, x2 = col[:-2], col[1:-1], col[2:]
    return np.stack([pair(x0, x1), pair(x1, x2), pair(x0, x2)])


def no_sig_cyclic(e, n):
    """공유변수 marginal 컨텍스트 간 L1(평균)."""
    marg = {v: [] for v in range(n)}
    for i in range(n):
        j = (i + 1) % n
        marg[i].append(e[i].sum(1)); marg[j].append(e[i].sum(0))
    tot = 0.0; cnt = 0
    for v in range(n):
        ms = marg[v]
        for a in range(len(ms)):
            for b in range(a + 1, len(ms)):
                tot += float(np.abs(ms[a] - ms[b]).sum()); cnt += 1
    return tot / cnt if cnt else 0.0


# ===========================================================================
# 셀 실행 (병렬 단위): (device, def, method) → CF + 부트스트랩
# ===========================================================================
def run_cell(args):
    dev, deftype, method, k, boot, seed = args
    Xb = np.load(os.path.join(PROC, dev + ".npy"))
    N = len(Xb)
    rng = np.random.default_rng(SEED + 7 * seed + hash(dev + deftype + method) % 99991)
    if deftype == "a":
        idx = select_subset(Xb, method, k, seed)
        n = len(idx); setup = K._cyclic_setup(n)
        Xs = Xb[:, idx]
        e = feature_cycle_model(Xs)
        cf = K.cyclic_cf(e, n, setup); nsig = no_sig_cyclic(e, n)
        cfs = []
        for _ in range(boot):
            ib = rng.integers(0, N, size=N)
            cfs.append(K.cyclic_cf(feature_cycle_model(Xs[ib]), n, setup))
    else:  # b: 시간창 3-cycle, method=highvar feature 하나
        idx = select_subset(Xb, "highvar", 1, seed)
        col = Xb[:, idx[0]]; n = 3; setup = K._cyclic_setup(3)
        e = time_cycle_model(col)
        cf = K.cyclic_cf(e, 3, setup); nsig = no_sig_cyclic(e, 3)
        cfs = []
        for _ in range(boot):
            # 블록 부트스트랩(행순서 보존 위해 시작점 랜덤 후 원길이 순환표집 대용: 단순 재표집)
            ib = np.sort(rng.integers(0, len(col), size=len(col)))
            cfs.append(K.cyclic_cf(time_cycle_model(col[ib]), 3, setup))
    cfs = np.array(cfs)
    ci = [float(np.percentile(cfs, 2.5)), float(np.percentile(cfs, 97.5))]
    cf_up = ci[1]
    return dict(device=dev, definition=deftype, method=method, k=int(n), N=int(N),
                subset=[int(i) for i in idx], cf=float(cf), cf_ci=ci,
                no_sig_L1=float(nsig), cf_upper=float(cf_up),
                gap_upper=float(C_GATE_E * cf_up**2))


# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="Gate R3 IoT CF 실측. 정박·정량화. 단정 금지.")
    ap.add_argument("--devices", type=int, default=3)
    ap.add_argument("--rows", type=int, default=500000)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(PROC, exist_ok=True)
    if args.smoke:
        args.devices, args.rows, args.boot, args.jobs = 1, 30000, 50, 4

    # self-check (cyclic CF-LP)
    anti = np.array([[0.0, 0.5], [0.5, 0.0]]); corr = np.array([[0.5, 0.0], [0.0, 0.5]])
    s3 = K._cyclic_setup(3)
    scf = K.cyclic_cf(np.stack([anti, anti, anti]), 3, s3)
    scc = K.cyclic_cf(np.stack([corr, corr, corr]), 3, s3)
    log("[self-check cyclic CF-LP] 3-cycle 좌절 CF=%.3f(기대1) 정합 CF=%.3f(기대0)" % (scf, scc))

    # 데이터 준비 (기기별 이진행렬 저장)
    zmd5 = md5(ZIP)
    log("[data] N-BaIoT zip MD5=%s size=%.0fMB" % (zmd5, os.path.getsize(ZIP) / 1048576))
    zf = zipfile.ZipFile(ZIP)
    devs = list_devices(zf)
    chosen = sorted(devs.keys())[:args.devices]
    meta = {}
    for dev in chosen:
        Xb, cols, nrow, pick = load_device(zf, devs[dev], args.rows, SEED)
        np.save(os.path.join(PROC, dev + ".npy"), Xb)
        meta[dev] = dict(rows=nrow, n_features=int(Xb.shape[1]), files=[os.path.basename(p) for p in pick])
        log("  [%s] rows=%d feat=%d files=%s" % (dev, nrow, Xb.shape[1], meta[dev]["files"]))
    zf.close()

    methods = ["highvar", "targeted", "random1", "random2", "random3"]
    tasks = [(dev, "a", m, args.k, args.boot, si) for dev in chosen for si, m in enumerate(methods)]
    tasks += [(dev, "b", "highvar", 3, args.boot, 99) for dev in chosen]
    log("\n[run] %d cells (def-a %d + def-b %d), jobs=%d, boot=%d" %
        (len(tasks), len(chosen) * 5, len(chosen), args.jobs, args.boot))
    t0 = time.time()
    if args.jobs > 1:
        with ProcessPoolExecutor(max_workers=args.jobs) as ex:
            cells = list(ex.map(run_cell, tasks))
    else:
        cells = [run_cell(t) for t in tasks]
    log("[run] %d cells done (%.0fs)" % (len(cells), time.time() - t0))

    for c in cells:
        log("  %-22s def-%s %-8s | CF=%.4f CI[%.4f,%.4f] no-sig=%.4f | gap_upper=%.2e"
            % (c["device"][:22], c["definition"], c["method"], c["cf"], c["cf_ci"][0], c["cf_ci"][1],
               c["no_sig_L1"], c["gap_upper"]))

    all_cf = [c["cf"] for c in cells]
    all_up = [c["cf_upper"] for c in cells]
    max_gap_up = max(c["gap_upper"] for c in cells)
    verdict = dict(
        cf_range=[float(min(all_cf)), float(max(all_cf))],
        cf_upper_max=float(max(all_up)),
        gap_upper_max=float(max_gap_up),
        all_near_zero=bool(max(all_up) < 0.05),
        reading=("전 데이터셋·정의에서 CF≈0 (CI 상단 max=%.4f<0.05) → 실 IoT 텔레메트리는 비맥락. "
                 "잔여 CF 기준 gap 상한 ≤ %.2e (c=1/8 경계근사). = IoT 꺼진 앵커 정박."
                 % (max(all_up), max_gap_up) if max(all_up) < 0.05 else
                 "일부 셀 CF 유의(CI상단 %.4f) — 유한표본/no-sig 잔차 교란 검토 필요(성급 해석 금지)." % max(all_up)),
    )
    payload = dict(
        note=("Gate R3 IoT CF 정박. 진짜 IoT(N-BaIoT) 다기기 CF≈0 실측 + CF²법칙 gap 상한. "
              "★발견 아님(정적결합=CF0 자명)·정박/정량화 목적. cyclic CF-LP=kink 무수정 재사용. supremacy 아님. 단정 금지."),
        params=dict(seed=SEED, rows=args.rows, k=args.k, boot=args.boot, c_gate_e=C_GATE_E,
                    definitions=dict(a="feature-block k-cycle", b="time-window 3-cycle(행순서)")),
        data_source=dict(dataset="N-BaIoT (UCI 442)",
                         url="https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip",
                         zip_md5=zmd5, devices=meta,
                         note_tononbot="TON_IoT/BoT-IoT: CloudStor(AARNET) 폐지로 접근 불가 → N-BaIoT 다기기로 진행(정직)."),
        self_check=dict(cyclic3_frustrated=scf, cyclic3_consistent=scc),
        cells=cells, verdict=verdict,
        limits=("N-BaIoT 다기기(단일 저장소; TON/BoT 접근불가)·행샘플 ≤%d·중앙값 이진화·부분집합 선택 의존. "
                "정적 결합분포는 이론상 CF=0(측정=정박/정량화). (a)(b)는 조작적 구성. c=1/8 경계근사(CF≤0.1). "
                "관측/앵커이지 quantum supremacy 아님." % args.rows),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    log("[판정] %s" % verdict["reading"])
    try:
        plot(payload)
    except Exception as e:
        log("[plot WARN] %r" % e)


def plot(P):
    cells = P["cells"]
    labels = ["%s\n%s/%s" % (c["device"][:10], c["definition"], c["method"][:4]) for c in cells]
    cf = np.array([c["cf"] for c in cells])
    lo = np.array([c["cf_ci"][0] for c in cells]); hi = np.array([c["cf_ci"][1] for c in cells])
    x = np.arange(len(cells))
    fig, ax = plt.subplots(figsize=(max(11, len(cells) * 0.5), 6))
    ax.bar(x, cf, color="#3B6EA5", alpha=.8)
    ax.errorbar(x, cf, yerr=[cf - lo, hi - cf], fmt="none", ecolor="#C1442E", capsize=2)
    ax.axhline(0.05, color="gray", ls="--", lw=1, label="CF=0.05 (≈0 기준)")
    ax.axhline(P["verdict"]["cf_upper_max"], color="#C1442E", ls=":", lw=1,
               label="max CI upper=%.4f" % P["verdict"]["cf_upper_max"])
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=6, rotation=90)
    ax.set_ylabel("contextual fraction  CF (with 95% CI)")
    ax.set_ylim(0, max(0.1, P["verdict"]["cf_upper_max"] * 1.3))
    ax.set_title("Gate R3 — real IoT (N-BaIoT) CF ≈ 0  [OFF anchor]\ngap upper bound ≤ c·CF²_upper = %.1e (c=1/8, boundary approx). NOT supremacy."
                 % P["verdict"]["gap_upper_max"])
    ax.legend(fontsize=8); ax.grid(alpha=.3, axis="y")
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=130); plt.close()
    log("[저장] %s" % OUT_PNG)


if __name__ == "__main__":
    main()
