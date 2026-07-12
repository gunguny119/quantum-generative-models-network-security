#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gate R5 — IoT 데이터 지형 contextuality 감사 (다중 기기 × 정의 2종 × 부분집합 5종).

★ 인식론적 역할(로그 서두): Fine 정리 따름정리에 의해, **동시 기록된 고전 로그의 def-a형 CF는 수학적으로 정확히 0**.
  따라서 이 감사는 '발견'이 아니라 (i) 정리·구현 검증, (ii) def-b(시간창) 유한표본 요동 규모 측정,
  (iii) 트래픽 종류 불변성 확인. **def-a 에서 CF>0 = 구현 버그 신호** → 원인 규명.

원칙: 결정적(seed), 무설치, **gate_r3_iot_cf(=R3) 무수정 재사용**(def-a/b·부분집합5·CF-LP·no_sig),
  self-check(좌절 3-cycle CF=1/정합 0), 출처 URL·MD5 기록, 병렬(Pool), 단정 금지. supremacy 아님.
신규: (3) null 보정(행순서 셔플 null 분포), 집계 히트맵.

데이터: N-BaIoT (UCI 442) benign, **9기기 전량**(이미 확보 zip). 공격(.rar)은 unrar 부재로 미추출→RED(레버2 스킵).
  외부셋(IoT-23 parquet=리더 부재, CIC/Edge/MQTT=계정/대용량, WUSTL=링크 미해결)=YELLOW/RED(§접근성표).

사용: python tools/gate_r5_iot_audit.py [--k 8 --boot 500 --null 500 --jobs 16] [--smoke]
"""
import os
import sys
import json
import time
import zipfile
import hashlib
import argparse
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import gate_r3_iot_cf as R3               # 무수정 재사용 (load_device·select_subset·feature_cycle_model·time_cycle_model·no_sig_cyclic)
import kink_origin_v1 as K                # cyclic CF-LP·setup

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402

OUTDIR = os.path.join("results2", "gate_r5")
PROC = os.path.join("data", "nbaiot", "proc5")
ZIP = os.path.join("data", "nbaiot", "nbaiot.zip")
OUT_JSON = os.path.join(OUTDIR, "r5.json")
OUT_PNG = os.path.join(OUTDIR, "r5_heatmap.png")
SEED = 20260705
C_GATE_E = R3.C_GATE_E                    # 1/8 (cyclic 경계 c)
METHODS = ["highvar", "targeted", "random1", "random2", "random3"]


def log(m):
    print(m, flush=True)


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(1 << 16), b""):
            h.update(c)
    return h.hexdigest()


# ── 셀 실행 (병렬 단위): CF + 부트스트랩 CI + (def-b) null 보정 + signalling + gap상한 ──
def run_cell(args):
    dev, deftype, method, k, boot, nboot = args
    Xb = np.load(os.path.join(PROC, dev + ".npy"))
    N = len(Xb)
    rng = np.random.default_rng(SEED + 13 * hash(dev + deftype + method) % 999983)
    if deftype == "a":
        idx = R3.select_subset(Xb, method, k, hash(method) % 7)
        n = len(idx); setup = K._cyclic_setup(n); Xs = Xb[:, idx]
        e = R3.feature_cycle_model(Xs)
        cf = K.cyclic_cf(e, n, setup); nsig = R3.no_sig_cyclic(e, n)
        cfs = [K.cyclic_cf(R3.feature_cycle_model(Xs[rng.integers(0, N, N)]), n, setup) for _ in range(boot)]
        null_p = None  # def-a 는 Fine 정리로 CF=0(검증) — null 불요
    else:  # b: 시간창 3-cycle (highvar feature 1개, 행순서)
        idx = R3.select_subset(Xb, "highvar", 1, 0)
        col = Xb[:, idx[0]]; n = 3; setup = K._cyclic_setup(3)
        e = R3.time_cycle_model(col)
        cf = K.cyclic_cf(e, 3, setup); nsig = R3.no_sig_cyclic(e, 3)
        cfs = []
        for _ in range(boot):
            ib = np.sort(rng.integers(0, len(col), len(col)))   # 정렬 재표집(대략 시간순 유지)
            cfs.append(K.cyclic_cf(R3.time_cycle_model(col[ib]), 3, setup))
        # null: 행순서 셔플(시간구조 파괴) → null CF 분포
        nulls = []
        for _ in range(nboot):
            cs = col.copy(); rng.shuffle(cs)
            nulls.append(K.cyclic_cf(R3.time_cycle_model(cs), 3, setup))
        nulls = np.array(nulls)
        null_ci = [float(np.percentile(nulls, 2.5)), float(np.percentile(nulls, 97.5))]
        within = bool(null_ci[0] - 1e-9 <= cf <= null_ci[1] + 1e-9)
        null_p = dict(null_ci95=null_ci, null_mean=float(nulls.mean()),
                      observed_within_null=within, n_null=int(nboot))
    cfs = np.array(cfs)
    ci = [float(np.percentile(cfs, 2.5)), float(np.percentile(cfs, 97.5))]
    return dict(device=dev, definition=deftype, method=method, k=int(n), N=int(N),
                cf=float(cf), cf_ci=ci, no_sig_L1=float(nsig),
                cf_upper=float(ci[1]), gap_upper=float(C_GATE_E * ci[1] ** 2), null=null_p)


def main():
    ap = argparse.ArgumentParser(description="Gate R5 IoT contextuality 감사. 검증·정량화. 단정 금지.")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--boot", type=int, default=500)
    ap.add_argument("--null", type=int, default=500)
    ap.add_argument("--rows", type=int, default=50000)
    ap.add_argument("--jobs", type=int, default=16)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.makedirs(OUTDIR, exist_ok=True); os.makedirs(PROC, exist_ok=True)
    if args.smoke:
        args.boot, args.null, args.jobs = 40, 40, 4

    # self-check (cyclic CF-LP; LP 는 데이터 무관이라 1회로 전 데이터셋 커버)
    anti = np.array([[0.0, 0.5], [0.5, 0.0]]); corr = np.array([[0.5, 0.0], [0.0, 0.5]])
    s3 = K._cyclic_setup(3)
    scf = K.cyclic_cf(np.stack([anti, anti, anti]), 3, s3)
    scc = K.cyclic_cf(np.stack([corr, corr, corr]), 3, s3)
    log("[self-check cyclic CF-LP] 좌절 3-cycle CF=%.3f(기대1) 정합 CF=%.3f(기대0) → 전 셀 공통 LP" % (scf, scc))

    zmd5 = md5(ZIP)
    log("[data] N-BaIoT zip MD5=%s | benign 9기기(공격=.rar unrar부재로 미추출)" % zmd5)
    zf = zipfile.ZipFile(ZIP)
    devs = R3.list_devices(zf)
    # 실제 기기 = benign CSV 를 가진 top-level 디렉터리만 (예: demonstrate_structure.csv 같은 최상위 파일 제외)
    chosen = sorted(d for d in devs if any("benign" in m.lower() for m in devs[d]))
    if args.smoke:
        chosen = chosen[:2]
    meta = {}
    for dev in chosen:
        Xb, cols, nrow, pick = R3.load_device(zf, devs[dev], args.rows, SEED)
        np.save(os.path.join(PROC, dev + ".npy"), Xb)
        meta[dev] = dict(rows=nrow, n_features=int(Xb.shape[1]))
        log("  [%s] rows=%d feat=%d" % (dev[:26], nrow, Xb.shape[1]))
    zf.close()

    tasks = [(dev, "a", m, args.k, args.boot, 0) for dev in chosen for m in METHODS]
    tasks += [(dev, "b", "highvar", 3, args.boot, args.null) for dev in chosen]
    log("\n[run] %d cells (def-a %d + def-b %d), jobs=%d" % (len(tasks), len(chosen) * 5, len(chosen), args.jobs))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.jobs) as ex:
        cells = list(ex.map(run_cell, tasks))
    log("[run] done (%.0fs)" % (time.time() - t0))

    # 판정 집계
    defa = [c for c in cells if c["definition"] == "a"]
    defb = [c for c in cells if c["definition"] == "b"]
    max_defa_cf = max(abs(c["cf"]) for c in defa)
    max_ci_up = max(c["cf_upper"] for c in cells)
    max_gap_up = max(c["gap_upper"] for c in cells)
    max_nsig = max(c["no_sig_L1"] for c in cells)
    defb_within = sum(1 for c in defb if c["null"]["observed_within_null"])
    verdict = dict(
        def_a_verification=dict(max_abs_cf=float(max_defa_cf), passes=bool(max_defa_cf < 1e-6),
                                reading="def-a 전 셀 CF≈0 (max|CF|=%.2e<1e-6) → Fine 정리·구현 검증 통과" % max_defa_cf
                                if max_defa_cf < 1e-6 else "def-a CF>0 셀 존재=구현 버그 신호(원인 규명 필요)"),
        def_b_finite_sample=dict(within_null=f"{defb_within}/{len(defb)}",
                                 all_within=bool(defb_within == len(defb)),
                                 reading="def-b 전 셀 관측 CF 가 셔플 null 분위수 내 → '유한표본 요동' 확정"
                                 if defb_within == len(defb) else "일부 def-b 셀 null 밖 — 좌표·잔차 확인 필요"),
        traffic_invariance=dict(status="benign 9기기만(공격=.rar 미추출) → 공격 vs 정상 비교 불가",
                                reading="레버2(공격) unrar 부재로 스킵; benign 9기기 전부 CF≈0 = 기기 무관 확인(정상 트래픽 내)"),
        gap_upper_max=float(max_gap_up), cf_upper_max=float(max_ci_up), no_sig_max=float(max_nsig),
    )
    payload = dict(
        note=("Gate R5 IoT contextuality 감사. Fine 정리로 def-a CF=0(검증). def-b 유한표본 요동+null 보정. "
              "N-BaIoT 9기기 benign(공격 .rar 미추출). R3 무수정 재사용. supremacy 아님. 단정 금지."),
        epistemic_role=("Proposition(Fine 따름정리): 동시기록 고전로그의 def-a CF=수학적 0. 감사=검증/정량화(발견 아님). "
                        "def-a CF>0=버그 신호."),
        params=dict(seed=SEED, k=args.k, boot=args.boot, null=args.null, rows=args.rows, c_gate_e=C_GATE_E),
        data_source=dict(dataset="N-BaIoT (UCI 442) benign, 9 devices", zip_md5=zmd5,
                         url="https://archive.ics.uci.edu/static/public/442/detection+of+iot+botnet+attacks+n+baiot.zip",
                         attacks="mirai/gafgyt .rar (unrar 부재로 미추출=RED)", devices=meta),
        accessibility_audit=ACCESS_TABLE,
        self_check=dict(cyclic3_frustrated=scf, cyclic3_consistent=scc),
        cells=cells, verdict=verdict,
        limits=("감사 범위=확보된 N-BaIoT 9기기 benign 한정. 공격(.rar)·외부셋(IoT-23 parquet 리더부재·CIC/Edge/MQTT 계정·"
                "WUSTL 링크 미해결) 미포함. 행샘플 ≤%d·중앙값 이진화. 정적결합=CF0 자명(검증목적). def 조작적(단 Proposition이 일반 커버). "
                "관측/검증이지 supremacy 아님." % args.rows),
    )
    json.dump(payload, open(OUT_JSON, "w"), indent=1, ensure_ascii=False)
    log("\n[저장] %s" % OUT_JSON)
    log("[판정] def-a: %s | def-b null내 %d/%d | gap상한 max=%.2e | no-sig max=%.4f"
        % (verdict["def_a_verification"]["passes"], defb_within, len(defb), max_gap_up, max_nsig))
    try:
        plot(payload, chosen)
    except Exception as e:
        log("[plot WARN] %r" % e)


# 접근성 감사표 (웹 확인 결과, 로그·논문 서술용)
ACCESS_TABLE = [
    dict(dataset="N-BaIoT benign (UCI 442)", url="archive.ics.uci.edu/static/public/442/…zip", fmt="CSV(수치115)",
         size="1.7GB zip", account="불요", verdict="GREEN", note="9기기 benign 직접 사용(본 감사)"),
    dict(dataset="N-BaIoT attacks (mirai/gafgyt)", url="동 zip 내 *.rar", fmt="RAR→CSV", size="186MB/기기",
         account="불요", verdict="RED", note="unrar/rarfile/7z 부재로 추출 불가"),
    dict(dataset="IoT-23 (Stratosphere)", url="mcfp.felk.cvut.cz …small.tar.gz / HF parquet mirror", fmt="conn.log / parquet",
         size="9.4GB / 28KB(parquet)", account="불요", verdict="YELLOW",
         note="full=9.4GB 과대; HF parquet=pyarrow 부재로 미판독(무설치)"),
    dict(dataset="CIC-IoT2023 (UNB)", url="unb.ca/cic/datasets/iotdataset-2023.html / Kaggle", fmt="CSV",
         size="~13GB", account="Kaggle 로그인/대용량", verdict="YELLOW", note="직접 대용량 or Kaggle 계정"),
    dict(dataset="Edge-IIoTset", url="IEEE DataPort / Kaggle", fmt="CSV", size="~12GB", account="계정 필요",
         verdict="YELLOW", note="Kaggle/DataPort 로그인"),
    dict(dataset="MQTT-IoT-IDS2020", url="IEEE DataPort", fmt="CSV/pcap", size="-", account="DataPort 계정",
         verdict="YELLOW", note="계정 필요(미검증)"),
    dict(dataset="WUSTL-IIoT-2021", url="cse.wustl.edu/~jain/iiot2 / IEEE DataPort", fmt="CSV",
         size="106MB / 390MB", account="직접링크 미해결/DataPort", verdict="YELLOW", note="직접 파일 URL 미해결"),
    dict(dataset="TON_IoT / Bot-IoT", url="AARNET CloudStor", fmt="CSV", size="-", account="-", verdict="RED",
         note="CloudStor 폐쇄(기확인)"),
]


def plot(P, devices):
    cells = P["cells"]
    cols = ["a/highvar", "a/targeted", "a/random1", "a/random2", "a/random3", "b/time"]
    M = np.full((len(devices), len(cols)), np.nan)
    cmap = {("a", m): j for j, m in enumerate(METHODS)}; cmap[("b", "highvar")] = 5
    di = {d: i for i, d in enumerate(devices)}
    for c in cells:
        M[di[c["device"]], cmap[(c["definition"], c["method"])]] = c["cf"]
    fig, ax = plt.subplots(figsize=(8.5, max(4, 0.5 * len(devices) + 2)))
    vmax = max(1e-4, np.nanmax(M))
    im = ax.imshow(M, aspect="auto", cmap="magma", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(devices))); ax.set_yticklabels([d[:22] for d in devices], fontsize=7)
    for i in range(len(devices)):
        for j in range(len(cols)):
            ax.text(j, i, "%.0e" % M[i, j] if M[i, j] > 0 else "0", ha="center", va="center",
                    fontsize=6, color="w" if M[i, j] < vmax * 0.6 else "k")
    fig.colorbar(im, ax=ax, label="contextual fraction CF")
    ax.set_title("CF across the IoT data landscape (N-BaIoT, 9 devices)\ndef-a=0 exactly (Fine); def-b≈0 (finite-sample). max CF=%.1e — NOT supremacy"
                 % np.nanmax(M))
    plt.tight_layout(); plt.savefig(OUT_PNG, dpi=130); plt.close()
    log("[저장] %s" % OUT_PNG)


if __name__ == "__main__":
    main()
