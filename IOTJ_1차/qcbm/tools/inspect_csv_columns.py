#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CSE-CIC-IDS2018 CSV 컬럼 사전 조사 — feature 확장 후보 발굴 (학습 없음)
=====================================================================
목적:
  현재 QCBM 8비트 인코딩(proto2+syn1+psh1+size4)의 4 feature로는 Infiltration AUC가
  0.43(분포기반 상한도 0.43)으로, "이 4 feature에 공격을 가를 정보가 없음"이 확정됨.
  feature를 10~12비트로 늘리기 전에, CSV의 어떤 컬럼이 benign vs infilteration을 잘 가르고
  이산화에 적합한지(고유값 수/분포)를 조사한다.

방법(학습 없음, 통계만):
  - 전체 컬럼 dtype / nunique / 결측·무한 비율
  - 수치형 컬럼: benign vs attack 그룹의 mean/std/median
  - 분리지표 = |mean_ben - mean_atk| / (pooled_std + eps)  (표준화 평균차; 후보발굴용)
  - 이산화 부적합(nunique<=2 또는 한 값 99%+) 표시
QCBM/학습은 절대 실행하지 않는다.
"""
import os
import sys
import json

import numpy as np
import pandas as pd

# experiment2 와 동일한 데이터 경로/라벨 규칙을 쓰기 위해 qcbm 루트로 chdir
QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(QCBM_DIR)

DATA = "data/cic_thursday.csv"          # experiment2.DATA 와 동일
OUT = "results2"
EPS = 1e-9
TOP_N = 20


def log(msg):
    print(msg, flush=True)


def main():
    if not os.path.exists(DATA):
        log("CSV 없음: %s -> 실행 불가. CSV가 있어야 한다." % DATA)
        sys.exit(1)
    os.makedirs(OUT, exist_ok=True)

    log("CSV 로드(전체 컬럼): %s" % DATA)
    df = pd.read_csv(DATA, low_memory=False)
    # 컬럼명 공백 strip
    df.columns = [str(c).strip() for c in df.columns]
    n_rows, n_cols = df.shape
    log("행=%d  컬럼=%d" % (n_rows, n_cols))

    if "Label" not in df.columns:
        log("'Label' 컬럼 없음. 실제 헤더: %s" % list(df.columns))
        sys.exit(1)

    # experiment2.load_frames 와 동일한 라벨 처리
    lab = df["Label"].astype(str).str.strip().str.lower()
    m_ben = (lab == "benign").to_numpy()
    m_atk = (lab == "infilteration").to_numpy()
    label_counts = lab.value_counts().to_dict()
    log("라벨 분포: %s" % {k: int(v) for k, v in label_counts.items()})
    log("benign=%d  infilteration=%d" % (int(m_ben.sum()), int(m_atk.sum())))

    rows = []
    for col in df.columns:
        s = df[col]
        dtype = str(s.dtype)
        nun = int(s.nunique(dropna=True))
        n = len(s)
        # 최빈값 점유율
        if nun > 0:
            top_frac = float(s.value_counts(dropna=True).iloc[0] / max(1, s.notna().sum()))
        else:
            top_frac = float("nan")

        # 수치 변환 시도 (errors=coerce). 실제 수치형이면 통계 계산.
        num = pd.to_numeric(s, errors="coerce")
        n_numeric = int(num.notna().sum())
        is_numeric = n_numeric > 0.5 * n          # 절반 이상 수치 변환되면 수치형 취급
        na_frac = float(num.isna().mean()) if is_numeric else float(s.isna().mean())
        inf_frac = float(np.isinf(num.to_numpy(dtype=float, na_value=np.nan)).mean()) \
            if is_numeric else 0.0

        rec = {"column": col, "dtype": dtype, "nunique": nun,
               "na_frac": round(na_frac, 6), "inf_frac": round(inf_frac, 6),
               "top_value_frac": round(top_frac, 6), "is_numeric": is_numeric}

        if is_numeric:
            v = num.to_numpy(dtype=float)
            v_fin = np.where(np.isinf(v), np.nan, v)     # inf는 통계에서 제외
            vb = v_fin[m_ben]; va = v_fin[m_atk]
            vb = vb[~np.isnan(vb)]; va = va[~np.isnan(va)]
            if len(vb) > 1 and len(va) > 1:
                mb, sb, medb = float(np.mean(vb)), float(np.std(vb)), float(np.median(vb))
                ma, sa, meda = float(np.mean(va)), float(np.std(va)), float(np.median(va))
                pooled = float(np.sqrt((sb * sb + sa * sa) / 2.0))
                sep = abs(mb - ma) / (pooled + EPS)
                rec.update({"mean_ben": mb, "std_ben": sb, "median_ben": medb,
                            "mean_atk": ma, "std_atk": sa, "median_atk": meda,
                            "pooled_std": pooled, "separation": float(sep)})
            else:
                rec.update({"separation": None})
        else:
            rec.update({"separation": None})

        # 이산화 부적합 판정
        rec["disc_unfit"] = bool(nun <= 2 or (top_frac == top_frac and top_frac >= 0.99))
        rows.append(rec)

    rep = pd.DataFrame(rows)

    # ---- 분리력 순위 ----
    ranked = rep[rep["separation"].notna()].sort_values(
        "separation", ascending=False).reset_index(drop=True)

    log("=" * 100)
    log("[분리력 상위 %d] |mean_ben-mean_atk|/(pooled_std+eps)" % TOP_N)
    log("  %-32s %-9s %-9s %-13s %-13s %-8s" %
        ("column", "dtype", "nunique", "mean_ben", "mean_atk", "sep"))
    for _, r in ranked.head(TOP_N).iterrows():
        log("  %-32s %-9s %-9d %-13.4g %-13.4g %-8.4f" %
            (r["column"][:32], r["dtype"], int(r["nunique"]),
             r["mean_ben"], r["mean_atk"], r["separation"]))

    unfit = rep[rep["disc_unfit"]]["column"].tolist()
    log("=" * 100)
    log("[이산화 부적합] nunique<=2 또는 한 값 99%%+ : %d개" % len(unfit))
    log("  %s" % unfit)

    # ---- 저장 ----
    out_json = os.path.join(OUT, "csv_columns_report.json")
    out_csv = os.path.join(OUT, "csv_columns_report.csv")
    payload = {
        "data": DATA, "n_rows": int(n_rows), "n_cols": int(n_cols),
        "columns": list(df.columns),
        "label_counts": {k: int(v) for k, v in label_counts.items()},
        "n_benign": int(m_ben.sum()), "n_attack": int(m_atk.sum()),
        "separation_metric": "|mean_ben-mean_atk|/(pooled_std+eps)",
        "top_separation": ranked.head(TOP_N).to_dict(orient="records"),
        "disc_unfit_columns": unfit,
        "all_columns": rows,
    }
    with open(out_json, "w") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    # csv는 분리력 정렬된 전체 컬럼 통계
    rep.sort_values("separation", ascending=False, na_position="last").to_csv(
        out_csv, index=False)
    log("저장: %s , %s" % (out_json, out_csv))
    log("DONE")


if __name__ == "__main__":
    main()
