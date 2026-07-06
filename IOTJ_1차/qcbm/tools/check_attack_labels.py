#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
현재 CSV의 공격 라벨 종류 확인 (학습 없음, 코드 수정 없음)
==========================================================
data/cic_thursday.csv 의 Label 컬럼만 읽어, 원본과 strip/lower 적용 후의
고유값/개수를 출력한다. DDoS/DoS/PortScan/BruteForce 등 강분리 공격 후보가
있는지 확인한다. QCBM/학습은 실행하지 않는다.
"""
import os
import sys
import pandas as pd

QCBM_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(QCBM_DIR)
DATA = "data/cic_thursday.csv"

STRONG_KEYS = ("ddos", "dos", "portscan", "port scan", "bruteforce",
               "brute force", "brute-force", "bot", "web", "sql", "xss")


def main():
    if not os.path.exists(DATA):
        print("CSV 없음: %s -> 실행 불가." % DATA)
        sys.exit(1)
    s = pd.read_csv(DATA, usecols=["Label"], low_memory=False)["Label"]
    print("=== RAW value_counts ===")
    print(s.value_counts(dropna=False))
    norm = s.astype(str).str.strip().str.lower()
    print("=== strip/lower value_counts ===")
    print(norm.value_counts())

    labels = set(norm.unique())
    hits = sorted(l for l in labels
                  if any(k in l for k in STRONG_KEYS))
    print("=== 강분리 공격 후보(ddos/dos/portscan/bruteforce/bot/web...) ===")
    print(hits if hits else "없음 (benign/infilteration 외 강분리 공격 라벨 없음)")
    print("DONE")


if __name__ == "__main__":
    main()
