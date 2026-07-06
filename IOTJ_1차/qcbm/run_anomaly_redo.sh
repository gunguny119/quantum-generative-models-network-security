#!/bin/bash
# 이상탐지 재평가 런처: 얕은(L=3) vs 제대로학습(L=6) 8큐비트 QCBM. tmux 세션에서 실행.
set -u
cd /home/elicer/IOTJ/qcbm
export PATH="$HOME/.local/bin:$PATH"
# 프로세스 병렬(재시작16) -> 워커당 단일 스레드
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "######################################################################"
echo "# 이상탐지 재평가(L=3 vs L=6) 시작: $(date)  host=$(hostname) cores=$(nproc)"
echo "######################################################################"

python3 -u anomaly_redo.py 2>&1 | tee -a anomaly_redo.log

echo "######################################################################"
echo "# 종료: $(date)  exit=${PIPESTATUS[0]}"
echo "######################################################################"
