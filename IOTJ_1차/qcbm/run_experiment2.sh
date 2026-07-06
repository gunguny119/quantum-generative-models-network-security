#!/bin/bash
# QCBM 확장 실험2 런처 (6 vs 8 큐비트 + 이상탐지). tmux 세션에서 실행.
set -u
cd /home/elicer/IOTJ/qcbm
export PATH="$HOME/.local/bin:$PATH"
# 프로세스 병렬(재시작16) -> 워커당 단일 스레드
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "######################################################################"
echo "# QCBM 확장 실험2 시작: $(date)  host=$(hostname) cores=$(nproc)"
echo "######################################################################"

python3 -u experiment2.py 2>&1 | tee -a run2.log

echo "######################################################################"
echo "# 종료: $(date)  exit=${PIPESTATUS[0]}"
echo "######################################################################"
