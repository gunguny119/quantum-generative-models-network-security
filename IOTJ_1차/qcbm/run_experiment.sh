#!/bin/bash
# QCBM 실험 백그라운드 런처 (tmux 세션 안에서 실행됨)
set -u
cd /home/elicer/IOTJ/qcbm

# 사용자 pip --user 설치 경로
export PATH="$HOME/.local/bin:$PATH"

# 프로세스 병렬(재시작 16개) 방식이므로 워커는 단일 스레드.
# -> 16개 프로세스가 16코어를 꽉 채운다 (오버서브스크립션 방지).
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "######################################################################"
echo "# QCBM 실험 시작: $(date)"
echo "# host=$(hostname)  cores=$(nproc)"
echo "######################################################################"

# -u : 버퍼링 없이 즉시 로그. tee 로 화면+파일 동시 기록.
python3 -u experiment.py \
    --restarts 16 \
    --epochs 300 \
    --layers 3 \
    --lr 0.1 \
    --log-every 25 \
    --outdir results \
    2>&1 | tee -a run.log

echo "######################################################################"
echo "# QCBM 실험 종료: $(date)  exit=${PIPESTATUS[0]}"
echo "######################################################################"
