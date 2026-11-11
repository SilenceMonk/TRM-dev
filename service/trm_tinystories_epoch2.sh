#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
exec .venv/bin/python -u trm_tinystories/train.py \
  --run runs/trm_tinystories/d256_seed0_epoch2 \
  --init-checkpoint runs/trm_tinystories/d256_seed0/latest.pt \
  --batch-size 224 --micro-batch 224 --seq-len 256 --steps 0 \
  --lr 3e-5 --warmup 1 --eval-every 100 --full-eval-every 1000 \
  --eval-tokens 262144 --eval-batch 32 --save-every 100 --compile --resume
