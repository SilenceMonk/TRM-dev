#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
exec .venv/bin/python -u trm_tinystories/train_ema.py \
  --run runs/trm_tinystories/d256_seed0_ema \
  --batch-size 224 --micro-batch 224 --seq-len 256 --seed 0 \
  --lr 3e-4 --warmup 200 --ema-decay 0.999 \
  --patience 3 --min-delta 0.001 --min-epochs 3 --max-epochs 0 \
  --lr-factor 0.5 --min-lr 1e-6 \
  --eval-every 100 --full-eval-every 1000 --eval-tokens 262144 --eval-batch 32 \
  --save-every 100 --compile --resume
