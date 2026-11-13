#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 TORCHINDUCTOR_COMPILE_THREADS=2
exec .venv/bin/python -u trm_tinystories/train_act.py \
  --run runs/trm_tinystories/d256_seed0_act16_ema \
  --batch-size 192 --micro-batch 192 --seq-len 256 --halt-max-steps 16 --seed 0 \
  --lr 3e-4 --warmup 200 --ema-decay 0.999 \
  --patience 3 --min-delta 0.001 --min-epochs 3 --max-epochs 0 \
  --lr-factor 0.5 --min-lr 1e-6 \
  --eval-every 20 --full-eval-every 1000 --eval-tokens 65536 --eval-batch 32 \
  --save-every 10 --log-every 1 --compile --resume
