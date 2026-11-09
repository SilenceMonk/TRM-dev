#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")/.."
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 TORCHINDUCTOR_COMPILE_THREADS=2
exec .venv/bin/python -u cifar10_looped_flows_v3_jit/train.py \
  --run runs/cifar10_looped_flows_v3_jit/seed0_gamma_only --batch-size 768 \
  --steps 100000 --save-every 1000 --eval-every 5000 --val-images 5000 \
  --p-mean -0.8 --p-std 0.8 --t-eps 0.05 --compile --resume
