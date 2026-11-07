#!/bin/bash
set -eo pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TORCHINDUCTOR_COMPILE_THREADS=2
cd /workspace/TRM-dev
exec /venv/main/bin/python -u cifar10_looped_flows/train.py \
  --run runs/cifar10_looped_flows/seed0 --batch-size 640 \
  --steps 100000 --save-every 1000 --eval-every 5000 --val-images 5000 \
  --compile --resume
