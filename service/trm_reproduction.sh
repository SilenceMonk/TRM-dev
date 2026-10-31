#!/bin/bash
set -eo pipefail
. /opt/supervisor-scripts/utils/logging.sh
. /opt/supervisor-scripts/utils/environment.sh
set -u
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=3
export MKL_NUM_THREADS=3
export TORCHINDUCTOR_COMPILE_THREADS=2
cd /workspace/project
/venv/main/bin/python trm/data.py
exec /venv/main/bin/python -u trm/train.py --micro-batch 256 --compile full --resume
