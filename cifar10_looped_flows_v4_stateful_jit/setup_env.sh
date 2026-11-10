#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
uv venv .venv --python 3.12 --allow-existing
uv pip install --python .venv/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .venv/bin/python -r cifar10_looped_flows_v4_stateful_jit/requirements.txt
# Compile a private copy for this GPU; preserve the original TRM vendor sources.
.venv/bin/python - <<'PY'
from pathlib import Path
import shutil
import torch
path = Path('.deps/adam_atan2')
shutil.copytree('trm/vendor_optimizer', path, dirs_exist_ok=True)
setup = path / 'setup.py'
arch = ''.join(map(str, torch.cuda.get_device_capability()))
setup.write_text(setup.read_text().replace('{"120"}', '{"'+arch+'"}'))
PY
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda}" MAX_JOBS=2 uv pip install --python .venv/bin/python --no-build-isolation .deps/adam_atan2
