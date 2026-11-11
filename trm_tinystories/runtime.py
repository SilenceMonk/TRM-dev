"""Shared configuration and atomic experiment artifacts."""
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / 'data/tinystories'


@dataclass
class ModelConfig:
    hidden_size: int = 256
    num_heads: int = 4
    layers: int = 2
    expansion: float = 4.
    H_cycles: int = 3
    L_cycles: int = 6
    seq_len: int = 256
    vocab_size: int = 8192


def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n')
    os.replace(tmp, path)


def atomic_save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def setup():
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')
