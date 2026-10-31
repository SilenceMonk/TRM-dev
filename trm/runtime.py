"""Configuration and provenance shared by training and evaluation."""
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / '.deps'))
sys.path.insert(0, str(ROOT / '.deps-cuda'))
sys.path.insert(0, str(Path(__file__).parent / 'vendor'))
SOURCE_SHA = 'c01103738605ba39d1430519b1ee0c62f4c707f8'
DATA_SHA = '58942f96baeb572ca3127e2a9e9c70f330783d6b'

def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(2**20), b''): h.update(b)
    return h.hexdigest()

def atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    os.replace(tmp, path)

def atomic_save(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    torch.save(obj, tmp)
    os.replace(tmp, path)

@dataclasses.dataclass
class Config:
    seed: int = 0
    effective_batch: int = 768
    micro_batch: int = 256
    epochs: int = 50000
    epochs_per_window: int = 5000
    lr: float = 1e-4
    warmup: int = 2000
    weight_decay: float = 1.0
    beta1: float = .9
    beta2: float = .95
    ema_rate: float = .999
    save_every: int = 250
    val_every: int = 6510
    log_every: int = 10
    eval_batch: int = 256
    compile_mode: str = 'full'
    data_dir: str = str(ROOT / 'data')
    source_sha: str = SOURCE_SHA
    data_sha: str = DATA_SHA

    @property
    def total_steps(self):
        # Match actual upstream loop: 10 windows, each drops its last partial batch.
        return (self.epochs // self.epochs_per_window) * (self.epochs_per_window * 1000 // self.effective_batch)


def model_config(batch_size):
    return dict(batch_size=batch_size, seq_len=81, vocab_size=11,
                num_puzzle_identifiers=1, puzzle_emb_ndim=512, puzzle_emb_len=16,
                H_cycles=3, L_cycles=6, H_layers=0, L_layers=2, hidden_size=512,
                expansion=4., num_heads=8, pos_encodings='none', mlp_t=True,
                halt_max_steps=16, halt_exploration_prob=.1,
                no_ACT_continue=True, forward_dtype='bfloat16')


def setup_device():
    torch.set_num_threads(3)
    torch.backends.cuda.matmul.allow_tf32 = True
    if not torch.cuda.is_available(): raise RuntimeError('CUDA GPU required')
    return torch.device('cuda')


def build_model(batch_size, compile_mode='none', loss=True):
    from model import TinyRecursiveReasoningModel_ACTV1
    from models.losses import ACTLossHead
    with torch.device('cuda'):
        core = TinyRecursiveReasoningModel_ACTV1(model_config(batch_size))
    model = ACTLossHead(core, loss_type='stablemax_cross_entropy') if loss else core
    if compile_mode == 'block':
        # Compile the shared reasoning module once per grad mode, avoid unrolling
        # 21 recurrent calls into a huge compilation graph.
        core.inner.L_level.compile(dynamic=False)
    elif compile_mode == 'full':
        model.compile(dynamic=False)
    elif compile_mode != 'none': raise ValueError(compile_mode)
    return model
