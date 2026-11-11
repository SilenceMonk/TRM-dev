"""TRM shared post-norm attention/SwiGLU, adapted to causal next-token prediction.

Original recurrence: H=3, L=6, two shared blocks. Only the final H round
has gradients. Each independent token window starts with fixed h/l buffers;
no future token, target, or previous minibatch state enters the recurrence.
"""
import math
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trm/vendor'))
from models.common import trunc_normal_init_
from models.layers import Attention, SwiGLU, CastedEmbedding, RotaryEmbedding, rms_norm
try:
    from .runtime import ModelConfig
except ImportError:
    from runtime import ModelConfig


class AttentionBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = Attention(cfg.hidden_size, cfg.hidden_size // cfg.num_heads,
                                   cfg.num_heads, cfg.num_heads, causal=True)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion)

    def forward(self, x, cos_sin):
        x = rms_norm(x + self.attention(cos_sin, x), 1e-5)
        return rms_norm(x + self.mlp(x), 1e-5)


class ReasoningModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(cfg) for _ in range(cfg.layers)])

    def forward(self, x, cos_sin):
        for layer in self.layers:
            x = layer(x, cos_sin)
        return x


class TRMLanguageModel(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg or ModelConfig()
        cfg = self.config
        assert cfg.hidden_size % cfg.num_heads == 0
        self.embedding = CastedEmbedding(cfg.vocab_size, cfg.hidden_size,
                                         1 / math.sqrt(cfg.hidden_size), torch.bfloat16)
        self.rotary = RotaryEmbedding(cfg.hidden_size // cfg.num_heads, cfg.seq_len, 10000.)
        self.core = ReasoningModule(cfg)
        for name in ('h_init', 'l_init'):
            self.register_buffer(name, trunc_normal_init_(torch.empty(cfg.hidden_size), std=1))

    def hidden(self, tokens):
        b, length = tokens.shape
        e = self.embedding(tokens) * math.sqrt(self.config.hidden_size)
        h, l = [v.to(e.dtype).view(1, 1, -1).expand(b, length, -1)
                for v in (self.h_init, self.l_init)]
        cos_sin = tuple(v[:length] for v in self.rotary())
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    l = self.core(l + h + e, cos_sin)
                h = self.core(h + l, cos_sin)
        for _ in range(self.config.L_cycles):
            l = self.core(l + h + e, cos_sin)
        return self.core(h + l, cos_sin)

    def forward(self, tokens):
        h = self.hidden(tokens)
        # Tied input/output embeddings; ordinary softmax probabilities for PPL.
        return F.linear(h, self.embedding.embedding_weight.to(h.dtype))

    def loss(self, tokens, targets):
        logits = self(tokens)
        return F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
