"""Causal token ACT with up to 16 locally supervised TRM cycles.

Every cycle retains H=3/L=6, with gradients only in the last H round.
Halted positions keep both recurrent states fixed inside every shared block.
The dense attention kernels still compute masked positions as prefix context.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F
try:
    from .model import Attention, SwiGLU, CastedEmbedding, RotaryEmbedding, rms_norm, trunc_normal_init_
    from .runtime import ModelConfig
except ImportError:
    from model import Attention, SwiGLU, CastedEmbedding, RotaryEmbedding, rms_norm, trunc_normal_init_
    from runtime import ModelConfig


@dataclass
class ACTConfig(ModelConfig):
    halt_max_steps: int = 16
    halt_exploration_prob: float = .1


@dataclass
class Carry:
    h: torch.Tensor
    l: torch.Tensor
    steps: torch.Tensor
    halted: torch.Tensor


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attention = Attention(cfg.hidden_size, cfg.hidden_size // cfg.num_heads,
                                   cfg.num_heads, cfg.num_heads, causal=True)
        self.mlp = SwiGLU(cfg.hidden_size, cfg.expansion)

    def forward(self, x, cos_sin, active):
        x = torch.where(active, rms_norm(x + self.attention(cos_sin, x), 1e-5), x)
        return torch.where(active, rms_norm(x + self.mlp(x), 1e-5), x)


class ReasoningModule(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.ModuleList([Block(cfg) for _ in range(cfg.layers)])

    def forward(self, state, injection, cos_sin, active):
        x = torch.where(active, state + injection, state)
        for layer in self.layers:
            x = layer(x, cos_sin, active)
        return x


class TokenACT(nn.Module):
    def __init__(self, cfg=None):
        super().__init__()
        self.config = cfg or ACTConfig()
        cfg = self.config
        if cfg.hidden_size % cfg.num_heads or cfg.halt_max_steps < 1:
            raise ValueError('Invalid ACT model dimensions/horizon')
        self.embedding = CastedEmbedding(cfg.vocab_size, cfg.hidden_size,
                                         1 / math.sqrt(cfg.hidden_size), torch.bfloat16)
        self.rotary = RotaryEmbedding(cfg.hidden_size // cfg.num_heads, cfg.seq_len, 10000.)
        self.core = ReasoningModule(cfg)
        for name in ('h_init', 'l_init'):
            self.register_buffer(name, trunc_normal_init_(torch.empty(cfg.hidden_size), std=1))
        self.q_head = nn.Linear(cfg.hidden_size, 1)
        nn.init.zeros_(self.q_head.weight)
        nn.init.constant_(self.q_head.bias, -5.)

    def initial_carry(self, tokens):
        shape = (*tokens.shape, self.config.hidden_size)
        h, l = [v.to(torch.bfloat16).view(1, 1, -1).expand(shape) for v in (self.h_init, self.l_init)]
        return Carry(h, l, torch.zeros_like(tokens, dtype=torch.int32), torch.zeros_like(tokens, dtype=torch.bool))

    def forward_cycle(self, tokens, carry):
        """One deep-supervision cycle. Caller backprops/steps before the next cycle."""
        e = self.embedding(tokens) * math.sqrt(self.config.hidden_size)
        h, l = carry.h.detach(), carry.l.detach()
        active = (~carry.halted).unsqueeze(-1)
        cos_sin = tuple(v[:tokens.shape[1]] for v in self.rotary())
        with torch.no_grad():
            for _ in range(self.config.H_cycles - 1):
                for _ in range(self.config.L_cycles):
                    l = self.core(l, h + e, cos_sin, active)
                h = self.core(h, l, cos_sin, active)
        for _ in range(self.config.L_cycles):
            l = self.core(l, h + e, cos_sin, active)
        h = self.core(h, l, cos_sin, active)
        logits = F.linear(h, self.embedding.embedding_weight.to(h.dtype))
        q = F.linear(h.float(), self.q_head.weight, self.q_head.bias).squeeze(-1)
        proposed = Carry(h.detach(), l.detach(), carry.steps + (~carry.halted).int(), carry.halted)
        return proposed, logits, q

    @torch.no_grad()
    def finish_cycle(self, proposed, q, training=False, force_max=False):
        at_cap = proposed.steps >= self.config.halt_max_steps
        halted = at_cap if force_max else (at_cap | (q > 0))
        if training and self.config.halt_max_steps > 1:
            minimum = (torch.rand_like(q) < self.config.halt_exploration_prob) * torch.randint_like(
                proposed.steps, low=2, high=self.config.halt_max_steps + 1)
            halted &= proposed.steps >= minimum
        return Carry(proposed.h, proposed.l, proposed.steps, proposed.halted | halted)

    @torch.no_grad()
    def forward_with_steps(self, tokens, force_max=False):
        """Label-free deterministic inference; each prefix uses its own halt decision."""
        carry = self.initial_carry(tokens)
        for _ in range(self.config.halt_max_steps):
            proposed, _, q = self.forward_cycle(tokens, carry)
            carry = self.finish_cycle(proposed, q, force_max=force_max)
            if carry.halted.all().item():
                break
        logits = F.linear(carry.h, self.embedding.embedding_weight.to(carry.h.dtype))
        return logits, carry.steps

    def forward(self, tokens):
        return self.forward_with_steps(tokens)[0]
