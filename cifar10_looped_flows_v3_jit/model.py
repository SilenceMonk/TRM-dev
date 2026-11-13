"""Stateless JiT denoiser wrapping attention TRM with configurable supervision cycles."""
import math
import sys
from pathlib import Path
import torch
from torch import nn
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trm/vendor'))
from models.layers import Attention, SwiGLU, CastedLinear, RotaryEmbedding, rms_norm
from models.common import trunc_normal_init_
try:
    from .jit_head import FinalLayer
except ImportError:
    from jit_head import FinalLayer


def patchify(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//4, 4, w//4, 4).permute(0, 2, 4, 1, 3, 5).reshape(b, 64, 48)


def unpatchify(x):
    return x.reshape(-1, 8, 8, 3, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32)


class AttentionBlock(nn.Module):
    """Original TRM attention block: post-norm attention and SwiGLU, no dropout."""
    def __init__(self):
        super().__init__()
        self.self_attn = Attention(512, 64, 8, 8, causal=False)
        self.mlp = SwiGLU(512, expansion=4)

    def forward(self, x, cos_sin):
        x = rms_norm(x + self.self_attn(cos_sin, x), 1e-5)
        return rms_norm(x + self.mlp(x), 1e-5)


class ReasoningModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([AttentionBlock(), AttentionBlock()])

    def forward(self, x, cos_sin):
        for layer in self.layers:
            x = layer(x, cos_sin)
        return x


class TRMJiT(nn.Module):
    Nsup = 4
    H_cycles = 3
    L_cycles = 6

    def __init__(self, Nsup=4):
        super().__init__()
        if Nsup not in (1, 4):
            raise ValueError("Nsup must be 1 or 4")
        self.Nsup = Nsup
        self.input = CastedLinear(48, 512, bias=False)
        self.time = nn.Sequential(CastedLinear(1, 512, bias=True), nn.SiLU(), CastedLinear(512, 512, bias=True))
        self.rotary_emb = RotaryEmbedding(64, 64, 10000.)
        self.core = ReasoningModule()
        self.output = FinalLayer(512, 4, 3)
        nn.init.zeros_(self.output.adaLN_modulation[-1].weight)
        nn.init.zeros_(self.output.adaLN_modulation[-1].bias)
        nn.init.zeros_(self.output.linear.weight)
        nn.init.zeros_(self.output.linear.bias)
        self.register_buffer('h_init', trunc_normal_init_(torch.empty(512), std=1))
        self.register_buffer('l_init', trunc_normal_init_(torch.empty(512), std=1))

    def initial_state(self, batch):
        return tuple(v.to(torch.bfloat16).view(1, 1, 512).expand(batch, 64, 512) for v in (self.h_init, self.l_init))

    def embeddings(self, x, t):
        return (math.sqrt(512)*self.input(patchify(x).to(torch.bfloat16)),
                self.time(t.reshape(-1, 1).to(torch.bfloat16))[:, None, :])

    def forward_cycle(self, x, t, state):
        """One supervised cycle: 2 no-grad H rounds, then 1 grad H round.

        State is detached at both boundaries. Training recomputes embeddings after
        each local optimizer update, always using the SAME x and t for Nsup cycles.
        """
        image, time = self.embeddings(x, t)
        e = image + time
        h, l = (v.detach() for v in state)
        cos_sin = self.rotary_emb()
        with torch.no_grad():
            for _ in range(self.H_cycles-1):
                for _ in range(self.L_cycles):
                    l = self.core(l+h+e, cos_sin)
                h = self.core(h+l, cos_sin)
        for _ in range(self.L_cycles):
            l = self.core(l+h+e, cos_sin)
        h = self.core(h+l, cos_sin)
        prediction = unpatchify(self.output(h.float(), time[:, 0, :].float()))
        return prediction, (h.detach(), l.detach())

    def forward(self, x, t):
        """Public x,t -> clean image prediction; no state crosses denoiser calls."""
        state = self.initial_state(len(x))
        for _ in range(self.Nsup):
            prediction, state = self.forward_cycle(x, t, state)
        return prediction


@torch.no_grad()
def sample(model, noise, steps=32, t_eps=0.05):
    if steps < 1 or t_eps <= 0:
        raise ValueError('steps and t_eps must be positive')
    x = noise.float().clone()
    for i in range(steps):
        t = torch.full((len(x),), i/steps, device=x.device)
        prediction = model(x, t)
        x = x + (prediction-x)/(steps*max(1-i/steps, t_eps))
    return x
