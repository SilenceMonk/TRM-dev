"""Continuous RGB Looped Flows, using the vendored TRM primitives unchanged."""
import math
import sys
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'trm/vendor'))
from models.layers import CastedLinear, rms_norm
from models.common import trunc_normal_init_


def patchify(x):
    b, c, h, w = x.shape
    return x.reshape(b, c, h//4, 4, w//4, 4).permute(0, 2, 4, 1, 3, 5).reshape(b, 64, 48)


def unpatchify(x):
    return x.reshape(-1, 8, 8, 3, 4, 4).permute(0, 3, 1, 4, 2, 5).reshape(-1, 3, 32, 32)


class SwiGLU(nn.Module):
    # Same TRM operation, explicit intermediate widths per experiment spec.
    def __init__(self, width, inner):
        super().__init__()
        self.gate_up_proj = CastedLinear(width, inner*2, bias=False)
        self.down_proj = CastedLinear(inner, width, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate)*up)


class Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.token = SwiGLU(64, 256)
        self.channel = SwiGLU(512, 1536)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = rms_norm(x + self.token(x), 1e-5).transpose(1, 2)
        return rms_norm(x + self.channel(x), 1e-5)


class LoopedFlow(nn.Module):
    def __init__(self, R=3, m=6):
        super().__init__()
        self.R, self.m = R, m
        self.input = CastedLinear(48, 512, bias=False)
        self.time = nn.Sequential(CastedLinear(1, 512, bias=True), nn.SiLU(), CastedLinear(512, 512, bias=True))
        self.core = nn.Sequential(Mixer(), Mixer())
        self.output = CastedLinear(512, 48, bias=False)
        self.register_buffer('h_init', trunc_normal_init_(torch.empty(512), std=1))
        self.register_buffer('l_init', trunc_normal_init_(torch.empty(512), std=1))
        self.refine = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.refine.weight)
        nn.init.zeros_(self.refine.bias)

    def initial_state(self, b):
        return tuple(x.to(torch.bfloat16).view(1, 1, 512).expand(b, 64, 512) for x in (self.h_init, self.l_init))

    def embeddings(self, x, t):
        image = math.sqrt(512)*self.input(patchify(x).to(torch.bfloat16))
        time = self.time(t.reshape(-1, 1).to(torch.bfloat16))[:, None, :]
        return image, time

    def forward(self, x, t, state):
        image, time = self.embeddings(x, t)
        e = image + time
        h, l = (v.detach() for v in state)
        with torch.no_grad():
            for _ in range(self.R-1):
                for _ in range(self.m):
                    l = self.core(l+h+e)
                h = self.core(h+l)
        for _ in range(self.m):
            l = self.core(l+h+e)
        h = self.core(h+l)
        image = unpatchify(self.output(h)).float()
        prediction = image + self.refine(image)
        return prediction, (h.detach(), l.detach())


@torch.no_grad()
def sample(model, noise, steps=32, reset_state=False):
    y = noise.float().clone()
    state = model.initial_state(len(y))
    for i in range(steps):
        if reset_state:
            state = model.initial_state(len(y))
        t = torch.full((len(y),), i/steps, device=y.device)
        prediction, state = model(y, t, state)
        y = prediction if i == steps-1 else y + (prediction-y)/(steps-i)
    return y
