# Adapted from LTH14/JiT @ cbc743a2ada5e9762697da2c83f8c4f8379e8c17
# Copyright (c) 2025 Tianhong Li. MIT license: reference/LICENSE.
# TRM adaptation: remove redundant final RMS normalization, keep channel gamma.
# The optional @torch.compile decorator is removed.
import torch
from torch import nn

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)


OUTPUT_HEAD_DESCRIPTION = ('JiT FinalLayer adapted for TRM: channel gamma only (no RMSNorm), '
                           'time shift/scale, zero-init biased linear; FP32')


class FinalLayer(nn.Module):
    """
    JiT modulation/projection with channel scale on already normalized TRM states.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(x * self.gamma, shift, scale)
        x = self.linear(x)
        return x
