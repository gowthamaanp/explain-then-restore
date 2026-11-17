"""Building blocks.

NAFNet blocks (Chen et al., ECCV 2022, "Simple Baselines for Image
Restoration"): LayerNorm, pointwise/depthwise convs, a multiplicative
SimpleGate in place of an activation, and simplified channel attention. No
BatchNorm anywhere -- batch statistics let a network read *which samples
share a batch*, and the previous codebase's degradation classifier was
caught doing exactly that.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm2d(nn.Module):
    def __init__(self, c, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = eps

    def forward(self, x):
        mu = x.mean(1, keepdim=True)
        var = x.var(1, keepdim=True, unbiased=False)
        x = (x - mu) / (var + self.eps).sqrt()
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half and multiply. Cheaper than GELU and, per
    NAFNet, at least as good."""

    def forward(self, x):
        a, b = x.chunk(2, dim=1)
        return a * b


class DWConv3x3(nn.Conv2d):
    """Depthwise 3x3 conv (zero padding 1) as nine shifted multiply-adds.

    Same parameters and state_dict as nn.Conv2d(c, c, 3, padding=1, groups=c),
    same output (checked to 1e-15 in float64). It exists for speed: the Prior is
    a gradient-step denoiser, so training differentiates through grad g, and
    cuDNN's grouped-conv kernels in that double backward took 78% of the step
    (3.9 s vs 0.68 s per micro-step at batch 4 x 160 px on a 3090). In eval mode it
    falls back to cuDNN (identical output, much less memory for whole-image solves).
    """

    def __init__(self, c):
        super().__init__(c, c, 3, padding=1, groups=c)

    def forward(self, x):
        if not self.training:
            # inference (the solver's single backward): cuDNN is fine and needs far less
            # memory than the nine-term graph on full-size images
            return F.conv2d(x, self.weight, self.bias, padding=1, groups=self.groups)
        H, W = x.shape[-2:]
        p = F.pad(x, (1, 1, 1, 1))
        w = self.weight.view(1, -1, 3, 3)
        out = self.bias.view(1, -1, 1, 1)
        for i in range(3):
            for j in range(3):
                out = out + p[:, :, i:i + H, j:j + W] * w[..., i, j, None, None]
        return out


class NAFBlock(nn.Module):
    def __init__(self, c, dw_expand=2, ffn_expand=2, drop=0.0):
        super().__init__()
        dw = c * dw_expand
        self.norm1 = LayerNorm2d(c)
        self.conv1 = nn.Conv2d(c, dw, 1)
        self.conv2 = DWConv3x3(dw)
        self.gate = SimpleGate()
        self.sca = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(dw // 2, dw // 2, 1))
        self.conv3 = nn.Conv2d(dw // 2, c, 1)

        ff = c * ffn_expand
        self.norm2 = LayerNorm2d(c)
        self.conv4 = nn.Conv2d(c, ff, 1)
        self.conv5 = nn.Conv2d(ff // 2, c, 1)

        self.beta = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, c, 1, 1))
        self.drop = nn.Dropout2d(drop) if drop > 0 else nn.Identity()

    def forward(self, x):
        h = self.conv2(self.conv1(self.norm1(x)))
        h = self.gate(h)
        h = h * self.sca(h)
        x = x + self.drop(self.conv3(h)) * self.beta
        h = self.gate(self.conv4(self.norm2(x)))
        return x + self.drop(self.conv5(h)) * self.gamma


class FiLM(nn.Module):
    """Per-channel scale/shift from a global vector (Perez et al. 2018).

    Zero-initialised so conditioning starts as an exact no-op -- but note
    that a zero-init projection whose INPUT is also always zero can never
    learn (its gradient is identically zero). That killed the degradation
    conditioning in the previous codebase. Here the goal vector is always
    non-zero, so the gradient path is live from step one.
    """

    def __init__(self, cond_dim, channels):
        super().__init__()
        self.proj = nn.Linear(cond_dim, channels * 2)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, cond):
        scale, shift = self.proj(cond).chunk(2, dim=-1)
        return x * (1 + scale[..., None, None]) + shift[..., None, None]


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.op = nn.Conv2d(c, c * 2, 2, stride=2)

    def forward(self, x):
        return self.op(x)


class Up(nn.Module):
    """PixelShuffle upsampling (Shi et al. 2016) -- no checkerboard
    artifacts, unlike transposed convolution (Odena et al. 2016)."""

    def __init__(self, c):
        super().__init__()
        self.op = nn.Sequential(nn.Conv2d(c, c * 2, 1, bias=False), nn.PixelShuffle(2))

    def forward(self, x):
        return self.op(x)


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    import math
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32)
                      / max(half - 1, 1))
    args = t.float().unsqueeze(-1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if dim % 2:
        emb = torch.cat([emb, torch.zeros_like(emb[:, :1])], dim=-1)
    return emb
