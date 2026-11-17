"""Post-solver refiner: corrects the residual error of ETR's blind chain inversion.

Input: the observation y (bicubic-upsampled to the output grid) and the solver's restoration x_hat;
output: x_hat + a learned residual. A small NAFNet-style U-Net (Chen et al. ECCV 2022 blocks, reused from
etr/models/blocks.py). It never replaces the world model: x_hat already undoes the estimated chain; the
refiner learns what the estimate got wrong (residual blur, tone / haze mismatch, leftover rain), from
synthetic chains only.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from etr.models.blocks import NAFBlock


class Refiner(nn.Module):
    def __init__(self, width=32, enc=(1, 1, 2, 4), mid=4, dec=(1, 1, 1, 1)):
        super().__init__()
        self.intro = nn.Conv2d(6, width, 3, padding=1)
        self.outro = nn.Conv2d(width, 3, 3, padding=1)
        nn.init.zeros_(self.outro.weight); nn.init.zeros_(self.outro.bias)     # starts as the identity on x_hat
        self.encs, self.downs, self.ups, self.decs = nn.ModuleList(), nn.ModuleList(), nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc:
            self.encs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(nn.Conv2d(c, 2 * c, 2, stride=2))
            c *= 2
        self.mid = nn.Sequential(*[NAFBlock(c) for _ in range(mid)])
        for n in dec:
            self.ups.append(nn.Sequential(nn.Conv2d(c, 2 * c, 1, bias=False), nn.PixelShuffle(2)))
            c //= 2
            self.decs.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
        self.pad = 2 ** len(enc)

    def forward(self, y, xh):
        H, W = xh.shape[-2:]
        ph, pw = (-H) % self.pad, (-W) % self.pad
        z = F.pad(torch.cat([y, xh], 1) * 2 - 1, (0, pw, 0, ph), mode="reflect")
        h = self.intro(z)
        skips = []
        for e, d in zip(self.encs, self.downs):
            h = e(h); skips.append(h); h = d(h)
        h = self.mid(h)
        for u, d, s in zip(self.ups, self.decs, reversed(skips)):
            h = d(u(h) + s)
        return xh + self.outro(h)[..., :H, :W]


def load_refiner(path, device="cuda"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Refiner(**ck["arch"]).to(device).eval()
    net.load_state_dict(ck["model"])
    return net


@torch.no_grad()
def refine(net, y, xh):
    """y: the observation (any grid); xh: the solver output on the output grid. Same y upsampling as in
    in the refiner's training data (torch bicubic, clamped)."""
    if y.shape[-2:] != xh.shape[-2:]:
        y = F.interpolate(y, size=xh.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    return net(y.clamp(0, 1), xh.clamp(0, 1)).clamp(0, 1)


@torch.no_grad()
def refine_d4(net, y, xh):
    """Rao-Blackwellised refiner: the mean over the 8 dihedral transforms of (y, xh), each mapped back
    (averaging the 8 views is used at test time)."""
    if y.shape[-2:] != xh.shape[-2:]:
        y = F.interpolate(y, size=xh.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    acc = 0
    for t in range(8):
        T = lambda z: torch.rot90(z.transpose(-2, -1) if t >= 4 else z, t % 4, (-2, -1))
        out = net(T(y.clamp(0, 1)), T(xh.clamp(0, 1))).clamp(0, 1)
        out = torch.rot90(out, -(t % 4), (-2, -1))
        acc = acc + (out.transpose(-2, -1) if t >= 4 else out)
    return acc / 8
