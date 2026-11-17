"""Kernel prior C: a spatial VAE over blur kernels (PSFs).

encode(k) -> (mu, logvar) on a spatial latent grid (zc x n x n, n = ceil(K/4));
decode(z, ksize) -> a valid PSF [B,1,K,K] (non-negative, sums to 1; exact
projection onto the simplex, etr/ops/psf.py).

Why a SPATIAL latent and a fully convolutional decoder: shapes are built from
local pieces (strokes, edges, arcs), so global shapes never seen in training but
made of familiar local pieces stay representable. Measured (79-px canvas,
latent-optimised L1): trained families 0.02-0.05, held-out disk / ring / curved
motion 0.01-0.05. Released checkpoint: checkpoints/kernel_vae.pt (zc=16, width=96, sizes 31..79).
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F

from etr.ops.psf import project_simplex


class _Res(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1), nn.SiLU(),
                               nn.GroupNorm(8, c), nn.Conv2d(c, c, 3, padding=1))
        nn.init.zeros_(self.b[-1].weight); nn.init.zeros_(self.b[-1].bias)

    def forward(self, x):
        return F.silu(x + self.b(x))


class VAESpatial(nn.Module):
    def __init__(self, ksize=31, zc=8, width=64):
        super().__init__()
        self.ksize, self.zc = ksize, zc
        self.enc = nn.Sequential(nn.Conv2d(1, width, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(width, width, 4, stride=2, padding=1), nn.SiLU(), _Res(width),
                                 nn.Conv2d(width, width, 4, stride=2, padding=1), nn.SiLU(), _Res(width),
                                 nn.Conv2d(width, 2 * zc, 1))
        self.dec_in = nn.Conv2d(zc, width, 3, padding=1)
        self.dec = nn.ModuleList([_Res(width), _Res(width), _Res(width), _Res(width)])
        self.out = nn.Conv2d(width, 1, 3, padding=1)
        self.latent_shape = (zc, 8, 8)

    def encode(self, k):
        """Any odd kernel size K: padded right/bottom to a multiple of 4; the
        latent grid is (K+pad)/4 on a side (8x8 for 31, 16x16 for 63)."""
        p = (-k.shape[-1]) % 4
        k = F.pad(k, (0, p, 0, p))
        h = self.enc(k)
        mu, lv = h.chunk(2, 1)
        return mu, lv.clamp(-12, 4)

    def decode(self, z, ksize=None):
        """Latent n x n -> kernel of odd size 4n-1 (31 for 8x8, 63 for 16x16),
        or `ksize` if given (must satisfy 0 <= 4n - ksize < 4)."""
        n4 = 4 * z.shape[-1]
        ksize = n4 - 1 if ksize is None else ksize
        assert 0 <= n4 - ksize < 4 and ksize % 2 == 1, (ksize, z.shape)
        h = F.silu(self.dec_in(z))
        h = self.dec[0](h)
        h = F.interpolate(h, scale_factor=2, mode="nearest"); h = self.dec[1](h)
        h = F.interpolate(h, scale_factor=2, mode="nearest"); h = self.dec[2](h); h = self.dec[3](h)
        return project_simplex(self.out(h)[..., :ksize, :ksize])


KernelPriorC = VAESpatial      # the name used in the docs
