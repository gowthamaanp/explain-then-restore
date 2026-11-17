"""The blur's phase-blind signature: an autocorrelation in OFFSET space.

autocorr_true(k)  the kernel's own autocorrelation (the ceiling input)
autocorr_obs(y)   autocorrelation of the luma gradients of a blurry photo
                  (Hann-windowed, offsets +-78 px), what the proposal heads read
Both are returned as [B,1,160,160] maps (157 offsets, zero-padded), sqrt-compressed
and normalised to a unit peak.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from etr.world.proposal import CANVAS


AC = 2 * CANVAS - 1


IN = 160


def autocorr_true(k):
    """k: [B,1,79,79] -> [B,1,160,160] autocorrelation (157 offsets, zero-padded)."""
    n = 2 * CANVAS
    Fk = torch.fft.rfft2(k, s=(n, n))
    a = torch.fft.irfft2(Fk.abs() ** 2, s=(n, n))
    a = torch.fft.fftshift(a, dim=(-2, -1))[..., 1:1 + AC, 1:1 + AC]         # centre at index 78
    return norm_ac(a)


def norm_ac(a):
    a = a / a.flatten(1).amax(1).clamp(min=1e-12).view(-1, 1, 1, 1)
    a = a.clamp(min=0).sqrt()                                                 # compress the dynamic range
    return F.pad(a, (0, IN - AC, 0, IN - AC))


def autocorr_obs(y):
    """Autocorrelation of the luma gradients of y (Hann-windowed), offsets +-78."""
    gy = 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]
    H, W = gy.shape[-2:]
    win = torch.hann_window(H, device=y.device)[:, None] * torch.hann_window(W, device=y.device)[None]
    a = 0
    for d in ((0, 1), (1, 0)):
        g = (gy - torch.roll(gy, d, (-2, -1)))[:, 1:, 1:]
        g = F.pad(g, (1, 0, 1, 0)) * win
        Fg = torch.fft.rfft2(g, s=(2 * H, 2 * W))
        a = a + torch.fft.irfft2(Fg.abs() ** 2, s=(2 * H, 2 * W))
    a = torch.fft.fftshift(a, dim=(-2, -1))
    c = H                                                                     # zero offset
    a = a[:, None, c - CANVAS + 1:c + CANVAS, c - CANVAS + 1:c + CANVAS]
    return norm_ac(a)
