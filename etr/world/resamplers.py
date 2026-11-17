"""Continuous resampler family (the world model's downsampling factor).

Downscaling by an integer factor s is  y = D_s( h_s * x ), D_s = keep every
s-th pixel at phase 0 on our grid, h_s a separable 2-D kernel. Real software
differs only in h_s:

  family   1-D profile f(t)                    params
  cubic    Keys cubic                          a  (MATLAB/PIL -0.5, OpenCV/PyTorch -0.75)
  linear   triangle (1-|t|)+
  box      area average                       (OpenCV INTER_AREA, PIL BOX)
  lanczos  sinc(t) sinc(t/n), |t|<n           n  (PIL LANCZOS 3, OpenCV LANCZOS4 4)
  gauss    exp(-t^2/2 sigma^2)                 sigma
and whether the profile is stretched by s (anti-aliased: h(t) = f(t/s)/s,
MATLAB / PIL / torch antialias=True) or not (h(t) = f(t), OpenCV and torch
default: an interpolator, aliasing). Taps sit at t = j - c + (s-1)/2: the
half-pixel-centre convention every one of these libraries uses, which
phase-0 decimation (etr/world/ops.py: resample) reproduces (verified per library in
tests/test_resamplers.py). Signed kernels are allowed here -- this is not a
PSF and never enters the non-negative kernel prior.
"""
from __future__ import annotations

import math

import torch

FAMILIES = ("cubic", "linear", "box", "lanczos", "gauss")


def profile(t, family, a=-0.5, n=3, sigma=0.5):
    at = t.abs()
    if family == "cubic":
        return (((a + 2) * at ** 3 - (a + 3) * at ** 2 + 1) * (at <= 1)
                + (a * at ** 3 - 5 * a * at ** 2 + 8 * a * at - 4 * a) * ((at > 1) & (at < 2)))
    if family == "linear":
        return (1 - at).clamp(min=0)
    if family == "box":
        return ((at < 0.5).float() + 0.5 * (at == 0.5).float())
    if family == "lanczos":
        x = math.pi * t
        s1 = torch.where(at < 1e-8, torch.ones_like(t), torch.sin(x) / x.where(at > 1e-8, torch.ones_like(x)))
        s2 = torch.where(at < 1e-8, torch.ones_like(t), torch.sin(x / n) / (x / n).where(at > 1e-8, torch.ones_like(x)))
        return s1 * s2 * (at < n)
    if family == "gauss":
        return torch.exp(-t ** 2 / (2 * sigma ** 2))
    raise ValueError(family)


def support(family, n=3, sigma=0.5):
    return {"cubic": 2, "linear": 1, "box": 0.5, "lanczos": n, "gauss": 3 * sigma}[family]


def taps_1d(s, family, antialias=True, a=-0.5, n=3, sigma=0.5, box_exact=True, device=None):
    st = s if antialias else 1
    half = math.ceil(support(family, n, sigma) * st + (s - 1) / 2) + 1
    L = 2 * half + 1
    c = half
    t = torch.arange(L, dtype=torch.float64, device=device) - c + (s - 1) / 2
    if family == "box" and antialias and box_exact:
        # exact area weights: overlap of [t-1/2, t+1/2] with [-s/2, s/2]
        w = (torch.minimum(t + 0.5, torch.full_like(t, s / 2)) - torch.maximum(t - 0.5, torch.full_like(t, -s / 2))).clamp(min=0)
    else:
        w = profile(t / st, family, a, n, sigma)
    return (w / w.sum()).float()


def resampler_psf(s, family="cubic", antialias=True, a=-0.5, n=3, sigma=0.5, device=None):
    """Separable 2-D kernel for O.resample(x, s, k): odd-sized, signed allowed."""
    h = taps_1d(s, family, antialias, a, n, sigma, device=device)
    return h[:, None] * h[None, :]
