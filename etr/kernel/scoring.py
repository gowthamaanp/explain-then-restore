"""Ground-truth-free kernel scores on the blur-stage observation (luma, Hann-tapered crop).

blur_sure      Stein's unbiased estimate of the FILTERED risk ||H f_H(y) - H0 x||^2 / N of the Wiener-type
               deconvolver f_H = K* / (|K|^2 + lam G), minimised over lam (Xue & Blu, IEEE TIP 2015). Its
               trace term penalises the no-blur explanation, so unlike any data-fit residual it does not
               collapse to the delta kernel. Phase-blind (depends on |K| only): flips are left to the critic.
gauss_evidence log p(y | k) under a Gaussian image prior with power spectrum c / (G + eps), c profiled over a
               grid: the MAP_k criterion of Levin et al. (CVPR 2009 / 2011), image marginalised.
trust_kernel   frequency-trust kernel: deblur only where the top candidates agree (cryo-EM practice:
               never correct beyond the validated frequency); real, sums to 1, so the solver's data term
               stays convex and its convergence argument holds.
All batched over candidates; additive Gaussian noise assumed (JPEG / heteroscedastic noise void the
unbiasedness: such images are calibrated separately).
"""
from __future__ import annotations

import math

import torch

from etr.world.proposal import CANVAS

R = CANVAS // 2


def prep(y, crop=384):
    """y [1,3,H,W] -> (Y = fft of the tapered zero-mean luma crop [h,w], mean(w^2), G [h,w])."""
    l = 0.299 * y[0, 0] + 0.587 * y[0, 1] + 0.114 * y[0, 2]
    H, W = l.shape
    c = min(crop, H, W)
    l = l[(H - c) // 2:(H - c) // 2 + c, (W - c) // 2:(W - c) // 2 + c]
    w = torch.hann_window(c, device=y.device)[:, None] * torch.hann_window(c, device=y.device)[None]
    l = (l - (l * w).sum() / w.sum()) * w
    fy = torch.fft.fftfreq(c, device=y.device)[:, None]; fx = torch.fft.fftfreq(c, device=y.device)[None]
    G = (2 - 2 * torch.cos(2 * math.pi * fy)) + (2 - 2 * torch.cos(2 * math.pi * fx))
    return torch.fft.fft2(l), float((w ** 2).mean()), G


def otf(k, shape):
    """k [M,79,79] -> |K|^2 and K on the grid `shape` (centred PSF)."""
    M = k.shape[0]
    kp = torch.zeros(M, *shape, device=k.device)
    kp[:, :CANVAS, :CANVAS] = k
    kp = torch.roll(kp, (-R, -R), (-2, -1))
    Kf = torch.fft.fft2(kp)
    return Kf.abs() ** 2, Kf


LAMS = torch.logspace(-4, 0, 9)


def blur_sure(Y, w2, G, k, sigma, lams=LAMS, chunk=64):
    """-> risk [M] (per pixel, minimised over lam), in units of the effective noise variance."""
    N = Y.numel()
    s2 = float(sigma) ** 2 * w2
    P = Y.abs() ** 2
    out = []
    for c0 in range(0, k.shape[0], chunk):
        K2, _ = otf(k[c0:c0 + chunk], Y.shape)
        best = None
        for lam in lams.tolist():
            Hf = K2 / (K2 + lam * G + 1e-12)
            r = ((1 - Hf) ** 2 * P).flatten(1).sum(1) / N ** 2 + 2 * s2 * Hf.flatten(1).sum(1) / N - s2
            best = r if best is None else torch.minimum(best, r)
        out.append(best / max(s2, 1e-12))
    return torch.cat(out)


CS = torch.logspace(-5, -1, 9)


def gauss_evidence(Y, w2, G, k, sigma, cs=CS, chunk=64, K2=None):
    """-> negative log evidence [M] (lower is better), the image prior's scale c profiled out."""
    N = Y.numel()
    s2 = float(sigma) ** 2 * w2
    P = Y.abs() ** 2 / N
    out = []
    for c0 in range(0, k.shape[0] if K2 is None else K2.shape[0], chunk):
        k2 = otf(k[c0:c0 + chunk], Y.shape)[0] if K2 is None else K2[c0:c0 + chunk]
        best = None
        for c in cs.tolist():
            v = k2 * (c / (G + 1e-3)) + s2
            nl = (torch.log(v) + P / v).flatten(1).mean(1)
            best = nl if best is None else torch.minimum(best, nl)
        out.append(best)
    return torch.cat(out)


def trust_kernel(pick, top, weights, power=1.0, grid=2 * CANVAS):
    """pick [79,79]; top [M,79,79] candidates with weights [M] -> frequency-trust kernel [79,79]."""
    _, Kp = otf(pick[None], (grid, grid))
    _, Kt = otf(top, (grid, grid))
    w = weights.view(-1, 1, 1) / weights.sum().clamp(min=1e-12)
    agree = ((w * Kt).sum(0).abs() / (w * Kt.abs()).sum(0).clamp(min=1e-12)).clamp(0, 1) ** power
    Ke = agree * Kp[0] + (1 - agree)
    k = torch.fft.ifft2(Ke).real
    k = torch.roll(k, (R, R), (-2, -1))[:CANVAS, :CANVAS]
    return k / k.sum().clamp(min=1e-12)
