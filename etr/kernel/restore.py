"""Cheap restoration and scoring utilities used inside the kernel module.

wiener      gradient-prior Wiener deconvolution (closed form, reflect-padded)
reblur      k * x (reflect-padded)
decode_k    C latents -> normalised kernels on the 79-px canvas
l1l2        l1/l2 ratio of gradients: low for sparse, ringing-free restorations
            (decides the 180-degree flip; normalised sparsity, Krishnan et al. 2011)
psnr_b      per-sample PSNR
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from etr.world.proposal import CANVAS

R = CANVAS // 2                       # 39


def luma_grads(img):
    l = 0.299 * img[:, 0] + 0.587 * img[:, 1] + 0.114 * img[:, 2]
    gx = F.pad(l[..., :, 1:] - l[..., :, :-1], (0, 1))
    gy = F.pad(l[..., 1:, :] - l[..., :-1, :], (0, 0, 0, 1))
    return gx, gy


def wiener(y, k, sigma, lam_scale=200.0, floor=1e-3):
    """Gradient-prior Wiener deconvolution, reflect-padded to limit boundary ringing."""
    B = y.shape[0]
    p = CANVAS
    yp = F.pad(y, [p] * 4, mode="reflect")
    H, W = yp.shape[-2:]
    kp = torch.zeros(B, 1, H, W, device=y.device)
    kp[..., :CANVAS, :CANVAS] = k
    kp = torch.roll(kp, (-R, -R), (-2, -1))
    Kf = torch.fft.fft2(kp)
    fy = torch.fft.fftfreq(H, device=y.device)[:, None]; fx = torch.fft.fftfreq(W, device=y.device)[None]
    G = (2 - 2 * torch.cos(2 * math.pi * fy)) + (2 - 2 * torch.cos(2 * math.pi * fx))
    lam = (lam_scale * sigma ** 2 + floor).view(B, 1, 1, 1)
    X = Kf.conj() * torch.fft.fft2(yp) / (Kf.abs() ** 2 + lam * G)
    return torch.fft.ifft2(X).real[..., p:-p, p:-p]


def decode_k(C, z):
    k = C.decode(z, CANVAS).clamp(min=0)
    return k / k.flatten(1).sum(1).clamp(min=1e-12).view(-1, 1, 1, 1)


def reblur(x, k):
    B = x.shape[0]
    r = CANVAS // 2
    xp = F.pad(x, [r] * 4, mode="reflect")
    return F.conv2d(xp.reshape(1, B * 3, *xp.shape[-2:]), k.flip(-2, -1).repeat_interleave(3, 0), groups=B * 3).view_as(x)


def psnr_b(a, b):
    return -10 * torch.log10(((a.clamp(0, 1) - b) ** 2).flatten(1).mean(1))


def l1l2(x, border=24):
    gx, gy = luma_grads(x)
    g = torch.cat([gx[..., border:-border, border:-border].flatten(1), gy[..., border:-border, border:-border].flatten(1)], 1)
    return g.abs().sum(1) / g.norm(dim=1).clamp(min=1e-12)


def sigma_mad(y):
    """Noise std per image from the median absolute diagonal Haar detail of the luma
    (Donoho's MAD estimate; the blind estimate the critic's Wiener step uses). [B]"""
    l = 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]
    H, W = l.shape[-2:]
    l = l[..., :H // 2 * 2, :W // 2 * 2]
    hh = (l[..., 0::2, 0::2] - l[..., 0::2, 1::2] - l[..., 1::2, 0::2] + l[..., 1::2, 1::2]) / 2
    return (hh.flatten(1).abs().median(1).values / 0.6745).clamp(1 / 255, 50 / 255)
