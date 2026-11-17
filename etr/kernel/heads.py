"""Proposal heads: signature map -> K kernel hypotheses as latents of kernel prior C.

ACHead is fully convolutional in offset space (no global pooling: pooling is what
made the first kernel head fail on thin kernels). wta_loss is the relaxed
winner-takes-all training loss on the decoded kernels, plus a confidence term.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from etr.world.proposal import CANVAS


class Res(nn.Module):
    def __init__(self, c, d=1):
        super().__init__()
        self.b = nn.Sequential(nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=d, dilation=d),
                               nn.GroupNorm(8, c), nn.SiLU(), nn.Conv2d(c, c, 3, padding=d, dilation=d))

    def forward(self, x):
        return x + self.b(x)


class ACHead(nn.Module):
    """[B,1,160,160] autocorrelation -> K latents [B,K,zc,20,20] + confidences [B,K].
    Fully convolutional in offset space; dilated blocks at 20x20 give every latent
    cell a view of every offset."""

    def __init__(self, K=8, zc=16, w=128, in_ch=1):
        super().__init__()
        self.K, self.zc = K, zc
        self.enc = nn.Sequential(
            nn.Conv2d(in_ch, 48, 3, padding=1), Res(48),
            nn.Conv2d(48, 64, 4, stride=2, padding=1), Res(64),                    # 80
            nn.Conv2d(64, 96, 4, stride=2, padding=1), Res(96),                    # 40
            nn.Conv2d(96, w, 4, stride=2, padding=1),                              # 20
            *[Res(w, d) for d in (1, 2, 4, 8, 1, 2, 4, 8)])
        self.out = nn.Conv2d(w, K * zc, 1)
        self.conf = nn.Linear(w, K)

    def forward(self, a, feat=False):
        h = self.enc(a)
        z = self.out(h).view(a.shape[0], self.K, self.zc, 20, 20)
        if feat:                                   # pooled trunk feature (for kernel retrieval)
            return z, self.conf(h.mean((-2, -1))), h.mean((-2, -1))
        return z, self.conf(h.mean((-2, -1)))


class Retrieval(nn.Module):
    """Image-kernel embedding for kernel RETRIEVAL from a bank (CLIP-style, symmetric InfoNCE).
    Image side: the blur head's pooled trunk feature; kernel side: kernel prior C's (frozen)
    latent of the kernel. About 0.07M parameters."""

    def __init__(self, w=128, zc=16, d=256):
        super().__init__()
        self.img = nn.Sequential(nn.Linear(w, d), nn.SiLU(), nn.Linear(d, d))
        self.ker = nn.Sequential(nn.Conv2d(zc, 48, 3, stride=2, padding=1), nn.SiLU(),
                                 nn.Conv2d(48, 64, 3, stride=2, padding=1), nn.SiLU())
        self.ker_fc = nn.Linear(64, d)

    def embed_img(self, h):
        return F.normalize(self.img(h), dim=-1)

    def embed_ker(self, zk):
        """zk: C's latent mean [N,zc,20,20]."""
        return F.normalize(self.ker_fc(self.ker(zk).mean((-2, -1))), dim=-1)


def wta_loss(z, conf, kt, C, eps=0.05):
    B, Kh = z.shape[:2]
    kh = C.decode(z.flatten(0, 1), CANVAS).view(B, Kh, CANVAS, CANVAS)
    l1 = (kh - kt[:, None, 0]).abs().flatten(2).sum(2)                             # [B,K]
    win = l1.argmin(1)
    w = torch.full_like(l1, eps / (Kh - 1)); w.scatter_(1, win[:, None], 1 - eps)
    return (w * l1).sum(1).mean() + 0.1 * F.cross_entropy(conf, win), l1
