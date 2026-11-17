"""Restoration critic: judges a candidate kernel by the artefacts its restoration shows.

Input: the blurry photo y, the Wiener restoration x_hat with the candidate kernel,
and the reblur residual |k * x_hat - y|. It never sees the kernel itself, so it
cannot learn to recognise kernel families -- only ghosts and ringing, which look
the same for every family. Output: predicted PSNR gap (dB) to the true-kernel
restoration. Trained on synthetic wrong kernels, then list-wise on the candidate pool the estimator builds.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from etr.kernel.restore import reblur


class Critic(nn.Module):
    """[y, x_hat, |reblur - y|] (7 ch, full res) -> predicted dB gap (<= 0 is good)."""

    def __init__(self, w=48):
        super().__init__()
        def blk(i, o, s):
            return nn.Sequential(nn.Conv2d(i, o, 3, stride=s, padding=1), nn.GroupNorm(8, o), nn.SiLU(),
                                 nn.Conv2d(o, o, 3, padding=1), nn.GroupNorm(8, o), nn.SiLU())
        self.f = nn.Sequential(blk(7, w, 1), blk(w, w, 2), blk(w, 2 * w, 2), blk(2 * w, 2 * w, 2), blk(2 * w, 4 * w, 2))
        self.head = nn.Sequential(nn.Linear(8 * w, 128), nn.SiLU(), nn.Linear(128, 1))

    def forward(self, y, xh, k):
        r = (reblur(xh, k) - y).abs().mean(1, keepdim=True) * 10
        h = self.f(torch.cat([y * 2 - 1, xh.clamp(-0.5, 1.5) * 2 - 1, r], 1))
        h = torch.cat([h.mean((-2, -1)), h.amax((-2, -1))], 1)              # mean + max: ghosts may be local
        return self.head(h)[:, 0]
