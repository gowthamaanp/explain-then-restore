"""Family-agnostic PSF descriptors: the target of ProposalNet's kernel head.

Defined for ANY non-negative PSF, so the proposal never inherits a learned
shape space's blind spots (a disk or ring has well-defined descriptors even
though no trained family is a disk or ring). All are translation-invariant
(centroid removed). D = 10:

  0 log s1          sqrt of the larger second-moment eigenvalue (size along the main axis)
  1 log s2          sqrt of the smaller eigenvalue
  2 a*cos 2th       orientation, weighted by anisotropy a = (l1-l2)/(l1+l2), so it is
  3 a*sin 2th       continuous and well-defined even for isotropic kernels
  4 log r95         radius holding 95% of the mass
  5-8 radial mass   fraction in normalised-radius bins r/r95 in [0,.25) [.25,.5) [.5,.75) [.75,inf)
                    (centre-filled vs flat-top vs ring)
  9 log Neff        effective support 1/sum k^2 (peaked vs spread)
"""
from __future__ import annotations

import torch

D = 10


def descriptors(k: torch.Tensor) -> torch.Tensor:
    """k: [B,1,K,K] or [K,K] PSF(s) -> [B, D]."""
    if k.dim() == 2:
        k = k[None, None]
    B, _, K, _ = k.shape
    k = k.clamp(min=0)
    k = k / k.flatten(1).sum(1).clamp(min=1e-12).view(B, 1, 1, 1)
    a = torch.arange(K, device=k.device, dtype=k.dtype)
    yy, xx = torch.meshgrid(a, a, indexing="ij")
    w = k[:, 0]
    cy = (w * yy).sum((1, 2)); cx = (w * xx).sum((1, 2))
    dy = yy[None] - cy[:, None, None]; dx = xx[None] - cx[:, None, None]
    myy = (w * dy * dy).sum((1, 2)); mxx = (w * dx * dx).sum((1, 2)); mxy = (w * dx * dy).sum((1, 2))
    tr, det = myy + mxx, myy * mxx - mxy ** 2
    disc = ((tr / 2) ** 2 - det).clamp(min=0).sqrt()
    l1, l2 = (tr / 2 + disc).clamp(min=1e-4), (tr / 2 - disc).clamp(min=1e-4)
    an = (l1 - l2) / (l1 + l2)
    th2 = torch.atan2(2 * mxy, mxx - myy)                       # 2*theta of the main axis
    r = (dy ** 2 + dx ** 2).sqrt()
    # r95 by sorting radii by mass
    rf, wf = r.flatten(1), w.flatten(1)
    o = rf.argsort(1)
    cm = wf.gather(1, o).cumsum(1)
    i95 = (cm < 0.95).sum(1).clamp(max=rf.shape[1] - 1)
    r95 = rf.gather(1, o).gather(1, i95[:, None])[:, 0].clamp(min=0.5)
    rn = r / r95[:, None, None]
    edges = (0.0, 0.25, 0.5, 0.75, float("inf"))
    rad = torch.stack([(w * ((rn >= edges[i]) & (rn < edges[i + 1]))).sum((1, 2)) for i in range(4)], 1)
    neff = 1.0 / (wf ** 2).sum(1).clamp(min=1e-12)
    return torch.cat([torch.stack([0.5 * l1.log(), 0.5 * l2.log(), an * th2.cos(), an * th2.sin(), r95.log()], 1),
                   rad, neff.log()[:, None]], 1)
