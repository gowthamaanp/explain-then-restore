"""PriorNet: the learned image prior. The only heavy network in ETR, and
the only place quality comes from.

    x_clean = PriorNet(x_noisy, sigma, goal)

THE DESIGN RULE, and the reason this file is short on features:

    PriorNet never sees the degradation.

Not the kernel, not the scale factor, not a task label, not the observation
`y`. Its inputs are the current image estimate and how much error is left in
it. That single constraint is what produces graceful degradation, and it is
worth stating why:

  * A feedforward restorer learns y -> x. Its input distribution IS the
    degradation distribution, so an unseen degradation is out-of-distribution
    input and it collapses.
  * A prior learns "somewhat-corrupted estimate" -> "cleaner estimate". When
    the degradation changes, its input distribution barely moves -- only the
    amount of residual error changes, a one-dimensional family it is
    explicitly conditioned on.

So the zero-shot property is not learned here; it is inherited from the
physics wrapper, and this network's job is only to be a good image model.
That also means: do not add degradation conditioning later "for a bit more
PSNR". It would trade the entire robustness claim for a fraction of a dB.

Conditioning follows DRUNet (Zhang et al. 2021): sigma enters as an extra
input channel rather than through FiLM, so the network can modulate its
behaviour spatially -- which matters because after a decimating operator the
residual error is genuinely non-uniform across the image.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from etr.models.blocks import NAFBlock, Down, Up, FiLM


class PriorNet(nn.Module):
    def __init__(self, in_ch: int = 3, width: int = 48,
                 enc_blocks=(2, 2, 4), mid_blocks: int = 6, dec_blocks=(2, 2, 2),
                 goal_dim: int = 64, identity_init: bool = True,
                 clamp_output: bool = True, n_cond: int = 1):
        super().__init__()
        # n_cond conditioning maps: 1 = sigma map (DRUNet-style); 2 = sigma map +
        # noise correlation-length map (0 = white noise). The correlation channel
        # lets one prior denoise white, heteroscedastic AND spatially correlated
        # (camera / demosaiced) noise without guessing which it is facing.
        self.n_cond = n_cond
        # clamp_output=False for GSDenoiser specifically: the SAME class of
        # bug identity_init guards against, one level deeper. This clamp is
        # fine standalone (over/under-saturated pixels just get no gradient
        # that step, a normal and usually benign thing), but GSDenoiser's
        # `torch.autograd.grad(Nx, x, ...)` backpropagates a SECOND time
        # through it -- if training ever pushes `inp+out` broadly outside
        # [0,1] (plausible in a double-backward landscape, and NOT
        # prevented by identity_init once past step 0), the VJP through the
        # saturated clamp collapses toward zero over those regions. Measured
        # directly in a real 400k-step run: `jac_pen` (this VJP's own norm)
        # decayed to exactly 0.00000 right as PSNR crashed to a sigma-
        # independent 4.10dB -- the same dead-constant-output signature as
        # the earlier `identity_init` bug, just triggered later by this
        # different mechanism instead of prevented by that fix.
        #
        # identity_init=False for GSDenoiser (gs_denoiser.py) specifically:
        # zero-initializing `self.out` makes N(x)=x and its Jacobian J_N=I
        # EXACTLY, for every x, at init -- fine standalone, but GSDenoiser's
        # D(x)=N(x)+J_N^T(x-N(x)) then has residual=x-N(x)=0 identically, so
        # d(D)/d(theta) = dN/dtheta - J_N^T @ dN/dtheta = dN/dtheta -
        # dN/dtheta = 0, EXACTLY, for every parameter, regardless of input --
        # a literal fixed point of the gradient, not a slow start. Confirmed
        # directly: a GSDenoiser built on this backbone showed exactly-zero
        # gradient norm from step 0, independent of loss formulation, batch,
        # or seed, only resolved by breaking this exact cancellation.
        # +1 channel for the sigma map (DRUNet-style noise-level conditioning)
        self.intro = nn.Conv2d(in_ch + n_cond, width, 3, padding=1)

        self.encoders, self.downs = nn.ModuleList(), nn.ModuleList()
        c = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))
            self.downs.append(Down(c))
            c *= 2

        self.middle = nn.Sequential(*[NAFBlock(c) for _ in range(mid_blocks)])
        # the goal steers the prior at the bottleneck only: it is a global
        # statement about WHAT KIND of restoration is wanted, not a spatial one
        self.goal_film = FiLM(goal_dim, c)

        self.decoders, self.ups = nn.ModuleList(), nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(Up(c))
            c //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(c) for _ in range(n)]))

        self.out = nn.Conv2d(width, in_ch, 3, padding=1)
        if identity_init:
            nn.init.zeros_(self.out.weight)
            nn.init.zeros_(self.out.bias)      # starts as the identity map
        self.padding = 2 ** len(enc_blocks)
        self.clamp_output = clamp_output

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                goal: torch.Tensor = None, corr: torch.Tensor = None) -> torch.Tensor:
        """x: [B,3,H,W] in [0,1]; sigma: [B] or [B,1,H,W]; goal: [B,goal_dim];
        corr: correlation length in px, [B] or [B,1,H,W] (None = 0, white noise)"""
        B, _, H, W = x.shape
        inp = x
        x, pad = self._pad(x)

        if sigma.dim() == 1:
            sigma = sigma.view(B, 1, 1, 1).expand(B, 1, *x.shape[-2:])
        elif sigma.shape[-2:] != x.shape[-2:]:
            sigma = F.interpolate(sigma, size=x.shape[-2:], mode="nearest")
        cond = [sigma]
        if self.n_cond >= 2:
            if corr is None:
                corr = torch.zeros_like(sigma)
            elif corr.dim() == 1:
                corr = corr.view(B, 1, 1, 1).expand(B, 1, *x.shape[-2:])
            elif corr.shape[-2:] != x.shape[-2:]:
                corr = F.interpolate(corr, size=x.shape[-2:], mode="nearest")
            cond.append(corr)
        h = self.intro(torch.cat([x] + cond, dim=1))

        skips = []
        for enc, down in zip(self.encoders, self.downs):
            h = enc(h)
            skips.append(h)
            h = down(h)

        h = self.middle(h)
        if goal is not None:
            h = self.goal_film(h, goal)

        for dec, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            h = up(h) + skip
            h = dec(h)

        out = self.out(h)
        out = out[..., pad[2]:pad[2] + H, pad[0]:pad[0] + W]
        result = inp + out                      # residual: predicts the error to remove
        return result.clamp(0, 1) if self.clamp_output else result

    def _pad(self, x):
        H, W = x.shape[-2:]
        p = self.padding
        ph, pw = (p - H % p) % p, (p - W % p) % p
        if ph == 0 and pw == 0:
            return x, (0, 0, 0, 0)
        return F.pad(x, (0, pw, 0, ph), mode="reflect"), (0, pw, 0, ph)

