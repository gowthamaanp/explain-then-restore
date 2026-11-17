"""Point-spread-function helpers shared by the world model and the kernel module."""
from __future__ import annotations

import torch


def psf2otf(k: torch.Tensor, shape) -> torch.Tensor:
    """k: [B,1,kh,kw] (or [B,kh,kw]) -> OTF [B,1,H,W] complex, for circular
    convolution with the kernel centred at (kh//2, kw//2)."""
    if k.dim() == 3:
        k = k.unsqueeze(1)
    B, _, kh, kw = k.shape
    H, W = shape[-2:]
    otf = k.new_zeros(B, 1, H, W)
    otf[..., :kh, :kw] = k
    otf = torch.roll(otf, shifts=(-(kh // 2), -(kw // 2)), dims=(-2, -1))
    return torch.fft.fft2(otf)


def project_simplex(k: torch.Tensor) -> torch.Tensor:
    """Euclidean projection of each kernel onto {k >= 0, sum k = 1}.

    A blur kernel is a normalised non-negative PSF. Enforcing that exactly
    (rather than via a softmax, which cannot represent a near-delta kernel
    without unbounded logits, and cannot represent the zero kernel at all)
    keeps the refinement inside the physically valid set at every step."""
    shape = k.shape
    v = k.flatten(1)
    n = v.shape[1]
    u = torch.sort(v, dim=1, descending=True).values
    css = u.cumsum(dim=1) - 1.0
    idx = torch.arange(1, n + 1, device=k.device, dtype=v.dtype).view(1, -1)
    cond = u - css / idx > 0
    rho_idx = cond.float().cumsum(dim=1).argmax(dim=1)
    theta = css.gather(1, rho_idx.view(-1, 1)) / (rho_idx.view(-1, 1).to(v.dtype) + 1.0)
    return (v - theta).clamp(min=0).view(shape)
