"""JPEG helpers for the world model's JPEG data term (etr/world/chain.py):
IJG quantisation tables, the orthonormal 8x8 DCT and block (i)DCT, JFIF colour transform."""
from __future__ import annotations

import torch

# JPEG Annex K base quantisation tables.
_Q_LUMA = torch.tensor([
    [16, 11, 10, 16, 24, 40, 51, 61], [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56], [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77], [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101], [72, 92, 95, 98, 112, 100, 103, 99]], dtype=torch.float32)
_Q_CHROMA = torch.tensor([
    [17, 18, 24, 47, 99, 99, 99, 99], [18, 21, 26, 66, 99, 99, 99, 99],
    [24, 26, 56, 99, 99, 99, 99, 99], [47, 66, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99],
    [99, 99, 99, 99, 99, 99, 99, 99], [99, 99, 99, 99, 99, 99, 99, 99]], dtype=torch.float32)


def quant_table(quality, device=None) -> torch.Tensor:
    """IJG quality (1..100) -> [B,3,8,8] tables (luma, chroma, chroma).
    `quality` may be a scalar or a [B] tensor, so a batch can mix qualities."""
    q = torch.as_tensor(quality, dtype=torch.float32, device=device).flatten().clamp(1, 100)
    scale = torch.where(q < 50, 5000.0 / q, 200.0 - 2.0 * q).view(-1, 1, 1, 1)
    base = torch.stack([_Q_LUMA, _Q_CHROMA, _Q_CHROMA]).to(q.device).unsqueeze(0)
    return ((base * scale + 50.0) / 100.0).floor().clamp(1, 255)


def _dct_matrix(device, dtype):
    n = 8
    x = torch.arange(n, device=device, dtype=dtype)
    u = x.view(-1, 1)
    D = torch.cos((2 * x.view(1, -1) + 1) * u * torch.pi / (2 * n))
    D *= torch.sqrt(torch.tensor(2.0 / n, device=device, dtype=dtype))
    D[0] /= torch.sqrt(torch.tensor(2.0, device=device, dtype=dtype))
    return D                                            # orthonormal DCT-II


def _blockify(x):
    """[B,C,H,W] -> [B,C,H/8,W/8,8,8]"""
    B, C, H, W = x.shape
    return x.view(B, C, H // 8, 8, W // 8, 8).permute(0, 1, 2, 4, 3, 5)


def _unblockify(b, H, W):
    B, C = b.shape[:2]
    return b.permute(0, 1, 2, 4, 3, 5).reshape(B, C, H, W)


def block_dct(x, D):
    b = _blockify(x)
    return torch.einsum("ij,...jk,lk->...il", D, b, D)


def block_idct(c, D):
    return torch.einsum("ji,...jk,kl->...il", D, c, D)


def rgb_to_ycc(x):
    """JFIF transform on a 0..255 tensor."""
    r, g, b = x[:, 0], x[:, 1], x[:, 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0
    cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0
    return torch.stack([y, cb, cr], 1)


def ycc_to_rgb(x):
    y, cb, cr = x[:, 0], x[:, 1], x[:, 2] - 128.0
    cb = cb - 128.0
    r = y + 1.402 * cr
    g = y - 0.344136 * cb - 0.714136 * cr
    b = y + 1.772 * cb
    return torch.stack([r, g, b], 1)
