"""Degradation-chain primitives.

Every function acts on [B,3,H,W] float tensors in [0,1] and is differentiable
in its image argument, so a data term built from them gets exact gradients
(and, for the linear ones, exact adjoints) from autograd rather than from
hand-written transposes that can silently drift.

Two conventions matter everywhere:
  * Circular convolution on a PADDED domain. The solver's variable lives on
    the observed region plus a margin `pad` wider than any kernel's radius,
    and the data term only compares the cropped observed region. Wrap-around
    therefore never reaches an observed pixel, and since the margin is free,
    ANY boundary condition the true process used (replicate, reflect-101,
    MATLAB symmetric) is representable exactly.
  * Decimation at phase 0 on that padded grid. With `pad` divisible by s the
    observed LR pixel i sits at padded HR coordinate pad + s*i, matching
    phase-0 decimation (and, with `bicubic_kernel`, MATLAB imresize to 2e-7).
"""
from __future__ import annotations

import math

import torch

from etr.ops.psf import psf2otf
from etr.ops import kernels as K

EPS_V = 1e-4                      # smallest V treated as non-black in tone inverses
RGB2Y = (0.299, 0.587, 0.114)
RGB2CB = (-0.168736, -0.331264, 0.5)
RGB2CR = (0.5, -0.418688, -0.081312)


# ---------------------------------------------------------------- linear --
def conv_circ(x: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
    """Circular convolution with an odd-sized PSF k [kh,kw] (or [1,1,kh,kw])."""
    k = k.view(1, 1, *k.shape[-2:]).to(x)
    assert k.shape[-1] % 2 == 1 and k.shape[-2] % 2 == 1, "PSF must be odd-sized"
    assert k.shape[-2] <= x.shape[-2] and k.shape[-1] <= x.shape[-1]
    otf = _otf_cached(k, x.shape)
    return torch.fft.ifft2(torch.fft.fft2(x) * otf).real


_OTF = {}


def _otf_cached(k, shape):
    """psf2otf with a small cache: the solver convolves with the same few PSFs thousands of times
    (every prox inner step); recomputing their FFT dominated the data-term cost."""
    if k.requires_grad:
        return psf2otf(k, shape)
    key = (k.data_ptr(), k._version, tuple(k.shape), tuple(shape[-2:]), k.dtype, k.device)
    o = _OTF.get(key)
    if o is None:
        if len(_OTF) > 64:
            _OTF.clear()
        o = _OTF[key] = (psf2otf(k, shape), k.detach().clone())
    elif not torch.equal(o[1], k):                         # address reused by a different kernel
        o = _OTF[key] = (psf2otf(k, shape), k.detach().clone())
    return o[0]


def resample(x: torch.Tensor, s: int, k: torch.Tensor) -> torch.Tensor:
    """Anti-alias with k, then keep every s-th pixel at phase 0."""
    return conv_circ(x, k)[..., ::s, ::s]


def bicubic_psf(s: int, device=None) -> torch.Tensor:
    """MATLAB-imresize downscaling kernel for factor s, sized to hold its
    full support (cubic support 2s each side, plus the (s-1)/2 offset)."""
    return K.bicubic_kernel(6 * s + 1, s, device)


def haze(x, t, A):
    """Atmospheric scattering: t*x + A*(1-t). t: scalar or [B,1,H,W]."""
    return x * t + A * (1 - t)


def haze_inv(y, t, A):
    return (y - A * (1 - t)) / t.clamp(min=1e-3) if torch.is_tensor(t) else (y - A * (1 - t)) / max(t, 1e-3)


def layer(x, L, M=None):
    """Additive scene layer (rain streaks) with a known clip/occlusion mask M:
    (1-M)(x+L) + M. M=1 where the result saturated at 1 (or is occluded)."""
    out = x + L
    return out if M is None else (1 - M) * out + M


# ------------------------------------------------------------------- tone --
def hsv_v(x):
    return x.amax(1, keepdim=True)


def curve(V, kind: str, p: dict):
    """Monotone tone curve on V in [0,1]."""
    if kind == "shift":
        return (V - p["s"]).clamp(min=0)
    if kind == "gamma":
        return V.clamp(min=0) ** (1.0 / p["gamma"])
    if kind == "linmap":
        return (V - p["vmin"]) / max(p["vmax"] - p["vmin"], 1e-6) * p["dmax"]
    if kind == "spline":
        return _spline(V, p["knots_y"])
    if kind == "gain":
        return (V * p["g"]).clamp(0, 1)
    if kind == "log":                                   # brightening log curve
        return torch.log1p(p["a"] * V.clamp(min=0)) / math.log1p(p["a"])
    if kind == "expc":                                  # its inverse: a darkening curve
        return torch.expm1(V.clamp(min=0) * math.log1p(p["a"])) / p["a"]
    if kind == "scurve":                                # contrast S-curve (HELD OUT family)
        c = p["c"]
        return 0.5 + 0.5 * torch.tanh(c * (V - 0.5)) / math.tanh(0.5 * c)
    if kind == "pwl":                                   # monotone piecewise linear, knots (kx, ky)
        return _pwl(V, p["kx"], p["ky"])
    if kind == "compose":                               # p["steps"]: list of curve dicts, applied in order
        for q in p["steps"]:
            V = curve(V, q["kind"], q)
        return V
    raise ValueError(kind)


def _pwl(V, kx, ky):
    kx = torch.as_tensor(kx, dtype=V.dtype, device=V.device)
    ky = torch.as_tensor(ky, dtype=V.dtype, device=V.device)
    v = V.clamp(kx[0].item(), kx[-1].item())
    i = (torch.searchsorted(kx.contiguous(), v.contiguous(), right=True) - 1).clamp(0, kx.numel() - 2)
    x0, x1 = kx[i], kx[i + 1]
    f = (v - x0) / (x1 - x0).clamp(min=1e-6)
    return ky[i] + f * (ky[i + 1] - ky[i])


def curve_inv(Vp, kind: str, p: dict):
    if kind == "shift":
        return Vp + p["s"]
    if kind == "gamma":
        return Vp.clamp(min=0) ** p["gamma"]
    if kind == "linmap":
        return Vp / p["dmax"] * (p["vmax"] - p["vmin"]) + p["vmin"]
    if kind == "spline":
        return _spline_inv(Vp, p["knots_y"])
    if kind == "gain":
        return Vp / p["g"]
    if kind == "log":
        return torch.expm1(Vp.clamp(min=0) * math.log1p(p["a"])) / p["a"]
    if kind == "expc":
        return torch.log1p(p["a"] * Vp.clamp(min=0)) / math.log1p(p["a"])
    if kind == "scurve":
        c = p["c"]
        z = ((Vp - 0.5) * 2 * math.tanh(0.5 * c)).clamp(-0.999999, 0.999999)
        return 0.5 + torch.atanh(z) / c
    if kind == "pwl":                                   # inverse = swap the axes
        return _pwl(Vp, p["ky"], p["kx"])
    if kind == "compose":
        for q in reversed(p["steps"]):
            Vp = curve_inv(Vp, q["kind"], q)
        return Vp
    raise ValueError(kind)


def _spline(V, ky):
    """Piecewise-linear monotone curve through (i/n, ky[i]), ky increasing."""
    ky = torch.as_tensor(ky, dtype=V.dtype, device=V.device)
    n = ky.numel() - 1
    t = (V.clamp(0, 1) * n)
    i = t.floor().clamp(max=n - 1).long()
    f = t - i
    return ky[i] * (1 - f) + ky[i + 1] * f


def _spline_inv(Vp, ky):
    ky = torch.as_tensor(ky, dtype=Vp.dtype, device=Vp.device)
    n = ky.numel() - 1
    v = Vp.clamp(ky[0].item(), ky[-1].item())
    i = (torch.searchsorted(ky.contiguous(), v.contiguous(), right=True) - 1).clamp(0, n - 1)
    lo, hi = ky[i], ky[i + 1]
    f = (v - lo) / (hi - lo).clamp(min=1e-6)
    return (i.to(v.dtype) + f) / n


def tone(x, kind, p, space="hsv_v"):
    """Apply a tone curve. hsv_v: scale each RGB pixel by c(V)/V, which keeps
    H and S fixed exactly (what an HSV-V edit does, up to cv2's 8-bit H/S
    quantisation). rgb: the curve per channel. luma: c(Y)/Y scaling."""
    if space == "rgb":
        return curve(x, kind, p)
    V = hsv_v(x) if space == "hsv_v" else _luma(x)
    return x * (curve(V, kind, p) / V.clamp(min=EPS_V)) * (V > EPS_V)


def tone_inv(y, kind, p, space="hsv_v"):
    """Inverse of `tone` and the per-pixel gain of that inverse (for the
    delta-method noise variance). Where the observed V is ~0 the colour is
    lost; the inverse returns 0 there and the gain is reported as 0 so the
    data term can drop those pixels."""
    if space == "rgb":
        xi = curve_inv(y, kind, p)
        gain = _num_deriv(lambda v: curve_inv(v, kind, p), y)
        return xi, gain
    Vp = hsv_v(y) if space == "hsv_v" else _luma(y)
    ok = Vp > EPS_V
    ratio = curve_inv(Vp, kind, p) / Vp.clamp(min=EPS_V) * ok
    return y * ratio, ratio.expand_as(y)


def tone_inv_var(y, var, kind, p, space="hsv_v"):
    """Delta-method variance of tone_inv(y) for independent per-channel noise of variance `var` on y:
    x_c = y_c r(S(y)) with S = max channel (hsv_v) or luma, r = c^-1(S)/S:
      dx_c/dy_c' = delta_cc' r + y_c r'(S) dS/dy_c'.
    hsv_v: the max channel's derivative is (c^-1)'(V) (not r); the others get r^2 var_c + (y_c r')^2 var_max.
    luma: the full 3x3 Jacobian. rgb: the per-channel derivative (as before)."""
    if space == "rgb":
        return var * _num_deriv(lambda v: curve_inv(v, kind, p), y) ** 2
    S = hsv_v(y) if space == "hsv_v" else _luma(y)
    ok = (S > EPS_V).to(y.dtype)
    Sc = S.clamp(min=EPS_V)
    r = curve_inv(Sc, kind, p) / Sc * ok
    d = _num_deriv(lambda v: curve_inv(v, kind, p), Sc) * ok                   # (c^-1)'(S)
    rp = (d - r) / Sc                                                            # r'(S)
    if space == "hsv_v":
        idx = y.argmax(1, keepdim=True)
        hot = torch.zeros_like(y).scatter_(1, idx, 1.0)
        var_m = var.gather(1, idx) if var.shape[1] == 3 else var
        return hot * d ** 2 * var + (1 - hot) * (r ** 2 * var + (y * rp) ** 2 * var_m)
    w = torch.tensor(RGB2Y, device=y.device, dtype=y.dtype).view(1, 3, 1, 1)
    var3 = var.expand_as(y)
    out = torch.zeros_like(y)
    for c2 in range(3):                                                          # sum over the input channel c'
        J = y * rp * w[:, c2:c2 + 1]                                             # y_c r' w_c'
        J = J + (torch.arange(3, device=y.device).view(1, 3, 1, 1) == c2).to(y.dtype) * r
        out = out + J ** 2 * var3[:, c2:c2 + 1]
    return out


def _luma(x):
    w = torch.tensor(RGB2Y, device=x.device, dtype=x.dtype).view(1, 3, 1, 1)
    return (x * w).sum(1, keepdim=True)


def _num_deriv(f, v, h=1e-3):
    return (f(v + h) - f(v - h)) / (2 * h)


# ------------------------------------------------------------------- jpeg --
def jpeg_cv2(x: torch.Tensor, q: int, subsampling: str = "420") -> torch.Tensor:
    """Real codec (OpenCV/libjpeg) for SYNTHESIS. Not differentiable; the data
    term uses `jpeg_coeff_residual` instead."""
    import cv2
    import numpy as np
    flag = {"420": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_420, "444": cv2.IMWRITE_JPEG_SAMPLING_FACTOR_444}[subsampling]
    out = []
    for im in x:
        a = (im.clamp(0, 1).permute(1, 2, 0).cpu().numpy()[..., ::-1] * 255).round().astype(np.uint8)
        ok, enc = cv2.imencode(".jpg", a, [int(cv2.IMWRITE_JPEG_QUALITY), int(q),
                                           int(cv2.IMWRITE_JPEG_SAMPLING_FACTOR), int(flag)])
        assert ok
        d = cv2.imdecode(enc, cv2.IMREAD_COLOR)[..., ::-1].astype(np.float32) / 255
        out.append(torch.from_numpy(d.copy()).permute(2, 0, 1))
    return torch.stack(out).to(x)


def ycc_linear(r255: torch.Tensor):
    """Linear part of the JFIF RGB->YCbCr transform (offsets cancel in a
    residual). r255: [B,3,H,W] in 0..255 units."""
    m = torch.tensor([RGB2Y, RGB2CB, RGB2CR], device=r255.device, dtype=r255.dtype)
    return torch.einsum("ij,bjhw->bihw", m, r255)


YCC_NOISE_GAIN = (sum(c * c for c in RGB2Y), sum(c * c for c in RGB2CB), sum(c * c for c in RGB2CR))
