"""Blur-stage observation: undo the factors that act AFTER the blur, so the blur module reads
(approximately) k * x + n instead of the full chain's output.

The blur module used to read the raw observation, with darkening, rain, downscaling and haze
still on it; on mixed chains its candidate pool rarely held a usable kernel. One function,
`linearise`, is used identically in training (from the TRUE chain with simulated Stage-B-like
errors: `lin_from_steps`) and at inference (from the chain estimator's estimate: `lin_from_theta`), so the
networks are trained on exactly what they will see.

Inverted (last applied first): trailing tone curve, additive rain layer, downscale (bicubic
upsample back to the blur grid), haze (t clamped >= 0.2 so the inverse does not explode noise).
A tone curve applied BEFORE the blur is left alone (y ~ k * T(x): its inverse would not give
k * x); haze is undone wherever it is (its transmission is smooth, so it nearly commutes).
Noise and JPEG are not inverted; the blur-stage noise level is measured on the result.
"""
from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from etr.kernel.restore import sigma_mad
from etr.world import ops as O

T_MIN = 0.2


def linearise(y, lin):
    """y [1,3,h,w] observation; lin: dict(s, tone=None|dict(kind,..., space), rain=None|L [1,1|3,h,w],
    haze=None|dict(A, t)). -> (y_lin [1,3,H,W] on the blur grid, sigma (scalar tensor))."""
    z = y
    if lin.get("tone") is not None:
        tp = lin["tone"]
        z = O.tone_inv(z, tp["kind"], tp, tp.get("space", "hsv_v"))[0]
    if lin.get("rain") is not None:
        L = lin["rain"].to(z)
        if L.shape[-2:] != z.shape[-2:]:
            L = F.interpolate(L, size=z.shape[-2:], mode="area")
        z = z - L.expand_as(z) * (y < 0.996).float()
    s = lin.get("s", 1)
    if s > 1:
        z = F.interpolate(z, scale_factor=s, mode="bicubic", align_corners=False)
    if lin.get("haze") is not None:
        A, t = lin["haze"]["A"], lin["haze"]["t"]
        if torch.is_tensor(t):
            t = t.to(z)
            if t.shape[-2:] != z.shape[-2:]:
                t = F.interpolate(t, size=z.shape[-2:], mode="bilinear", align_corners=False)
            t = t.clamp(min=T_MIN)
        else:
            t = max(float(t), T_MIN)
        z = O.haze_inv(z, t, A)
    z = torch.nan_to_num(z, nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)   # clamp alone keeps NaN
    return z, sigma_mad(z)[0]


def lin_from_theta(th, y, s):
    """Inference: the same inversions from the chain estimator's estimate (etr/world/infer.py: estimate_theta),
    following build_chain's canonical order (tone trails the linear factors unless tone_first, JPEG is
    present, or there are no linear factors)."""
    pres = set(th["present"])
    lin = dict(s=s, tone=None, rain=None, haze=None)
    lin_factors = bool(pres & {"blur", "haze"}) or s > 1
    if "tone" in pres and not (th["order"][0] or "jpeg" in pres or not lin_factors):
        lin["tone"] = th["tone"]
    if "layer" in pres and th.get("rain") is not None:
        lin["rain"] = th["rain"]
    if "haze" in pres:
        lin["haze"] = dict(A=th["haze"]["A"], t=th["haze"]["t"])
    return lin


def _smooth_noise(shape, rng, device):
    g = torch.Generator(device="cpu").manual_seed(rng.randrange(2 ** 31))
    n = torch.rand(1, 1, 6, 6, generator=g) * 2 - 1
    return F.interpolate(n, size=shape, mode="bicubic", align_corners=False).to(device)


# miss rates of the chain estimator on synthetic validation, used to simulate its errors in training
MISS = dict(tone=0.15, haze=0.12, layer=0.2)


def lin_from_steps(steps, rng: random.Random, perturb=True):
    """Training: inversions from the TRUE chain (steps as filled in by generator.synthesize), with
    Stage-B-like errors when perturb. Returns (lin, ref_haze) where ref_haze is the haze applied
    BEFORE the blur (so the clean blur-stage reference can be dehazed the same way), or None."""
    kinds = [q.kind for q in steps]
    b = kinds.index("blur")
    lin = dict(s=1, tone=None, rain=None, haze=None)
    ref_haze = None
    for i, q in enumerate(steps):
        if q.kind == "resample" and i > b:
            lin["s"] *= q.p["s"]
        elif q.kind == "tone" and i > b:
            if perturb and rng.random() < MISS["tone"]:
                continue
            tp = dict(q.p)
            if perturb:                                          # knot-level error, as the chain estimator makes
                V = torch.linspace(0, 1, 17)
                u = O.curve(V, tp["kind"], tp).clamp(0, 1) + torch.randn(17, generator=torch.Generator().manual_seed(
                    rng.randrange(2 ** 31))) * 0.05
                u = torch.cummax(u.clamp(0, 1), 0).values + 1e-4 * torch.arange(17)
                tp = dict(kind="pwl", kx=V.tolist(), ky=u.tolist(), space=q.p.get("space", "hsv_v"))
            lin["tone"] = tp
        elif q.kind == "layer" and torch.is_tensor(q.p.get("L")):
            if perturb and rng.random() < MISS["layer"]:
                continue
            L = q.p["L"][:, :1]
            lin["rain"] = L * (rng.uniform(0.4, 1.0) if perturb else 1.0)
        elif q.kind == "haze":
            A, t = q.p["A"], q.p["t"]
            if i < b:
                ref_haze = dict(A=A, t=t)
            if perturb and rng.random() < MISS["haze"]:
                continue
            if perturb:
                A = min(max(A + rng.uniform(-0.05, 0.05), 0.3), 1.0)
                if torch.is_tensor(t):
                    t = (t * (1 + 0.1 * _smooth_noise(t.shape[-2:], rng, t.device))).clamp(0.02, 1)
                else:
                    t = min(max(t * (1 + rng.uniform(-0.1, 0.1)), 0.02), 1.0)
            lin["haze"] = dict(A=A, t=t)
    # the observation grid: the layer acts where it was applied; if it was applied before a resample its
    # map is at the higher resolution and `linearise` area-downsamples it to the observation grid
    return lin, ref_haze


def blur_stage_reference(x_pre, ref_haze):
    """Clean blur-stage reference for scoring candidates: the blur step's input, dehazed like y_lin."""
    if ref_haze is None:
        return x_pre
    t = ref_haze["t"]
    if torch.is_tensor(t):
        t = t.to(x_pre).clamp(min=T_MIN)
        if t.shape[-2:] != x_pre.shape[-2:]:
            t = F.interpolate(t, size=x_pre.shape[-2:], mode="bilinear", align_corners=False)
    else:
        t = max(float(t), T_MIN)
    return torch.nan_to_num(O.haze_inv(x_pre, t, ref_haze["A"]), nan=0.0, posinf=1.0, neginf=0.0).clamp(0, 1)
