"""Synthetic blur for training and evaluating the kernel module.

SIZES / WEIGHTS: the kernel canvases drawn in training (31..79 px).
blur_noise: y = k * x + n with reflect padding (valid region), uint8-rounded.
eval_sets: fixed per-family kernel sets (TRAIN families and HELD-out families
disk / ring / curved motion) used by every kernel-module evaluation.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from etr.ops import kernels as K
from etr.world.proposal import canvas, CANVAS


SIZES, WEIGHTS = (31, 47, 63, 65, 75, 79), (0.35, 0.2, 0.15, 0.1, 0.1, 0.1)


def blur_noise(x, k, sig):
    """x: [B,3,256,256], k: [B,1,79,79] -> blurred (reflect-padded, valid) + noise, 8-bit."""
    B = x.shape[0]
    r = CANVAS // 2
    xp = F.pad(x, [r] * 4, mode="reflect")
    y = F.conv2d(xp.reshape(1, B * 3, *xp.shape[-2:]), k.flip(-2, -1).repeat_interleave(3, 0), groups=B * 3)
    y = y.view(B, 3, *x.shape[-2:])
    y = y + torch.randn_like(y) * sig.view(B, 1, 1, 1)
    return (y.clamp(0, 1) * 255).round() / 255


def eval_sets(n=32):
    g = torch.Generator().manual_seed(3)
    r = lambda a, b: a + (b - a) * torch.rand((), generator=g).item()
    ang = lambda: r(0, math.pi)
    fam = {
        "train/gauss":        lambda: K.gaussian(31, r(1, 3)),
        "train/aniso":        lambda: K.anisotropic_gaussian(31, r(1, 4), r(0.6, 2), ang()),
        "train/motion5-15":   lambda: K.motion(31, int(r(5, 15)), ang()),
        "train/motion15-29":  lambda: K.motion(31, int(r(15, 29)), ang()),
        "train/motion30-60":  lambda: K.motion(63, int(r(30, 60)), ang()),
        "train/traj31":       lambda: K.random_trajectory(31, r(8, 24), g),
        "train/traj63":       lambda: K.random_trajectory(63, r(16, 48), g),
        "train/sampler79":    lambda: K.sample_kernel(79, "train", g),
        "train/disk":         lambda: K.disk(31, r(2, 8)),
        "HELD/ring":          lambda: K.ring(31, *(lambda ro: (ro, ro * r(0.3, 0.85)))(r(4, 13))),
        "HELD/curved":        lambda: K.motion_curved(31, r(10, 25), r(0.3, 1.0), ang()),
        "HELD/sampler63":     lambda: K.sample_kernel(63, "heldout", g),
    }
    out = {}
    for name, f in fam.items():
        ks = []
        for _ in range(n):
            k = f().float().clamp(min=0)
            ks.append(canvas(k / k.sum()))
        out[name] = torch.stack(ks)[:, None]
    return out


def sample_blur_chain(rng, hw, device="cuda", max_tries=200):
    """A world-model training chain (etr/world/generator.py: TRAIN families only) that
    contains exactly one blur acting at FULL resolution (no resample before it), mixed
    in random order with whatever else the generator draws: tone curves, haze, rain,
    noise, JPEG, downsampling: the composite setting the blur module meets at test time."""
    from etr.world.generator import sample_chain
    for _ in range(max_tries):
        st = sample_chain(rng, hw, device)
        kinds = [s.kind for s in st]
        if kinds.count("blur") != 1 or "resample" in kinds[:kinds.index("blur")]:
            continue
        return st
    from etr.world.chain import Step
    from etr.world.generator import Q8
    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
    return [Step("blur", dict(k=K.sample_kernel(31, "train", g).to(device)), Q8),
            Step("noise", dict(type="gauss", sigma=5 / 255, var=(5 / 255) ** 2), Q8)]


def synth_blur_chain(x, rng, linearised=False, perturb=True, depth=None, p_simple=0.0):
    """x [1,3,H,W] clean -> (y_sig [1,3,H,W], k [1,1,79,79], y [observation]) and, when linearised,
    also (x_ref [1,3,H,W], sigma): y_sig is the blur-stage observation (etr/kernel/linearise.py:
    the factors after the blur undone with the TRUE chain plus chain-estimator-like errors), x_ref the clean
    blur-stage reference (the blur step's input, dehazed like y_sig), sigma the noise level measured
    on y_sig. Without linearised: y_sig is the raw observation, bicubic-upsampled back to H x W.
    p_simple: share of plain blur + Gaussian noise chains (keeps the clean regime)."""
    from etr.world.generator import synthesize, Q8
    from etr.world.chain import Step
    if rng.random() < p_simple:
        g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
        sig = rng.uniform(1, 25) / 255
        st = [Step("blur", dict(k=K.sample_kernel(rng.choices(SIZES, WEIGHTS)[0], "train", g).to(x.device)), Q8),
              Step("noise", dict(type="gauss", sigma=sig, var=sig ** 2), Q8)]
    else:
        st = sample_blur_chain(rng, x.shape[-2:], x.device)
    y = synthesize(x, st, rng, depth=depth)
    k = canvas([q for q in st if q.kind == "blur"][0].p["k"].float())[None, None].to(x.device)
    if linearised:
        from etr.kernel.linearise import blur_stage_reference, lin_from_steps, linearise
        lin, ref_haze = lin_from_steps(st, rng, perturb)
        y_sig, sig = linearise(y, lin)
        x_ref = blur_stage_reference([q for q in st if q.kind == "blur"][0].p["_pre"], ref_haze)
        if y_sig.shape[-2:] != x_ref.shape[-2:]:                    # odd sizes after resampling
            y_sig = F.interpolate(y_sig, size=x_ref.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
        return y_sig, k, y, x_ref, sig
    s = 1
    for q in st:
        if q.kind == "resample":
            s *= q.p["s"]
    y_sig = y
    if s > 1:
        y_sig = F.interpolate(y, size=x.shape[-2:], mode="bicubic", align_corners=False).clamp(0, 1)
    return y_sig, k, y
