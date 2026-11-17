"""Solver hyperparameters chosen from the estimated chain alone (usable blind), tuned once on synthetic chains over
validation images (configs/solver_hp.json).

Category: JPEG if present, else resampling ("sr"), else affine steps with the propagated noise below / above 3/255
("lin_low" / "lin_noise"), else pure denoising. Chains whose propagated noise exceeds `heavy_sigma` use the
two-stage setting (denoise the observation first, then solve with the residual variance `pre_denoise_rho`^2 v).
"""
from __future__ import annotations

import json

from etr.world.solver import noise_scale

_CACHE = {}


def _load(path):
    if path not in _CACHE:
        _CACHE[path] = json.load(open(path))
    return _CACHE[path]


def category(steps, dt, heavy_sigma):
    kinds = {s.kind for s in steps}
    s_ref = noise_scale(dt)
    heavy = s_ref > heavy_sigma
    if "jpeg" in kinds:
        return "jpeg", heavy
    if "resample" in kinds:
        return "sr", heavy
    if kinds & {"blur", "haze", "layer"}:
        return ("lin_low" if s_ref < 3 / 255 else "lin_noise"), heavy
    return "denoise", heavy


def hp_for(steps, dt, path="configs/solver_hp.json", max_iter=40):
    H = _load(path)
    cat, heavy = category(steps, dt, H["heavy_sigma"])
    if heavy and cat in ("lin_low", "lin_noise"):
        hp = dict(H["heavy"]["blur+n30"])
    elif heavy and cat == "sr":
        hp = dict(H["heavy"]["x4+n30"])
    elif cat == "denoise":
        hp = dict(H["heavy"]["denoise25"])
    else:
        hp = dict(H["categories"][cat])
    hp["max_iter"] = max_iter
    return hp, cat
