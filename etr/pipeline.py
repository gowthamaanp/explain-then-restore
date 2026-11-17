"""ETR inference: explain an image (estimate its degradation chain), then restore it (invert the chain).

    from etr.pipeline import ETR
    etr = ETR()                                  # configs/etr.json, checkpoints/
    out = etr.restore(y, scale=1)                # y: [1,3,H,W] RGB in [0,1] on the GPU
    out["image"], out["solver"], out["chain"]    # refined output, solver output, estimated chain (JSON-safe)

`scale` is the ratio of the requested output size to the input size (1, 2, 4 or 8); it is not estimated.
"""
from __future__ import annotations

import json
import os
import time

import torch
import torch.nn.functional as F

from etr.kernel.estimator import KernelEstimator
from etr.models.refiner import load_refiner, refine, refine_d4
from etr.world.chain import DataTerm
from etr.world.infer import blind_chain, load_proposal
from etr.world.prior_io import load_prior
from etr.world.solver import solve_hp
from etr.world.solver_hp import hp_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def chain_summary(th, steps):
    """JSON-safe account of the estimated chain: the ordered steps with their parameters, the detection logits and any
    fallback taken (a blur step dropped because no kernel beat the no-blur delta)."""
    out = dict(chain=[q.kind for q in steps], present=sorted(th["present"]),
               presence_logits=dict(zip(("blur", "resample", "noise", "jpeg", "tone", "haze", "layer"),
                                        [round(float(v), 3) for v in th.get("presence_logits", [])])),
               fallback=th.get("fallback", []), steps=[])
    for q in steps:
        d = dict(kind=q.kind)
        if q.kind == "blur":
            k = q.p["k"].detach().float().cpu()
            yy, xx = torch.meshgrid(torch.arange(k.shape[-2]), torch.arange(k.shape[-1]), indexing="ij")
            c = (k.shape[-2] // 2, k.shape[-1] // 2)
            m = k.sum()
            d.update(centroid=[round(float((k * yy).sum() / m) - c[0], 2), round(float((k * xx).sum() / m) - c[1], 2)],
                     support_px=int((k > k.max() * 0.05).sum()))
        elif q.kind == "resample":
            d.update(scale=int(q.p["s"]), family=q.p["family"], antialias=bool(q.p["antialias"]))
        elif q.kind == "tone":
            d.update(curve=q.p.get("kind"), **{kk: v for kk, v in q.p.items() if kk in ("kx", "ky", "knots_y", "space")})
        elif q.kind == "haze":
            t = q.p["t"]
            d.update(airlight=round(float(q.p["A"]), 4), transmission_mean=round(float(t.mean() if torch.is_tensor(t) else t), 4))
        elif q.kind == "noise":
            d.update(type=q.p["type"], **{kk: round(float(v), 6) for kk, v in q.p.items() if kk in ("sigma", "alpha", "beta")})
        elif q.kind == "jpeg":
            d.update(quality=int(q.p["q"]), subsampling=q.p.get("subsampling"))
        elif q.kind == "layer":
            d.update(rain_mean=round(float(q.p["L"].mean()), 5))
        out["steps"].append(d)
    return out


class ETR:
    def __init__(self, config=os.path.join(ROOT, "configs", "etr.json"), device="cuda", tile_batch=4):
        assert device == "cuda", "ETR runs on a CUDA GPU"
        cfg = json.load(open(config))
        P = lambda p: p if os.path.isabs(p) else os.path.join(ROOT, p)
        ck = {k: P(v) for k, v in cfg["checkpoints"].items()}
        self.cfg, self.device, self.tile_batch = cfg, device, tile_batch
        self.hp_path = P(cfg["solver_hp"])
        self.prior, self.goal = load_prior(ck["prior"], device=device)
        self.estimator = load_proposal(ck["chain_estimator"], device)
        self.kernels = KernelEstimator(ck["kernel_vae"], ck["signature_head"], ck["critic"], ck.get("kernel_bank", ""),
                                       **cfg.get("estimator", {}))
        self.refiner = load_refiner(ck["refiner"], device) if ck.get("refiner") else None

    @torch.no_grad()
    def explain(self, y, scale=1, seed=0):
        """y [1,3,H,W] in [0,1] -> (steps, theta): the estimated, ordered chain."""
        torch.manual_seed(seed)                      # the kernel module's latent perturbations
        return blind_chain(self.estimator, self.kernels, y, scale, dict(self.cfg["inference"]))

    def invert(self, y, steps):
        """Minimise the chain's data term under the image prior (monotone proximal gradient)."""
        if not steps:
            return y
        dt = DataTerm(y, steps, pad=48)
        hp, _ = hp_for(steps, dt, path=self.hp_path, max_iter=self.cfg.get("max_iter", 40))
        extra = dict(batch=self.tile_batch, amp=True) if self.cfg.get("amp", False) else dict(batch=1)
        return solve_hp(dt, self.prior, self.goal, dict(hp, **extra))[0]

    def restore(self, y, scale=1, seed=0, refine_output=True):
        """y [1,3,H,W] RGB in [0,1] (on the GPU). Returns dict(image, solver, chain, steps, seconds)."""
        y = y.to(self.device).float().clamp(0, 1)
        t0 = time.time()
        steps, th = self.explain(y, scale, seed)
        x = self.invert(y, steps)
        hw = (y.shape[-2] * scale, y.shape[-1] * scale)
        if x.shape[-2:] != hw:
            x = F.interpolate(x, size=hw, mode="bicubic", align_corners=False)
        x = x.clamp(0, 1)
        xr = x
        if refine_output and self.refiner is not None:
            xr = (refine_d4 if self.cfg.get("refine_d4", True) else refine)(self.refiner, y, x)
        torch.cuda.synchronize()
        return dict(image=xr, solver=x, chain=chain_summary(th, steps), steps=steps, seconds=time.time() - t0)
