"""Phase 2b: convergent GS-PnP proximal gradient.

    minimise  F(z) = beta * f(z) + g(x(z))   over z in [0,1]^n
    f          convex quadratic data term (etr.world.chain.DataTerm)
    g(x)       = sum_t 0.5 * || m_t * (x_t - N_sigma(x_t)) ||^2, the gradient-
                 step potential of the image prior, evaluated on overlapping
                 tiles with partition-of-unity weights m_t (a full-resolution
                 VJP does not fit in memory). This is a smooth function in its
                 own right, so the theorem is applied to exactly this g.
    x(z)       identity, or the inverse of the prefix tone curves

    z+ = prox_{tau*beta*f + box}( z - tau * grad g(z) )      (Hurault et al. 2022)

with backtracking on tau until F(z) - F(z+) >= (gamma/tau) ||z+ - z||^2.
Every accepted step satisfies that sufficient-decrease inequality, so F is
monotone, sum ||z+ - z||^2 < inf, and (KL property: analytic N, quadratic f,
box) the iterates converge to a critical point. The run stops when the
relative decrease of F falls below `tol`; the iteration count is reported.
"""
from __future__ import annotations

import math
import time

import torch

from etr.world.chain import DataTerm

# TF32 convolutions carry ~1e-3 relative error, enough to make the
# sufficient-decrease test (and any monotonicity claim) meaningless.
torch.backends.cudnn.allow_tf32 = False
torch.backends.cuda.matmul.allow_tf32 = False


class GSPotential:
    def __init__(self, net, goal_vec, sigma: float, tile: int = 384, overlap: int = 48, batch: int = 4,
                 amp: bool = False):
        self.net, self.goal, self.sigma = net, goal_vec, float(sigma)
        # ETR_TILE_BATCH caps the tile batch on a shared GPU (a batch only groups tiles: the partition of unity and the
        # per-tile gradients are the same; starting above what fits makes every solve hit OOM, halve and empty the cache)
        import os
        cap = int(os.environ.get("ETR_TILE_BATCH", "0"))
        batch = min(batch, cap) if cap > 0 else batch
        self.tile, self.ov, self.batch = tile, overlap, batch
        # amp: fp16 autocast for the prior GRADIENT only (the potential's VALUES, which the
        # sufficient-decrease test and the monotone-F record use, stay fp32); the solver turns it
        # off for the final iterations so the stationarity claim is made with exact gradients
        self.amp = amp
        self._masks = {}

    def _layout(self, H, W, dev):
        key = (H, W)
        if key in self._masks:
            return self._masks[key]
        t = self.tile
        th, tw = min(t, H), min(t, W)
        step_h, step_w = max(th - self.ov, 1), max(tw - self.ov, 1)
        ys = list(range(0, H - th + 1, step_h))
        xs = list(range(0, W - tw + 1, step_w))
        if ys[-1] != H - th:
            ys.append(H - th)
        if xs[-1] != W - tw:
            xs.append(W - tw)

        def ramp(n):
            r = torch.ones(n, device=dev)
            o = min(self.ov, n // 2)
            if o > 0:
                e = torch.linspace(0, 1, o + 2, device=dev)[1:-1]
                r[:o] = e
                r[-o:] = e.flip(0)
            return r
        wt = ramp(th)[:, None] * ramp(tw)[None, :]
        tot = torch.zeros(H, W, device=dev)
        boxes = [(y, x) for y in ys for x in xs]
        for y, x in boxes:
            tot[y:y + th, x:x + tw] += wt
        masks = [(wt / tot[y:y + th, x:x + tw]) for y, x in boxes]   # partition of unity
        self._masks[key] = (boxes, th, tw, masks)
        return self._masks[key]

    def __call__(self, x: torch.Tensor, need_grad: bool = True):
        """Returns g(x) (float) and, if need_grad, grad g(x). The tile batch adapts: halved and
        remembered on a CUDA out-of-memory error (shared GPU)."""
        while True:
            try:
                return self._call(x, need_grad)
            except torch.cuda.OutOfMemoryError:
                if self.batch <= 1:
                    raise
                self.batch = max(1, self.batch // 2)
                torch.cuda.empty_cache()

    def _call(self, x: torch.Tensor, need_grad: bool = True):
        B, C, H, W = x.shape
        assert B == 1
        boxes, th, tw, masks = self._layout(H, W, x.device)
        gval = 0.0
        grad = torch.zeros_like(x) if need_grad else None
        for i in range(0, len(boxes), self.batch):
            bx = boxes[i:i + self.batch]
            m = torch.stack(masks[i:i + self.batch])[:, None]                # [b,1,th,tw]
            xt = torch.cat([x[..., y:y + th, xx:xx + tw] for y, xx in bx]).detach()
            sig = torch.full((xt.shape[0],), self.sigma, device=x.device)
            gv = self.goal.expand(xt.shape[0], -1)
            if need_grad:
                with torch.enable_grad():
                    xt.requires_grad_(True)
                    with torch.autocast("cuda", dtype=torch.float16, enabled=self.amp):
                        out = self.net(xt, sig, gv)
                    r = xt - out.float()
                    gt = 0.5 * (m * r * r).sum()
                    (dg,) = torch.autograd.grad(gt, xt)
                    dg = dg.float()
                for j, (y, xx) in enumerate(bx):
                    grad[..., y:y + th, xx:xx + tw] += dg[j:j + 1]
            else:
                with torch.no_grad():
                    r = xt - self.net(xt, sig, gv)
                    gt = 0.5 * (m * r * r).sum()
            gval += gt.item()
        return gval, grad

    @torch.no_grad()
    def denoise(self, x):
        """D(x) = x - grad g(x): one GS denoising pass (the PnP floor)."""
        _, gr = self(x, need_grad=True)
        return (x - gr).clamp(0, 1)


def prox_box_quadratic(dt: DataTerm, v, c, z0, Lf, tol=1e-6, max_it=300):
    """argmin_u 0.5||u - v||^2 + c f(u)  s.t. u in [0,1]^n. Strongly convex
    (mu=1, L=1+c*Lf): projected FISTA with the constant strongly-convex
    momentum. Stops on relative change < tol."""
    L = 1.0 + c * Lf
    kappa = L
    beta_m = (math.sqrt(kappa) - 1) / (math.sqrt(kappa) + 1)
    u = z0.clamp(0, 1).clone()
    u_prev = u.clone()
    it = 0
    for it in range(1, max_it + 1):
        w = u + beta_m * (u - u_prev)
        gw = (w - v) + c * dt.grad_f(w)
        u_prev = u
        u = (w - gw / L).clamp(0, 1)
        if (u - u_prev).norm() <= tol * max(u.norm().item(), 1e-12):
            break
    return u, it


torch.backends.cudnn.benchmark = True


def solve(dt: DataTerm, pot: GSPotential, beta: float, max_iter: int = 100, tol: float = 1e-4,
          tau0: float = 1.0, gamma: float = 0.1, tau_max: float = 2.0, z0=None, log=False, prox_max_it: int = 300):
    """Phase 2b. Returns (x_hat cropped to the observed HR region, info)."""
    t0 = time.time()
    z = (dt.z_init() if z0 is None else z0).detach()
    Lf = dt.lipschitz(z)

    def g_and_grad(z):
        if not dt.prefix:
            return pot(z, need_grad=True)
        with torch.enable_grad():
            zz = z.detach().requires_grad_(True)
            x = dt.x_from_z(zz)
            gv, gx = pot(x.detach(), need_grad=True)
            (gz,) = torch.autograd.grad(x, zz, grad_outputs=gx)
        return gv, gz

    def g_only(z):
        return pot(dt.x_from_z(z) if dt.prefix else z, need_grad=False)[0]

    with torch.no_grad():
        fz = dt.f(z).item()
    gz, grad = g_and_grad(z)
    if getattr(pot, "amp", False):
        gz = g_only(z)                                     # exact fp32 value for the descent test
    F = beta * fz + gz
    hist = [F]
    tau, n_bt, inner = tau0, 0, 0
    monotone = True
    it = 0
    for it in range(1, max_iter + 1):
        while True:
            v = z - tau * grad
            zn, k = prox_box_quadratic(dt, v, tau * beta, z, Lf, max_it=prox_max_it)
            inner += k
            with torch.no_grad():
                fn = dt.f(zn).item()
            gn = g_only(zn)
            Fn = beta * fn + gn
            step2 = ((zn - z) ** 2).sum().item()
            accepted = F - Fn >= gamma / tau * step2 or step2 < 1e-16
            if accepted:
                break
            tau *= 0.5
            n_bt += 1
            if tau < 1e-6:
                break
        if not accepted:
            # no step size gives sufficient decrease: z is (numerically) critical
            break
        monotone &= Fn <= F + 1e-9 * abs(F)
        rel = (F - Fn) / max(abs(hist[0]), 1e-12)
        z, F = zn, Fn
        hist.append(F)
        if log:
            print(f"  it {it:3d} F={F:.6e} f={fn:.4e} g={gn:.4e} tau={tau:.3g} inner={k}")
        if rel < tol:
            if getattr(pot, "amp", False):                 # finish with exact gradients
                pot.amp = False
                gz, grad = g_and_grad(z)
                continue
            break
        if getattr(pot, "amp", False) and rel < 10 * tol:
            pot.amp = False
        tau = min(tau * 1.25, tau_max)
        gz, grad = g_and_grad(z)
    x = dt.x_from_z(z) if dt.prefix else z
    info = dict(iters=it, F=hist, monotone=bool(monotone), backtracks=n_bt, inner=inner,
                tau=tau, Lf=Lf, sec=time.time() - t0)
    return dt.crop_hr(x).clamp(0, 1), info


DEFAULT_HP = dict(kappa=1.0, lam=0.2, sigma_min=2 / 255, max_iter=100, tol=1e-4)


def noise_scale(dt: DataTerm) -> float:
    """Effective observation noise std (in [0,1] units) the prior strength is
    tied to: RMS of the data-term standard deviations over valid pixels, or
    over DCT coefficients when the chain ends in JPEG."""
    if dt.jpeg is not None:
        # coefficients quantised to index 0 sit in the dead zone, where the true
        # error variance is ~0.2 (luma) / ~0.04 (chroma) of the uniform Q^2/12 (measured, tests/test_world.py
        # t_jpeg); counting them at full Q^2/12 would pin sigma_d at its cap
        yc = dt._jpeg_coeffs(dt.y_tilde)
        from etr.ops.jpeg import quant_table
        Q = quant_table(dt.jpeg.p["q"], dt.y.device)[0]
        tot, n = 0.0, 0
        for w, c, Qi, rho in zip(dt._jw, yc, (Q[0], Q[1]), (0.2, 0.04)):   # luma, chroma
            v = (1.0 / w).expand_as(c)
            zero = (c / Qi).round().abs() == 0
            tot += (v * torch.where(zero, rho, 1.0)).sum().item()
            n += c.numel()
        return math.sqrt(tot / n) / 255.0
    m = dt.valid > 0
    return float(dt.var[m].mean().sqrt()) if m.any() else 1 / 255


def solve_hp(dt: DataTerm, net, goal_vec, hp=None, **kw):
    """Phase 2b with the prior strength set by the rule
    sigma_d = clamp(kappa * sigma_ref, sigma_min, 50/255), beta = lam * sigma_d^2
    (hyperparameters tuned once on synthetic chains over validation images)."""
    hp = dict(DEFAULT_HP, **(hp or {}))
    s_ref = noise_scale(dt)
    if not dt.core and dt.jpeg is None and dt.prefix:
        dt = DataTerm(dt.y, dt.steps, pad=dt.pad, invert_prefix_if_no_core=True)
    if not dt.core and not dt.prefix and dt.jpeg is None:
        # H = I: the MMSE estimate under the prior is Tweedie's D_sigma(y~) in closed
        # form (measured on DF2K-val: 30.85 dB vs 21.76 for the MAP iterates at sigma 25),
        # with the per-pixel noise std map (Poisson, tone-amplified) as sigma.
        smap = (dt.var.mean(1, keepdim=True).sqrt() * hp.get("direct_scale", 1.0)).clamp(1 / 255, 50 / 255)
        x = mmse_readout(net, goal_vec, dt.y_tilde, smap, hp.get("batch", 4))
        return x, dict(iters=0, F=[0.0], monotone=True, backtracks=0, inner=0, sec=0.0,
                       sigma_ref=s_ref, sigma_d=float(smap.mean()), mode="direct")
    if hp.get("pre_denoise_rho") and dt.core and dt.jpeg is None:
        # heavy noise: the MAP point of the GS potential is a poor estimator when
        # noise dominates (DF2K-val, blur+sigma30: 21.25 dB vs 22.93 for the floor).
        # Stage 1: Tweedie MMSE denoising of y~ with its own noise map. Stage 2:
        # the convergent deconvolution on that observation with the residual
        # variance rho^2 * var.
        smap = dt.var.mean(1, keepdim=True).sqrt().clamp(1 / 255, 50 / 255)
        yd = mmse_readout(net, goal_vec, dt.y_tilde, smap, hp.get("batch", 4))
        dt = dt.with_observation(yd, hp["pre_denoise_rho"] ** 2)
        s_ref = noise_scale(dt)
    sd = min(max(hp["kappa"] * s_ref, hp["sigma_min"]), 50 / 255)
    pot = GSPotential(net, goal_vec, sd, batch=hp.get("batch", 4), tile=hp.get("tile", 384), amp=hp.get("amp", False))
    x, info = solve(dt, pot, beta=hp["lam"] * sd ** 2, max_iter=hp["max_iter"], tol=hp["tol"],
                    prox_max_it=hp.get("prox_max_it", 300), **kw)
    info.update(sigma_ref=s_ref, sigma_d=sd, mode="pgd", x_map=x)
    if hp.get("readout", 0) > 0:
        # MMSE readout of the converged MAP point; the guarantee is on x_map
        x = mmse_readout(net, goal_vec, x, max(hp["readout"] * s_ref, 1 / 255), hp.get("batch", 4))
    return x, info


def mmse_readout(net, goal_vec, x, sigma, batch=4):
    """One GS denoising pass D_sigma(x) = x - grad g(x); sigma scalar or a
    [1,1,H,W] map. Tiled exactly like the potential."""
    if torch.is_tensor(sigma):
        pot = _MapPotential(net, goal_vec, sigma, batch=batch)
    else:
        pot = GSPotential(net, goal_vec, sigma, batch=batch)
    return pot.denoise(x)


class _MapPotential(GSPotential):
    """GSPotential with a per-pixel sigma map (tiles crop the map too)."""

    def __init__(self, net, goal_vec, smap, **kw):
        super().__init__(net, goal_vec, float(smap.mean()), **kw)
        self.smap = smap

    def _call(self, x, need_grad=True):            # GSPotential.__call__ adds the adaptive tile batch
        B, C, H, W = x.shape
        boxes, th, tw, masks = self._layout(H, W, x.device)
        gval, grad = 0.0, torch.zeros_like(x)
        for i in range(0, len(boxes), self.batch):
            bx = boxes[i:i + self.batch]
            m = torch.stack(masks[i:i + self.batch])[:, None]
            xt = torch.cat([x[..., y:y + th, xx:xx + tw] for y, xx in bx]).detach()
            st = torch.cat([self.smap[..., y:y + th, xx:xx + tw] for y, xx in bx])
            gv = self.goal.expand(xt.shape[0], -1)
            with torch.enable_grad():
                xt.requires_grad_(True)
                r = xt - self.net(xt, st, gv)
                gt = 0.5 * (m * r * r).sum()
                (dg,) = torch.autograd.grad(gt, xt)
            for j, (y, xx) in enumerate(bx):
                grad[..., y:y + th, xx:xx + tw] += dg[j:j + 1]
            gval += gt.item()
        return gval, grad
