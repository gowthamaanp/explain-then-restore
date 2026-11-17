"""Training degradation generator: DF2K patch -> (y, theta).

Draws 1-4 factors from the TRAINED families only, applies them in a random
order (the order pi is part of theta; JPEG, when present, is always last),
and returns the chain as a list of `Step`s with the true parameters, so the
same object supervises the proposal network and builds the oracle data term.

Orders the convex data term cannot express (a tone curve sandwiched between
linear factors, or a tone after a linear factor with JPEG at the end; see
etr/world/chain.py) are re-drawn, so every training sample is one the
solver can represent. Nothing here reads any benchmark's code or data.
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F

from etr.ops import kernels as K
from etr.world import ops as O
from etr.world.chain import Step, parse
from etr.world.resamplers import resampler_psf
from etr.world.proposal_consts import RS_FAMILIES

Q8 = 1.0 / (12 * 255.0 ** 2)
P_FACTOR = dict(blur=0.6, noise=0.6, jpeg=0.3, resample=0.35, tone=0.3, haze=0.25, layer=0.2)


# ------------------------------------------------------------ fields --
def smooth_field(h, w, rng: random.Random, device, octaves=4):
    """Random smooth field in [0,1]: sum of bicubically upsampled noise."""
    f = torch.zeros(1, 1, h, w, device=device)
    amp, tot = 1.0, 0.0
    g = torch.Generator(device=device).manual_seed(rng.randrange(2 ** 31))
    for o in range(octaves):
        n = 2 + 2 ** o
        f = f + amp * F.interpolate(torch.rand(1, 1, n, n, generator=g, device=device), size=(h, w),
                                    mode="bicubic", align_corners=False)
        tot += amp
        amp *= 0.5
    f = f / tot
    return (f - f.min()) / (f.max() - f.min()).clamp(min=1e-6)


def guided_filter(I, p, r=8, eps=1e-3):
    """He et al. guided filter (gray guide I, input p), box radius r."""
    def box(x):
        return F.avg_pool2d(F.pad(x, [r] * 4, mode="reflect"), 2 * r + 1, stride=1)
    mI, mp = box(I), box(p)
    a = (box(I * p) - mI * mp) / (box(I * I) - mI * mI + eps)
    b = mp - a * mI
    return box(a) * I + box(b)


def random_tone_knots(rng: random.Random, n=8):
    """Monotone curve through 9 knots: random positive increments with a
    gamma-like bend, scaled to a random maximum (darkening or flattening)."""
    gam = rng.uniform(1.0, 3.0) if rng.random() < 0.7 else rng.uniform(0.6, 1.0)
    base = [(i / n) ** gam for i in range(n + 1)]
    inc = [max(base[i + 1] - base[i], 1e-3) * rng.uniform(0.6, 1.4) for i in range(n)]
    c = [0.0]
    for d in inc:
        c.append(c[-1] + d)
    top = rng.uniform(0.12, 0.45) if rng.random() < 0.4 else rng.uniform(0.3, 1.0)   # strong darkening is common
    shift = rng.uniform(0.0, 0.15) if rng.random() < 0.3 else 0.0
    ky = [max(v / c[-1] * top - shift, 0.0) for v in c]
    for i in range(1, len(ky)):                      # strictly increasing after the clip
        ky[i] = max(ky[i], ky[i - 1] + 1e-4)
    return ky


def _param_curve(rng: random.Random, fam: str):
    if fam == "gamma":
        return dict(kind="gamma", gamma=math.exp(rng.uniform(math.log(0.25), math.log(1.6))))
    if fam == "shift":
        return dict(kind="shift", s=rng.uniform(0.02, 0.6))
    if fam == "linmap":
        vmin = rng.uniform(0.0, 0.25) if rng.random() < 0.5 else 0.0
        return dict(kind="linmap", vmin=vmin, vmax=rng.uniform(max(vmin + 0.4, 0.6), 1.0), dmax=rng.uniform(0.15, 1.0))
    if fam == "gain":                                   # exposure gain (clipped above 1)
        return dict(kind="gain", g=math.exp(rng.uniform(math.log(0.1), math.log(1.6))))
    if fam == "log":
        return dict(kind=rng.choice(["log", "expc"]), a=math.exp(rng.uniform(math.log(0.5), math.log(30.0))))
    if fam == "scurve":                                 # HELD OUT: generalisation table only
        return dict(kind="scurve", c=rng.uniform(2.0, 8.0))
    raise ValueError(fam)


TONE_TRAIN = (("spline", 0.30), ("gamma", 0.12), ("shift", 0.12), ("linmap", 0.12),
              ("gain", 0.12), ("log", 0.12), ("compose", 0.10))


def _with_knots(p):
    """Add `knots_y` (the curve sampled at i/8, clipped, strictly increasing) to a parametric curve."""
    V = torch.linspace(0, 1, 9)
    ky = O.curve(V, p["kind"], p).clamp(0, 1).tolist()
    ky = [min(v, 1 - (len(ky) - 1 - i) * 1e-4) for i, v in enumerate(ky)]
    for i in range(1, len(ky)):
        ky[i] = max(ky[i], ky[i - 1] + 1e-4)
    p["knots_y"] = ky
    return p


def random_tone_curve(rng: random.Random, split: str = "train"):
    """A tone curve from a broad DECLARED prior of common photo-editing curves:
    random monotone spline, gamma, constant shift (clipped), linear remap, exposure gain, log /
    inverse-log, and compositions of two of them. The S-curve family is HELD OUT (split="heldout")
    for a generalisation table. Returns the parameters plus `knots_y` (the curve sampled at i/8,
    clipped to [0, 1], strictly increasing) so every family shares one target format."""
    if split == "heldout":
        p = _param_curve(rng, "scurve")
    else:
        fam = rng.choices([f for f, _ in TONE_TRAIN], weights=[w for _, w in TONE_TRAIN])[0]
        if fam == "spline":
            return dict(kind="spline", knots_y=random_tone_knots(rng))
        if fam == "compose":
            a, b = rng.sample(["gamma", "shift", "linmap", "gain", "log"], 2)
            p = dict(kind="compose", steps=[_param_curve(rng, a), _param_curve(rng, b)])
        else:
            p = _param_curve(rng, fam)
    V = torch.linspace(0, 1, 9)
    ky = O.curve(V, p["kind"], p).clamp(0, 1).tolist()
    ky = [min(v, 1 - (len(ky) - 1 - i) * 1e-4) for i, v in enumerate(ky)]      # room to stay below 1
    for i in range(1, len(ky)):
        ky[i] = max(ky[i], ky[i - 1] + 1e-4)
    p["knots_y"] = ky
    return p


def rain_streaks_fast(h, w, n, rng: random.Random, device, length=(10, 40), angle_spread=0.35,
                      intensity=(0.05, 0.35), steps=48, base=None):
    """Vectorised version of etr.ops.layers.rain_streaks (same model: n
    near-parallel bilinear-splatted streaks around one base angle, additive,
    >= 0), one scatter instead of a Python loop per streak (12.7 s -> ms)."""
    g = torch.Generator(device=device).manual_seed(rng.randrange(2 ** 31))
    u = lambda *sh: torch.rand(*sh, generator=g, device=device)
    base = rng.random() * math.pi if base is None else base
    ang = base + (u(n) - 0.5) * angle_spread
    ln = length[0] + u(n) * (length[1] - length[0])
    amp = intensity[0] + u(n) * (intensity[1] - intensity[0])
    cy, cx = u(n) * h, u(n) * w
    t = torch.linspace(-0.5, 0.5, steps, device=device)[None]
    ys = cy[:, None] + t * ln[:, None] * torch.sin(ang)[:, None]
    xs = cx[:, None] + t * ln[:, None] * torch.cos(ang)[:, None]
    # per-streak density: ~1 sample per pixel of length, like the loop version
    wgt = (amp * ln / steps)[:, None].expand_as(ys)
    L = torch.zeros(h * w, device=device)
    y0, x0 = ys.floor().long(), xs.floor().long()
    fy, fx = ys - y0, xs - x0
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = y0 + dy, x0 + dx
            ww = wgt * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
            ok = (yi >= 0) & (yi < h) & (xi >= 0) & (xi < w)
            L.index_add_(0, (yi * w + xi)[ok], ww[ok])
    return L.view(1, h, w).clamp(max=1.0)


# ------------------------------------------------------------ sampler --
def _factor(kind, rng, x_hw, s_now, device):
    h, w = x_hw
    if kind == "blur":
        g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
        ksz = rng.choices((31, 47, 63, 65, 75, 79), weights=(0.35, 0.2, 0.15, 0.1, 0.1, 0.1))[0]  # ranges scale with the canvas
        return Step("blur", dict(k=K.sample_kernel(ksz, "train", g).to(device)))
    if kind == "resample":
        s = rng.choice([2, 4, 8] if min(h, w) % 8 == 0 else [2, 4])
        fam = rng.choices(RS_FAMILIES, weights=(0.45, 0.15, 0.15, 0.15, 0.10))[0]
        aa = True if fam == "gauss" else rng.random() < 0.7
        p = dict(s=s, family=fam, antialias=aa, a=rng.uniform(-0.9, -0.3), n=rng.choice((2, 3, 4)),
                 sigma=rng.uniform(0.3, 0.7))
        p["k"] = resampler_psf(s, fam, aa, p["a"], p["n"], p["sigma"], device=device)
        return Step("resample", p)
    if kind == "noise":
        if rng.random() < 0.6:
            sig = rng.uniform(1, 50) / 255
            return Step("noise", dict(type="gauss", sigma=sig, var=sig ** 2), Q8)
        return Step("noise", dict(type="pg", alpha=rng.uniform(1e-4, 2e-2), beta=rng.uniform(0, 15) / 255), Q8)
    if kind == "jpeg":
        return Step("jpeg", dict(q=rng.randint(10, 95), subsampling="420" if rng.random() < 0.7 else "444"))
    if kind == "tone":
        return Step("tone", dict(random_tone_curve(rng), space=rng.choice(["hsv_v", "luma", "rgb"])), Q8)
    if kind == "haze":
        A = rng.uniform(0.6, 1.0)
        u = rng.random()
        if u < 0.15:
            return Step("haze", dict(t=rng.uniform(0.15, 0.9), A=A, field="homog"), Q8)
        if u < 0.4:
            return Step("haze", dict(t=None, A=A, beta=rng.uniform(0.3, 2.0), field="smooth"), Q8)
        # scene depth (a monocular depth map of the clean image): t = exp(-beta (d0 + (1-d0) d));
        # synthesize() falls back to "smooth" when the image has no depth map
        return Step("haze", dict(t=None, A=A, beta=rng.uniform(0.5, 3.0), d0=rng.uniform(0.0, 0.3), field="depth"), Q8)
    if kind == "layer":
        # rain falls near-vertically (a minority of samples: any direction, e.g. a tilted camera) and
        # ranges from light to dense; per-pixel density spans ~0.5x - 12x the first generator's
        vertical = rng.random() < 0.7
        base = math.pi / 2 + rng.uniform(-0.5, 0.5) if vertical else rng.random() * math.pi
        return Step("layer", dict(L=None, n=int(math.exp(rng.uniform(math.log(30), math.log(1000)))), base=base,
                                  spread=rng.uniform(0.05, 0.6),
                                  length=rng.choice([(8, 30), (10, 40), (20, 60)]),
                                  intensity=rng.choice([(0.05, 0.35), (0.1, 0.5), (0.03, 0.2)])), Q8)
    raise ValueError(kind)


def _supported(order):
    try:
        parse(order)
        return True
    except NotImplementedError:
        return False


def sample_chain(rng: random.Random, x_hw, device="cuda", max_tries=50, p_over=None):
    """A random chain of 1-4 factors from the declared generator (TRAIN families), in an order the data term accepts."""
    probs = dict(P_FACTOR, **(p_over or {}))
    for _ in range(max_tries):
        kinds = [k for k, p in probs.items() if rng.random() < p]
        if not 1 <= len(kinds) <= 4:
            continue
        body = [k for k in kinds if k != "jpeg"]
        rng.shuffle(body)
        if "blur" in body and "resample" in body and rng.random() < 0.5:     # camera blur, then downscale
            body.remove("blur"); body.insert(body.index("resample"), "blur")
        order = body + (["jpeg"] if "jpeg" in kinds else [])
        steps = [_factor(k, rng, x_hw, 1, device) for k in order]
        # a blur AFTER a resample acts on the LR grid: cap its canvas so it fits
        # the LR image (and its reach stays physical); ranges scale with it
        c = 1
        for i, st in enumerate(steps):
            if st.kind == "resample":
                c *= st.p["s"]
            elif st.kind == "blur" and c > 1:
                lr = min(x_hw) // c
                cap = min(31, lr - 1 if lr % 2 == 0 else lr - 2)
                if st.p["k"].shape[-1] > cap:
                    g = torch.Generator().manual_seed(rng.randrange(2 ** 31))
                    st.p["k"] = K.sample_kernel(max(cap, 5), "train", g).to(device)
        if _supported(steps):
            return steps
    return [_factor("noise", rng, x_hw, 1, device)]


# ---------------------------------------------------------- synthesis --
@torch.no_grad()
def synthesize(x: torch.Tensor, steps, rng: random.Random, pad: int = 48, round_p: float = 0.5,
               clip_intermediate: bool = True, depth: torch.Tensor | None = None, tone_samples: int = 0):
    """Apply `steps` to x [1,3,H,W]. Fills in each step's spatial maps (haze
    t, rain L and clip mask M, Poisson-Gaussian variance) at the resolution
    where it acts, as the oracle would see them. Returns y (uint8-valued).

    depth: optional [1,1,H,W] scene depth in [0,1] (1 = far) for field="depth" haze.
    Each step also gets `effect`: the mean absolute change it made to its own input
    (for noise, the mean |n|), so labels can mark an invisible factor as absent.
    tone_samples > 0 stores that many pixel samples before / after a tone step
    (p["_pre"], p["_post"], [1,3,1,N]) for the image-space tone loss."""
    dev = x.device
    g = torch.Generator(device=dev).manual_seed(rng.randrange(2 ** 31))
    from etr.world.chain import required_pad
    pad = max(pad, required_pad(steps)[0])
    z = F.pad(x, [pad] * 4, mode="reflect")
    p = pad
    for i, s in enumerate(steps):
        h, w = z.shape[-2] - 2 * p, z.shape[-1] - 2 * p
        crop = lambda a: a[..., p:p + h, p:p + w]
        z_in = crop(z)
        if s.kind == "haze" and s.p.get("field") == "depth" and depth is None:
            s.p["field"], s.p["beta"] = "smooth", min(s.p["beta"], 2.0)
        if s.kind == "blur":
            s.p["_pre"] = z_in.detach()                 # clean input of the blur step (blur-stage reference)
            z = O.conv_circ(z, s.p["k"])
        elif s.kind == "resample":
            z = O.resample(z, s.p["s"], s.p["k"])
            p //= s.p["s"]
        elif s.kind == "tone":
            z = O.tone(z, s.p["kind"], s.p, s.p["space"])
            if tone_samples:
                idx = torch.randint(0, h * w, (tone_samples,), generator=g, device=dev)
                pick = lambda a: a.flatten(2)[..., idx].clamp(0, 1)[:, :, None]
                s.p["_pre"], s.p["_post"] = pick(z_in), pick(crop(z))
        elif s.kind == "haze":
            if s.p["field"] == "smooth":
                d = smooth_field(h, w, rng, dev)
                d = guided_filter(O._luma(crop(z)).clamp(0, 1), d).clamp(0, 1)
                s.p["t"] = torch.exp(-s.p["beta"] * d)
            elif s.p["field"] == "depth":
                d = F.interpolate(depth.to(dev), size=(h, w), mode="area") if depth.shape[-2:] != (h, w) else depth.to(dev)
                s.p["t"] = torch.exp(-s.p["beta"] * (s.p["d0"] + (1 - s.p["d0"]) * d))
            t = F.pad(s.p["t"], [p] * 4, mode="replicate") if torch.is_tensor(s.p["t"]) else s.p["t"]
            z = O.haze(z, t, s.p["A"])
        elif s.kind == "layer":
            L = rain_streaks_fast(h, w, s.p["n"], rng, dev, length=s.p.get("length", (10, 40)), angle_spread=s.p.get("spread", 0.35),
                                  intensity=s.p.get("intensity", (0.05, 0.35)), base=s.p.get("base"))[None].expand(1, 3, h, w).contiguous()
            Lp = F.pad(L, [p] * 4)
            out = z + Lp
            Mp = (out >= 1).float()
            z = out.clamp(max=1)
            s.p["L"], s.p["M"] = L, crop(Mp)
        elif s.kind == "noise":
            if s.p["type"] == "pg":
                var = s.p["alpha"] * z.clamp(min=0) + s.p["beta"] ** 2
                s.p["var"] = crop(var)
            else:
                var = s.p["var"]
            z = z + torch.randn(z.shape, generator=g, device=dev) * (var ** 0.5 if torch.is_tensor(var) else math.sqrt(var))
        elif s.kind == "jpeg":
            z = O.jpeg_cv2(crop(z).clamp(0, 1), s.p["q"], s.p["subsampling"])        # the codec needs [0,1]
            s.effect = float((z - z_in.clamp(0, 1)).abs().mean())
            return z
        if s.kind == "noise":
            s.effect = float((var.mean() if torch.is_tensor(var) else var) ** 0.5) * 0.798    # E|n|
        elif s.kind != "resample":
            s.effect = float((crop(z) - z_in).abs().mean())
        last = i == len(steps) - 1
        if clip_intermediate or last:
            z = z.clamp(0, 1)
        # an 8-bit store right after a tone edit is the common case (and what leaves the
        # histogram fingerprint the tone head reads): p = 0.8 there, round_p elsewhere
        rounded = last or rng.random() < (0.8 if s.kind == "tone" else round_p)
        if rounded:
            z = (z * 255).round() / 255
        s.err_var = Q8 if rounded else 0.0          # what this step really injected
    return z[..., p:-p, p:-p] if p > 0 else z
