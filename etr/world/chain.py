"""An ordered degradation chain and its data term.

A chain is a list of `Step`s in the order they were applied to the clean
image. `DataTerm` turns (chain, observation y) into the convex quadratic
f(z) that Phase 2 minimises, following these order rules:

  prefix tones   tone curves applied BEFORE any linear factor become a change
                 of variables: the solver works on z = T(x), so the data term
                 stays quadratic in z and the prior is applied to x = T^-1(z).
  core           linear factors (blur, resample, haze, layer), with noise
                 steps interleaved: H(z) is affine in z.
  trailing       tone curves / noise AFTER the last linear factor: the tones
                 are inverted on y (y~ = T^-1(y)), the noise is propagated to
                 y~ by the delta method.
  jpeg           must be the last step; becomes a block-DCT weighting with
                 the quantisation variance Q^2/12 per coefficient, 4:2:0
                 chroma handled by 2x2 pooling.

A tone that sits between two linear factors, or before a JPEG with a linear
factor in front of it, makes the data term non-convex; `DataTerm` refuses it
loudly rather than solving something else.

Noise variances are propagated through the linear factors pixel-wise, which
ignores the spatial correlation a blur introduces. This only affects the
data WEIGHTING (never the operator), and it is the documented approximation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from etr.world import ops as O
from etr.ops.jpeg import quant_table, block_dct, _dct_matrix

LINEAR = ("blur", "resample", "haze", "layer")
VAR_FLOOR = 1.0 / (120 * 255.0 ** 2)
KINDS = LINEAR + ("tone", "noise", "jpeg")
# variance through a trailing tone inverse: "ratio" (the inverse's ratio c^-1(V)/V used as the gain on every
# channel; the released model) or "jacobian" (the exact delta method)
TONE_VAR = "ratio"


@dataclass
class Step:
    kind: str
    p: dict = field(default_factory=dict)
    # white variance ([0,1]^2 units) injected right after this step: uint8
    # rounding, codec/colour-space quantisation, measured model error
    err_var: float = 0.0

    def __post_init__(self):
        assert self.kind in KINDS, self.kind


def parse(steps):
    """Split a chain into (prefix tones, core, trailing, jpeg) per the order
    rules in the module docstring; raises NotImplementedError for orders whose
    data term is not convex. The single source of truth for what the solver
    can represent (the training generator calls it too)."""
    kinds = [s.kind for s in steps]
    if "jpeg" in kinds:
        assert kinds.index("jpeg") == len(kinds) - 1 and kinds.count("jpeg") == 1, \
            "JPEG must be the single last step"
    jpeg = steps[-1] if kinds and kinds[-1] == "jpeg" else None
    body = steps[:-1] if jpeg is not None else steps
    lin_idx = [i for i, s in enumerate(body) if s.kind in LINEAR]
    n_prefix = 0
    while n_prefix < len(body) and body[n_prefix].kind == "tone":
        n_prefix += 1
    last_lin = lin_idx[-1] if lin_idx else n_prefix - 1
    prefix, core, trailing = body[:n_prefix], body[n_prefix:last_lin + 1], body[last_lin + 1:]
    if any(s.kind == "tone" for s in core):
        raise NotImplementedError("tone between linear factors: non-convex data term")
    if jpeg is not None and any(s.kind == "tone" for s in trailing):
        raise NotImplementedError("tone between a non-tone factor and JPEG: non-convex")
    return prefix, core, trailing, jpeg


def required_pad(steps, margin=8):
    """Smallest HR margin that keeps circular wrap-around out of the observed
    region: the summed reach (in HR pixels) of every kernel in the chain --
    a kernel of radius r acting after a cumulative downscale c reaches r*c HR
    pixels -- plus a margin, rounded up to a multiple of the total scale."""
    c, reach = 1, 0
    for s in steps:
        if s.kind in ("blur", "resample"):
            reach += (s.p["k"].shape[-1] // 2) * c
        if s.kind == "resample":
            c *= s.p["s"]
    need = reach + margin
    return int(math.ceil(need / c) * c), c


def _pad(m, pad, mode="replicate"):
    if not torch.is_tensor(m) or m.dim() < 4 or pad == 0:
        return m
    return F.pad(m, [pad] * 4, mode=mode)


def _crop(x, pad):
    return x if pad == 0 else x[..., pad:-pad, pad:-pad]


class DataTerm:
    def __init__(self, y: torch.Tensor, steps: list, pad: int = 48, censor: bool = True,
                 invert_prefix_if_no_core: bool = False):
        """invert_prefix_if_no_core: with no linear factor at all, prefix tones can
        be inverted on y instead of becoming a change of variables (both are exact
        then); the direct MMSE path needs the image-domain form."""
        assert y.dim() == 4 and y.shape[0] == 1, "one image at a time"
        self.y = y
        self.steps = steps
        dev, dt = y.device, y.dtype
        self.prefix, self.core, self.trailing, self.jpeg = parse(steps)
        if invert_prefix_if_no_core and not self.core and self.jpeg is None and self.prefix:
            self.trailing, self.prefix = self.prefix + self.trailing, []

        # ---- geometry: observed HR size, padded stage sizes
        s_tot = 1
        for s in self.core:
            if s.kind == "resample":
                s_tot *= s.p["s"]
        self.s_tot = s_tot
        need, _ = required_pad(self.core)
        pad = max(pad, need)
        pad = int(math.ceil(pad / s_tot) * s_tot)
        self.pad = pad
        h, w = y.shape[-2:]
        self.hr_hw = (h * s_tot, w * s_tot)

        # spatial maps of each core step, padded to their stage (replicate:
        # the margin is free in the solver, so any extension is valid)
        self._core_padded = []
        stage_pad = pad
        for s in self.core:
            q = dict(s.p)
            if s.kind == "haze" and torch.is_tensor(q["t"]):
                q["t"] = _pad(q["t"].to(dev, dt), stage_pad)
            if s.kind == "layer":
                q["L"] = _pad(q["L"].to(dev, dt), stage_pad)
                if q.get("M") is not None:
                    q["M"] = _pad(q["M"].to(dev, dt), stage_pad)
            if s.kind in ("blur", "resample"):
                q["k"] = q["k"].to(dev, dt)
            if s.kind == "noise" and torch.is_tensor(q.get("var")):
                q["var"] = _pad(q["var"].to(dev, dt), stage_pad)
            self._core_padded.append(q)
            if s.kind == "resample":
                stage_pad //= s.p["s"]
        self.obs_pad = stage_pad

        # ---- observation side: invert trailing tones on y, propagate variance
        with torch.no_grad():
            yt = y.clone()
            var_t = torch.zeros_like(y)
            valid = torch.ones_like(y)
            for s in reversed(self.trailing):
                var_t = var_t + s.err_var
                if s.kind == "noise":
                    var_t = var_t + _as_map(s.p["var"], y)
                elif s.kind == "tone":
                    if TONE_VAR == "jacobian":                 # the exact delta method
                        var_t = O.tone_inv_var(yt, var_t.expand_as(yt), s.p["kind"], s.p, s.p.get("space", "hsv_v"))
                    yt, gain = O.tone_inv(yt, s.p["kind"], s.p, s.p.get("space", "hsv_v"))
                    if TONE_VAR != "jacobian":
                        var_t = var_t * gain ** 2
                    valid = valid * (gain > 0)
            self.y_tilde = yt
            var_core = self._core_variance()
            # floor at 1/10 of uint8 rounding: a noise-free chain would otherwise give
            # weights ~1e10 and an ill-conditioned prox for no modelling gain
            self.var = (var_core + var_t).clamp(min=VAR_FLOOR)
            if censor and self.jpeg is None:
                # saturated observations (noise, signed-resampler overshoot, exposure)
                # carry only a one-sided constraint; drop them. Censoring a pixel
                # whose true value is exactly 0 or 1 loses almost nothing.
                valid = valid * ((y > 0.5 / 255) & (y < 1 - 0.5 / 255))
            self.valid = valid
            self.w = valid / self.var
            if self.jpeg is not None:
                self._init_jpeg()

    # ------------------------------------------------------------ forward
    def H(self, z: torch.Tensor) -> torch.Tensor:
        """Affine core map on the padded HR variable -> cropped observation."""
        for s, q in zip(self.core, self._core_padded):
            if s.kind == "blur":
                z = O.conv_circ(z, q["k"])
            elif s.kind == "resample":
                z = O.resample(z, q["s"], q["k"])
            elif s.kind == "haze":
                z = O.haze(z, q["t"], q["A"])
            elif s.kind == "layer":
                z = O.layer(z, q["L"], q.get("M"))
        return _crop(z, self.obs_pad)

    def _core_variance(self):
        """Marginal noise variance at the observation, for every noise source
        (noise steps and per-step injected errors) propagated through the core.

        Per source we track: v (variance map), h (kernel composed since the last
        decimation), R (normalised autocorrelation of the field h acts on; delta
        for white noise), m (pointwise factor^2: haze t, clip mask).
          marginal variance = conv(v, h^2) * gamma * m,
          gamma = (A_h * R)(0) / A_h(0),  A_h = h * flip(h)  (gamma = 1 for white noise)
        Kernels are COMPOSED before squaring -- squaring them one at a time ignores
        the correlation the first blur introduces (measured: noise > blur > x4 gave
        chi2 = 18). At a decimation by s the field's autocorrelation A_h * R is
        subsampled every s pixels, so correlation carried onto the LR grid is kept
        (exact for stationary noise). Pointwise factors are applied as their square
        at the output: exact when they vary slowly relative to the kernel."""
        Hp, Wp = self.hr_hw[0] + 2 * self.pad, self.hr_hw[1] + 2 * self.pad
        like = self.y.new_zeros(1, 3, Hp, Wp)
        one = self.y.new_ones(1, 1)
        sources = []
        pre = float(sum(s.err_var for s in self.prefix))
        if pre > 0:
            sources.append(dict(v=like + pre, h=one, R=one, m=None))

        def marginal(src):
            h, R = src["h"], src["R"]
            var = O.conv_circ(src["v"], h ** 2) if h.numel() > 1 else src["v"]
            if R.numel() > 1 and h.numel() > 1:
                A = _compose(h, h.flip(-1, -2))
                C = _compose(A, R)
                gamma = float(C[C.shape[0] // 2, C.shape[1] // 2] / A[A.shape[0] // 2, A.shape[1] // 2])
                var = var * gamma
            return var * src["m"] if src["m"] is not None else var

        def _flush(src, marginal):
            if src["h"].numel() == 1 and src["m"] is None:
                return
            h = src["h"]
            v = marginal(src)
            if h.numel() > 1:
                A = _compose(h, h.flip(-1, -2))
                C = _compose(A, src["R"]) if src["R"].numel() > 1 else A
                src["R"] = C / C[C.shape[0] // 2, C.shape[1] // 2]
            src.update(v=v, h=self.y.new_ones(1, 1), m=None)

        for s, q in zip(self.core, self._core_padded):
            if s.kind == "blur":
                for src in sources:
                    src["h"] = _compose(src["h"], q["k"])
            elif s.kind in ("haze", "layer"):
                f2 = (q["t"] ** 2) if s.kind == "haze" else ((1 - q["M"]) ** 2 if q.get("M") is not None else None)
                if f2 is not None:
                    for src in sources:
                        # flush: the marginal variance at THIS stage is exact; the
                        # factor then multiplies it before any later convolution
                        # (a sharp clip mask followed by a blur must not zero the
                        # variance the blur mixes in from unsaturated neighbours)
                        _flush(src, marginal)
                        src["v"] = src["v"] * f2
            elif s.kind == "resample":
                st = q["s"]
                out = []
                for src in sources:
                    src["h"] = _compose(src["h"], q["k"])
                    var = marginal(src)[..., ::st, ::st]
                    A = _compose(src["h"], src["h"].flip(-1, -2))
                    C = _compose(A, src["R"]) if src["R"].numel() > 1 else A
                    c = C.shape[-1] // 2
                    idx = torch.arange(c % st, C.shape[-1], st, device=C.device)
                    R = C[idx][:, idx]
                    R = R / R[R.shape[0] // 2, R.shape[1] // 2]
                    out.append(dict(v=var, h=one, R=R, m=None))
                sources = out
                like = like[..., ::st, ::st]
            elif s.kind == "noise":
                sources.append(dict(v=like + _as_map(q["var"], like), h=one, R=one, m=None))
            if s.err_var > 0:
                sources.append(dict(v=like + s.err_var, h=one, R=one, m=None))
        tot = torch.zeros_like(like)
        for src in sources:
            tot = tot + marginal(src)
        return _crop(tot, self.obs_pad).clamp(min=0)

    # --------------------------------------------------------------- f(z)
    def f(self, z: torch.Tensor) -> torch.Tensor:
        r = self.H(z) - self.y_tilde
        if self.jpeg is None:
            return 0.5 * (self.w * r * r).sum()
        return 0.5 * sum((wc * c * c).sum() for wc, c in zip(self._jw, self._jpeg_coeffs(r)))

    def _jpeg_coeffs(self, r):
        r = F.pad(r * 255.0, (0, self._jpad[1], 0, self._jpad[0]))
        ycc = O.ycc_linear(r)
        cY = block_dct(ycc[:, :1], self._D)
        ch = ycc[:, 1:]
        if self.jpeg.p.get("subsampling", "420") == "420":
            ch = F.avg_pool2d(ch, 2)
        cC = block_dct(ch, self._D)
        return cY, cC

    def _init_jpeg(self):
        h, w = self.y.shape[-2:]
        self._jpad = ((-h) % 16, (-w) % 16)
        self._D = _dct_matrix(self.y.device, self.y.dtype)
        Q = quant_table(self.jpeg.p["q"], self.y.device)[0]           # [3,8,8], 0..255 units
        vm = F.pad(self.var.mean(1, keepdim=True) * 255.0 ** 2, (0, self._jpad[1], 0, self._jpad[0]),
                   mode="replicate")
        gY, gCb, gCr = O.YCC_NOISE_GAIN
        sub = self.jpeg.p.get("subsampling", "420") == "420"
        vY = F.avg_pool2d(vm, 8)[..., None, None] * gY                  # [1,1,nb,nb,1,1]
        vC = F.avg_pool2d(vm, 16 if sub else 8)[..., None, None] * (0.5 * (gCb + gCr)) / (4 if sub else 1)
        wY = 1.0 / (vY + Q[0] ** 2 / 12.0)
        wC = 1.0 / (vC + Q[1] ** 2 / 12.0)
        self._jw = (wY, wC.expand(1, 2, *wC.shape[2:]))

    # ----------------------------------------------------------- helpers
    def with_observation(self, y_tilde_new, var_scale):
        """Same chain, a pre-denoised observation: y~ replaced, the observation
        variance scaled by var_scale (the residual noise left by the denoiser).
        The data term stays a fixed convex quadratic, so Phase 2's guarantee holds."""
        import copy
        d = copy.copy(self)
        d.y_tilde = y_tilde_new
        d.var = (self.var * var_scale).clamp(min=VAR_FLOOR)
        d.w = self.valid / d.var
        return d

    def x_from_z(self, z):
        for s in reversed(self.prefix):
            z = O.tone_inv(z, s.p["kind"], s.p, s.p.get("space", "hsv_v"))[0]
        return z

    def z_from_x(self, x):
        for s in self.prefix:
            x = O.tone(x, s.p["kind"], s.p, s.p.get("space", "hsv_v"))
        return x

    def crop_hr(self, z):
        return _crop(z, self.pad)

    @torch.no_grad()
    def z_init(self):
        """Pointwise pseudo-inverse of the core, padded by replicate."""
        v = self.y_tilde
        obs_maps = [s.p for s in self.core]
        for s, q in zip(reversed(self.core), reversed(obs_maps)):
            if s.kind == "resample":
                v = F.interpolate(v, scale_factor=q["s"], mode="bicubic", align_corners=False)
            elif s.kind == "haze":
                t = q["t"].to(v) if torch.is_tensor(q["t"]) else q["t"]
                v = O.haze_inv(v, t, q["A"])
            elif s.kind == "layer":
                v = v - q["L"].to(v) * (1 - (q["M"].to(v) if q.get("M") is not None else 0))
            v = v.clamp(0, 1)
        return F.pad(v, [self.pad] * 4, mode="replicate")

    def grad_f(self, z):
        with torch.enable_grad():
            z = z.detach().requires_grad_(True)
            (g,) = torch.autograd.grad(self.f(z), z)
        return g

    @torch.no_grad()
    def lipschitz(self, z_like, iters=20):
        """Largest eigenvalue of the Hessian of f (f is quadratic)."""
        g0 = self.grad_f(torch.zeros_like(z_like))
        v = torch.randn_like(z_like)
        v = v / v.norm()
        lam = 0.0
        for _ in range(iters):
            hv = self.grad_f(v) - g0
            lam = (hv * v).sum().item()
            v = hv / hv.norm().clamp(min=1e-20)
        return lam * 1.05


def _compose(h, k):
    """Full linear convolution of two odd-sized kernels (result odd-sized)."""
    h2 = h.view(1, 1, *h.shape[-2:]).to(k.dtype if torch.is_tensor(k) else h.dtype)
    k2 = k.view(1, 1, *k.shape[-2:]).to(h2.device)
    return F.conv2d(F.pad(h2, [k2.shape[-1] - 1] * 2 + [k2.shape[-2] - 1] * 2), k2.flip(-1, -2))[0, 0]


def _as_map(v, like):
    return v.to(like) if torch.is_tensor(v) else torch.full_like(like, float(v))
