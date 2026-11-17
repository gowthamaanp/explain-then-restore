"""ProposalNet: amortised inference q_phi(theta | y).

Reads ONLY the observation (plus the scale s implied by the requested output
size) and outputs a distribution over the world-model parameters, never a
restored image. Every continuous head is (mean, log-variance) trained by
Gaussian NLL, every discrete head by cross-entropy, so the outputs are
proposals with uncertainty for Phase 1's particles, not point estimates.

Heads (targets come from etr/world/generator.py chains via `targets()`):
  presence   7 logits: blur, resample, noise, jpeg, tone, haze, layer
  blur       family-agnostic PSF descriptors (etr/world/kernel_desc.py), mean + log-var;
             no kernel-shape prior involved, so no dependence on its blind spots
  resample   resampler family {cubic, linear, box, lanczos, gauss} (etr/world/resamplers.py),
             anti-aliased or not, cubic a, lanczos order {2,3,4}, gauss sigma
  noise      type {gauss, pg}; log sigma, log alpha, log beta
  jpeg       q/100; subsampling {420, 444}
  tone       9 knot values; space {hsv_v, luma, rgb}. Optional sub-head (tone_hist=True):
             a small MLP on the V / luma histograms of y adds a correction to the
             knots (brightness statistics are global, not local texture); trained
             also by an image-space loss |T_hat^-1(T(x)) - x| (tone_image_loss)
  haze       airlight A; transmission map t at 1/8 resolution (mean, log-var)
  order      tone_first (tone before every linear factor),
             noise_before_resample, noise_before_tone
Rain layer: presence + a dense streak map L_hat at observation resolution (RainHead, a small
decoder over the encoder skips; rain=True), the additive layer of the data term.
The kernel SHAPE is estimated by the separate blur module (etr/kernel); the
optional kernel head here (n_hyp > 0) is the first attempt, kept for reference
(it failed on thin kernels).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from etr.models.blocks import NAFBlock
from etr.world.kernel_desc import descriptors, D as DESC_D

KINDS = ("blur", "resample", "noise", "jpeg", "tone", "haze", "layer")
SPACES = ("hsv_v", "luma", "rgb")
SCALES = (1, 2, 4, 8)
from etr.world.proposal_consts import RS_FAMILIES, LANCZOS_N


def dark_channel(y, r=7):
    m = y.amin(1, keepdim=True)
    return -F.max_pool2d(-F.pad(m, [r] * 4, mode="replicate"), 2 * r + 1, stride=1)


def spectrum_features(y, n_r=24, n_t=8):
    """Radially/angularly binned log-magnitude spectrum of the luma (blur width
    and direction live here); normalised by frequency, not pixel position."""
    g = (0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2])
    g = g - g.mean((-1, -2), keepdim=True)
    B, h, w = g.shape
    mag = torch.log1p(torch.fft.rfft2(g, norm="ortho").abs()).flatten(1)
    fy = torch.fft.fftfreq(h, device=y.device)[:, None]
    fx = torch.fft.rfftfreq(w, device=y.device)[None, :]
    r = (fy ** 2 + fx ** 2).sqrt() / (0.5 * math.sqrt(2))
    th = torch.atan2(fy.expand_as(r), fx.expand_as(r)) % math.pi
    ri = (r * n_r).long().clamp(max=n_r - 1)
    ti = (th / math.pi * n_t).long().clamp(max=n_t - 1)
    idx = (ri * n_t + ti).flatten()
    out = torch.zeros(B, n_r * n_t, device=y.device)
    out.scatter_add_(1, idx[None].expand(B, -1), mag)
    cnt = torch.bincount(idx, minlength=n_r * n_t).clamp(min=1).float()
    return out / cnt


def brightness_hist(y, bins=32):
    """Normalised histograms of HSV-V and luma, [B, 2*bins] (input features of the tone sub-head)."""
    V = y.amax(1)
    Yl = 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]
    out = []
    for b in range(y.shape[0]):
        h = [torch.histc(c[b].clamp(0, 1), bins, 0, 1) for c in (V, Yl)]
        out.append(torch.cat(h) / V[b].numel())
    return torch.stack(out)


N_Q = 17            # tone v2: the inverse curve is predicted at 17 quantiles of the observed V


def tone_features(y):
    """Global brightness statistics of the WHOLE observation for ToneHeadV2 -> (feat [B,803], q [B,17]).
    exact 256-bin HSV-V histogram (8-bit levels) and its high-pass part (the comb left by a tone
    edit followed by an 8-bit store; image forensics), empty-bin mask, 17 V quantiles, clipping
    mass at 0 and 1, and a 16-bin noise-level function (Haar-HH MAD per intensity bin; noise that
    entered before the curve is scaled by the curve's slope)."""
    V = y.amax(1)                                                        # [B,h,w]
    Yl = 0.299 * y[:, 0] + 0.587 * y[:, 1] + 0.114 * y[:, 2]
    feats, qs = [], []
    lv = torch.linspace(0, 1, N_Q, device=y.device)
    for b in range(y.shape[0]):
        v = V[b].flatten()
        h = torch.bincount((v.clamp(0, 1) * 255).round().long(), minlength=256).float() / v.numel()
        hs = F.avg_pool1d(F.pad(h[None, None], (4, 4), mode="replicate"), 9, 1)[0, 0]
        hp = torch.log1p(1e3 * h) - torch.log1p(1e3 * hs)
        empty = (h == 0).float()
        q = torch.quantile(v[:: max(1, v.numel() // 200000)].float(), lv)
        q = torch.cummax(q, 0).values + 1e-4 * torch.arange(N_Q, device=y.device)
        clip = torch.stack([(v <= 1 / 255).float().mean(), (v >= 254 / 255).float().mean()])
        g = Yl[b]
        H, W = (g.shape[0] // 2) * 2, (g.shape[1] // 2) * 2
        g = g[:H, :W]
        hh = (g[0::2, 0::2] - g[0::2, 1::2] - g[1::2, 0::2] + g[1::2, 1::2]).abs().flatten() / 2
        mu = ((g[0::2, 0::2] + g[0::2, 1::2] + g[1::2, 0::2] + g[1::2, 1::2]) / 4).flatten()
        bi = (mu.clamp(0, 0.9999) * 16).long()
        nlf = torch.zeros(16, device=y.device)
        for k in range(16):
            sel = hh[bi == k]
            if sel.numel() >= 64:
                nlf[k] = sel.median() / 0.6745
        feats.append(torch.cat([h * 50, hp, empty, q, clip, nlf * 20]))
        qs.append(q)
    return torch.stack(feats), torch.stack(qs)


class ToneHeadV2(nn.Module):
    """Inverse tone curve from global statistics + the trunk's feature g. K hypotheses of the
    INVERSE curve u = T^-1(q) at the observed V quantiles q, monotone by construction, plus
    mixture logits. Trained with an image-weighted loss (the curve's error on this image's own
    pixels), winner-takes-all over hypotheses, and an MSE on the mixture mean (what is decoded)."""

    def __init__(self, g_dim=512, K=4, hidden=128):
        super().__init__()
        self.K = K
        self.mlp = nn.Sequential(nn.Linear(803 + g_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, K * (N_Q + 1) + K))

    def forward(self, y, g):
        f, q = tone_features(y)
        o = self.mlp(torch.cat([f, g], 1))
        B, K = y.shape[0], self.K
        pi = o[:, -K:]
        a = o[:, :-K].view(B, K, N_Q + 1)
        u0 = torch.sigmoid(a[..., 0])
        u1 = u0 + (1 - u0) * torch.sigmoid(a[..., 1] + 2.0)
        inc = torch.softmax(a[..., 2:], -1)                              # [B,K,N_Q-1]
        u = torch.cat([u0[..., None], u0[..., None] + (u1 - u0)[..., None] * torch.cumsum(inc, -1)], -1)
        return dict(tone_q=q, tone_u=u, tone_pi=pi, u_mu=(torch.softmax(pi, -1)[..., None] * u).sum(1))


def monotone_knots(k):
    """Knot values [.., n] -> strictly increasing (the inverse needs it)."""
    k = torch.cummax(k, -1).values
    return k + 1e-4 * torch.arange(k.shape[-1], device=k.device, dtype=k.dtype)


CANVAS = 79          # every blur kernel (31..79 px) is centred on this canvas for the kernel head


class RainHead(nn.Module):
    """Dense additive rain-layer estimate L_hat in [0, 1] at full resolution (~0.25M parameters):
    a U-Net style decoder over the encoder skips (full, 1/2, 1/4 resolution) and the 1/8 bottleneck.
    Thin streaks need the full-resolution skip; their orientation and extent need the context.
    The output layer is zero-initialised with bias -4 (L_hat ~ 0.02), so adding it changes nothing."""

    def __init__(self, c):
        super().__init__()
        self.pf, self.p2 = nn.Conv2d(c[3], 64, 1), nn.Conv2d(c[2], 64, 1)
        self.d2 = nn.Sequential(nn.Conv2d(128, 64, 3, padding=1), nn.SiLU())
        self.p1 = nn.Conv2d(c[1], 48, 1)
        self.d1 = nn.Sequential(nn.Conv2d(112, 48, 3, padding=1), nn.SiLU(), nn.Conv2d(48, 48, 3, padding=1), nn.SiLU())
        self.d0 = nn.Sequential(nn.Conv2d(48 + c[0], 32, 3, padding=1), nn.SiLU(), nn.Conv2d(32, 32, 3, padding=1), nn.SiLU())
        self.out = nn.Conv2d(32, 1, 3, padding=1)
        nn.init.zeros_(self.out.weight)
        nn.init.constant_(self.out.bias, -4.0)

    def forward(self, f, skips):
        up = lambda a: F.interpolate(a, scale_factor=2, mode="bilinear", align_corners=False)
        e0, e1, e2 = skips
        x = self.d2(torch.cat([up(self.pf(f)), self.p2(e2)], 1))
        x = self.d1(torch.cat([up(x), self.p1(e1)], 1))
        x = self.d0(torch.cat([up(x), e0], 1))
        return torch.sigmoid(self.out(x))


class KernelHead(nn.Module):
    """K kernel hypotheses as latents of the kernel prior C (spatial grid zc x G x G,
    G = 20 for the 79-px canvas: C decodes 4G-1 px), plus a confidence per hypothesis.

    Kernel evidence lives in local edges, so the image features are pooled with a
    learned spatial attention (4 heads) rather than a plain mean: edge-rich regions
    dominate, flat ones are ignored. Hypotheses share one decoder and differ by a
    learned per-hypothesis embedding, so K adds only K x 512 parameters."""

    def __init__(self, in_c, g_dim, K=8, zc=16, grid=20, w=64, seed=5):
        super().__init__()
        self.K, self.zc, self.grid, self.w, self.seed = K, zc, grid, w, seed
        self.att = nn.Conv2d(in_c, 4, 1)
        d = g_dim + 4 * in_c
        self.trunk = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, w * seed * seed))
        self.hyp = nn.Parameter(0.02 * torch.randn(K, w * seed * seed))
        ups = []
        n_up = int(math.log2(grid // seed))
        for _ in range(n_up):
            ups += [nn.Conv2d(w, w, 3, padding=1), nn.SiLU(), nn.Upsample(scale_factor=2, mode="nearest")]
        self.dec = nn.Sequential(*ups, nn.Conv2d(w, w, 3, padding=1), nn.SiLU(), nn.Conv2d(w, zc, 1))
        self.conf = nn.Linear(d, K)

    def forward(self, f, g):
        B, C, h, w = f.shape
        a = torch.softmax(self.att(f).flatten(2), -1)                      # [B,4,hw]
        pooled = torch.einsum("bah,bch->bac", a, f.flatten(2)).flatten(1)   # [B,4C]
        d = torch.cat([g, pooled], 1)
        base = self.trunk(d)                                                # [B, w*s*s]
        seeds = (base[:, None] + self.hyp[None]).view(B * self.K, self.w, self.seed, self.seed)
        z = self.dec(seeds).view(B, self.K, self.zc, self.grid, self.grid)
        return z, self.conf(d)


class ProposalNet(nn.Module):
    def __init__(self, width=48, n_desc=DESC_D, n_knots=9, n_hyp=0, zc=16, grid=(CANVAS + 1) // 4,
                 tone_hist=False, rain=False, tone_v2=False):
        super().__init__()
        self.n_desc, self.n_knots = n_desc, n_knots
        c = [width, width * 2, width * 4, width * 4]
        self.stem = nn.Conv2d(3 + 1 + 1, c[0], 3, padding=1)            # y, dark channel, local std
        self.enc = nn.ModuleList()
        self.down = nn.ModuleList()
        for i in range(3):
            self.enc.append(nn.Sequential(NAFBlock(c[i]), NAFBlock(c[i])))
            self.down.append(nn.Conv2d(c[i], c[i + 1], 2, stride=2))
        self.mid = nn.Sequential(*[NAFBlock(c[3]) for _ in range(4)])
        # dense t-map head at 1/8 resolution
        self.t_head = nn.Sequential(nn.Conv2d(c[3], 64, 3, padding=1), nn.SiLU(), nn.Conv2d(64, 2, 1))
        self.spec = nn.Sequential(nn.Linear(24 * 8, 128), nn.SiLU(), nn.Linear(128, 128))
        self.s_emb = nn.Embedding(len(SCALES), 32)
        d = c[3] * 2 + 128 + 32
        self.fuse = nn.Sequential(nn.Linear(d, 512), nn.SiLU(), nn.Linear(512, 512), nn.SiLU())
        n_out = (len(KINDS) + 2 * n_desc + (len(RS_FAMILIES) + 2 + 2 + len(LANCZOS_N) + 2) + 2 + 3 * 2 + 2 + 2 + 2 * n_knots + len(SPACES)
                 + 2 + 3)
        self.head = nn.Linear(512, n_out)
        self.khead = KernelHead(c[3], 512, K=n_hyp, zc=zc, grid=grid) if n_hyp > 0 else None
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)
        self.rain_head = RainHead(c) if rain else None
        self.tone_v2 = ToneHeadV2(512) if tone_v2 else None
        self.tone_mlp = None
        if tone_hist:                       # ~0.15M; zero-initialised so a warm start is unchanged
            self.tone_mlp = nn.Sequential(nn.Linear(64 + 512, 192), nn.SiLU(), nn.Linear(192, 192), nn.SiLU(),
                                          nn.Linear(192, 2 * n_knots))
            nn.init.zeros_(self.tone_mlp[-1].weight)
            nn.init.zeros_(self.tone_mlp[-1].bias)

    def forward(self, y: torch.Tensor, s: int) -> dict:
        """y: [1,3,h,w] observation in [0,1] (one image: sizes differ per sample)."""
        B = y.shape[0]
        lstd = F.avg_pool2d(F.pad(y.mean(1, keepdim=True), [2] * 4, mode="reflect"), 5, 1)
        lstd = (F.avg_pool2d(F.pad(y.mean(1, keepdim=True) ** 2, [2] * 4, mode="reflect"), 5, 1)
                - lstd ** 2).clamp(min=0).sqrt()
        h, w = y.shape[-2:]
        ph, pw = (-h) % 8, (-w) % 8
        x = torch.cat([y, dark_channel(y), lstd], 1)
        x = F.pad(x, (0, pw, 0, ph), mode="replicate")
        f = self.stem(x)
        skips = []
        for e, dn in zip(self.enc, self.down):
            f = e(f)
            skips.append(f)
            f = dn(f)
        f = self.mid(f)
        tmap = self.t_head(f)                                             # [B,2,h/8,w/8]
        desc = torch.cat([f.mean((-1, -2)), f.std((-1, -2))], 1)
        sid = torch.full((B,), SCALES.index(s), device=y.device, dtype=torch.long)
        g = self.fuse(torch.cat([desc, self.spec(spectrum_features(y)), self.s_emb(sid)], 1))
        o = self.head(g)
        i = 0

        def take(n):
            nonlocal i
            v = o[:, i:i + n]
            i += n
            return v
        out = dict(presence=take(len(KINDS)), g=g)
        out["desc_mu"], out["desc_lv"] = take(self.n_desc), take(self.n_desc).clamp(-10, 4)
        out["rs_family"] = take(len(RS_FAMILIES))
        out["rs_aa"] = take(2)
        out["rs_a_mu"], out["rs_a_lv"] = take(1)[:, 0], take(1)[:, 0].clamp(-10, 4)
        out["rs_n"] = take(len(LANCZOS_N))
        out["rs_sig_mu"], out["rs_sig_lv"] = take(1)[:, 0], take(1)[:, 0].clamp(-10, 4)
        out["noise_type"] = take(2)
        nz = take(6)
        out["noise_mu"], out["noise_lv"] = nz[:, :3], nz[:, 3:].clamp(-10, 4)   # log sigma, log alpha, log beta
        out["q_mu"], out["q_lv"] = torch.sigmoid(take(1)[:, 0]), take(1)[:, 0].clamp(-10, 4)
        out["subsampling"] = take(2)
        km, kl = take(self.n_knots), take(self.n_knots)
        if self.tone_mlp is not None:
            dk = self.tone_mlp(torch.cat([brightness_hist(y), g], 1))
            km, kl = km + dk[:, :self.n_knots], kl + dk[:, self.n_knots:]
        out["knots_mu"] = torch.sigmoid(km)
        out["knots_lv"] = kl.clamp(-10, 4)
        out["space"] = take(len(SPACES))
        out["A_mu"], out["A_lv"] = torch.sigmoid(take(1)[:, 0]), take(1)[:, 0].clamp(-10, 4)
        out["order"] = take(3)
        assert i == o.shape[1]
        tm = F.interpolate(tmap, size=(h + ph, w + pw), mode="bilinear", align_corners=False)[..., :h, :w]
        out["t_mu"], out["t_lv"] = torch.sigmoid(tm[:, :1]), tm[:, 1:].clamp(-10, 4)
        if self.tone_v2 is not None:
            out.update(self.tone_v2(y, g))
        if self.rain_head is not None:
            out["rain"] = self.rain_head(f, skips)[..., :h, :w]                  # [B,1,h,w]
        if self.khead is not None:
            out["kz"], out["kconf"] = self.khead(f, g)                   # [B,K,zc,G,G], [B,K]
        return out


# ----------------------------------------------------------- targets / loss --
def targets(steps):
    """Chain (list of Step, as synthesised) -> training targets + masks."""
    kinds = [s.kind for s in steps]
    t = dict(presence=torch.tensor([float(k in kinds) for k in KINDS]))
    st = {s.kind: s for s in steps}
    lin = [i for i, k in enumerate(kinds) if k in ("blur", "resample", "haze", "layer")]
    if "blur" in st:
        kb = st["blur"].p["k"].float()
        t["desc"] = descriptors(kb)[0].cpu()
        t["kc"] = canvas(kb)
    if "resample" in st:
        r = st["resample"].p
        t["rs_family"] = RS_FAMILIES.index(r["family"])
        t["rs_aa"] = int(bool(r["antialias"]))
        if r["family"] == "cubic":
            t["rs_a"] = float(r["a"])
        if r["family"] == "lanczos":
            t["rs_n"] = LANCZOS_N.index(int(r["n"]))
        if r["family"] == "gauss":
            t["rs_sig"] = float(r["sigma"])
    if "noise" in st:
        n = st["noise"].p
        t["noise_type"] = 0 if n["type"] == "gauss" else 1
        t["noise"] = (torch.tensor([math.log(n["sigma"]), 0.0, 0.0]) if t["noise_type"] == 0 else
                      torch.tensor([0.0, math.log(n["alpha"]), math.log(max(n["beta"], 1e-3))]))
    if "jpeg" in st:
        t["q"] = st["jpeg"].p["q"] / 100.0
        t["subsampling"] = 0 if st["jpeg"].p["subsampling"] == "420" else 1
    if "tone" in st:
        t["knots"] = torch.tensor(st["tone"].p["knots_y"], dtype=torch.float32)
        t["space"] = SPACES.index(st["tone"].p["space"])
        if "_pre" in st["tone"].p:          # pixel samples before / after the curve (image-space loss)
            t["tone_pre"], t["tone_post"] = st["tone"].p["_pre"], st["tone"].p["_post"]
    if "haze" in st:
        t["A"] = st["haze"].p["A"]
        t["t"] = st["haze"].p["t"]
    if "layer" in st and torch.is_tensor(st["layer"].p.get("L")):          # streak map at observation resolution
        f = 1
        for q in steps[kinds.index("layer") + 1:]:
            if q.kind == "resample":
                f *= q.p["s"]
        L = st["layer"].p["L"][:, :1]
        t["rain"] = F.avg_pool2d(L, f) if f > 1 else L
    ti = kinds.index("tone") if "tone" in kinds else None
    ni = kinds.index("noise") if "noise" in kinds else None
    ri = kinds.index("resample") if "resample" in kinds else None
    om = torch.zeros(3); ov = torch.zeros(3)
    if ti is not None and lin:
        om[0], ov[0] = 1, float(ti < lin[0])
    if ni is not None and ri is not None:
        om[1], ov[1] = 1, float(ni < ri)
    if ni is not None and ti is not None:
        om[2], ov[2] = 1, float(ni < ti)
    t["order_mask"], t["order"] = om, ov
    return t


def index(out, b):
    """Per-sample view of a batched forward (loss_fn expects batch size 1)."""
    return {k: v[b:b + 1] for k, v in out.items()}


def canvas(k, size=None):
    """Centre a PSF on a size x size canvas (zero padding) and move its centroid to
    the centre by an INTEGER roll (sub-pixel interpolation smears thin kernels)."""
    size = CANVAS if size is None else size
    k = k.view(k.shape[-2:]).float()
    assert k.shape[-1] <= size and k.shape[-1] % 2 == 1, (k.shape, size)
    p = (size - k.shape[-1]) // 2
    k = F.pad(k, [p] * 4) if p > 0 else k
    a = torch.arange(size, dtype=k.dtype, device=k.device) - (size - 1) / 2
    s = k.sum().clamp(min=1e-12)
    cy, cx = float((k.sum(1) * a).sum() / s), float((k.sum(0) * a).sum() / s)
    return torch.roll(k / s, (-int(round(cy)), -int(round(cx))), (0, 1))


def kernel_hyp_loss(out, t, b, C, eps=0.05, w_lat=0.1):
    """Relaxed winner-takes-all over the K kernel hypotheses of sample b.
    Each hypothesis is drawn by the (frozen) kernel prior C; the loss is the pixel
    L1 to the true kernel. The best hypothesis gets weight 1-eps, the others
    eps/(K-1) each, so no hypothesis dies and none is forced to the average (which
    is what turns single-output kernel regressors into blobs). A small latent term
    pulls the winner toward C's own code of the true kernel; a cross-entropy
    teaches the confidence head which hypothesis wins."""
    z = out["kz"][b]                                                     # [K,zc,G,G]
    K = z.shape[0]
    kh = C.decode(z, CANVAS)[:, 0]                                       # [K,79,79]
    kt = t["kc"].to(kh.device)
    l1 = (kh - kt[None]).abs().flatten(1).sum(1)                          # [K]
    win = int(l1.argmin())
    w = torch.full((K,), eps / max(K - 1, 1), device=kh.device); w[win] = 1 - eps
    with torch.no_grad():
        zt = C.encode(kt[None, None])[0][0]
    lat = ((z[win] - zt) ** 2).mean()
    ce = F.cross_entropy(out["kconf"][b:b + 1], torch.tensor([win], device=kh.device))
    return (w * l1).sum() + w_lat * lat + 0.1 * ce, float(l1[win]), win


def gnll(mu, target, lv):
    return 0.5 * (lv + (target - mu) ** 2 / lv.exp())


def tone_image_loss(knots_mu, space, pre, post):
    """Mean |T_hat^-1(T(x)) - x| on pixel samples (the error that matters when undoing a
    darkening), differentiable in the knots."""
    from etr.world import ops as O
    xi, _ = O.tone_inv(post, "spline", dict(knots_y=monotone_knots(knots_mu)), space)
    return (xi.clamp(0, 1) - pre).abs().mean()


def rain_loss(l_hat, target):
    """Streak-weighted L1 on the layer (streaks cover a few percent of the pixels; plain L1 would
    settle on L_hat = 0)."""
    return 20.0 * ((1 + 10 * target) * (l_hat - target).abs()).mean()


def tone_v2_loss(out, t, dev, eps=0.05):
    """Image-weighted inverse-curve loss on the tone step's own pixel samples."""
    from etr.world import ops as O
    sp = SPACES[t["space"]]
    pre, post = t["tone_pre"].to(dev), t["tone_post"].to(dev)
    q = out["tone_q"][0]
    errs = []
    for k in range(out["tone_u"].shape[1]):
        xi, _ = O.tone_inv(post, "pwl", dict(kx=out["tone_u"][0, k], ky=q), sp)   # forward T: u_i -> q_i
        errs.append(((xi.clamp(0, 1) - pre) ** 2).mean())
    errs = torch.stack(errs)
    K = len(errs)
    win = int(errs.argmin())
    w = torch.full((K,), eps / max(K - 1, 1), device=dev); w[win] = 1 - eps
    xm, _ = O.tone_inv(post, "pwl", dict(kx=out["u_mu"][0], ky=q), sp)
    mean_err = ((xm.clamp(0, 1) - pre) ** 2).mean()
    ce = F.cross_entropy(out["tone_pi"][0:1], torch.tensor([win], device=dev)) if K > 1 else 0.0
    return 30.0 * ((w * errs).sum() + mean_err) + 0.1 * ce


def loss_fn(out, t, dev, w_tone_img=0.0, w_rain=0.0, w_rain_energy=0.0, w_haze_inv=0.0, y=None):
    """Sum of NLL / CE terms, each only where its factor is present."""
    L = {}
    L["presence"] = F.binary_cross_entropy_with_logits(out["presence"][0], t["presence"].to(dev), reduction="sum")
    if "desc" in t:
        L["desc"] = gnll(out["desc_mu"][0], t["desc"].to(dev), out["desc_lv"][0]).mean()
    if "rs_family" in t:
        ce = lambda logits, c: F.cross_entropy(logits, torch.tensor([c], device=dev))
        L["rs_family"] = ce(out["rs_family"], t["rs_family"])
        L["rs_aa"] = ce(out["rs_aa"], t["rs_aa"])
        if "rs_a" in t:
            L["rs_a"] = gnll(out["rs_a_mu"][0], torch.tensor(t["rs_a"], device=dev), out["rs_a_lv"][0])
        if "rs_n" in t:
            L["rs_n"] = ce(out["rs_n"], t["rs_n"])
        if "rs_sig" in t:
            L["rs_sig"] = gnll(out["rs_sig_mu"][0], torch.tensor(t["rs_sig"], device=dev), out["rs_sig_lv"][0])
    if "noise_type" in t:
        L["noise_type"] = F.cross_entropy(out["noise_type"], torch.tensor([t["noise_type"]], device=dev))
        sl = [0] if t["noise_type"] == 0 else [1, 2]
        L["noise"] = gnll(out["noise_mu"][0, sl], t["noise"].to(dev)[sl], out["noise_lv"][0, sl]).mean()
    if "q" in t:
        L["q"] = gnll(out["q_mu"][0], torch.tensor(t["q"], device=dev), out["q_lv"][0])
        L["subsampling"] = F.cross_entropy(out["subsampling"], torch.tensor([t["subsampling"]], device=dev))
    if "knots" in t and "tone_u" in out:
        L["space"] = F.cross_entropy(out["space"], torch.tensor([t["space"]], device=dev))
        if "tone_pre" in t:
            L["tone_v2"] = tone_v2_loss(out, t, dev)
    elif "knots" in t:
        L["knots"] = gnll(out["knots_mu"][0], t["knots"].to(dev), out["knots_lv"][0]).mean()
        L["space"] = F.cross_entropy(out["space"], torch.tensor([t["space"]], device=dev))
        if w_tone_img > 0 and "tone_pre" in t:
            L["tone_img"] = w_tone_img * tone_image_loss(out["knots_mu"][0], SPACES[t["space"]],
                                                         t["tone_pre"].to(dev), t["tone_post"].to(dev))
    if "A" in t:
        L["A"] = gnll(out["A_mu"][0], torch.tensor(t["A"], device=dev), out["A_lv"][0])
        tt = t["t"]
        if torch.is_tensor(tt):
            tt = tt.to(dev)
            if tt.shape[-2:] != out["t_mu"].shape[-2:]:      # haze applied before a resample
                tt = F.interpolate(tt, size=out["t_mu"].shape[-2:], mode="area")
            tt = tt.expand_as(out["t_mu"])
        else:
            tt = torch.full_like(out["t_mu"], float(tt))
        L["t"] = gnll(out["t_mu"], tt, out["t_lv"]).mean()
        if w_haze_inv > 0 and y is not None:            # error in the DEHAZED image (A and t together, 1/t amplified)
            yy = y.to(dev)
            if yy.shape[-2:] != tt.shape[-2:]:
                yy = F.interpolate(yy, size=tt.shape[-2:], mode="area")
            inv = lambda tm, A: ((yy - A * (1 - tm)) / tm).clamp(0, 1)
            L["haze_inv"] = w_haze_inv * (inv(out["t_mu"].clamp(0.05, 1), out["A_mu"][0])
                                          - inv(tt.clamp(0.05, 1), float(t["A"]))).abs().mean()
    if w_rain > 0 and "rain" in out:
        tr = t["rain"].to(dev) if "rain" in t else torch.zeros_like(out["rain"])
        if tr.shape[-2:] != out["rain"].shape[-2:]:
            tr = F.interpolate(tr, size=out["rain"].shape[-2:], mode="area")
        L["rain"] = w_rain * rain_loss(out["rain"], tr)
        if w_rain_energy > 0 and float(tr.sum()) > 0:            # the head under-estimated streak energy (54%)
            L["rain_energy"] = w_rain_energy * (out["rain"].sum() - tr.sum()).abs() / tr.sum()
    if t["order_mask"].sum() > 0:
        m = t["order_mask"].to(dev)
        L["order"] = (F.binary_cross_entropy_with_logits(out["order"][0], t["order"].to(dev), reduction="none") * m).sum()
    return sum(L.values()), {k: float(v) for k, v in L.items()}
