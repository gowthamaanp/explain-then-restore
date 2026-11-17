"""Blur-kernel construction. Pure geometry -- no learning, no fitting."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from etr.config import MODEL


def _grid(size: int, device=None):
    a = torch.arange(size, dtype=torch.float32, device=device) - (size - 1) / 2
    return torch.meshgrid(a, a, indexing="ij")


def gaussian(size: int, sigma: float, device=None) -> torch.Tensor:
    xx, yy = _grid(size, device)
    k = torch.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    return k / k.sum()


def anisotropic_gaussian(size: int, sx: float, sy: float, angle: float, device=None) -> torch.Tensor:
    xx, yy = _grid(size, device)
    c, s = math.cos(angle), math.sin(angle)
    xr, yr = xx * c + yy * s, -xx * s + yy * c
    k = torch.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    return k / k.sum()


def motion(size: int, length: int, angle: float, device=None) -> torch.Tensor:
    """Linear camera-shake smear"""
    k = torch.zeros(size, size, device=device)
    c = size // 2
    length = max(1, min(length, size))
    for i in range(length):
        t = i - length / 2
        xi = int(math.floor(c + t * math.cos(angle) + 0.5))
        yi = int(math.floor(c + t * math.sin(angle) + 0.5))
        if 0 <= xi < size and 0 <= yi < size:
            k[yi, xi] += 1.0
    if k.sum() == 0:
        k[c, c] = 1.0
    return k / k.sum()


def disk(size: int, radius: float, device=None) -> torch.Tensor:
    """Defocus. Hard spectral nulls (Bessel zeros)."""
    xx, yy = _grid(size, device)
    k = ((xx ** 2 + yy ** 2) <= radius ** 2).float()
    if k.sum() == 0:
        k[size // 2, size // 2] = 1.0
    return k / k.sum()


def delta(size: int, device=None) -> torch.Tensor:
    """The identity kernel."""
    k = torch.zeros(size, size, device=device)
    k[size // 2, size // 2] = 1.0
    return k


def shift_kernel(k: torch.Tensor, dy: float, dx: float) -> torch.Tensor:
    """Sub-pixel shift a [size,size] PSF by (dy,dx) via bicubic resampling.

    Exists because a decimation-PHASE convention (where the LR grid sits
    relative to the HR grid) is mathematically identical to a sub-pixel shift
    of the effective kernel: y = D_phase(K*x) = D_0(shift(K, phase)*x). ETR
    solves at phase 0 (etr/world/ops.py: resample), so any data generated under a
    different convention -- KernelGAN/DIV2KRK's (sf-1)/2, a real camera's
    unknown one, anything -- is only representable if the kernel manifold and
    the estimator have seen shifted kernels (sample_kernel shifts half of the
    generic training kernels by up to +-2 px for this reason).
    """
    size = k.shape[-1]
    kk = k if k.dim() == 4 else k.view(1, 1, size, size)
    yy, xx = torch.meshgrid(torch.linspace(-1, 1, size, device=k.device),
                            torch.linspace(-1, 1, size, device=k.device), indexing="ij")
    grid = torch.stack([xx - 2 * dx / (size - 1), yy - 2 * dy / (size - 1)], dim=-1)[None]
    out = F.grid_sample(kk, grid, mode="bicubic", padding_mode="zeros",
                        align_corners=True).clamp(min=0)
    out = (out / out.sum().clamp(min=1e-12)).view(k.shape)
    return out


def _quad(size: int, sx: float, sy: float, angle: float, device=None):
    """Rotated, scaled squared radius"""
    xx, yy = _grid(size, device)
    c, s = math.cos(angle), math.sin(angle)
    xr, yr = xx * c + yy * s, -xx * s + yy * c
    return (xr / max(sx, 1e-3)) ** 2 + (yr / max(sy, 1e-3)) ** 2


def generalized_gaussian(size: int, sx: float, sy: float, angle: float,
                         beta: float, device=None) -> torch.Tensor:
    """exp(-0.5 * quad^beta). beta=1 is the ordinary Gaussian; beta<1 is
    heavy-tailed, beta>1 tends toward a box."""
    k = torch.exp(-0.5 * _quad(size, sx, sy, angle, device).clamp(min=0) ** beta)
    return k / k.sum().clamp(min=1e-12)


def plateau(size: int, sx: float, sy: float, angle: float,
            beta: float, device=None) -> torch.Tensor:
    """1 / (1 + quad^beta) -- a flat-topped, heavy-tailed PSF."""
    k = 1.0 / (1.0 + _quad(size, sx, sy, angle, device).clamp(min=0) ** beta)
    return k / k.sum().clamp(min=1e-12)


def sinc(size: int, cutoff: float, device=None) -> torch.Tensor:
    """Ideal low-pass (jinc) filter. 
    `cutoff` is the angular cutoff frequency in (0, pi]."""
    c = (size - 1) / 2
    a = torch.arange(size, dtype=torch.float64, device=device) - c
    yy, xx = torch.meshgrid(a, a, indexing="ij")
    r = (xx ** 2 + yy ** 2).sqrt()
    k = cutoff * torch.special.bessel_j1(cutoff * r.clamp(min=1e-9)) / (2 * math.pi * r.clamp(min=1e-9))
    k[r == 0] = cutoff ** 2 / (4 * math.pi)          # analytic limit at the origin
    k = k.clamp(min=0).float()
    return k / k.sum().clamp(min=1e-12)


def ring(size: int, r_out: float, r_in: float, device=None) -> torch.Tensor:
    """Annulus -- defocus through an obstructed aperture (catadioptric
    lenses, and any optic with a central stop)."""
    xx, yy = _grid(size, device)
    rr = xx ** 2 + yy ** 2
    k = ((rr <= r_out ** 2) & (rr >= min(r_in, r_out - 1e-3) ** 2)).float()
    if k.sum() == 0:
        k[size // 2, size // 2] = 1.0
    return k / k.sum()


def motion_curved(size: int, length: float, curvature: float, angle: float,
                  steps: int = 96, device=None) -> torch.Tensor:
    """Camera shake along a CURVED trajectory, rasterised with bilinear
    splatting."""
    k = torch.zeros(size, size, device=device)
    c = (size - 1) / 2
    t = torch.linspace(-0.5, 0.5, steps, device=device)
    # quadratic bow perpendicular to the direction of travel
    u = t * length
    v = curvature * length * (t ** 2 - 0.25)
    ca, sa = math.cos(angle), math.sin(angle)
    xs, ys = c + u * ca - v * sa, c + u * sa + v * ca

    x0, y0 = xs.floor().long(), ys.floor().long()
    fx, fy = xs - x0.float(), ys - y0.float()
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            w = (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
            if ok.any():
                k.view(-1).index_add_(0, (yi[ok] * size + xi[ok]), w[ok])
    if k.sum() == 0:
        k[size // 2, size // 2] = 1.0
    return k / k.sum()


def random_blob(size: int, n: int = 3, rng: torch.Generator = None,
                device=None) -> torch.Tensor:
    """A sum of small offset Gaussians of random weight."""
    r = lambda a, b: a + (b - a) * torch.rand((), generator=rng).item()
    xx, yy = _grid(size, device)
    k = torch.zeros(size, size, device=device)
    spread = size / 6.0
    for _ in range(max(1, n)):
        cx, cy = r(-spread, spread), r(-spread, spread)
        s = r(0.5, size / 6.0)
        k = k + r(0.3, 1.0) * torch.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * s ** 2))
    return k / k.sum().clamp(min=1e-12)


def convolve_kernels(a: torch.Tensor, b: torch.Tensor, size: int) -> torch.Tensor:
    """Compose two PSFs (e.g. defocus during camera shake), cropped back to
    `size` and renormalised."""
    import torch.nn.functional as F
    pad = size // 2
    ap = F.pad(a.view(1, 1, *a.shape), [pad] * 4)
    k = F.conv2d(ap, b.view(1, 1, *b.shape)).squeeze()
    c = k.shape[-1] // 2
    lo, hi = c - size // 2, c - size // 2 + size
    k = k[lo:hi, lo:hi]
    return k / k.sum().clamp(min=1e-8)


def add_kernel_noise(k: torch.Tensor, noise_level: float = None,
                     rng: torch.Generator = None) -> torch.Tensor:
    """Multiplicative uniform noise on an existing kernel shape, renormalised
    (the mechanism of KernelGAN's kernel generator), with the noise level itself
    random (U[0.1, 0.5]). Applied to some TRAIN draws so kernel prior C also covers
    noisy, irregular kernels as a class."""
    r = rng
    if noise_level is None:
        noise_level = 0.1 + torch.rand((), generator=r).item() * 0.4
    noise = -noise_level + torch.rand(k.shape, generator=r, device=k.device) * noise_level * 2
    raw = (k * (1 + noise)).clamp(min=0)  # disk/delta bases have exact zeros; stay exact
    s = raw.sum()
    return raw / s if s > 0 else k


# ---------------------------------------------------------------------------
# Parameter ranges of the TRAIN families on the 31-px canvas (they scale with the
# canvas, see _scaled_ranges). `sample_extrap_kernel` draws just beyond them and is
# used only by the calibration split.
# ---------------------------------------------------------------------------
KP_TRAIN_RANGES = dict(
    gauss_sigma=(0.3, 12.0),
    aniso_sigma=(0.3, 12.0),
    gengauss_beta=(0.4, 4.0),
    plateau_beta=(0.8, 2.5),
    sinc_cutoff=(0.3, 3.2),
    motion_len=(3, MODEL.ksize),  # up to the full kernel -- tracks ksize,
                                   # not a literal, so it can't drift out of
                                   # sync with it again
    disk_radius=(1.0, 14.0),
    ring_outer=(2.5, 14.0),
    curve=(0.0, 1.2),
    blob_n=(2, 6),
)


def sample_extrap_kernel(size: int, rng: torch.Generator = None, device=None) -> torch.Tensor:
    """A MILD extrapolation probe: parameters just beyond `KP_TRAIN_RANGES`,
    one axis at a time.

    This exists for model selection -- without a held-out signal there is no
    way to tell a manifold that learned the shape space from one that
    memorised the pool, and no principled place to stop training.

    Two properties are deliberate, both learned the hard way. It perturbs ONE
    axis at a time: the first version pushed length AND curvature past their
    caps together and scored 1.429 of a possible 2.0, dominating the mean and
    pinning selection to an undertrained checkpoint. And it contains no family
    that is absent from training: selecting against a family you never train
    on measures generality you are not optimising for, so it can only degrade
    as the model specialises, which is exactly what was observed.
    """
    R = KP_TRAIN_RANGES
    u = torch.rand((), generator=rng).item()
    r = lambda a, b: a + (b - a) * torch.rand((), generator=rng).item()
    ang = lambda: torch.rand((), generator=rng).item() * math.pi
    over = lambda ab, f=1.25: (ab[1], ab[1] * f)     # just past the trained edge

    if u < 0.22:
        return gaussian(size, r(*over(R["gauss_sigma"])), device)
    if u < 0.40:
        return anisotropic_gaussian(size, r(*over(R["aniso_sigma"])),
                                    r(0.3, 1.0), ang(), device)
    if u < 0.55:
        return disk(size, r(*over(R["disk_radius"])), device)
    if u < 0.70:
        ro = r(*over(R["ring_outer"]))
        return ring(size, ro, ro * r(0.88, 0.95), device)     # thinner annulus
    if u < 0.82:
        return motion_curved(size, r(R["motion_len"][1] * 0.8, R["motion_len"][1]),
                             r(*over(R["curve"])), ang(), device=device)
    if u < 0.92:
        return sinc(size, r(R["sinc_cutoff"][1], R["sinc_cutoff"][1] * 1.2), device)
    return generalized_gaussian(size, r(0.3, 6.0), r(0.3, 6.0), ang(),
                                r(*over(R["gengauss_beta"])), device)


# ---------------------------------------------------------------------------
# Family splits. The generalisation claim is at the FAMILY level:
#   train   -- every trained component (the kernel VAE, the signature head, the critic,
#              the chain estimator) sees only these families (+ generic random trajectories /
#              apertures, filtered to stay away from the held-out shapes)
#   calib   -- only for calibration
#   heldout -- ring, curved motion: never trained or calibrated on; evaluation only.
#              (Disk, the textbook defocus model, is a TRAIN family; generalisation is
#              evaluated on ring + curved motion.)
# ---------------------------------------------------------------------------

HELDOUT_FAMILIES = ("ring", "motion_curved")
CALIB_FAMILIES = ("blob", "extrap")
TRAIN_FAMILIES = ("gauss", "aniso", "gengauss", "plateau", "sinc", "motion",
                  "motion_profiled", "gauss_x_motion", "delta", "defocus")

# Caps that keep a trained family from approximating a held-out one: an
# isotropic generalised Gaussian at beta=4 is exp(-0.5 r^8), which is a disk
# in all but name.
SPLIT_TRAIN_RANGES = dict(KP_TRAIN_RANGES, gengauss_beta=(0.4, 2.0), plateau_beta=(0.8, 2.0))


def _scaled_ranges(sc: float) -> dict:
    """SPLIT_TRAIN_RANGES for a canvas sc x the native 31: widths and lengths
    scale by sc, the jinc cutoff (a frequency) by 1/sc; shape exponents do not."""
    R = dict(SPLIT_TRAIN_RANGES)
    if sc == 1.0:
        return R
    for k in ("gauss_sigma", "aniso_sigma"):
        R[k] = (R[k][0], R[k][1] * sc)
    R["sinc_cutoff"] = (R["sinc_cutoff"][0] / sc, R["sinc_cutoff"][1])
    R["motion_len"] = (R["motion_len"][0], int(round(R["motion_len"][1] * sc)))
    return R


def motion_profiled(size: int, length: float, angle: float, profile: str = "gauss",
                    decay: float = 0.5, one_sided: bool = True, steps: int = 96,
                    device=None) -> torch.Tensor:
    """Straight-line motion with a non-uniform velocity profile, recentred on
    its centroid. A camera that decelerates (or an exposure that ramps)
    deposits more energy at one end of the streak; `motion` assumes constant
    speed. Recentring removes the global translation a one-sided streak
    implies, which is unidentifiable from y alone (an image shifted by a few
    pixels is just as natural) and is therefore not part of the kernel.

    profile: "uniform" | "linear" (weight falls linearly to `decay`) |
             "gauss" (weight exp(-t^2 / 2 (decay*length)^2) from the start).
    """
    t = torch.linspace(0, 1, steps, device=device)
    if profile == "uniform":
        w = torch.ones_like(t)
    elif profile == "linear":
        w = 1 - (1 - decay) * t
    else:
        w = torch.exp(-0.5 * (t / max(decay, 1e-3)) ** 2)
    u = (t if one_sided else t - 0.5) * length
    u = u - (w * u).sum() / w.sum()                     # centroid at the origin
    c = (size - 1) / 2
    ca, sa = math.cos(angle), math.sin(angle)
    xs, ys = c + u * ca, c + u * sa
    k = torch.zeros(size, size, device=device)
    x0, y0 = xs.floor().long(), ys.floor().long()
    fx, fy = xs - x0.float(), ys - y0.float()
    for dx in (0, 1):
        for dy in (0, 1):
            xi, yi = x0 + dx, y0 + dy
            ww = w * (fx if dx else 1 - fx) * (fy if dy else 1 - fy)
            ok = (xi >= 0) & (xi < size) & (yi >= 0) & (yi < size)
            if ok.any():
                k.view(-1).index_add_(0, (yi[ok] * size + xi[ok]), ww[ok])
    if k.sum() == 0:
        k[size // 2, size // 2] = 1.0
    return k / k.sum()


def _cubic(x: torch.Tensor) -> torch.Tensor:
    """Keys cubic, a = -0.5 (MATLAB imresize's 'cubic')."""
    ax = x.abs()
    return ((1.5 * ax ** 3 - 2.5 * ax ** 2 + 1) * (ax <= 1)
            + (-0.5 * ax ** 3 + 2.5 * ax ** 2 - 4 * ax + 2) * ((ax > 1) & (ax <= 2)))


def bicubic_kernel(size: int, sf: int, device=None) -> torch.Tensor:
    """The anti-aliasing kernel of bicubic downscaling by `sf` (MATLAB
    imresize semantics: the cubic is stretched by sf). Offset by (sf-1)/2
    from the grid centre -- MATLAB's LR pixel i sits at HR coordinate
    sf*i + (sf-1)/2 -- so that phase-0 decimation (etr/world/ops.py: resample) reproduces
    imresize to 2e-7 in the interior. Has negative
    lobes, so it is a resampler factor, never a PSF in the simplex."""
    c = (size - 1) / 2
    a = torch.arange(size, dtype=torch.float32, device=device) - c + (sf - 1) / 2
    h = _cubic(a / sf)
    k = h[:, None] * h[None, :]
    return k / k.sum()


def sample_kernel(size: int, split: str = "train", rng: torch.Generator = None,
                  device=None) -> torch.Tensor:
    """One non-negative PSF from a family split (see the split constants above).
    Bicubic downscaling is NOT a PSF (its cubic has negative lobes); it is a
    separate known resampler factor of the world model, see `bicubic_kernel`."""
    sc = size / 31.0                      # spatial ranges scale with the kernel canvas
    R = _scaled_ranges(sc)
    u = torch.rand((), generator=rng).item()
    r = lambda ab: ab[0] + (ab[1] - ab[0]) * torch.rand((), generator=rng).item()
    ang = lambda: torch.rand((), generator=rng).item() * math.pi

    if split == "heldout":
        if u < 0.4:
            ro = r((2.5, 14.0 * sc))
            return ring(size, ro, ro * r((0.3, 0.85)), device)
        return motion_curved(size, r(R["motion_len"]), r((0.2, 1.2)), ang(), device=device)
    if split == "calib":
        if u < 0.5:
            return random_blob(size, int(r(R["blob_n"])), rng, device)
        k = sample_extrap_kernel(size, rng, device)
        return k
    assert split == "train", split

    if torch.rand((), generator=rng).item() < 0.12:
        # physical defocus: a disk (circular aperture) softened by a small Gaussian
        # (lens aberration / sensor integration)
        k = convolve_kernels(disk(size, r((1.0, 14.0 * sc)), device),
                             gaussian(size, r((0.1, 0.8)), device), size)
        if torch.rand((), generator=rng).item() < 0.5:
            k = shift_kernel(k, -2.0 + 4.0 * torch.rand((), generator=rng).item(),
                             -2.0 + 4.0 * torch.rand((), generator=rng).item())
        return k

    if torch.rand((), generator=rng).item() < 0.35:
        # generic shapes (trajectories / non-circular apertures / compositions),
        # filtered away from every held-out exemplar
        k = sample_generic(size, rng)
        if torch.rand((), generator=rng).item() < 0.5:
            k = shift_kernel(k, -2.0 + 4.0 * torch.rand((), generator=rng).item(),
                             -2.0 + 4.0 * torch.rand((), generator=rng).item())
        return k.to(device) if device is not None else k
    u = torch.rand((), generator=rng).item()
    if u < 0.12:
        k = gaussian(size, r(R["gauss_sigma"]), device)
    elif u < 0.26:
        k = anisotropic_gaussian(size, r(R["aniso_sigma"]), r(R["aniso_sigma"]), ang(), device)
    elif u < 0.36:
        k = generalized_gaussian(size, r(R["aniso_sigma"]), r(R["aniso_sigma"]), ang(),
                                 r(R["gengauss_beta"]), device)
    elif u < 0.44:
        k = plateau(size, r(R["aniso_sigma"]), r(R["aniso_sigma"]), ang(),
                    r(R["plateau_beta"]), device)
    elif u < 0.52:
        k = sinc(size, r(R["sinc_cutoff"]), device)
    elif u < 0.68:
        k = motion(size, int(r(R["motion_len"])), ang(), device)
    elif u < 0.84:
        prof = ("uniform", "linear", "gauss")[int(torch.randint(3, (), generator=rng))]
        k = motion_profiled(size, r(R["motion_len"]), ang(), prof, r((0.15, 0.8)),
                            one_sided=torch.rand((), generator=rng).item() < 0.7, device=device)
    elif u < 0.96:
        k = convolve_kernels(gaussian(size, r((0.4, 1.5 * sc)), device),
                             motion(size, int(r((3, 15 * sc))), ang(), device), size)
    else:
        k = delta(size, device)
    if torch.rand((), generator=rng).item() < 0.35:
        k = add_kernel_noise(k, rng=rng)
    if torch.rand((), generator=rng).item() < 0.5:
        k = shift_kernel(k, -2.0 + 4.0 * torch.rand((), generator=rng).item(),
                         -2.0 + 4.0 * torch.rand((), generator=rng).item())
    return k


# ---------------------------------------------------------------------------
# Generic shape families (world-model plan, kernel generalisation step 1).
# A learned estimator generalises to shapes that are combinations of what it
# has seen; named parametric families alone leave held-out shapes alien. These
# generic families cover the space broadly but are filtered so that NO sample
# is close to a held-out exemplar (disk, ring, curved quadratic streak): the
# held-out claim stays honest. See `_heldout_bank` / `_novel_enough`.
# ---------------------------------------------------------------------------

def _splat_points(size, ys, xs, w=None, device=None):
    k = torch.zeros(size * size, device=device)
    c = (size - 1) / 2
    ys = ys - (ys * (w if w is not None else 1)).sum() / (w.sum() if w is not None else ys.numel()) + c
    xs = xs - (xs * (w if w is not None else 1)).sum() / (w.sum() if w is not None else xs.numel()) + c
    w = torch.ones_like(ys) if w is None else w
    y0, x0 = ys.floor(), xs.floor()
    fy, fx = ys - y0, xs - x0
    for dy in (0, 1):
        for dx in (0, 1):
            yi, xi = (y0 + dy).long(), (x0 + dx).long()
            ww = w * (fy if dy else 1 - fy) * (fx if dx else 1 - fx)
            ok = (yi >= 0) & (yi < size) & (xi >= 0) & (xi < size)
            k.index_add_(0, (yi * size + xi)[ok], ww[ok])
    k = k.view(size, size)
    return k / k.sum().clamp(min=1e-12)


def random_trajectory(size, extent, rng: torch.Generator = None, device=None, n=256):
    """Camera-shake trajectory in the spirit of Boracchi & Foi (2012): random
    accelerations (with occasional impulsive jerks) integrated twice, scaled to
    `extent` px, sampled with a random exposure-time density along the path."""
    r = lambda: torch.rand((), generator=rng).item()
    acc = torch.randn(n, 2, generator=rng)
    jerk = (torch.rand(n, generator=rng) < 0.02).float()[:, None] * torch.randn(n, 2, generator=rng) * 8
    vel = torch.cumsum(acc + jerk, 0) * (0.3 + r())
    vel = vel + torch.randn(1, 2, generator=rng) * n ** 0.5 * r()          # global drift
    pos = torch.cumsum(vel, 0)
    pos = pos - pos.mean(0)
    span = (pos.max(0).values - pos.min(0).values).max().clamp(min=1e-6)
    pos = pos / span * extent
    w = 0.4 + torch.rand(n, generator=rng)
    w = torch.nn.functional.avg_pool1d(w[None, None], 9, 1, 4)[0, 0]       # smooth exposure density
    return _splat_points(size, pos[:, 0].to(device), pos[:, 1].to(device), w.to(device), device)


def random_aperture(size, radius, rng: torch.Generator = None, device=None, ss=4):
    """Filled, non-circular aperture: either a jittered regular polygon (3-8
    sides) or a smooth star-shaped outline r(phi) = R(1 + sum_m a_m cos(m phi +
    phi_m)), rendered with 4x supersampling for anti-aliased edges. The radial
    deviation is forced to be >= 30% so it is never a disk in disguise (and
    sample_generic still rejects anything close to a held-out exemplar)."""
    r = lambda a, b: a + (b - a) * torch.rand((), generator=rng).item()
    n = size * ss
    a = (torch.arange(n, dtype=torch.float32) + 0.5) / ss - size / 2
    yy, xx = torch.meshgrid(a, a, indexing="ij")
    rr, ph = (yy ** 2 + xx ** 2).sqrt(), torch.atan2(yy, xx)
    if torch.rand((), generator=rng).item() < 0.5:
        m = int(r(3, 7))                       # 3-6 sides: clearly non-circular
        rot = r(0, 2 * math.pi)
        sector = 2 * math.pi / m
        aa = ((ph - rot) % sector) - sector / 2
        edge = radius * math.cos(math.pi / m) / torch.cos(aa)
        edge = edge * (1 + r(-0.15, 0.15))
    else:
        edge = torch.ones_like(ph)
        dev_total = 0.0
        for mm in range(2, 7):
            amp = r(0.0, 0.6) / mm
            edge = edge + amp * torch.cos(mm * ph + r(0, 2 * math.pi))
            dev_total += amp
        if dev_total < 0.3:                        # strongly non-circular outlines only
            edge = 1 + (edge - 1) * (0.3 / max(dev_total, 1e-3))
        edge = edge.clamp(min=0.15)
        edge = radius * edge
    m_ = (rr <= edge).float()
    k = torch.nn.functional.avg_pool2d(m_[None, None], ss)[0, 0]
    return (k / k.sum().clamp(min=1e-12)).to(device)


_HELD_BANK = {}


def _heldout_bank(size):
    """Exemplars of the held-out families at this canvas (cached)."""
    if size in _HELD_BANK:
        return _HELD_BANK[size]
    sc = size / 31.0
    ks = []
    for ro in torch.arange(2.5, 14.0 * sc + 1e-6, 1.0).tolist():
        for fr in (0.3, 0.5, 0.7, 0.85):
            ks.append(ring(size, ro, ro * fr))
    for L in torch.linspace(6, 31 * sc, 8).tolist():
        for cv in (0.2, 0.45, 0.7, 0.95, 1.2):
            for an in torch.linspace(0, math.pi, 12)[:-1].tolist():
                ks.append(motion_curved(size, L, cv, an))
    _HELD_BANK[size] = torch.stack(ks).flatten(1)
    return _HELD_BANK[size]


def _novel_enough(k, eps=0.45):
    """True if k is at least eps (L1) from every held-out exemplar, allowing
    +-1 px shifts (a shifted disk is still a disk)."""
    B = _heldout_bank(k.shape[-1])
    kc = k.detach().cpu()
    best = float("inf")
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            d = (B - torch.roll(kc, (dy, dx), (0, 1)).flatten()[None]).abs().sum(1).min().item()
            best = min(best, d)
    return best >= eps


def sample_generic(size, rng: torch.Generator = None, device=None, max_tries=20):
    """One generic kernel (trajectory / aperture / composition), rejected while
    it is within eps of a held-out exemplar."""
    sc = size / 31.0
    r = lambda a, b: a + (b - a) * torch.rand((), generator=rng).item()
    for _ in range(max_tries):
        u = torch.rand((), generator=rng).item()
        if u < 0.45:
            k = random_trajectory(size, r(4, 26 * sc), rng)
        elif u < 0.75:
            k = random_aperture(size, r(1.5, 12 * sc), rng)
        else:
            k = convolve_kernels(random_aperture(size, r(1.0, 5 * sc), rng),
                                 random_trajectory(size, r(4, 18 * sc), rng), size)
        if _novel_enough(k):
            return k.to(device) if device is not None else k
    return gaussian(size, r(0.5, 3.0 * sc), device)
