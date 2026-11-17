"""ETR demo: degrade a clean photo with a random chain from the declared generator, restore it blind, and compare the
estimated chain with the true one.

    python demo/run_demo.py --image path/to/clean_photo.png --preset haze_blur --out demo_out/

Writes demo_out/<name>_s<seed>/
    panel.png          degraded | chain inversion | ETR | clean, with the true and the estimated chain written below
    degraded.png       the input given to ETR (low resolution when the chain resamples)
    restored.png       ETR's output;  solver.png  the chain inversion before the refiner
    chain_true.json    the chain that made the input;  chain_estimated.json  ETR's estimate (it never sees the truth)
Presets (moderate composites): haze_blur, dark_noise, motion_jpeg, rain_lowres. --preset random draws a chain from the
declared generator, whose range is wide (up to four factors, some severe); --chain "blur,noise,jpeg" draws the named
factors from the generator in the order written.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from etr.pipeline import ETR, chain_summary                                   # noqa: E402
from etr.ops import kernels as K                                              # noqa: E402
from etr.world import generator as G                                          # noqa: E402
from etr.world.chain import Step                                              # noqa: E402
from etr.world.resamplers import resampler_psf                                # noqa: E402

Q8 = G.Q8


def preset_chain(name):
    """Moderate composite chains, in the order of image formation."""
    dev = "cuda"
    gauss = lambda s: Step("noise", dict(type="gauss", sigma=s / 255, var=(s / 255) ** 2), Q8)
    if name == "haze_blur":
        return [Step("haze", dict(t=0.6, A=0.85, field="homog"), Q8), Step("blur", dict(k=K.gaussian(31, 2.5).to(dev)), Q8),
                gauss(4)]
    if name == "dark_noise":
        return [Step("tone", dict(G._with_knots(dict(kind="gain", g=0.35)), space="hsv_v"), Q8), gauss(12)]
    if name == "motion_jpeg":
        return [Step("blur", dict(k=K.motion(31, 13, 30.0).to(dev)), Q8), gauss(3),
                Step("jpeg", dict(q=40, subsampling="420"))]
    if name == "rain_lowres":
        rs = dict(s=4, family="cubic", antialias=True, a=-0.5, n=3, sigma=0.5)
        rs["k"] = resampler_psf(4, "cubic", True, -0.5, 3, 0.5, device=dev)
        return [Step("layer", dict(L=None, n=300, base=1.7, spread=0.15, length=(10, 40), intensity=(0.05, 0.35)), Q8),
                Step("resample", rs, Q8)]
    raise SystemExit(f"unknown preset {name}")

FONT = cv2.FONT_HERSHEY_SIMPLEX


def describe(d):
    k = d["kind"]
    if k == "blur":
        return f"blur ({d['support_px']} px kernel)"
    if k == "resample":
        return f"downscale x{d['scale']} ({d['family']})"
    if k == "noise":
        return (f"noise (Gaussian, sigma {255 * d['sigma']:.1f}/255)" if d["type"] == "gauss" else
                f"noise (Poisson-Gaussian, a {d['alpha']:.4f}, b {255 * d['beta']:.1f}/255)")
    if k == "jpeg":
        return f"JPEG (q {d['quality']})"
    if k == "tone":
        return f"tone curve ({d.get('space', '')})"
    if k == "haze":
        return f"haze (A {d['airlight']:.2f}, mean t {d['transmission_mean']:.2f})"
    if k == "layer":
        return "rain layer"
    return k


def fixed_chain(kinds, rng, hw):
    """The declared generator's own factor sampler, in the order given (checked against the grammar)."""
    steps = [G._factor(k, rng, hw, 1, "cuda") for k in kinds]
    if not G._supported(steps):
        raise SystemExit(f"order {kinds} has no convex data term; put tone curves first or last (before JPEG)")
    return steps


def to_img(x, hw):
    a = (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()[..., ::-1]
    return cv2.resize(a, (hw[1], hw[0]), interpolation=cv2.INTER_CUBIC) if a.shape[:2] != tuple(hw) else a


def psnr_y(a, b):
    """Luminance PSNR on BGR uint8 images: BT.601 studio-swing Y (16-235), as in the paper's tables."""
    y = lambda u: 16 + (65.481 * u[..., 2] + 128.553 * u[..., 1] + 24.966 * u[..., 0]) / 255.0
    d = y(a.astype(np.float64)) - y(b.astype(np.float64))
    return 20 * np.log10(255.0 / max(float(np.sqrt(np.mean(d ** 2))), 1e-6))


def panel(imgs, titles, lines, width=1600):
    h, w = imgs[0].shape[:2]
    tw = width // len(imgs)
    th = int(round(h * tw / w))
    row = np.concatenate([cv2.resize(i, (tw, th), interpolation=cv2.INTER_AREA) for i in imgs], 1)
    head = np.full((34, row.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(titles):
        cv2.putText(head, t, (i * tw + 8, 24), FONT, 0.6, (0, 0, 0), 1, cv2.LINE_AA)
    foot = np.full((22 * len(lines) + 12, row.shape[1], 3), 255, np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(foot, t, (8, 22 * (i + 1)), FONT, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
    return np.concatenate([head, row, foot], 0)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True, help="a clean photograph")
    ap.add_argument("--preset", default="haze_blur", help="haze_blur | dark_noise | motion_jpeg | rain_lowres | random")
    ap.add_argument("--chain", default="", help='e.g. "haze,blur,noise": these factors from the generator, in this order')
    ap.add_argument("--seed", type=int, default=0, help="draws the random parameters (noise, rain, random chains)")
    ap.add_argument("--max_side", type=int, default=768, help="the photo is cropped to at most this size")
    ap.add_argument("--out", default="demo_out")
    a = ap.parse_args()

    bgr = cv2.imread(a.image, cv2.IMREAD_COLOR)
    assert bgr is not None, a.image
    H, W = (min(s, a.max_side) // 8 * 8 for s in bgr.shape[:2])           # divisible by every resampling factor
    y0, x0 = (bgr.shape[0] - H) // 2, (bgr.shape[1] - W) // 2
    clean = bgr[y0:y0 + H, x0:x0 + W]
    x = torch.from_numpy(clean[..., ::-1].astype(np.float32).copy() / 255).permute(2, 0, 1)[None].cuda()

    rng = random.Random(a.seed)
    if a.chain:
        steps = fixed_chain(a.chain.split(","), rng, (H, W))
    elif a.preset == "random":
        steps = G.sample_chain(rng, (H, W), "cuda")
    else:
        steps = preset_chain(a.preset)
    tag = a.chain.replace(",", "-") if a.chain else a.preset
    y = G.synthesize(x, steps, rng)
    scale = W // y.shape[-1]
    true = chain_summary(dict(present=[q.kind for q in steps]), steps)

    etr = ETR()
    r = etr.restore(y, scale=scale, seed=a.seed)

    out = os.path.join(a.out, f"{os.path.splitext(os.path.basename(a.image))[0]}_{tag}_s{a.seed}")
    os.makedirs(out, exist_ok=True)
    deg, sol, res = to_img(y, (H, W)), to_img(r["solver"], (H, W)), to_img(r["image"], (H, W))
    cv2.imwrite(os.path.join(out, "degraded.png"), to_img(y, y.shape[-2:]))
    cv2.imwrite(os.path.join(out, "solver.png"), sol)
    cv2.imwrite(os.path.join(out, "restored.png"), res)
    json.dump(true, open(os.path.join(out, "chain_true.json"), "w"), indent=1)
    json.dump(dict(r["chain"], seconds=round(r["seconds"], 2)), open(os.path.join(out, "chain_estimated.json"), "w"), indent=1)
    p = [psnr_y(i, clean) for i in (deg, sol, res)]
    lines = ["true chain:      " + "  ->  ".join(describe(d) for d in true["steps"]),
             "estimated chain: " + ("  ->  ".join(describe(d) for d in r["chain"]["steps"]) or "none")]
    if r["chain"]["fallback"]:
        lines.append("dropped (no kernel beat the no-blur option): " + ", ".join(r["chain"]["fallback"]))
    img = panel([deg, sol, res, clean], [f"degraded ({p[0]:.2f} dB)", f"chain inversion ({p[1]:.2f} dB)",
                                         f"ETR ({p[2]:.2f} dB)", "clean"], lines)
    cv2.imwrite(os.path.join(out, "panel.png"), img)
    print("\n".join(lines))
    print(f"Y-PSNR  degraded {p[0]:.2f}  inversion {p[1]:.2f}  ETR {p[2]:.2f} dB   ({r['seconds']:.1f} s)   -> {out}/panel.png")


if __name__ == "__main__":
    main()
