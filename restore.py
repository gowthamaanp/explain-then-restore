"""Restore images with ETR and write the estimated degradation chain next to each output.

    python restore.py --input path/to/image_or_folder --out results/ [--scale 4]

For every input image `name.png` it writes
    results/name.png              the restored image
    results/name_solver.png       the chain inversion before the refiner
    results/name_chain.json       the estimated chain: ordered steps and parameters, detection logits, fallbacks
    results/name_kernel.png       the estimated blur kernel, when the chain has a blur step
--scale is the ratio of the output size to the input size (for low-resolution inputs); it is not estimated.
"""
from __future__ import annotations

import argparse
import json
import os

import cv2
import numpy as np
import torch

from etr.pipeline import ETR

EXT = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp")


def read_rgb(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    assert bgr is not None, f"cannot read {path}"
    return torch.from_numpy(bgr[..., ::-1].astype(np.float32).copy() / 255).permute(2, 0, 1)[None].cuda()


def write_rgb(path, x):
    a = (x[0].clamp(0, 1) * 255).round().byte().permute(1, 2, 0).cpu().numpy()[..., ::-1]
    cv2.imwrite(path, a)


def write_kernel(path, k, zoom=4):
    k = k.detach().float().cpu().numpy()
    a = (255 * k / max(float(k.max()), 1e-12)).astype(np.uint8)
    cv2.imwrite(path, cv2.resize(a, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_NEAREST))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="an image or a folder of images")
    ap.add_argument("--out", default="results")
    ap.add_argument("--scale", type=int, default=1, choices=(1, 2, 4, 8))
    ap.add_argument("--config", default=None, help="default: configs/etr.json")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_refiner", action="store_true", help="return the chain inversion only")
    ap.add_argument("--tile_batch", type=int, default=4, help="lower it on GPUs with little memory")
    a = ap.parse_args()
    files = ([os.path.join(a.input, f) for f in sorted(os.listdir(a.input)) if f.lower().endswith(EXT)]
             if os.path.isdir(a.input) else [a.input])
    os.makedirs(a.out, exist_ok=True)
    etr = ETR(**({"config": a.config} if a.config else {}), tile_batch=a.tile_batch)
    for f in files:
        name = os.path.splitext(os.path.basename(f))[0]
        r = etr.restore(read_rgb(f), scale=a.scale, seed=a.seed, refine_output=not a.no_refiner)
        write_rgb(os.path.join(a.out, name + ".png"), r["image"])
        write_rgb(os.path.join(a.out, name + "_solver.png"), r["solver"])
        json.dump(dict(r["chain"], seconds=round(r["seconds"], 2)), open(os.path.join(a.out, name + "_chain.json"), "w"), indent=1)
        for q in r["steps"]:
            if q.kind == "blur":
                write_kernel(os.path.join(a.out, name + "_kernel.png"), q.p["k"])
        print(f"{name}: {' -> '.join(r['chain']['chain']) or 'no degradation'}  ({r['seconds']:.1f} s)", flush=True)


if __name__ == "__main__":
    main()
