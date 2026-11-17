"""Evaluate ETR on paired degraded / ground-truth images with the paper's scorer.

    python evaluate.py --lq path/to/degraded --gt path/to/ground_truth --out results/eval [--recursive]

Pairs files by relative path (--gt may also hold every ground truth flat, matched by file name). The scale of each
pair is the ratio of the ground-truth size to the input size. Scoring: luminance PSNR and SSIM on BT.601
studio-swing Y (pyiqa `psnr` / `ssim` with test_y_channel=True, color_space="ycbcr"), outputs rounded to 8 bits, no
border crop. Writes the outputs, one chain JSON per image and summary.json (means per top-level folder and overall).
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import numpy as np
import pyiqa
import torch

from etr.pipeline import ETR
from restore import EXT, read_rgb, write_rgb


def pairs(lq, gt, recursive):
    out = []
    for dp, _, fs in os.walk(lq):
        for f in sorted(fs):
            if not f.lower().endswith(EXT):
                continue
            rel = os.path.relpath(os.path.join(dp, f), lq)
            g = os.path.join(gt, rel) if os.path.exists(os.path.join(gt, rel)) else os.path.join(gt, f)
            assert os.path.exists(g), f"no ground truth for {rel}"
            out.append((rel, os.path.join(dp, f), g))
        if not recursive:
            break
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lq", required=True)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", default="results/eval")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tile_batch", type=int, default=4)
    a = ap.parse_args()
    psnr = pyiqa.create_metric("psnr", test_y_channel=True, color_space="ycbcr", device="cuda")
    ssim = pyiqa.create_metric("ssim", test_y_channel=True, color_space="ycbcr", device="cuda")
    q = lambda x: (x.clamp(0, 1) * 255).round() / 255
    etr = ETR(tile_batch=a.tile_batch)
    rows, P = {}, pairs(a.lq, a.gt, a.recursive)
    for i, (rel, fl, fg) in enumerate(P):
        y, gt = read_rgb(fl), read_rgb(fg)
        s = int(round(gt.shape[-1] / y.shape[-1]))
        assert y.shape[-2] * s == gt.shape[-2] and y.shape[-1] * s == gt.shape[-1], (rel, y.shape, gt.shape)
        r = etr.restore(y, scale=s, seed=a.seed)
        o = os.path.join(a.out, rel)
        os.makedirs(os.path.dirname(o), exist_ok=True)
        write_rgb(os.path.splitext(o)[0] + ".png", r["image"])
        json.dump(r["chain"], open(os.path.splitext(o)[0] + "_chain.json", "w"), indent=1)
        with torch.no_grad():
            rows[rel] = dict(psnr_y=float(psnr(q(r["image"]), gt)), ssim_y=float(ssim(q(r["image"]), gt)),
                             psnr_y_solver=float(psnr(q(r["solver"]), gt)), chain=r["chain"]["chain"],
                             seconds=round(r["seconds"], 2))
        print(f"[{i + 1}/{len(P)}] {rel}: {rows[rel]['psnr_y']:.2f} dB  {' -> '.join(rows[rel]['chain'])}", flush=True)
    groups = defaultdict(list)
    for rel, v in rows.items():
        groups[rel.split(os.sep)[0] if os.sep in rel else "all"].append(v)
        if os.sep in rel:
            groups["all"].append(v)
    summ = {g: dict(n=len(v), **{m: float(np.mean([x[m] for x in v])) for m in ("psnr_y", "ssim_y", "psnr_y_solver", "seconds")})
            for g, v in groups.items()}
    json.dump(dict(summary=summ, rows=rows), open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    for g, v in summ.items():
        print(f"{g}: n={v['n']}  Y-PSNR {v['psnr_y']:.2f}  Y-SSIM {v['ssim_y']:.4f}  (inversion only {v['psnr_y_solver']:.2f})")


if __name__ == "__main__":
    main()
