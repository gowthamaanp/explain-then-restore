# ETR: Explain Then Restore

ETR restores an image corrupted by several unknown degradations at once. From the single input it **explains** the
image: it estimates an explicit, ordered chain of physical degradations (haze, blur, resampling, rain, noise, tone
curve, JPEG). It then **restores** the image by inverting that chain with a plug-and-play solver whose objective
decreases monotonically, under a learned image prior. A small residual refiner corrects what the estimate got wrong.
The estimated chain is returned with the restored image, so each output can be checked factor by factor.

One model, 25.1M parameters, no language model, trained only on synthetic degradations of generic photographs.

## Setup

```bash
pip install -r requirements.txt       # torch, numpy, opencv-python; pyiqa for evaluate.py
```

A CUDA GPU is required. A 768x1024 image takes about 35 s on one RTX 3090 and fits in 12 GB; pass `--tile_batch 1`
or `2` on smaller GPUs (the result is unchanged, only slower). The checkpoints are in `checkpoints/` (146 MB).

## Restore your images

```bash
python restore.py --input photo.png --out results/
python restore.py --input folder/ --out results/ --scale 4     # low-resolution inputs: output is 4x larger
```

For each image this writes the restored image, the chain inversion before the refiner (`*_solver.png`), the estimated
chain (`*_chain.json`) and, when the chain contains a blur, the estimated kernel (`*_kernel.png`). `--scale` is the
ratio of the requested output size to the input size; ETR does not estimate it.

A chain file looks like this (the haze + blur + noise demo below):

```json
{
 "chain": ["haze", "blur", "noise"],
 "present": ["blur", "haze", "noise"],
 "presence_logits": {"blur": 2.252, "resample": -17.339, "noise": 11.269, "jpeg": -7.459,
                     "tone": -2.064, "haze": 4.555, "layer": -4.146},
 "fallback": [],
 "steps": [
  {"kind": "haze", "airlight": 0.8408, "transmission_mean": 0.5629},
  {"kind": "blur", "centroid": [-0.21, -0.65], "support_px": 71},
  {"kind": "noise", "type": "gauss", "sigma": 0.016527}
 ]
}
```

`chain` is the order in which ETR believes the degradations were applied, `presence_logits` the detection scores
(a factor is kept above its threshold in `configs/etr.json`), and `fallback` lists factors that were detected but
dropped (a blur for which no candidate kernel beat the no-blur option).

## Demo

```bash
python demo/run_demo.py --image clean_photo.png --preset haze_blur --out demo_out/
python demo/run_demo.py --image clean_photo.png --preset random --seed 3 --out demo_out/
python demo/run_demo.py --image clean_photo.png --chain "haze,blur,noise,jpeg" --seed 1 --out demo_out/
```

The demo degrades a clean photo, restores it blind, and writes `panel.png`: the degraded input, the chain inversion,
ETR's output and the clean photo, with the true and the estimated chain written below and the luminance PSNR of each.
ETR never sees the true chain. Presets are moderate composites (`haze_blur`, `dark_noise`, `motion_jpeg`,
`rain_lowres`); `--preset random` draws a chain from ETR's declared degradation generator, whose range is wide (up to
four factors, some severe); `--chain` draws the named factors from the generator in the order written.

## Evaluate on paired data

```bash
python evaluate.py --lq path/to/degraded --gt path/to/ground_truth --out results/eval --recursive
```

Scoring follows the paper: PSNR and SSIM on BT.601 studio-swing luminance (pyiqa, `test_y_channel=True`,
`color_space="ycbcr"`), outputs rounded to 8 bits, no border crop; the scale of each pair is the ratio of the
ground-truth size to the input size. For MiO100, point `--lq` at the released degraded images (`Group A`, `Group B`,
`Group C` folders) and `--gt` at the clean images; `summary.json` reports each group. Results reproduce the paper's
to within GPU nondeterminism: cuDNN autotuning (enabled by the solver) makes results vary between runs in the last
bits. On 16 MiO100 images (one per combination) this release matches the paper's outputs to 0.003 dB on average; on
one image a near-tied kernel choice flipped (0.04 dB).

## Known limitations

- Global darkening is the factor the chain estimator misses most often: a darkened scene can look like a dark one.
- Degradations outside the chain grammar (adherent lens raindrops, depth-varying defocus) have no operator; ETR then
  substitutes the nearest operators it has and can be worse than its input.
- The refiner improves the chain inversion on average but not on every image; `*_solver.png` (or `--no_refiner`) gives
  the inversion alone.

## Code

```
etr/pipeline.py          ETR: explain (estimate the chain) -> invert (solver) -> refine
etr/world/infer.py       chain estimator outputs -> ordered chain; detection thresholds and fallbacks
etr/world/proposal.py    chain estimator network (presence, parameters, order flags, transmission, rain layer, inverse tone curve)
etr/world/chain.py       the chain grammar and its convex data term, with analytic noise-variance propagation
etr/world/solver.py      monotone proximal-gradient solver under the gradient-step image prior; MMSE readout
etr/world/solver_hp.py   solver settings per chain category (configs/solver_hp.json)
etr/world/generator.py   the declared degradation generator (training families; held-out families excluded)
etr/kernel/              blur module: linearised observation, signature head, kernel VAE, retrieval bank, critic
etr/models/              image prior, kernel VAE, refiner, shared blocks
etr/ops/                 kernel families, resamplers, JPEG helpers
configs/etr.json         checkpoints, detection thresholds and inference settings of the paper's main configuration
```

Held-out families: the ring and curved-motion kernels and the S-curve tone curve are excluded from every trained
component and from the retrieval bank (`etr/ops/kernels.py`, `etr/world/generator.py`).
