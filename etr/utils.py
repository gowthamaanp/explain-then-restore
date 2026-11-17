"""Shared training helpers."""
from __future__ import annotations

import math
import os
import random

import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def save_ckpt(path, **kw):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(kw, path)


def load_ckpt(path, map_location="cpu"):
    return torch.load(path, map_location=map_location, weights_only=True)


def cosine_lr(step, warmup, total, base, min_lr=0.0):
    if step < warmup:
        return base * (step + 1) / max(warmup, 1)
    p = min(max((step - warmup) / max(total - warmup, 1), 0.0), 1.0)
    return min_lr + 0.5 * (base - min_lr) * (1 + math.cos(math.pi * p))


class Meter:
    def __init__(self): self.reset()
    def reset(self): self.s, self.n = 0.0, 0
    def update(self, v, n=1): self.s += float(v) * n; self.n += n
    @property
    def avg(self): return self.s / max(self.n, 1)


def infinite(loader):
    while True:
        for b in loader:
            yield b


# ---------------------------------------------------------------------------
# resume
#
# `ckpt_every` was in every stage config and used by none of them: there was
# no periodic checkpoint and no way to resume, so a crash 20 hours into a
# 400K-step run started again from zero. These write a `<out>_last.pt`
# alongside the best-so-far checkpoint and restore model + optimizer + step.
# ---------------------------------------------------------------------------
def resume_path(out: str) -> str:
    return out.replace(".pt", "_last.pt")


def save_resume(out: str, step: int, opt, extra: dict = None, **modules):
    """modules: name -> nn.Module. `extra` is any metadata to carry along."""
    payload = {k: m.state_dict() for k, m in modules.items()}
    payload.update(step=step, opt=opt.state_dict(), **(extra or {}))
    save_ckpt(resume_path(out), **payload)


def try_resume(out: str, opt, device="cpu", **modules) -> int:
    """Returns the step to continue from, or 0 if there is nothing to resume.
    Missing/corrupt state is reported and ignored rather than raising -- a
    bad resume file should not block a fresh run."""
    p = resume_path(out)
    if not os.path.exists(p):
        return 0
    try:
        ck = torch.load(p, map_location=device, weights_only=False)
        for name, m in modules.items():
            m.load_state_dict(ck[name])
        opt.load_state_dict(ck["opt"])
        step = int(ck["step"])
        print(f"[resume] {p} -> continuing from step {step}")
        return step
    except Exception as e:
        print(f"[resume] ignoring {p}: {type(e).__name__}: {e}")
        return 0
