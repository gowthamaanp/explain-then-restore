"""Kernel bank for retrieval: N kernels from the TRAIN families (held-out ring / curved motion never
enter), stored as (size, seed) so each kernel is regenerated exactly by `bank_kernel`, plus its
retrieval embedding (etr/kernel/heads.py: Retrieval.embed_ker of the kernel VAE's latent).

    python -m etr.kernel.bank --vae checkpoints/kernel_vae.pt --head checkpoints/signature_head.pt --n 100000
"""
from __future__ import annotations

import argparse, random

import torch

from etr.kernel.heads import Retrieval
from etr.kernel.synth import SIZES, WEIGHTS
from etr.models.kernel_prior import VAESpatial
from etr.ops import kernels as K
from etr.utils import load_ckpt
from etr.world.proposal import CANVAS, canvas


def bank_kernel(size, seed):
    return canvas(K.sample_kernel(int(size), "train", torch.Generator().manual_seed(int(seed))))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", default="checkpoints/kernel_vae.pt")
    ap.add_argument("--head", default="checkpoints/signature_head.pt")
    ap.add_argument("--n", type=int, default=100000)
    ap.add_argument("--out", default="checkpoints/kernel_bank.pt")
    a = ap.parse_args()
    dev = "cuda"
    ck = load_ckpt(a.vae)
    C = VAESpatial(ksize=CANVAS, zc=16, width=ck.get("width", 96)).to(dev); C.load_state_dict(ck["model"]); C.eval()
    ret = Retrieval().to(dev); ret.load_state_dict(load_ckpt(a.head)["retrieval"]); ret.eval()
    r = random.Random(20261021)
    sizes = [r.choices(SIZES, WEIGHTS)[0] for _ in range(a.n)]
    seeds = [r.randrange(2 ** 31) for _ in range(a.n)]
    embs = []
    for c0 in range(0, a.n, 512):
        ks = torch.stack([bank_kernel(s, z) for s, z in zip(sizes[c0:c0 + 512], seeds[c0:c0 + 512])]).to(dev)
        embs.append(ret.embed_ker(C.encode(ks[:, None])[0]).half().cpu())
        if c0 % 10240 == 0:
            print(f"{c0}/{a.n}", flush=True)
    torch.save(dict(sizes=torch.tensor(sizes, dtype=torch.int16), seeds=torch.tensor(seeds, dtype=torch.int64),
                    emb=torch.cat(embs)), a.out)
    print("saved", a.out, flush=True)


if __name__ == "__main__":
    main()
