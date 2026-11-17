"""The blind kernel estimator (the "blur module"), run on the linearised observation (etr/kernel/linearise.py).

candidates   the signature head (32 hypotheses) on the gradient autocorrelation of up to 5 crops (the centre and up
             to 4 of highest gradient energy) x 8 dihedral views, mapped back; consensus vote over all of them
retrieval    the 32 nearest kernels of a 10^5-kernel bank under the image-kernel embedding
perturb      latent perturbations of the consensus top-8 in the kernel VAE
safe set     the no-blur delta and the top-8 shrunk towards it are always candidates
pick         the highest sum of the standardised consensus and critic scores; the delta winning means no blur
"""
from __future__ import annotations

import torch

from etr.kernel.critic import Critic
from etr.kernel.heads import ACHead, Retrieval
from etr.kernel.restore import decode_k, wiener
from etr.kernel.signature import autocorr_obs
from etr.models.kernel_prior import VAESpatial
from etr.utils import load_ckpt
from etr.world.proposal import CANVAS

dev = "cuda"


def T(img, t):
    """Dihedral transform t in 0..7 on the last two dims."""
    r, tr = t % 4, t >= 4
    if tr:
        img = img.transpose(-2, -1)
    return torch.rot90(img, r, (-2, -1))


def T_inv(img, t):
    r, tr = t % 4, t >= 4
    img = torch.rot90(img, -r, (-2, -1))
    return img.transpose(-2, -1) if tr else img


def candidates(Pn, C, y, views):
    """y [n,3,h,w] -> cand [n, V*Kh, 79, 79], conf [n, V*Kh] (softmax within view)."""
    ks, cs = [], []
    for t in views:
        z, conf = Pn(autocorr_obs(T(y, t)))
        n, Kh = z.shape[:2]
        k = decode_k(C, z.flatten(0, 1)).view(n, Kh, CANVAS, CANVAS)
        ks.append(T_inv(k, t)); cs.append(torch.softmax(conf, 1))
    return torch.cat(ks, 1), torch.cat(cs, 1)


def energy_crops(y, c=480, n_extra=4):
    """Centre crop plus the n_extra highest gradient-energy c x c crops (on a stride-c/2 grid), so the
    signature pools evidence from more of the image than one centre crop."""
    H, W = y.shape[-2:]
    c = min(c, H, W)
    boxes = [((H - c) // 2, (W - c) // 2)]
    if n_extra > 0 and (H > c or W > c):
        l = 0.299 * y[0, 0] + 0.587 * y[0, 1] + 0.114 * y[0, 2]
        e = (l[1:, :-1] - l[:-1, :-1]) ** 2 + (l[:-1, 1:] - l[:-1, :-1]) ** 2
        st = max(c // 2, 1)
        cand = [(y0, x0) for y0 in range(0, H - c + 1, st) for x0 in range(0, W - c + 1, st)]
        sc = sorted(cand, key=lambda b: -float(e[b[0]:b[0] + c - 1, b[1]:b[1] + c - 1].mean()))
        for b in sc:
            if b not in boxes:
                boxes.append(b)
            if len(boxes) > n_extra:
                break
    return torch.cat([y[..., a:a + c, b:b + c] for a, b in boxes]), boxes


class KernelEstimator:
    """Blur module: signature head + kernel VAE + retrieval bank + critic."""

    def __init__(self, vae_path, head_path, critic_path, bank_path="", n_crops=5, n_score=64, n_retrieve=32, crop=480):
        ck = load_ckpt(vae_path)
        self.C = VAESpatial(ksize=CANVAS, zc=16, width=ck.get("width", 96)).to(dev)
        self.C.load_state_dict(ck["model"]); self.C.eval()
        hk = load_ckpt(head_path)
        self.P = ACHead(K=32).to(dev); self.P.load_state_dict(hk["model"]); self.P.eval()
        self.ret = None
        if "retrieval" in hk and bank_path:
            self.ret = Retrieval().to(dev); self.ret.load_state_dict(hk["retrieval"]); self.ret.eval()
            self.bank = torch.load(bank_path, map_location="cpu", weights_only=False)
            self.bank_emb = self.bank["emb"].to(dev).float()
        self.critic = Critic().to(dev); self.critic.load_state_dict(load_ckpt(critic_path)["model"]); self.critic.eval()
        self.n_crops, self.n_score, self.n_retrieve, self.crop = n_crops, n_score, n_retrieve, crop

    @torch.no_grad()
    def estimate(self, y, sig):
        """y: the linearised observation [1,3,H,W]; sig: its noise level. -> (kernel [79,79], info). info: the critic
        score, the margin over no blur (the delta), whether the delta won, and the top-8 kernels with their weights."""
        crops, _ = energy_crops(y, self.crop, self.n_crops - 1)
        cand, conf = candidates(self.P, self.C, crops, list(range(8)))       # [n,V*K,79,79]
        cand, conf = cand.flatten(0, 1), conf.flatten(0, 1) / crops.shape[0]
        extra = []
        if self.ret is not None:
            _, _, h = self.P(autocorr_obs(crops), feat=True)
            ei = torch.nn.functional.normalize(self.ret.embed_img(h).mean(0), dim=0)
            top = (self.bank_emb @ ei).topk(self.n_retrieve).indices.tolist()
            from etr.kernel.bank import bank_kernel
            extra.append(torch.stack([bank_kernel(self.bank["sizes"][i], self.bank["seeds"][i]) for i in top]).to(dev))
        flat = cand.flatten(1)
        vote = (conf[None] * torch.exp(-torch.cdist(flat, flat, p=1) / 0.5)).sum(1)
        top8 = vote.topk(8).indices
        zt = self.C.encode(cand[top8][:, None])[0]
        pert = [decode_k(self.C, zt + s * torch.randn_like(zt))[:, 0] for s in (0.3, 0.6) for _ in range(3)]
        d = torch.zeros(1, CANVAS, CANVAS, device=dev); d[0, CANVAS // 2, CANVAS // 2] = 1
        ladder = [(1 - l) * cand[top8] + l * d for l in (0.25, 0.5, 0.75)]
        ks = torch.cat([cand[vote.topk(min(self.n_score, len(vote))).indices]] + extra + pert + ladder + [d])
        flat2 = ks.flatten(1)
        v2 = (conf[None] * torch.exp(-torch.cdist(flat2, flat, p=1) / 0.5)).sum(1)
        y0 = crops[:1]
        sc = []
        for c0 in range(0, len(ks), 16):
            kk = ks[c0:c0 + 16][:, None]; m = len(kk)
            yy = y0.expand(m, -1, -1, -1)
            sc.append(self.critic(yy, wiener(yy, kk, torch.full((m,), float(sig), device=dev)), kk))
        sc = torch.cat(sc)
        z = lambda v: (v - v.mean()) / v.std().clamp(min=1e-6)
        tot = z(v2) + z(sc)
        i = int(tot.argmax())
        topm = tot.topk(8).indices
        info = dict(critic=float(sc[i]), critic_delta=float(sc[-1]), margin=float(sc[i] - sc[-1]),
                    total_margin=float(tot[i] - tot[-1]), is_delta=bool(i == len(ks) - 1),
                    top=ks[topm].detach(), top_w=torch.softmax(sc[topm] / 0.5, 0).detach(), pool=ks.detach())
        return ks[i], info
