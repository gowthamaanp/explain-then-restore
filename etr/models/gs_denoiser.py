"""GSDenoiser: a provably-convergent-in-PnP denoiser with NO architectural
capacity tax, per Hurault, Leclaire & Papadakis, "Gradient Step Denoiser for
convergent Plug-and-Play" (ICLR 2022, arXiv:2110.03220).

Why not an architecturally <=1-Lipschitz network: an earlier attempt to force
every layer to be a contraction crushed capacity (RGB PSNR stuck near 13 dB).

The insight that makes this unnecessary: PnP-HQS/PGD convergence does NOT
need the denoiser itself to be a contraction. It needs the denoiser to be
the gradient of a scalar potential with a FINITE (not necessarily <=1)
Lipschitz constant on that gradient -- the algorithm's own step size
compensates for whatever that constant turns out to be, rather than the
network being forced small. Concretely:

    g(x)    = 0.5 * ||x - N(x)||^2        for an ORDINARY, unconstrained N
    D(x)    = x - grad(g)(x)
            = N(x) + J_N(x)^T (x - N(x))

`N` can be architecturally identical to `PriorNet` -- no spectral
normalization, no convex-combination merges, no wavelet shrinkage. The
paper's own measured cost of the guarantee is close to zero: GS-DRUNet
matches DRUNet to within 0.01-0.07dB on CBSD68 Gaussian denoising, and
GS-PnP beats DPIR on super-resolution (27.54 vs 27.30dB) while proving
convergence DPIR does not have.

`J_N^T @ (x - N(x))` is one vector-Jacobian product, computed via
`torch.autograd.grad` -- roughly the cost of one extra backward pass through
`N`, not an architectural constraint. This is why the paper compensates with
a "DRUNet-light" backbone for wall-clock parity; here `N` stays the full,
already-validated `PriorNet` width since the 16M budget has ample room
(GSDenoiser adds ZERO parameters over its backbone -- g's formula is a pure
function of N, no new learnable weights).
"""
from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint as _checkpoint

from etr.models.prior import PriorNet


class GSDenoiser(nn.Module):
    def __init__(self, **prior_kwargs):
        super().__init__()
        # identity_init=False: see PriorNet's docstring on this flag -- its
        # default zero-initialized output layer makes D's gradient EXACTLY
        # zero at init for every parameter (a real, general cancellation
        # from N(x)=x and J_N=I holding identically at that point, not an
        # approximation), a dead trap this specific wrapper cannot use.
        # clamp_output=False: see PriorNet's docstring on this flag -- its
        # default output clamp is a dead trap here too, one level deeper
        # than identity_init's, and triggered LATER in training instead of
        # at step 0 (confirmed in a real run: jac_pen decayed to exactly
        # 0.0 and PSNR crashed to 4.10dB starting around step 19400).
        prior_kwargs = dict(prior_kwargs)
        prior_kwargs["identity_init"] = False
        prior_kwargs["clamp_output"] = False
        self.net = PriorNet(**prior_kwargs)

    def forward(self, x: torch.Tensor, sigma: torch.Tensor,
                goal: torch.Tensor = None, use_checkpoint: bool = False,
                return_jac_penalty: bool = False, corr: torch.Tensor = None):
        """use_checkpoint gradient-checkpoints ONLY the inner `self.net(...)`
        call, never this whole method -- checkpointing the whole forward
        also works; narrowing it to `self.net` is the standard pattern for a
        function that itself performs a nested backward, and it reproduces
        the un-checkpointed gradient.
        """
        # Ambient grad mode BEFORE forcing it on below -- if the caller is
        # under `torch.no_grad()` (pure inference), we still need a local
        # grad-enabled forward to compute the VJP, but must not hand back a
        # tensor that pretends to carry gradient history the caller never
        # asked for. If the caller IS building a graph (training, or
        # inference under `torch.enable_grad()`), `create_graph=True` keeps
        # this whole computation differentiable so loss.backward() can flow
        # back through it.
        need_graph = torch.is_grad_enabled()
        with torch.enable_grad():
            x_ = x if x.requires_grad else x.clone().requires_grad_(True)
            if use_checkpoint and need_graph:
                Nx = _checkpoint(self.net, x_, sigma, goal, corr, use_reentrant=False)
            else:
                Nx = self.net(x_, sigma, goal, corr)
            residual = x_ - Nx
            (JTr,) = torch.autograd.grad(
                Nx, x_, grad_outputs=residual,
                create_graph=need_graph, retain_graph=True)
        out = Nx + JTr
        # NOT clamped during training, same reasoning as PriorNet's
        # clamp_output=False above: D(x) has no inherent [0,1] bound (it's
        # x-grad(g), not a squashed map), and a training-time clamp here
        # would be exactly the same class of dead trap this whole change is
        # fixing, just at the outermost level instead of inside `self.net`.
        # Clamped only for eval/inference, where a correct display/metric
        # range matters and gradient is a non-issue.
        if not self.training:
            out = out.clamp(0, 1)
        out = out if need_graph else out.detach()
        if return_jac_penalty:
            # ||J_N^T @ residual||^2, mean-reduced -- a direct, already-
            # computed handle on the exact quantity that was measured to run
            # away during real training (local-Lipschitz-of-gradient probe:
            # mean 0.84 -> mean 15.3, max 211.9, within 5000 steps, in
            # lockstep with the loss diverging and PSNR crashing BELOW the
            # identity baseline). Penalizing it directly discourages J_N's
            # operator norm from growing unboundedly, without an architectural
            # cap. (Not used by the current training: train_prior_world.py.)
            jac_penalty = (JTr ** 2).mean() if need_graph else JTr.detach().pow(2).mean()
            return out, jac_penalty
        return out


@torch.no_grad()
def local_lipschitz_probe(model: GSDenoiser, goal_dim: int, dev, n: int = 32,
                          hw=(64, 64)):
    """d(D(x))/d(x) sampled at random points/directions -- informational
    only. There is no bound this number
    is supposed to stay under: GSDenoiser's convergence guarantee comes from
    choosing the PnP step size relative to whatever this constant turns out
    to be (Theorem 1's `tau < 1/(lambda*L)`), not from constraining the
    network to keep it small. This exists so that constant is actually
    measured and reportable, not assumed.

    Requires grad internally (D's own VJP does), so this cannot be wrapped
    in `torch.no_grad()` end-to-end -- the decorator above only protects the
    finite-difference probe itself.
    """
    from etr.models.goal import GoalEncoder
    goal = GoalEncoder(dim=goal_dim).to(dev)
    model.eval(); goal.eval()
    ratios = []
    with torch.enable_grad():
        for i in range(n):
            g = torch.Generator(device=dev).manual_seed(9000 + i)
            x = torch.rand(1, 3, *hw, generator=g, device=dev)
            d = torch.randn(1, 3, *hw, generator=g, device=dev)
            d = d / d.flatten().norm().clamp(min=1e-12) * 0.01
            sigma = torch.tensor([0.1], device=dev)
            gv = goal("fidelity", 1, dev)
            y0 = model(x, sigma, gv)
            y1 = model((x + d).clamp(0, 1), sigma, gv)
            ratios.append(((y1 - y0).flatten().norm() / d.flatten().norm()).item())
    model.train(); goal.train()
    return sum(ratios) / len(ratios), max(ratios)
