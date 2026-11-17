"""Load the gradient-step image prior for the solver."""
from __future__ import annotations

import torch

from etr import config as C
from etr.models.gs_denoiser import GSDenoiser
from etr.models.goal import GoalEncoder
from etr.utils import load_ckpt


def load_prior(path="checkpoints/prior.pt", device="cuda"):
    ck = load_ckpt(path)
    gs = GSDenoiser(**ck["arch"]).to(device)
    gs.load_state_dict(ck["prior"])
    goal = GoalEncoder(dim=C.MODEL.goal_dim).to(device)
    goal.load_state_dict(ck["goal"])
    gs.eval(); goal.eval()
    for p in list(gs.parameters()) + list(goal.parameters()):
        p.requires_grad_(False)
    with torch.no_grad():
        gv = goal("fidelity", 1, device)
    return gs.net, gv
