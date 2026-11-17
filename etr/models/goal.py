"""GoalEncoder: the Prior's conditioning vector.

A point on the simplex over (fidelity, perceptual, conservative), mapped to a
64-d vector that conditions PriorNet at its bottleneck (FiLM). The current
pipeline always uses the "fidelity" preset (PSNR-oriented restoration); the
other presets are kept so a perceptual variant can be trained later without an
architecture change. Classifier-free-guidance style null embedding: with
probability drop_prob in training the goal is replaced by a learned null vector.
"""
from __future__ import annotations

import torch
import torch.nn as nn

PRESETS = {
    "fidelity": (1.0, 0.0, 0.0),
    "perceptual": (0.0, 1.0, 0.0),
    "conservative": (0.0, 0.0, 1.0),
}


class GoalEncoder(nn.Module):
    def __init__(self, dim: int = 64, n_quality: int = 3, drop_prob: float = 0.1):
        super().__init__()
        self.dim = dim
        self.drop_prob = drop_prob
        self.net = nn.Sequential(nn.Linear(n_quality, 128), nn.SiLU(), nn.Linear(128, dim))
        self.null = nn.Parameter(torch.zeros(dim))

    def forward(self, quality, batch_size: int = None, device=None) -> torch.Tensor:
        q = self.as_vector(quality, batch_size, device)
        if q is None:
            return self.null.expand(batch_size, self.dim)
        g = self.net(q)
        if self.training and self.drop_prob > 0:
            drop = torch.rand(g.shape[0], 1, device=g.device) < self.drop_prob
            g = torch.where(drop, self.null.expand_as(g), g)
        return g

    def guided(self, quality, batch_size: int, device, scale: float = 1.0) -> torch.Tensor:
        """g_null + scale * (g_cond - g_null). scale=1 is ordinary
        conditioning; >1 pushes further along the requested axis."""
        q = self.as_vector(quality, batch_size, device)
        null = self.null.expand(batch_size, self.dim)
        if q is None:
            return null
        return null + scale * (self.net(q) - null)

    @staticmethod
    def as_vector(quality, batch_size=None, device=None):
        if quality is None:
            return None
        if isinstance(quality, str):
            quality = PRESETS[quality]
        if not torch.is_tensor(quality):
            quality = torch.tensor(quality, dtype=torch.float32, device=device)
        quality = quality.to(device).float()
        if quality.dim() == 1:
            quality = quality.unsqueeze(0)
        if batch_size is not None and quality.shape[0] == 1 and batch_size > 1:
            quality = quality.expand(batch_size, -1)
        return quality
