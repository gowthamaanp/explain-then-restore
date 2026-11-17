"""Paths, seed and the image prior's architecture. Set ETR_DATA to the folder that holds the datasets."""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Tuple

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_ROOT = os.environ.get("ETR_DATA", os.path.join(ROOT, "data"))
CHECKPOINTS = os.path.join(ROOT, "checkpoints")

SEED = 1226


@dataclass
class ModelConfig:
    ksize: int = 31            # base kernel canvas of etr/ops/kernels.py's family ranges (they scale with size)
    goal_dim: int = 64         # GoalEncoder output (the Prior's conditioning vector)

    # Prior backbone (PriorNet inside GSDenoiser): 9.59M parameters
    prior_width: int = 48
    prior_enc_blocks: Tuple[int, ...] = (2, 2, 6)
    prior_mid_blocks: int = 6
    prior_dec_blocks: Tuple[int, ...] = (2, 2, 4)
    prior_n_cond: int = 2      # conditioning maps: noise std + noise correlation length

    def prior_kwargs(self):
        return dict(width=self.prior_width, enc_blocks=tuple(self.prior_enc_blocks),
                    mid_blocks=self.prior_mid_blocks, dec_blocks=tuple(self.prior_dec_blocks),
                    goal_dim=self.goal_dim, n_cond=self.prior_n_cond)


MODEL = ModelConfig()
