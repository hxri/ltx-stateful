"""
KG Condition Encoder — projects CLIP scene embeddings into the
transformer's conditioning dimension via a small learned MLP.
"""

import torch
import torch.nn as nn


class KGConditionEncoder(nn.Module):
    """Project CLIP/DINOv2 embeddings (768-d) into the model conditioning space.

    Parameters
    ----------
    input_dim : int
        Dimension of input embeddings (768 for CLIP-L / DINOv2-base).
    model_dim : int
        Dimension of the target conditioning space (e.g. 4096 for LTX-2).
    """

    def __init__(self, input_dim: int = 768, model_dim: int = 4096):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, model_dim),
            nn.SiLU(),
            nn.Linear(model_dim, model_dim),
        )

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        emb : Tensor [B, input_dim]

        Returns
        -------
        Tensor [B, model_dim]
        """
        return self.proj(emb)
