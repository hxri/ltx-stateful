"""
Latent Observer — extracts KG-update signals from video frames using DINOv2.

Instead of a custom linear projection over raw latents (which requires
matching the VAE latent layout), we use DINOv2 on decoded RGB frames.
This is more robust and produces semantically meaningful embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentObserver(nn.Module):
    """Observe decoded video frames and produce KG-update embeddings.

    Uses DINOv2 under the hood via the KG semantic encoder.

    Parameters
    ----------
    kg_dim : int
        Output embedding dimension for the knowledge graph (default 768
        to match DINOv2/CLIP).
    device : torch.device
        Device for inference.
    """

    def __init__(self, kg_dim: int = 768, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.kg_dim = kg_dim
        self.device = device
        self._encoder = None

    def _ensure_encoder(self):
        if self._encoder is None:
            from ltx_pipelines.kg.kg_semantic_encoder import DINOFrameEncoder
            self._encoder = DINOFrameEncoder(self.device)

    @torch.no_grad()
    def forward(self, frame) -> torch.Tensor:
        """Encode a single RGB frame into a KG embedding.

        Parameters
        ----------
        frame : PIL.Image or np.ndarray (H, W, 3) uint8 RGB

        Returns
        -------
        Tensor [1, kg_dim]
        """
        self._ensure_encoder()
        return self._encoder(frame)  # [1, 768]

    @torch.no_grad()
    def observe_batch(self, frames: list) -> torch.Tensor:
        """Encode multiple frames and return mean-pooled embedding.

        Parameters
        ----------
        frames : list of PIL.Image or np.ndarray

        Returns
        -------
        Tensor [1, kg_dim]
        """
        self._ensure_encoder()
        embeddings = [self._encoder(f) for f in frames]
        stacked = torch.cat(embeddings, dim=0)  # [N, 768]
        return F.normalize(stacked.mean(dim=0, keepdim=True), dim=-1)
