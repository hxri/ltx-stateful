import torch
import torch.nn as nn

class LatentObserver(nn.Module):
    def __init__(self, latent_dim=4096, kg_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, kg_dim),
            nn.SiLU(),
            nn.Linear(kg_dim, kg_dim)
        )

    def forward(self, latent):
        pooled = latent.mean(dim=list(range(2, latent.ndim)))
        return self.net(pooled)
