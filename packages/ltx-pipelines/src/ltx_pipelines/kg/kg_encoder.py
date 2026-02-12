import torch
import torch.nn as nn

class KGConditionEncoder(nn.Module):
    def __init__(self, kg_dim=512, model_dim=4096):
        super().__init__()
        self.proj = nn.Linear(kg_dim, model_dim)

    def forward(self, kg_vec):
        return self.proj(kg_vec)
