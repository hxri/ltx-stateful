import torch
import os

class GraphStore:
    def __init__(self, path="kg_state.pt"):
        self.path = path

    def save(self, graph):
        torch.save(graph.serialize(), self.path)

    def load(self, graph):
        if os.path.exists(self.path):
            graph.load(torch.load(self.path))
