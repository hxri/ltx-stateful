import torch
import time

class SceneNode:
    def __init__(self, node_id, embedding):
        self.id = node_id
        self.embedding = embedding.cpu()
        self.last_seen = time.time()

class DynamicSceneGraph:
    def __init__(self, dim=512):
        self.dim = dim
        self.nodes = {}
        self.global_state = torch.zeros(dim)
        self.decay = 0.98

    def update_global(self, obs_emb):
        self.global_state = self.decay * self.global_state + (1 - self.decay) * obs_emb.cpu()

    def add_or_update_node(self, node_id, emb):
        if node_id not in self.nodes:
            self.nodes[node_id] = SceneNode(node_id, emb)
        else:
            self.nodes[node_id].embedding = 0.9 * self.nodes[node_id].embedding + 0.1 * emb.cpu()

    def readout(self):
        return self.global_state.clone()

    def serialize(self):
        return {
            "global": self.global_state,
            "nodes": {k: v.embedding for k,v in self.nodes.items()}
        }

    def load(self, state):
        self.global_state = state["global"]
        for k,v in state["nodes"].items():
            self.nodes[k] = SceneNode(k, v)
