from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional


@dataclass
class KGNode:
    id: str
    type: str
    attributes: Dict
    last_frame_path: Optional[str] = None  # ← NEW: Path to last frame PNG


@dataclass
class KGEdge:
    source: str
    target: str
    relation: str
    weight: float = 1.0


@dataclass
class KGState:
    timestamp: float
    story_step: int
    nodes: List[KGNode]
    edges: List[KGEdge]

    def to_dict(self):
        return {
            "timestamp": self.timestamp,
            "story_step": self.story_step,
            "nodes": [asdict(n) for n in self.nodes],
            "edges": [asdict(e) for e in self.edges],
        }
