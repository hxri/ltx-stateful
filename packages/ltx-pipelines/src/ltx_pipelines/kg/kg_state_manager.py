import json
import time
from pathlib import Path
from typing import List

from .kg_schema import KGState


class KGStateManager:
    def __init__(self, filepath="kg_state.json"):
        self.filepath = Path(filepath)

    def write_state(self, state: KGState):
        tmp = self.filepath.with_suffix(".tmp")

        with open(tmp, "w") as f:
            json.dump(state.to_dict(), f)

        tmp.replace(self.filepath)

    def exists(self):
        return self.filepath.exists()

    def read_state(self):
        if not self.exists():
            return None

        with open(self.filepath) as f:
            return json.load(f)
