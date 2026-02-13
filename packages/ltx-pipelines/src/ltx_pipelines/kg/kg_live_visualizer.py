import os
import time
import json
import logging
from typing import Dict, List

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import cv2


# =========================
# CONFIG
# =========================
KG_FILE = "kg_state.json"
OUTPUT_VIDEO = "kg_story_visualization.mp4"

POLL_INTERVAL = 0.5
FPS = 10
VIDEO_ENABLED = True

FIG_SIZE = (8, 8)


# =========================
# LOGGING
# =========================
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KG_VISUALIZER")


# =========================
# SAFE VIDEO WRITER
# =========================
class SafeVideoWriter:
    def __init__(self, path, fps):
        self.path = path
        self.fps = fps
        self.writer = None
        self.frames_written = 0

    def write(self, frame):
        h, w, _ = frame.shape

        if self.writer is None:
            logger.info("Initializing video writer...")
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.writer = cv2.VideoWriter(self.path, fourcc, self.fps, (w, h))

        self.writer.write(frame)
        self.frames_written += 1

    def close(self):
        if self.writer:
            self.writer.release()

        if os.path.exists(self.path):
            size_mb = os.path.getsize(self.path) / 1e6
            logger.info(f"Saved → {self.path} ({size_mb:.2f} MB, {self.frames_written} frames)")
        else:
            logger.warning("No video created")


# =========================
# LOAD KG STATE SAFELY
# =========================
def load_kg_state() -> Dict:
    if not os.path.exists(KG_FILE):
        return {}

    try:
        with open(KG_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return {}


# =========================
# BUILD GRAPH
# =========================
def build_graph(state: Dict) -> nx.DiGraph:
    G = nx.DiGraph()

    nodes = state.get("nodes", [])
    edges = state.get("edges", [])

    for n in nodes:
        if isinstance(n, dict):
            G.add_node(n["id"], **n)
        else:
            G.add_node(n)

    for e in edges:
        if isinstance(e, dict):
            G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))
        else:
            G.add_edge(e[0], e[1])

    return G


# =========================
# RENDER FRAME (STABLE)
# =========================
def render_frame(G: nx.DiGraph):
    fig = plt.figure(figsize=FIG_SIZE)
    ax = fig.add_subplot(111)
    ax.clear()

    if len(G.nodes) == 0:
        ax.text(0.5, 0.5, "Waiting for Knowledge Graph Update",
                ha="center", va="center", fontsize=16)
        ax.axis("off")
    else:
        pos = nx.spring_layout(G, seed=42)

        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=700)
        nx.draw_networkx_edges(G, pos, ax=ax, arrows=True)
        nx.draw_networkx_labels(G, pos, ax=ax)

        edge_labels = nx.get_edge_attributes(G, "relation")
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax)

        ax.set_title(f"Nodes: {len(G.nodes)} | Edges: {len(G.edges)}")

    fig.tight_layout()

    # 🔥 Stable canvas extraction
    fig.canvas.draw()

    width, height = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    frame = buf.reshape((height, width, 4))
    frame = frame[:, :, :3]  # Drop alpha

    plt.close(fig)

    return frame


# =========================
# LIVE LOOP
# =========================
def run():
    logger.info("Starting KG Live Visualizer")

    video_writer = SafeVideoWriter(OUTPUT_VIDEO, FPS) if VIDEO_ENABLED else None

    last_hash = None

    while True:
        state = load_kg_state()

        state_hash = hash(json.dumps(state, sort_keys=True)) if state else None

        if state_hash != last_hash:
            G = build_graph(state)
            frame = render_frame(G)

            cv2.imshow("KG Live", frame)

            if video_writer:
                video_writer.write(frame)

            last_hash = state_hash

        if cv2.waitKey(1) & 0xFF == 27:
            break

        time.sleep(POLL_INTERVAL)

    if video_writer:
        video_writer.close()

    cv2.destroyAllWindows()


# =========================
# MAIN
# =========================
if __name__ == "__main__":
    run()
