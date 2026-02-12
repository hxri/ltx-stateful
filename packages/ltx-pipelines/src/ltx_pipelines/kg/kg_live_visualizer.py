"""
Headless KG Visualization Recorder

Reads live KG state JSON and records visualization as MP4.
Works on shell-only servers (no display needed).
"""

import time
import json
import io
from pathlib import Path
import logging

import numpy as np
import cv2

import networkx as nx
import matplotlib

# Headless backend
matplotlib.use("Agg")

import matplotlib.pyplot as plt

from ltx_pipelines.kg.kg_state_manager import KGStateManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

KG_JSON_PATH = "kg_state_live.json"
OUTPUT_VIDEO = "kg_story_visualization.mp4"

FPS = 2                 # Frames per KG update
POLL_INTERVAL = 0.5     # Seconds (more frequent polling)

VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720

MAX_WAIT_TIME = 300     # 5 minutes max wait before timeout

# ─────────────────────────────────────────────────────────────────

kg_manager = KGStateManager(KG_JSON_PATH)


# ─────────────────────────────────────────────────────────────────
# GRAPH BUILDER
# ─────────────────────────────────────────────────────────────────


def build_graph(state):
    """Build networkx graph from KG state."""
    G = nx.DiGraph()

    if "nodes" in state:
        for n in state["nodes"]:
            G.add_node(n["id"], **n.get("attributes", {}))

    if "edges" in state:
        for e in state["edges"]:
            G.add_edge(e["source"], e["target"], relation=e.get("relation", ""))

    return G


# ─────────────────────────────────────────────────────────────────
# FRAME RENDERING
# ─────────────────────────────────────────────────────────────────


def render_graph_frame(state):
    """Render KG state as matplotlib figure → numpy image (BGR for OpenCV)."""
    
    fig = plt.figure(figsize=(VIDEO_WIDTH / 100, VIDEO_HEIGHT / 100), dpi=100)
    ax = fig.add_subplot(111)

    G = build_graph(state)

    if len(G.nodes()) == 0:
        # Empty graph — just show placeholder
        ax.text(0.5, 0.5, "Waiting for KG data...", ha="center", va="center", fontsize=16)
        ax.set_title("KG Story Visualization")
    else:
        # Draw graph
        pos = nx.spring_layout(G, seed=42, k=2, iterations=50)

        nx.draw_networkx_nodes(
            G, pos,
            node_color="lightblue",
            node_size=2500,
            ax=ax
        )

        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        nx.draw_networkx_edges(G, pos, edge_color="gray", ax=ax, arrows=True)

        edge_labels = {
            (u, v): d.get("relation", "")
            for u, v, d in G.edges(data=True)
        }

        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=7, ax=ax)

        story_step = state.get("story_step", "?")
        num_nodes = len(G.nodes())
        ax.set_title(f"KG Story Step: {story_step} | Nodes: {num_nodes}")

    ax.axis("off")

    # ── Convert figure → numpy image (RGB) ────────────────────────
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
    plt.close(fig)

    buf.seek(0)
    
    # Read PNG as numpy array (RGB)
    import imageio.v2 as imageio
    img_rgb = imageio.imread(buf)
    
    # Ensure correct size
    if img_rgb.shape != (VIDEO_HEIGHT, VIDEO_WIDTH, 3):
        img_rgb = cv2.resize(img_rgb, (VIDEO_WIDTH, VIDEO_HEIGHT))
    
    # Convert RGB to BGR for OpenCV
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    return img_bgr


def create_video_writer():
    """Create video writer with fallback codecs."""
    
    # Try multiple codecs in order of preference
    codecs = [
        ("mp4v", "MP4V"),
        ("h264", "H.264"),
        ("MJPG", "Motion JPEG"),
        ("DIVX", "DIVX"),
        ("X264", "X264"),
    ]
    
    for codec_str, codec_name in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec_str)
            writer = cv2.VideoWriter(
                OUTPUT_VIDEO,
                fourcc,
                FPS,
                (VIDEO_WIDTH, VIDEO_HEIGHT)
            )
            
            if writer.isOpened():
                logger.info(f"✅ Opened video writer with codec: {codec_name}")
                return writer
            else:
                logger.warning(f"⚠️  Codec {codec_name} failed to open")
        except Exception as e:
            logger.warning(f"⚠️  Codec {codec_name} error: {e}")
    
    # Last resort: no codec specified (system default)
    logger.info("Trying system default codec...")
    writer = cv2.VideoWriter(OUTPUT_VIDEO, -1, FPS, (VIDEO_WIDTH, VIDEO_HEIGHT))
    if writer.isOpened():
        logger.info("✅ Opened video writer with system default codec")
        return writer
    
    return None


# ─────────────────────────────────────────────────────────────────
# MAIN RECORD LOOP
# ─────────────────────────────────────────────────────────────────


def main():
    """Main recording loop: watch KG JSON and write MP4."""
    
    print("")
    print("🎥 Starting KG Live Visualizer Recording...")
    print(f"📄 Watching: {KG_JSON_PATH}")
    print(f"🎬 Output: {OUTPUT_VIDEO}")
    print(f"⏱️  Poll interval: {POLL_INTERVAL}s, FPS: {FPS}")
    print(f"⏱️  Max wait: {MAX_WAIT_TIME}s")
    print("")

    # Create video writer
    writer = create_video_writer()
    if writer is None:
        print(f"❌ Failed to create video writer with any codec")
        print("   Try: apt-get install -y ffmpeg libsm6 libxext6")
        return

    last_timestamp = None
    last_gen_order = None
    frame_count = 0
    update_count = 0
    start_time = time.time()
    no_update_count = 0

    try:
        while True:
            elapsed = time.time() - start_time
            
            # Timeout if no updates for too long
            if elapsed > MAX_WAIT_TIME:
                logger.info(f"⏱️  Timeout after {elapsed:.0f}s with no updates, stopping...")
                break

            try:
                state = kg_manager.read_state()
            except Exception as e:
                logger.warning(f"⚠️  Error reading KG state: {e}")
                time.sleep(POLL_INTERVAL)
                continue

            if state is None:
                no_update_count += 1
                if no_update_count % 10 == 0:
                    logger.debug(f"⏳ Waiting for KG state ({elapsed:.0f}s elapsed)...")
                time.sleep(POLL_INTERVAL)
                continue

            # Reset counter when we get data
            no_update_count = 0

            timestamp = state.get("timestamp")
            gen_order = state.get("generation_order", [])

            # Detect change: either timestamp changed OR generation order grew
            state_changed = (
                timestamp != last_timestamp or 
                len(gen_order) > len(last_gen_order or [])
            )

            if not state_changed:
                time.sleep(POLL_INTERVAL)
                continue

            last_timestamp = timestamp
            last_gen_order = gen_order
            update_count += 1

            logger.info(f"📌 KG Update #{update_count}: {len(gen_order)} scenes, timestamp={timestamp}")

            # Render frame
            try:
                frame = render_graph_frame(state)
                
                # Verify frame
                if frame is None or frame.size == 0:
                    logger.error("Rendered frame is empty")
                    continue
                
                logger.debug(f"   Frame shape: {frame.shape}")
                
            except Exception as e:
                logger.error(f"⚠️  Error rendering frame: {e}", exc_info=True)
                time.sleep(POLL_INTERVAL)
                continue

            # Write multiple copies for smoother playback
            for i in range(FPS):
                success = writer.write(frame)
                if not success:
                    logger.error(f"❌ Failed to write frame {frame_count}")
                else:
                    frame_count += 1

            logger.info(f"   ✅ {FPS} frames written (total: {frame_count})")

    except KeyboardInterrupt:
        logger.info("\n🛑 Recording stopped by user")

    except Exception as e:
        logger.error(f"❌ Fatal error: {e}", exc_info=True)

    finally:
        logger.info("Finalizing video...")
        writer.release()
        time.sleep(1)  # Give file system time to flush
        
        # Verify output file exists and has size
        output_path = Path(OUTPUT_VIDEO)
        if output_path.exists():
            file_size = output_path.stat().st_size / (1024 * 1024)  # MB
            logger.info(f"✅ Video saved → {OUTPUT_VIDEO} ({file_size:.2f} MB, {frame_count} frames)")
            
            if file_size < 0.1:
                logger.warning(f"⚠️  Video file is very small ({file_size:.2f} MB) - may be corrupted")
        else:
            logger.error(f"❌ Output file not created: {OUTPUT_VIDEO}")
        
        logger.info(f"📊 Total KG updates detected: {update_count}")
        logger.info(f"📊 Total frames written: {frame_count}")
        logger.info(f"⏱️  Total time: {time.time() - start_time:.1f}s")


# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    main()
