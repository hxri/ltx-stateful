"""
Story Driver for Distilled KG Pipeline

Runs multi-step storyline generations while:
- Maintaining pipeline-level KG
- Capturing last frames for continuity
- Exporting live KG snapshots for visualization
- Minimizing GPU memory churn
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict

import torch

from ltx_pipelines.distilled_kg import DistilledPipelineKG
from ltx_pipelines.utils.helpers import get_device

# ─────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────

CHECKPOINT_PATH = "ltx-2-19b-distilled-fp8.safetensors"
SPATIAL_UPSAMPLER = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT = "gemma-3-12b-it-qat-q4_0-unquantized"

OUTPUT_DIR = Path("story_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

KG_SNAPSHOT_PATH = Path("kg_state_live.json")

HEIGHT = 1280
WIDTH = 768
NUM_FRAMES = 96
FRAME_RATE = 24.0

BASE_SEED = 13370
STEP_DELAY_SEC = 2

# ─────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("story_driver")

device = get_device()

# ─────────────────────────────────────────────────────────────────
# STORYLINE DEFINITION
# ─────────────────────────────────────────────────────────────────


def build_storyline() -> List[Dict]:
    """
    Define multi-generation storyline with scene continuity.
    Each scene builds on the last frame of the previous scene.
    """

    return [
        dict(
            scene_id="scene_0_idle_front",
            prompt=(
                "A professional model standing still facing the camera in an elegant outfit. "
                "Natural lighting. High fashion photography. Studio background."
            ),
            seed_offset=0,
        ),
        dict(
            scene_id="scene_1_turn_back",
            prompt=(
                "The same model smoothly turns around to show the back of the outfit. "
                "Same pose and positioning. Continuous motion from previous scene."
            ),
            seed_offset=1,
        ),
        dict(
            scene_id="scene_2_turn_side",
            prompt=(
                "The model continues turning to show the side profile of the outfit. "
                "Smooth continuous rotation. Same position."
            ),
            seed_offset=2,
        ),
        dict(
            scene_id="scene_3_return_front",
            prompt=(
                "The model completes the rotation returning to face forward. "
                "Smooth natural motion. Final pose showing full outfit."
            ),
            seed_offset=3,
        ),
    ]


# ─────────────────────────────────────────────────────────────────
# KG SNAPSHOT EXPORT
# ─────────────────────────────────────────────────────────────────


def export_live_kg_snapshot(pipeline: DistilledPipelineKG):
    """
    Export pipeline KG for external live visualizer.
    Atomic write prevents partial reads.
    """

    tmp_path = KG_SNAPSHOT_PATH.with_suffix(".tmp")

    with open(tmp_path, "w") as f:
        json.dump(pipeline.kg.serialize(), f, indent=2)

    tmp_path.replace(KG_SNAPSHOT_PATH)


# ─────────────────────────────────────────────────────────────────
# MEMORY SAFETY HELPERS
# ─────────────────────────────────────────────────────────────────


def cleanup_memory():
    """
    Prevent long-run fragmentation.
    """

    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()


# ─────────────────────────────────────────────────────────────────
# MAIN STORY EXECUTION
# ─────────────────────────────────────────────────────────────────


@torch.inference_mode()
def run_story():
    logger.info("🚀 Initializing DistilledPipelineKG")

    pipeline = DistilledPipelineKG(
        checkpoint_path=CHECKPOINT_PATH,
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
        gemma_root=GEMMA_ROOT,
        loras=[],  # ← Changed from None to []
        quantization=None,
    )

    storyline = build_storyline()

    logger.info("📖 Running storyline with %d scenes", len(storyline))
    logger.info("   Last frame from each scene will condition the next")

    for step_idx, step in enumerate(storyline):

        scene_id = step["scene_id"]
        prompt = step["prompt"]
        seed = BASE_SEED + step["seed_offset"]

        logger.info("")
        logger.info("════════════════════════════════════════")
        logger.info("🎬 STORY STEP %d / %d", step_idx + 1, len(storyline))
        logger.info("Scene: %s", scene_id)
        logger.info("Seed: %d", seed)

        output_path = OUTPUT_DIR / f"{scene_id}.mp4"

        # ─────────────────────────────────────────────────────
        # GENERATE (with last-frame continuity)
        # ─────────────────────────────────────────────────────

        pipeline.generate_and_track(
            scene_id=scene_id,
            prompt=prompt,
            seed=seed,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
            images=[],  # ← Changed from None to []
            output_path=str(output_path),
            use_previous_frame=(step_idx > 0),  # Use prev frame for all except first
        )

        # ─────────────────────────────────────────────────────
        # EXPORT KG SNAPSHOT FOR LIVE VISUALIZER
        # ─────────────────────────────────────────────────────

        export_live_kg_snapshot(pipeline)

        # ─────────────────────────────────────────────────────
        # MEMORY MAINTENANCE
        # ─────────────────────────────────────────────────────

        cleanup_memory()

        logger.info("✅ Step %d complete", step_idx + 1)
        logger.info("   Video: %s", output_path)
        logger.info("   Last frame path: %s", pipeline.kg.scenes[scene_id].get("last_frame_path", "NOT SET"))

        if step_idx < len(storyline) - 1:
            logger.info("   Waiting %d seconds before next scene...", STEP_DELAY_SEC)
            time.sleep(STEP_DELAY_SEC)

    # ───────────────────────────────────────────────────────
    # FINAL KG EXPORT
    # ───────────────────────────────────────────────────────

    final_kg_path = OUTPUT_DIR / "final_story_kg.json"

    with open(final_kg_path, "w") as f:
        json.dump(pipeline.kg.serialize(), f, indent=2)

    logger.info("════════════════════════════════════════")
    logger.info("📊 Final KG saved → %s", final_kg_path)
    logger.info("🎉 Story generation complete!")
    logger.info("")
    logger.info("Summary:")
    logger.info("  Scenes: %d", len(pipeline.kg.generation_order))
    logger.info("  Output dir: %s", OUTPUT_DIR)
    logger.info("  All scenes use previous frame for continuity")
    
    # Print final KG state
    logger.info("")
    logger.info("Final KG State:")
    for scene_id, scene_info in pipeline.kg.scenes.items():
        logger.info(f"  {scene_id}: last_frame={scene_info.get('last_frame_path')}")


# ─────────────────────────────────────────────────────────────────
# ENTRY
# ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run_story()
