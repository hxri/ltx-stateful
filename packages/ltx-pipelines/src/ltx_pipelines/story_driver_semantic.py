"""Story driver with rich semantic conditioning and detailed prompts."""

import json
import torch
import time
from pathlib import Path
from typing import List, Dict
import logging

from ltx_pipelines.distilled_kg_semantic import DistilledPipelineKGSemantic
from ltx_pipelines.kg.kg_prompt_parser import PromptParser
from ltx_pipelines.kg.kg_prompt_generator import DetailedPromptGenerator
from ltx_pipelines.kg.kg_schema import StoryState

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("story_driver_semantic")

CHECKPOINT_PATH = "ltx-2-19b-distilled-fp8.safetensors"
SPATIAL_UPSAMPLER = "ltx-2-spatial-upscaler-x2-1.0.safetensors"
GEMMA_ROOT = "gemma-3-12b-it-qat-q4_0-unquantized"

OUTPUT_DIR = Path("story_outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

FINAL_KG_PATH = OUTPUT_DIR / "final_story_kg.json"

HEIGHT = 1280
WIDTH = 768
NUM_FRAMES = 96
FRAME_RATE = 24.0
BASE_SEED = 13370


def build_storyline() -> List[Dict]:
    """Define story scenes with DIVERSE, concrete motions.
    
    Each scene has a unique, physically distinct motion — no repetitive spinning.
    The motion_override ensures the prompt generator picks the right template.
    The raw prompt is for the parser; the motion_override drives generation.
    """
    return [
        # ─── Scene 0: Static front-facing pose ───────────────
        dict(
            scene_id="scene_0_idle_front",
            prompt=(
                "A professional model standing still facing the camera "
                "in an elegant brown formal outfit. Natural lighting. "
                "High fashion photography. Studio background."
            ),
            seed_offset=0,
            motion_override="static",
            ref_strength=0.0,  # First scene, no reference
        ),
        # ─── Scene 1: Squat down and rise ─────────────────────
        dict(
            scene_id="scene_1_squat",
            prompt=(
                "The model bends their knees and squats down low, "
                "then rises back up to standing. Deep squat motion."
            ),
            seed_offset=1,
            motion_override="squatting",
            ref_strength=0.25,
        ),
        # ─── Scene 2: Raise arms overhead ────────────────────
        dict(
            scene_id="scene_2_arms_up",
            prompt=(
                "The model raises both arms up high above their head, "
                "stretching upward, hands reaching toward the ceiling. "
                "Arms lifting motion."
            ),
            seed_offset=2,
            motion_override="raising_arms",
            ref_strength=0.25,
        ),
        # ─── Scene 3: Look up then look down ─────────────────
        dict(
            scene_id="scene_3_look_around",
            prompt=(
                "The model tilts their head upward looking at the ceiling, "
                "then looks downward toward the floor. Head tilting motion."
            ),
            seed_offset=3,
            motion_override="looking_around",
            ref_strength=0.25,
        ),
        # ─── Scene 4: Lift one leg (balance pose) ────────────
        dict(
            scene_id="scene_4_leg_raise",
            prompt=(
                "The model lifts one leg up, bending the knee, "
                "holding a balance pose on one foot. One leg raised."
            ),
            seed_offset=4,
            motion_override="leg_raise",
            ref_strength=0.25,
        ),
        # ─── Scene 5: Wave hand ──────────────────────────────
        dict(
            scene_id="scene_5_wave",
            prompt=(
                "The model raises one hand and waves side to side, "
                "a friendly waving gesture. Hand waving motion."
            ),
            seed_offset=5,
            motion_override="waving",
            ref_strength=0.25,
        ),
        # ─── Scene 6: Bend forward (bow) ─────────────────────
        dict(
            scene_id="scene_6_bend",
            prompt=(
                "The model bends forward at the waist, bowing down, "
                "then straightens back up. Forward bending motion."
            ),
            seed_offset=6,
            motion_override="bending",
            ref_strength=0.25,
        ),
        # ─── Scene 7: Final static pose ──────────────────────
        dict(
            scene_id="scene_7_final_pose",
            prompt=(
                "The model returns to a confident standing pose "
                "facing the camera. Still, composed, final pose."
            ),
            seed_offset=7,
            motion_override="static",
            ref_strength=0.25,
        ),
    ]


@torch.inference_mode()
def run_story_semantic():
    logger.info("🚀 Initializing DistilledPipelineKGSemantic with rich semantic schema")

    pipeline = DistilledPipelineKGSemantic(
        checkpoint_path=CHECKPOINT_PATH,
        spatial_upsampler_path=SPATIAL_UPSAMPLER,
        gemma_root=GEMMA_ROOT,
        loras=[],
        quantization=None,
        kg_hidden_dim=512,
    )

    storyline = build_storyline()
    story_state = StoryState(
        story_id="fashion_showcase",
        created_at=time.time(),
        updated_at=time.time(),
        title="Fashion Model Pose Showcase",
        description="Professional model demonstrating diverse poses and movements"
    )

    logger.info(f"📖 Running storyline with {len(storyline)} diverse motion scenes")

    prev_scene_node = None

    for step_idx, step in enumerate(storyline):
        scene_id = step["scene_id"]
        seed = BASE_SEED + step["seed_offset"]
        output_path = OUTPUT_DIR / f"{scene_id}.mp4"
        ref_strength = step.get("ref_strength", 0.25)

        # ───────────────────────────────────────────────────────
        # PARSE SCENE WITH RICH SEMANTICS
        # ───────────────────────────────────────────────────────
        logger.info(f"\n{'='*60}")
        logger.info(f"📊 Scene {step_idx + 1}/{len(storyline)}: {scene_id}")
        logger.info(f"{'='*60}")
        
        scene_node = PromptParser.parse_scene(
            prompt=step["prompt"],
            scene_id=scene_id,
            timestamp=time.time(),
            sequence_num=step_idx,
            prev_scene=prev_scene_node
        )
        
        # Apply motion override — this is critical for correct prompt generation
        motion_override = step.get("motion_override")
        if motion_override:
            from ltx_pipelines.kg.kg_schema import MotionType
            scene_node.pose.motion_type = MotionType(motion_override)
            
            # Set appropriate flags per motion type
            scene_node.pose.is_turning = False
            scene_node.pose.is_walking = False
            scene_node.pose.is_gesturing = False
            scene_node.pose.motion_direction = None
            
            if motion_override == "turning":
                scene_node.pose.is_turning = True
                scene_node.pose.motion_direction = "turning"
            elif motion_override == "walking":
                scene_node.pose.is_walking = True
                scene_node.pose.motion_direction = "forward"
            elif motion_override in ("waving", "gesture", "raising_arms"):
                scene_node.pose.is_gesturing = True
            
            logger.info(f"   ⚡ Motion override: {motion_override}")
        
        story_state.scenes[scene_id] = scene_node
        story_state.generation_order.append(scene_id)

        # ───────────────────────────────────────────────────────
        # GENERATE MOTION-FIRST DETAILED PROMPT
        # ───────────────────────────────────────────────────────
        detailed_prompt = DetailedPromptGenerator.generate_detailed_prompt(
            scene_node,
            include_context=True
        )
        
        logger.info(f"\n📝 FINAL PROMPT FOR GENERATION:")
        logger.info(f"   {detailed_prompt}")
        logger.info(f"   Reference frame strength: {ref_strength}")

        # ───────────────────────────────────────────────────────
        # GENERATE
        # ───────────────────────────────────────────────────────
        pipeline.generate_and_track_semantic(
            scene_id=scene_id,
            prompt=detailed_prompt,
            seed=seed,
            height=HEIGHT,
            width=WIDTH,
            num_frames=NUM_FRAMES,
            frame_rate=FRAME_RATE,
            scene_node=scene_node,
            kg_state={"nodes": [], "edges": []},
            images=[],
            output_path=str(output_path),
            use_previous_frame=(step_idx > 0),
            use_kg_conditioning=True,
            ref_strength=ref_strength,
        )

        # Update history
        story_state.updated_at = time.time()
        prev_scene_node = scene_node
        torch.cuda.empty_cache()

    # ─────────────────────────────────────────────────────────
    # SAVE RICH KG
    # ─────────────────────────────────────────────────────────
    story_dict = story_state.to_dict()
    
    for scene_id, scene_node in story_state.scenes.items():
        detailed = DetailedPromptGenerator.generate_detailed_prompt(scene_node)
        story_dict["scenes"][scene_id]["detailed_prompt"] = detailed
    
    with open(FINAL_KG_PATH, "w") as f:
        json.dump(story_dict, f, indent=2)

    logger.info(f"\n✅ Story generation complete!")
    logger.info(f"📊 Rich KG saved → {FINAL_KG_PATH}")
    logger.info(f"📈 Scenes: {len(story_state.scenes)}")
    
    # Print scene summary
    logger.info("\n📋 Scene Summary:")
    for sid, sn in story_state.scenes.items():
        logger.info(f"  {sid}: motion={sn.pose.motion_type.value}")


if __name__ == "__main__":
    run_story_semantic()