"""
Distilled pipeline with dynamic knowledge graph state tracking.
Maintains a knowledge graph across sequential generations for scene continuity.
"""

import logging
import torch
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Optional

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.args import default_2_stage_distilled_arg_parser
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.helpers import get_device

logger = logging.getLogger(__name__)
device = get_device()


class SimpleSceneGraph:
    """Lightweight scene graph tracking generation history with last-frame continuity."""

    def __init__(self):
        self.scenes = {}
        self.generation_order = []
        self.last_frame_dir = Path("./kg_last_frames")
        self.last_frame_dir.mkdir(exist_ok=True, parents=True)

    def add_scene(self, scene_id: str, prompt: str, seed: int, output_path: str = None):
        """Add a scene to the graph."""
        self.scenes[scene_id] = {
            "prompt": prompt,
            "seed": seed,
            "timestamp": datetime.now().isoformat(),
            "output_path": output_path,
            "last_frame_path": None,
        }
        self.generation_order.append(scene_id)
        logger.info(f"📌 Scene added: {scene_id}")

    def capture_last_frame(self, scene_id: str, frame_array) -> Optional[str]:
        """
        Capture and save the last frame of a scene.
        
        Args:
            scene_id: Scene identifier
            frame_array: Can be:
                - Single frame: numpy array [H, W, C] in RGB format (uint8)
                - Batch of frames: [B, H, W, C] — will extract last frame
                - torch tensor (any shape above)
        
        Returns:
            Path to saved frame PNG
        """
        if scene_id not in self.scenes:
            logger.warning(f"Scene {scene_id} not in KG")
            return None

        try:
            # Convert to numpy if torch tensor
            if torch.is_tensor(frame_array):
                frame_array = frame_array.cpu().numpy()
            
            frame_array = np.asarray(frame_array, dtype=np.uint8)
            
            logger.info(f"   Frame shape: {frame_array.shape}, dtype: {frame_array.dtype}")
            
            # Handle batch of frames [B, H, W, C] — extract last frame
            if frame_array.ndim == 4:
                logger.info(f"   Detected batch of {frame_array.shape[0]} frames, extracting last...")
                frame_array = frame_array[-1]  # Take last frame from batch
            
            # Now should be [H, W, C]
            if frame_array.ndim != 3:
                logger.error(f"Unexpected frame shape after extraction: {frame_array.shape}")
                return None
            
            logger.info(f"   Single frame shape: {frame_array.shape}")
            
            # Convert RGB to BGR for OpenCV
            if frame_array.shape[2] == 3:
                frame_bgr = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)
            else:
                frame_bgr = frame_array

            # Save PNG
            frame_path = self.last_frame_dir / f"{scene_id}_last_frame.png"
            success = cv2.imwrite(str(frame_path), frame_bgr)
            
            if not success:
                logger.error(f"Failed to write frame to {frame_path}")
                return None

            # Verify file was written
            if not frame_path.exists():
                logger.error(f"Frame file does not exist after write: {frame_path}")
                return None

            # Update scene metadata
            self.scenes[scene_id]["last_frame_path"] = str(frame_path)
            logger.info(f"📸 Last frame saved: {frame_path.name} ({frame_array.shape})")

            return str(frame_path)

        except Exception as e:
            logger.error(f"Failed to capture last frame for {scene_id}: {e}", exc_info=True)
            return None

    def get_last_frame_for_next_scene(self) -> Optional[str]:
        """
        Get last frame from the most recently completed scene.
        Returns path if available, None otherwise.
        """
        if not self.generation_order:
            logger.debug("No scenes in generation order yet")
            return None

        # Get most recent scene
        last_scene_id = self.generation_order[-1]
        scene_info = self.scenes.get(last_scene_id, {})
        last_frame_path = scene_info.get("last_frame_path")

        if last_frame_path and Path(last_frame_path).exists():
            logger.info(f"🔗 Using last frame from {last_scene_id} as reference")
            return last_frame_path
        else:
            logger.debug(f"Last frame not available for {last_scene_id} (path: {last_frame_path})")
            return None

    def get_scene(self, scene_id: str):
        """Get scene info."""
        return self.scenes.get(scene_id)

    def serialize(self):
        """Export graph as dict."""
        return {
            "scenes": self.scenes,
            "generation_order": self.generation_order,
            "created_at": datetime.now().isoformat(),
        }


class DistilledPipelineKG(DistilledPipeline):
    """Distilled pipeline with scene graph tracking and last-frame continuity."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kg = SimpleSceneGraph()
        logger.info("✓ Scene graph initialized with last-frame tracking")

    def generate_and_track(
        self,
        scene_id: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list = None,
        output_path: str = None,
        use_previous_frame: bool = True,
    ):
        """
        Generate video, track in KG, and capture last frame for continuity.
        """
        images = images or []

        # ─────────────────────────────────────────────────────────
        # Add last frame from previous scene as reference if available
        # ─────────────────────────────────────────────────────────
        if use_previous_frame:
            prev_last_frame = self.kg.get_last_frame_for_next_scene()
            if prev_last_frame:
                # Add as reference image at frame 0 with moderate strength
                images.append((prev_last_frame, 0, 0.7))
                logger.info(f"   📷 Added previous scene's last frame as reference")
            else:
                logger.info(f"   ⚠️ No previous frame available for continuity")

        logger.info(f"🎬 Generating scene: {scene_id}")
        logger.info(f"   Prompt: {prompt[:80]}...")
        logger.info(f"   Reference images: {len(images)}")

        # ─────────────────────────────────────────────────────────
        # Call parent pipeline
        # ─────────────────────────────────────────────────────────
        tiling_config = TilingConfig.default()
        video, audio = super().__call__(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=images,
            tiling_config=tiling_config,
            enhance_prompt=False,
        )

        # ─────────────────────────────────────────────────────────
        # Collect frames and encode video
        # Note: video is an iterator of batches [B, H, W, C]
        # ─────────────────────────────────────────────────────────
        logger.info(f"   Collecting frames...")
        all_frames = list(video)
        
        logger.info(f"   Collected {len(all_frames)} batches")
        if all_frames:
            logger.info(f"   First batch type: {type(all_frames[0])}, shape: {getattr(all_frames[0], 'shape', 'N/A')}")

        # ─────────────────────────────────────────────────────────
        # IMPORTANT: Extract last frame BEFORE encoding
        # because encode_video will consume the iterator
        # ─────────────────────────────────────────────────────────
        last_frame_for_kg = None
        if all_frames:
            # Get the last batch
            last_batch = all_frames[-1]
            
            # Convert to numpy if needed
            if torch.is_tensor(last_batch):
                last_batch = last_batch.cpu().numpy()
            
            # Extract last frame from last batch [H, W, C]
            if last_batch.ndim == 4:
                # Batch of frames [B, H, W, C]
                last_frame_for_kg = last_batch[-1]  # Last frame of last batch
            elif last_batch.ndim == 3:
                # Already single frame [H, W, C]
                last_frame_for_kg = last_batch
            
            logger.info(f"   Extracted last frame shape: {getattr(last_frame_for_kg, 'shape', 'N/A')}")

        # ─────────────────────────────────────────────────────────
        # Track in KG BEFORE encoding
        # ─────────────────────────────────────────────────────────
        self.kg.add_scene(
            scene_id=scene_id,
            prompt=prompt,
            seed=seed,
            output_path=output_path,
        )

        # Capture last frame BEFORE encoding destroys the frames
        if last_frame_for_kg is not None:
            logger.info(f"   Capturing last frame...")
            frame_path = self.kg.capture_last_frame(scene_id, last_frame_for_kg)
            if frame_path:
                logger.info(f"   ✅ Last frame captured: {frame_path}")
            else:
                logger.warning(f"   ⚠️ Failed to capture last frame")
        else:
            logger.warning(f"   ⚠️ No frames to capture")

        # ─────────────────────────────────────────────────────────
        # NOW encode video (this consumes all_frames)
        # ─────────────────────────────────────────────────────────
        if output_path:
            video_chunks_number = get_video_chunks_number(num_frames, tiling_config)
            logger.info(f"💾 Encoding to {output_path}")
            encode_video(
                video=iter(all_frames),
                fps=frame_rate,
                audio=audio,
                audio_sample_rate=AUDIO_SAMPLE_RATE,
                output_path=output_path,
                video_chunks_number=video_chunks_number,
            )
            logger.info(f"   ✅ Video encoded successfully")

        logger.info(f"✅ Scene {scene_id} completed and tracked")
        return output_path


@torch.inference_mode()
def main() -> None:
    """CLI entry point for knowledge graph-aware distilled pipeline."""
    logging.getLogger().setLevel(logging.INFO)
    parser = default_2_stage_distilled_arg_parser()
    args = parser.parse_args()

    logger.info("🎬 Initializing Distilled Pipeline with Scene Graph")
    pipeline = DistilledPipelineKG(
        checkpoint_path=args.checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=args.lora,
        quantization=args.quantization,
    )

    logger.info(f"📝 Prompt: {args.prompt[:80]}...")

    # Generate and track
    output_path = args.output_path or "kg_output.mp4"
    pipeline.generate_and_track(
        scene_id="scene_0",
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        output_path=output_path,
        use_previous_frame=False,  # First scene has no previous
    )

    # Save knowledge graph
    kg_path = Path(output_path).stem + "_kg.json"
    import json
    with open(kg_path, "w") as f:
        json.dump(pipeline.kg.serialize(), f, indent=2)

    logger.info(f"✅ Video saved: {output_path}")
    logger.info(f"📊 Scene graph saved: {kg_path}")


if __name__ == "__main__":
    main()
