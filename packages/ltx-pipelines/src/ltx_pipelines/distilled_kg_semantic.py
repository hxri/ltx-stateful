"""
Distilled pipeline with KG semantic conditioning.
Uses CLIP + DINOv2 for encoding and consistency scoring.
"""

import logging
import torch
import numpy as np
import json
from pathlib import Path
from typing import Optional

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled_kg import DistilledPipelineKG
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.kg.kg_semantic_encoder import KGConditioner, scene_node_to_text

logger = logging.getLogger(__name__)


class DistilledPipelineKGSemantic(DistilledPipelineKG):
    """
    Enhanced distilled pipeline that uses CLIP + DINOv2 for:
    1. Semantic encoding of scene nodes (CLIP text)
    2. Consistency scoring of outputs (CLIP text↔image, DINOv2 frame↔frame)
    3. Temporal coherence across scenes
    """

    def __init__(self, *args, kg_hidden_dim=512, **kwargs):
        super().__init__(*args, **kwargs)

        # Determine device
        device = torch.device("cpu")
        try:
            if torch.cuda.is_available():
                device = torch.device("cuda")
        except Exception:
            pass

        # KGConditioner uses CLIP + DINOv2 (pretrained, no random weights)
        self.kg_conditioner = KGConditioner(device=device)
        self._kg_device = device
        logger.info(f"✅ KG Semantic Conditioner initialized (CLIP + DINOv2, device={device})")

    def load_kg_state(self, kg_json_path: str) -> dict:
        """Load KG state from JSON file."""
        try:
            with open(kg_json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load KG state: {e}")
            return {}

    def _frame_to_pil(self, frame_array):
        """Convert a frame array to PIL Image for CLIP/DINOv2."""
        from PIL import Image

        if torch.is_tensor(frame_array):
            frame_array = frame_array.cpu().numpy()

        frame_array = np.asarray(frame_array)

        # Handle float [0,1] frames
        if frame_array.dtype in (np.float32, np.float64):
            frame_array = (frame_array * 255).clip(0, 255).astype(np.uint8)

        # Handle batch [B, H, W, C] — take last
        if frame_array.ndim == 4:
            frame_array = frame_array[-1]

        # Handle [C, H, W] → [H, W, C]
        if frame_array.ndim == 3 and frame_array.shape[0] in (1, 3):
            frame_array = frame_array.transpose(1, 2, 0)

        # Handle grayscale
        if frame_array.ndim == 2:
            frame_array = np.stack([frame_array] * 3, axis=-1)

        return Image.fromarray(frame_array)

    def generate_and_track_semantic(
        self,
        scene_id: str,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        scene_node=None,
        kg_state: dict = None,
        images: list = None,
        output_path: str = None,
        use_previous_frame: bool = True,
        use_kg_conditioning: bool = True,
        ref_strength: float = 0.3,
    ):
        """
        Generate video with KG semantic conditioning and CLIP/DINOv2 scoring.
        """
        images = images or []

        # ─────────────────────────────────────────────────────────
        # ENCODE KG SEMANTICS (CLIP text embedding)
        # ─────────────────────────────────────────────────────────
        kg_text = None
        if use_kg_conditioning and scene_node is not None:
            try:
                kg_text = scene_node_to_text(scene_node)
                logger.info(f"   🧠 KG text: {kg_text[:100]}...")
            except Exception as e:
                logger.warning(f"   ⚠️ Failed to convert scene node to text: {e}")

        # ─────────────────────────────────────────────────────────
        # Add last frame as reference WITH CONFIGURABLE STRENGTH
        # ─────────────────────────────────────────────────────────
        if use_previous_frame and ref_strength > 0:
            prev_last_frame = self.kg.get_last_frame_for_next_scene()
            if prev_last_frame:
                images.append((prev_last_frame, 0, ref_strength))
                logger.info(f"   📷 Reference frame added (strength={ref_strength})")
            else:
                logger.info(f"   ⚠️ No previous frame available")
        elif use_previous_frame and ref_strength <= 0:
            logger.info(f"   📷 Skipping reference frame (strength={ref_strength})")

        logger.info(f"🎬 Generating scene: {scene_id}")
        logger.info(f"   Prompt: {prompt[:120]}...")
        logger.info(f"   Reference images: {len(images)}, ref_strength: {ref_strength}")

        # ─────────────────────────────────────────────────────────
        # Call parent pipeline
        # ─────────────────────────────────────────────────────────
        tiling_config = TilingConfig.default()
        video, audio = super(DistilledPipelineKG, self).__call__(
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
        # Collect frames
        # ─────────────────────────────────────────────────────────
        logger.info(f"   Collecting frames...")
        all_frames = list(video)

        # ─────────────────────────────────────────────────────────
        # SCORE CONSISTENCY WITH CLIP + DINOv2
        # ─────────────────────────────────────────────────────────
        consistency_scores = []
        if kg_text and all_frames:
            logger.info(f"   📊 Scoring consistency with CLIP...")

            sample_indices = [0, len(all_frames) // 2, -1]

            for idx in sample_indices:
                try:
                    frame_batch = all_frames[idx]
                    pil_frame = self._frame_to_pil(frame_batch)

                    # CLIP text ↔ image score
                    score = self.kg_conditioner.scorer.score_text_image(kg_text, pil_frame)
                    consistency_scores.append(float(score))
                    logger.info(f"      Frame {idx}: CLIP consistency={score:.3f}")

                except Exception as e:
                    logger.warning(f"      Frame {idx}: scoring error: {e}")
                    continue

            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                logger.info(f"   📊 Average CLIP consistency: {avg_consistency:.3f}")

        # ─────────────────────────────────────────────────────────
        # DINOv2 temporal consistency (compare with previous scene)
        # ─────────────────────────────────────────────────────────
        temporal_score = None
        if all_frames and use_previous_frame:
            prev_last_frame_path = self.kg.get_last_frame_for_next_scene()
            if prev_last_frame_path:
                try:
                    from PIL import Image
                    prev_pil = Image.open(prev_last_frame_path).convert("RGB")
                    curr_pil = self._frame_to_pil(all_frames[0])
                    temporal_score = self.kg_conditioner.score_frames(prev_pil, curr_pil)
                    logger.info(f"   🔗 DINOv2 temporal consistency: {temporal_score:.3f}")
                except Exception as e:
                    logger.warning(f"   ⚠️ Temporal scoring failed: {e}")

        # ─────────────────────────────────────────────────────────
        # Extract last frame
        # ─────────────────────────────────────────────────────────
        last_frame_for_kg = None
        if all_frames:
            last_batch = all_frames[-1]
            if torch.is_tensor(last_batch):
                last_batch = last_batch.cpu().numpy()

            if last_batch.ndim == 4:
                last_frame_for_kg = last_batch[-1]
            elif last_batch.ndim == 3:
                last_frame_for_kg = last_batch

        # ─────────────────────────────────────────────────────────
        # Track in KG
        # ─────────────────────────────────────────────────────────
        self.kg.add_scene(scene_id, prompt, seed, output_path)

        if last_frame_for_kg is not None:
            self.kg.capture_last_frame(scene_id, last_frame_for_kg)

        # ─────────────────────────────────────────────────────────
        # Encode video
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

        # ─────────────────────────────────────────────────────────
        # Store scores in KG
        # ─────────────────────────────────────────────────────────
        if consistency_scores:
            avg = sum(consistency_scores) / len(consistency_scores)
            self.kg.scenes[scene_id]["clip_consistency_score"] = avg
        if temporal_score is not None:
            self.kg.scenes[scene_id]["dino_temporal_score"] = temporal_score

        logger.info(f"✅ Scene {scene_id} completed")
        return output_path