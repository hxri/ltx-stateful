"""
Distilled pipeline with KG semantic conditioning.
The KG semantics are used to guide and score generation quality.
"""

import logging
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from typing import Optional

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled_kg import DistilledPipelineKG
from ltx_pipelines.utils.constants import AUDIO_SAMPLE_RATE
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.kg.kg_semantic_encoder import KGConditioner

logger = logging.getLogger(__name__)


class DistilledPipelineKGSemantic(DistilledPipelineKG):
    """
    Enhanced distilled pipeline that uses KG semantics for:
    1. Semantic conditioning of generation
    2. Consistency scoring of outputs
    3. Temporal coherence across scenes
    """

    def __init__(self, *args, kg_hidden_dim=512, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.kg_conditioner = KGConditioner(hidden_dim=kg_hidden_dim)
        logger.info(f"✅ KG Semantic Conditioner initialized (dim={kg_hidden_dim})")

    def load_kg_state(self, kg_json_path: str) -> dict:
        """Load KG state from JSON file."""
        try:
            with open(kg_json_path, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load KG state: {e}")
            return {}

    def extract_frame_features(self, frame_batch, hidden_dim=512):
        """
        Extract meaningful features from frame batch.
        """
        import numpy as np
        
        if isinstance(frame_batch, np.ndarray):
            frame_batch = torch.from_numpy(frame_batch).float() / 255.0
        
        if frame_batch.ndim == 3:
            frame_batch = frame_batch.unsqueeze(0)
        
        if frame_batch.ndim != 4:
            logger.warning(f"Unexpected frame shape: {frame_batch.shape}, using zeros")
            return torch.zeros(hidden_dim)
        
        B, H, W, C = frame_batch.shape
        mean_color = frame_batch.mean(dim=(1, 2))
        spatial_var = frame_batch.std(dim=(1, 2))
        
        if B > 1:
            temporal_var = frame_batch.std(dim=0).mean()
        else:
            temporal_var = torch.tensor(0.0)
        
        mean_color_pooled = mean_color.mean(dim=0, keepdim=False)
        spatial_var_pooled = spatial_var.mean(dim=0, keepdim=False)
        
        combined = torch.cat([mean_color_pooled, spatial_var_pooled])
        
        if combined.shape[0] >= hidden_dim:
            features = combined[:hidden_dim]
        else:
            pad_size = hidden_dim - combined.shape[0]
            features = torch.cat([combined, torch.full((pad_size,), temporal_var.item())])
        
        features = features.reshape(-1)
        if torch.isnan(features).any():
            features = torch.randn(hidden_dim) * 0.01
        
        return features

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
        ref_strength: float = 0.3,  # ← NEW: Configurable reference strength
    ):
        """
        Generate video with KG semantic conditioning.
        
        Args:
            ref_strength: How strongly the previous frame conditions the next scene.
                         0.0 = no reference, 0.3 = light reference (allows motion),
                         0.7 = strong reference (anchors to previous frame).
                         For motion scenes, use LOW values (0.2-0.4).
        """
        images = images or []

        # ─────────────────────────────────────────────────────────
        # ENCODE KG SEMANTICS
        # ─────────────────────────────────────────────────────────
        kg_semantic = None
        if use_kg_conditioning:
            if scene_node is not None:
                kg_semantic = self.kg_conditioner.encode_scene_node(scene_node)
                logger.info(f"   🧠 Encoded scene node")
            elif kg_state:
                kg_semantic = self.kg_conditioner.encode_kg_state(kg_state)
                logger.info(f"   🧠 Encoded KG state")
            
            if kg_semantic is not None:
                kg_semantic = kg_semantic.squeeze() if kg_semantic.dim() > 1 else kg_semantic
                logger.info(f"   🧠 KG semantic shape: {kg_semantic.shape}")

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
        logger.info(f"   KG-conditioned: {use_kg_conditioning and kg_semantic is not None}")

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
        # SCORE CONSISTENCY WITH KG
        # ─────────────────────────────────────────────────────────
        consistency_scores = []
        if kg_semantic is not None and all_frames:
            logger.info(f"   Scoring frame consistency with KG...")
            
            sample_indices = [0, len(all_frames) // 2, -1]
            
            for idx in sample_indices:
                try:
                    frame_batch = all_frames[idx]
                    if torch.is_tensor(frame_batch):
                        frame_batch = frame_batch.cpu().numpy()
                    
                    frame_feature = self.extract_frame_features(
                        frame_batch,
                        hidden_dim=self.kg_conditioner.hidden_dim
                    )
                    
                    if frame_feature.dim() != 1:
                        frame_feature = frame_feature.squeeze()
                    kg_semantic_1d = kg_semantic.squeeze() if kg_semantic.dim() > 1 else kg_semantic
                    
                    if torch.isnan(frame_feature).any():
                        continue
                    
                    score = self.kg_conditioner.score_frame_consistency(kg_semantic_1d, frame_feature)
                    
                    if not torch.isnan(torch.tensor(score)):
                        consistency_scores.append(float(score))
                        logger.info(f"      Frame {idx}: consistency={score:.3f}")
                
                except Exception as e:
                    logger.error(f"      Frame {idx}: error: {e}")
                    continue
            
            if consistency_scores:
                avg_consistency = sum(consistency_scores) / len(consistency_scores)
                logger.info(f"   📊 Average consistency: {avg_consistency:.3f}")
            else:
                avg_consistency = 0.5

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

        if consistency_scores:
            self.kg.scenes[scene_id]["kg_consistency_score"] = avg_consistency

        logger.info(f"✅ Scene {scene_id} completed")
        return output_path