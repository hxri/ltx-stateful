"""
Enhanced KG Semantic Encoder that uses rich structured attributes.
Supports expanded motion vocabulary.
"""

import torch
import torch.nn as nn
import logging
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class RichSemanticEncoder(nn.Module):
    """Encodes rich semantic attributes into embeddings."""

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        emb_dim = hidden_dim // 8  # Each embedding is 64 dims

        # ─────────────────────────────────────────────────────
        # CHARACTER EMBEDDINGS (5 attributes)
        # ─────────────────────────────────────────────────────
        self.gender_embedding = nn.Embedding(4, emb_dim)
        self.ethnicity_embedding = nn.Embedding(7, emb_dim)
        self.age_group_embedding = nn.Embedding(7, emb_dim)
        self.height_embedding = nn.Embedding(4, emb_dim)
        self.body_type_embedding = nn.Embedding(5, emb_dim)

        # ─────────────────────────────────────────────────────
        # OUTFIT EMBEDDINGS (6 attributes)
        # ─────────────────────────────────────────────────────
        self.clothing_style_embedding = nn.Embedding(6, emb_dim)
        self.formality_embedding = nn.Embedding(4, emb_dim)
        self.top_type_embedding = nn.Embedding(8, emb_dim)
        self.bottom_type_embedding = nn.Embedding(8, emb_dim)
        self.shoe_type_embedding = nn.Embedding(6, emb_dim)
        self.accessory_embedding = nn.Embedding(16, emb_dim)

        # ─────────────────────────────────────────────────────
        # POSE/MOTION EMBEDDINGS (expanded vocab: 16 motion types)
        # ─────────────────────────────────────────────────────
        self.motion_type_embedding = nn.Embedding(16, emb_dim * 2)     # 2x weight, 16 types
        self.motion_speed_embedding = nn.Embedding(4, emb_dim * 2)     # 2x weight
        self.head_angle_embedding = nn.Embedding(8, emb_dim)           # expanded for look up/down
        self.gaze_embedding = nn.Embedding(5, emb_dim)
        self.torso_angle_embedding = nn.Embedding(5, emb_dim)
        self.motion_direction_embedding = nn.Embedding(8, emb_dim * 2) # 2x weight, expanded

        # ─────────────────────────────────────────────────────
        # ENVIRONMENT EMBEDDINGS (4 attributes)
        # ─────────────────────────────────────────────────────
        self.lighting_embedding = nn.Embedding(7, emb_dim)
        self.background_embedding = nn.Embedding(7, emb_dim)
        self.camera_distance_embedding = nn.Embedding(4, emb_dim)
        self.camera_angle_embedding = nn.Embedding(4, emb_dim)

        # ─────────────────────────────────────────────────────
        # FUSION NETWORKS
        # ─────────────────────────────────────────────────────
        self.character_fusion = nn.Sequential(
            nn.Linear(5 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.outfit_fusion = nn.Sequential(
            nn.Linear(6 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        # Pose: motion_type(2x) + speed(2x) + head + torso + direction(2x) = 8*emb_dim
        pose_input_dim = emb_dim * 2 + emb_dim * 2 + emb_dim + emb_dim + emb_dim * 2
        self.pose_fusion = nn.Sequential(
            nn.Linear(pose_input_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )

        self.environment_fusion = nn.Sequential(
            nn.Linear(4 * emb_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim)
        )

        self.scene_fusion = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 3),
            nn.ReLU(),
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        self.motion_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

    def string_to_id(self, s: Optional[str], mapping_size: int) -> int:
        """Convert string to consistent ID."""
        if s is None:
            return mapping_size - 1
        return hash(s) % (mapping_size - 1)

    def forward(self, scene_node) -> torch.Tensor:
        """Encode a scene node into semantic embedding."""
        
        device = next(self.parameters()).device
        
        # CHARACTER
        char = scene_node.character
        gender_id = self.string_to_id(char.gender.value if char.gender else None, 4)
        ethnicity_id = self.string_to_id(char.ethnicity.value if char.ethnicity else None, 7)
        age_id = self.string_to_id(char.age_group.value if char.age_group else None, 7)
        height_id = self.string_to_id(char.height, 4)
        body_type_id = self.string_to_id(char.body_type, 5)
        
        char_combined = torch.cat([
            self.gender_embedding(torch.tensor([gender_id], device=device, dtype=torch.long)).squeeze(0),
            self.ethnicity_embedding(torch.tensor([ethnicity_id], device=device, dtype=torch.long)).squeeze(0),
            self.age_group_embedding(torch.tensor([age_id], device=device, dtype=torch.long)).squeeze(0),
            self.height_embedding(torch.tensor([height_id], device=device, dtype=torch.long)).squeeze(0),
            self.body_type_embedding(torch.tensor([body_type_id], device=device, dtype=torch.long)).squeeze(0),
        ], dim=-1)
        char_embed = self.character_fusion(char_combined)
        
        # OUTFIT
        outfit = scene_node.outfit
        style_id = self.string_to_id(outfit.style, 6)
        formality_id = self.string_to_id(outfit.formality, 4)
        top_id = self.string_to_id(outfit.top_type, 8)
        bottom_id = self.string_to_id(outfit.bottom_type, 8)
        shoe_id = self.string_to_id(outfit.shoes, 6)
        
        accessory_ids = [self.string_to_id(acc, 16) for acc in outfit.accessories]
        accessory_ids = accessory_ids if accessory_ids else [15]
        accessory_emb = self.accessory_embedding(torch.tensor(accessory_ids[:5], device=device, dtype=torch.long)).mean(dim=0)
        
        outfit_combined = torch.cat([
            self.clothing_style_embedding(torch.tensor([style_id], device=device, dtype=torch.long)).squeeze(0),
            self.formality_embedding(torch.tensor([formality_id], device=device, dtype=torch.long)).squeeze(0),
            self.top_type_embedding(torch.tensor([top_id], device=device, dtype=torch.long)).squeeze(0),
            self.bottom_type_embedding(torch.tensor([bottom_id], device=device, dtype=torch.long)).squeeze(0),
            self.shoe_type_embedding(torch.tensor([shoe_id], device=device, dtype=torch.long)).squeeze(0),
            accessory_emb,
        ], dim=-1)
        outfit_embed = self.outfit_fusion(outfit_combined)
        
        # POSE/MOTION
        pose = scene_node.pose
        motion_id = self.string_to_id(pose.motion_type.value if pose.motion_type else None, 16)
        speed_id = self.string_to_id(pose.motion_speed, 4)
        head_id = self.string_to_id(pose.head_angle, 8)
        torso_id = self.string_to_id(pose.torso_angle, 5)
        motion_dir_id = self.string_to_id(pose.motion_direction, 8)
        
        pose_combined = torch.cat([
            self.motion_type_embedding(torch.tensor([motion_id], device=device, dtype=torch.long)).squeeze(0),
            self.motion_speed_embedding(torch.tensor([speed_id], device=device, dtype=torch.long)).squeeze(0),
            self.head_angle_embedding(torch.tensor([head_id], device=device, dtype=torch.long)).squeeze(0),
            self.torso_angle_embedding(torch.tensor([torso_id], device=device, dtype=torch.long)).squeeze(0),
            self.motion_direction_embedding(torch.tensor([motion_dir_id], device=device, dtype=torch.long)).squeeze(0),
        ], dim=-1)
        pose_embed = self.pose_fusion(pose_combined)
        
        # ENVIRONMENT
        env = scene_node.environment
        lighting_id = self.string_to_id(env.lighting_type.value if env.lighting_type else None, 7)
        bg_id = self.string_to_id(env.background_type.value if env.background_type else None, 7)
        camera_dist_id = self.string_to_id(env.camera_distance, 4)
        camera_angle_id = self.string_to_id(env.camera_angle, 4)
        
        env_combined = torch.cat([
            self.lighting_embedding(torch.tensor([lighting_id], device=device, dtype=torch.long)).squeeze(0),
            self.background_embedding(torch.tensor([bg_id], device=device, dtype=torch.long)).squeeze(0),
            self.camera_distance_embedding(torch.tensor([camera_dist_id], device=device, dtype=torch.long)).squeeze(0),
            self.camera_angle_embedding(torch.tensor([camera_angle_id], device=device, dtype=torch.long)).squeeze(0),
        ], dim=-1)
        env_embed = self.environment_fusion(env_combined)
        
        # FUSE ALL
        scene_combined = torch.cat([char_embed, outfit_embed, pose_embed, env_embed], dim=-1)
        scene_semantic = self.scene_fusion(scene_combined)
        
        motion_gate = self.motion_gate(pose_embed)
        scene_semantic = scene_semantic * (1.0 + motion_gate)
        
        if torch.isnan(scene_semantic).any():
            logger.warning("NaN in scene semantic, using small random values")
            scene_semantic = torch.randn_like(scene_semantic) * 0.01
        
        logger.info(f"✅ Encoded: motion={pose.motion_type.value}, gate_mean={motion_gate.mean():.3f}")
        
        return scene_semantic


class KGConsistencyScorer(nn.Module):
    """Scores frame consistency with KG expectations."""

    def __init__(self, hidden_dim=512):
        super().__init__()
        self.scorer_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

    def forward(self, kg_semantic: torch.Tensor, frame_feature: torch.Tensor) -> torch.Tensor:
        device = kg_semantic.device
        dtype = kg_semantic.dtype
        
        if kg_semantic.dim() != 1:
            kg_semantic = kg_semantic.squeeze()
        if frame_feature.dim() != 1:
            frame_feature = frame_feature.squeeze()
        
        frame_feature = frame_feature.to(device=device, dtype=dtype)
        combined = torch.cat([kg_semantic, frame_feature], dim=0)
        score = self.scorer_net(combined)
        
        if torch.isnan(score).any():
            return torch.tensor(0.5, device=device, dtype=dtype)
        
        return score.squeeze()


class KGConditioner:
    """Main interface for KG semantic conditioning."""

    def __init__(self, hidden_dim=512):
        self.encoder = RichSemanticEncoder(hidden_dim=hidden_dim)
        self.scorer = KGConsistencyScorer(hidden_dim=hidden_dim)
        self.hidden_dim = hidden_dim
        self.scene_history = []
        self.max_history = 10

    def encode_scene_node(self, scene_node) -> torch.Tensor:
        kg_semantic = self.encoder(scene_node)
        if kg_semantic.dim() > 1:
            kg_semantic = kg_semantic.squeeze()

        self.scene_history.append(kg_semantic.detach())
        if len(self.scene_history) > self.max_history:
            self.scene_history.pop(0)

        logger.info(f"🧠 Encoded scene node: shape={kg_semantic.shape}")
        return kg_semantic

    def encode_kg_state(self, kg_state: Dict) -> torch.Tensor:
        device = next(self.encoder.parameters()).device
        kg_semantic = torch.zeros(self.hidden_dim, device=device)
        return kg_semantic

    def score_frame_consistency(self, kg_semantic: torch.Tensor, frame_feature: torch.Tensor) -> float:
        with torch.no_grad():
            score = self.scorer(kg_semantic, frame_feature)
        score_val = float(score.cpu().numpy()) if not torch.isnan(score) else 0.5
        return score_val

    def reset(self):
        self.scene_history = []