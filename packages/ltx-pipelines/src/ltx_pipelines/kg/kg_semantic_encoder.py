"""
KG Semantic Encoder using CLIP (text) and DINOv2 (vision).

- CLIP encodes scene node attributes as text → semantic vector
- DINOv2 encodes video frames → visual feature for KG updates
- Consistency scorer compares CLIP text vs DINOv2 frame embeddings
"""

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# Lazy globals — loaded once on first use
_clip_model = None
_clip_processor = None
_dino_model = None
_dino_processor = None


def _load_clip(device: torch.device = torch.device("cpu")):
    """Lazy-load CLIP model and processor."""
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor

        model_name = "openai/clip-vit-large-patch14"
        logger.info(f"Loading CLIP: {model_name}")
        _clip_processor = CLIPProcessor.from_pretrained(model_name)
        _clip_model = CLIPModel.from_pretrained(model_name).eval().to(device)
        for p in _clip_model.parameters():
            p.requires_grad_(False)
    return _clip_model, _clip_processor


def _load_dino(device: torch.device = torch.device("cpu")):
    """Lazy-load DINOv2 model and processor."""
    global _dino_model, _dino_processor
    if _dino_model is None:
        from transformers import AutoImageProcessor, AutoModel

        model_name = "facebook/dinov2-base"
        logger.info(f"Loading DINOv2: {model_name}")
        _dino_processor = AutoImageProcessor.from_pretrained(model_name)
        _dino_model = AutoModel.from_pretrained(model_name).eval().to(device)
        for p in _dino_model.parameters():
            p.requires_grad_(False)
    return _dino_model, _dino_processor


def scene_node_to_text(scene_node) -> str:
    """Convert a SceneNode's attributes into a natural-language description
    suitable for CLIP text encoding.

    Parameters
    ----------
    scene_node : kg_schema.SceneNode  (or any object with .to_dict())

    Returns
    -------
    str — A single descriptive sentence.
    """
    d = scene_node.to_dict() if hasattr(scene_node, "to_dict") else scene_node

    parts: list[str] = []

    # Character
    char = d.get("character", {})
    gender = char.get("gender", "person")
    if gender in ("unknown", None):
        gender = "person"
    age = char.get("age_group", "")
    if age and age != "unknown":
        age = age.replace("_", " ")
        parts.append(f"A {age} {gender}")
    else:
        parts.append(f"A {gender}")

    hair = char.get("hair_color")
    hair_style = char.get("hair_style")
    if hair and hair != "unknown":
        h = f"{hair_style} {hair}" if hair_style else hair
        parts.append(f"with {h} hair")

    body = char.get("body_type")
    if body and body != "unknown":
        parts.append(f"{body} build")

    # Outfit
    outfit = d.get("outfit", {})
    top = outfit.get("top_type")
    top_color = outfit.get("top_color")
    if top:
        t = f"{top_color} {top}" if top_color else top
        parts.append(f"wearing a {t}")

    bottom = outfit.get("bottom_type")
    bottom_color = outfit.get("bottom_color")
    if bottom:
        b = f"{bottom_color} {bottom}" if bottom_color else bottom
        parts.append(f"and {b}")

    shoes = outfit.get("shoes")
    if shoes:
        parts.append(f"with {shoes}")

    # Pose / motion
    pose = d.get("pose", {})
    motion = pose.get("motion_type", "static")
    if motion and motion not in ("unknown", "static"):
        parts.append(f"{motion.replace('_', ' ')}")

    # Environment
    env = d.get("environment", {})
    bg = env.get("background_type")
    if bg and bg != "unknown":
        parts.append(f"in a {bg} setting")

    lighting = env.get("lighting_type")
    if lighting and lighting != "unknown":
        parts.append(f"with {lighting} lighting")

    return " ".join(parts) + "."


class CLIPSceneEncoder(nn.Module):
    """Encode scene-node text descriptions into CLIP text embeddings."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            _load_clip(self.device)
            self._loaded = True

    @torch.no_grad()
    def forward(self, text: str) -> torch.Tensor:
        """Return L2-normalised CLIP text embedding [1, 768]."""
        self._ensure_loaded()
        model, processor = _clip_model, _clip_processor
        inputs = processor(text=[text], return_tensors="pt", truncation=True, max_length=77)
        inputs = {k: v.to(self.device) for k, v in inputs.items() if k != "pixel_values"}
        emb = model.get_text_features(**inputs)  # [1, 768]
        return F.normalize(emb, dim=-1)

    @torch.no_grad()
    def encode_image(self, image) -> torch.Tensor:
        """Return L2-normalised CLIP image embedding [1, 768].

        Parameters
        ----------
        image : PIL.Image or np.ndarray (H, W, 3) uint8 RGB
        """
        self._ensure_loaded()
        from PIL import Image as PILImage
        import numpy as np

        model, processor = _clip_model, _clip_processor
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        emb = model.get_image_features(**inputs)  # [1, 768]
        return F.normalize(emb, dim=-1)


class DINOFrameEncoder(nn.Module):
    """Encode video frames into DINOv2 CLS embeddings."""

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self._loaded = False

    def _ensure_loaded(self):
        if not self._loaded:
            _load_dino(self.device)
            self._loaded = True

    @torch.no_grad()
    def forward(self, image) -> torch.Tensor:
        """Return DINOv2 CLS token embedding [1, 768].

        Parameters
        ----------
        image : PIL.Image or np.ndarray (H, W, 3) uint8 RGB
        """
        self._ensure_loaded()
        from PIL import Image as PILImage
        import numpy as np

        model, processor = _dino_model, _dino_processor
        if isinstance(image, np.ndarray):
            image = PILImage.fromarray(image)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        cls_token = outputs.last_hidden_state[:, 0, :]  # [1, 768]
        return F.normalize(cls_token, dim=-1)


class KGConsistencyScorer(nn.Module):
    """Score consistency between KG scene description and an actual frame.

    Uses CLIP text ↔ image cosine similarity for cross-modal scoring
    and DINOv2 frame ↔ frame cosine similarity for temporal consistency.
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        super().__init__()
        self.device = device
        self.clip_encoder = CLIPSceneEncoder(device)
        self.dino_encoder = DINOFrameEncoder(device)

    @torch.no_grad()
    def score_text_image(self, text: str, image) -> float:
        """Cosine similarity between CLIP text and CLIP image embeddings.

        Returns float in [-1, 1]; higher = more consistent.
        """
        text_emb = self.clip_encoder(text)  # [1, 768]
        img_emb = self.clip_encoder.encode_image(image)  # [1, 768]
        return (text_emb @ img_emb.T).item()

    @torch.no_grad()
    def score_scene_image(self, scene_node, image) -> float:
        """Score how well an image matches a SceneNode description."""
        text = scene_node_to_text(scene_node)
        return self.score_text_image(text, image)

    @torch.no_grad()
    def score_frame_pair(self, frame_a, frame_b) -> float:
        """DINOv2 cosine similarity between two frames.

        High score → visually consistent (same character / environment).
        """
        emb_a = self.dino_encoder(frame_a)
        emb_b = self.dino_encoder(frame_b)
        return (emb_a @ emb_b.T).item()


class KGConditioner:
    """Main interface for KG semantic conditioning.

    Public API
    ----------
    encode_scene_node(scene_node) → [1, 768]  CLIP text embedding
    encode_frame(image)           → [1, 768]  DINOv2 CLS embedding
    score(scene_node, image)      → float      CLIP cross-modal similarity
    score_frames(frame_a, frame_b)→ float      DINOv2 temporal consistency
    """

    def __init__(self, device: torch.device = torch.device("cpu")):
        self.device = device
        self.clip_encoder = CLIPSceneEncoder(device)
        self.dino_encoder = DINOFrameEncoder(device)
        self.scorer = KGConsistencyScorer(device)

    def encode_scene_node(self, scene_node) -> torch.Tensor:
        """CLIP text embedding of the scene node description."""
        text = scene_node_to_text(scene_node)
        return self.clip_encoder(text)

    def encode_frame(self, image) -> torch.Tensor:
        """DINOv2 CLS embedding of a video frame."""
        return self.dino_encoder(image)

    def score(self, scene_node, image) -> float:
        """Cross-modal consistency: KG description vs actual frame."""
        return self.scorer.score_scene_image(scene_node, image)

    def score_frames(self, frame_a, frame_b) -> float:
        """Visual consistency between two frames."""
        return self.scorer.score_frame_pair(frame_a, frame_b)

    def unload(self):
        """Free GPU memory held by CLIP and DINOv2."""
        global _clip_model, _clip_processor, _dino_model, _dino_processor
        _clip_model = None
        _clip_processor = None
        _dino_model = None
        _dino_processor = None
        torch.cuda.empty_cache()
        logger.info("CLIP + DINOv2 unloaded from GPU")