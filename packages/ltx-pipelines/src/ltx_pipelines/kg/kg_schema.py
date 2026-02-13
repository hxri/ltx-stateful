"""
Enhanced Knowledge Graph Schema with rich semantic attributes.
Captures detailed scene and character properties for coherent multi-scene generation.
"""

from dataclasses import dataclass, asdict, field
from typing import Dict, List, Optional, Set
from enum import Enum
import json


# ─────────────────────────────────────────────────────────────────
# ENUMS FOR STRUCTURED ATTRIBUTES
# ─────────────────────────────────────────────────────────────────

class Gender(str, Enum):
    """Character gender classification."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"
    UNKNOWN = "unknown"


class Ethnicity(str, Enum):
    """Character ethnicity classification."""
    CAUCASIAN = "caucasian"
    AFRICAN = "african"
    ASIAN = "asian"
    MIDDLE_EASTERN = "middle_eastern"
    LATIN = "latin"
    MIXED = "mixed"
    UNKNOWN = "unknown"


class AgeGroup(str, Enum):
    """Character age group classification."""
    CHILD = "child"
    TEENAGER = "teenager"
    YOUNG_ADULT = "young_adult"
    ADULT = "adult"
    MIDDLE_AGED = "middle_aged"
    SENIOR = "senior"
    UNKNOWN = "unknown"


class Lighting(str, Enum):
    """Scene lighting conditions."""
    NATURAL = "natural"
    STUDIO = "studio"
    SOFT = "soft"
    HARSH = "harsh"
    BACKLIT = "backlit"
    SIDELIT = "sidelit"
    UNKNOWN = "unknown"


class Background(str, Enum):
    """Scene background type."""
    STUDIO = "studio"
    INDOOR = "indoor"
    OUTDOOR = "outdoor"
    ABSTRACT = "abstract"
    BLURRED = "blurred"
    PLAIN = "plain"
    UNKNOWN = "unknown"


class MotionType(str, Enum):
    """Type of motion in scene."""
    STATIC = "static"
    WALKING = "walking"
    RUNNING = "running"
    TURNING = "turning"
    DANCING = "dancing"
    GESTURE = "gesture"
    SQUATTING = "squatting"
    RAISING_ARMS = "raising_arms"
    LOOKING_AROUND = "looking_around"
    LEG_RAISE = "leg_raise"
    BENDING = "bending"
    WAVING = "waving"
    STEPPING = "stepping"
    COMPLEX = "complex"
    UNKNOWN = "unknown"


# ─────────────────────────────────────────────────────────────────
# CHARACTER SCHEMA
# ─────────────────────────────────────────────────────────────────

@dataclass
class CharacterAttributes:
    """Rich character attributes extracted from prompts."""
    
    # Identity
    character_id: str  # Unique ID for this character across scenes
    name: Optional[str] = None
    
    # Physical traits
    gender: Gender = Gender.UNKNOWN
    ethnicity: Ethnicity = Ethnicity.UNKNOWN
    age_group: AgeGroup = AgeGroup.UNKNOWN
    height: Optional[str] = None  # e.g., "tall", "average", "petite"
    body_type: Optional[str] = None  # e.g., "athletic", "slim", "curvy"
    
    # Appearance details
    skin_tone: Optional[str] = None  # e.g., "light", "medium", "dark"
    hair_color: Optional[str] = None  # e.g., "brown", "blonde", "black"
    hair_style: Optional[str] = None  # e.g., "long", "short", "curly", "straight"
    eye_color: Optional[str] = None  # e.g., "blue", "brown", "green"
    
    # Distinguishing features
    facial_features: Set[str] = field(default_factory=set)  # e.g., "beard", "glasses", "freckles"
    distinguishing_marks: Set[str] = field(default_factory=set)  # e.g., "tattoo", "scar", "mole"
    
    # Expression/emotion
    expression: Optional[str] = None  # e.g., "neutral", "smiling", "serious"
    emotion: Optional[str] = None  # e.g., "happy", "sad", "confident"
    
    def to_dict(self):
        return {
            "character_id": self.character_id,
            "name": self.name,
            "gender": self.gender.value,
            "ethnicity": self.ethnicity.value,
            "age_group": self.age_group.value,
            "height": self.height,
            "body_type": self.body_type,
            "skin_tone": self.skin_tone,
            "hair_color": self.hair_color,
            "hair_style": self.hair_style,
            "eye_color": self.eye_color,
            "facial_features": list(self.facial_features),
            "distinguishing_marks": list(self.distinguishing_marks),
            "expression": self.expression,
            "emotion": self.emotion,
        }


# ─────────────────────────────────────────────────────────────────
# OUTFIT/CLOTHING SCHEMA
# ─────────────────────────────────────────────────────────────────

@dataclass
class OutfitAttributes:
    """Rich clothing and outfit attributes."""
    
    outfit_id: str  # Unique ID for tracking outfit consistency
    
    # Overall style
    style: Optional[str] = None  # e.g., "casual", "formal", "sporty", "vintage", "modern"
    dress_code: Optional[str] = None  # e.g., "business", "evening", "beach", "street"
    formality: Optional[str] = None  # e.g., "formal", "semi-formal", "casual"
    
    # Top/upper body
    top_type: Optional[str] = None  # e.g., "shirt", "blouse", "sweater", "jacket"
    top_color: Optional[str] = None
    top_pattern: Optional[str] = None  # e.g., "solid", "striped", "floral"
    top_material: Optional[str] = None  # e.g., "cotton", "silk", "wool"
    
    # Bottom/lower body
    bottom_type: Optional[str] = None  # e.g., "pants", "skirt", "shorts", "dress"
    bottom_color: Optional[str] = None
    bottom_pattern: Optional[str] = None
    bottom_material: Optional[str] = None
    
    # Footwear
    shoes: Optional[str] = None  # e.g., "heels", "sneakers", "boots"
    shoe_color: Optional[str] = None
    
    # Accessories
    accessories: Set[str] = field(default_factory=set)  # e.g., "hat", "scarf", "jewelry", "bag"
    
    # Additional details
    layers: Optional[int] = None  # Number of clothing layers
    overall_color_scheme: Optional[str] = None  # e.g., "monochrome", "complementary", "analogous"
    
    def to_dict(self):
        return {
            "outfit_id": self.outfit_id,
            "style": self.style,
            "dress_code": self.dress_code,
            "formality": self.formality,
            "top_type": self.top_type,
            "top_color": self.top_color,
            "top_pattern": self.top_pattern,
            "top_material": self.top_material,
            "bottom_type": self.bottom_type,
            "bottom_color": self.bottom_color,
            "bottom_pattern": self.bottom_pattern,
            "bottom_material": self.bottom_material,
            "shoes": self.shoes,
            "shoe_color": self.shoe_color,
            "accessories": list(self.accessories),
            "layers": self.layers,
            "overall_color_scheme": self.overall_color_scheme,
        }


# ─────────────────────────────────────────────────────────────────
# POSE/MOTION SCHEMA
# ─────────────────────────────────────────────────────────────────

@dataclass
class PoseAttributes:
    """Pose and motion attributes."""
    
    pose_id: str  # Unique ID for tracking pose consistency
    
    # Overall motion
    motion_type: MotionType = MotionType.UNKNOWN
    motion_speed: Optional[str] = None  # e.g., "slow", "moderate", "fast"
    motion_direction: Optional[str] = None  # e.g., "forward", "backward", "left", "right", "turning"
    
    # Head position
    head_angle: Optional[str] = None  # e.g., "frontal", "profile_left", "profile_right", "3_4_view"
    head_tilt: Optional[str] = None  # e.g., "level", "tilted_left", "tilted_right"
    gaze_direction: Optional[str] = None  # e.g., "forward", "down", "up", "to_side"
    
    # Arm position
    arm_left_position: Optional[str] = None  # e.g., "down", "bent_90", "raised", "behind"
    arm_right_position: Optional[str] = None
    hand_gesture: Optional[str] = None  # e.g., "open", "closed", "pointing", "waving"
    
    # Leg position
    leg_left_position: Optional[str] = None  # e.g., "straight", "bent", "extended"
    leg_right_position: Optional[str] = None
    stance_width: Optional[str] = None  # e.g., "narrow", "shoulder_width", "wide"
    
    # Torso
    torso_angle: Optional[str] = None  # e.g., "upright", "leaning_forward", "leaning_back"
    
    # Additional motion context
    is_walking: bool = False
    is_turning: bool = False
    is_gesturing: bool = False
    weight_distribution: Optional[str] = None  # e.g., "balanced", "left_leg", "right_leg"
    
    def to_dict(self):
        return {
            "pose_id": self.pose_id,
            "motion_type": self.motion_type.value,
            "motion_speed": self.motion_speed,
            "motion_direction": self.motion_direction,
            "head_angle": self.head_angle,
            "head_tilt": self.head_tilt,
            "gaze_direction": self.gaze_direction,
            "arm_left_position": self.arm_left_position,
            "arm_right_position": self.arm_right_position,
            "hand_gesture": self.hand_gesture,
            "leg_left_position": self.leg_left_position,
            "leg_right_position": self.leg_right_position,
            "stance_width": self.stance_width,
            "torso_angle": self.torso_angle,
            "is_walking": self.is_walking,
            "is_turning": self.is_turning,
            "is_gesturing": self.is_gesturing,
            "weight_distribution": self.weight_distribution,
        }


# ─────────────────────────────────────────────────────────────────
# SCENE/ENVIRONMENT SCHEMA
# ─────────────────────────────────────────────────────────────────

@dataclass
class EnvironmentAttributes:
    """Scene environment and camera attributes."""
    
    # Lighting
    lighting_type: Lighting = Lighting.UNKNOWN
    light_direction: Optional[str] = None  # e.g., "frontal", "side", "back", "overhead"
    light_intensity: Optional[str] = None  # e.g., "bright", "moderate", "dim"
    light_color_temp: Optional[str] = None  # e.g., "warm", "cool", "neutral"
    shadows: Optional[str] = None  # e.g., "soft", "sharp", "none"
    
    # Background
    background_type: Background = Background.UNKNOWN
    background_color: Optional[str] = None
    background_blur: Optional[str] = None  # e.g., "sharp", "slight_blur", "heavy_blur"
    
    # Space/setting
    setting_type: Optional[str] = None  # e.g., "studio", "home", "office", "outdoor", "street"
    space_scale: Optional[str] = None  # e.g., "intimate", "medium", "expansive"
    visible_elements: Set[str] = field(default_factory=set)  # e.g., "furniture", "plants", "windows"
    
    # Camera/framing
    camera_angle: Optional[str] = None  # e.g., "eye_level", "high_angle", "low_angle"
    camera_distance: Optional[str] = None  # e.g., "close_up", "medium", "wide", "extreme_wide"
    camera_movement: Optional[str] = None  # e.g., "static", "pan", "zoom", "follow"
    frame_composition: Optional[str] = None  # e.g., "rule_of_thirds", "centered", "leading_lines"
    
    # Color scheme
    dominant_colors: Set[str] = field(default_factory=set)
    color_harmony: Optional[str] = None  # e.g., "monochromatic", "analogous", "complementary"
    saturation: Optional[str] = None  # e.g., "vibrant", "muted", "desaturated"
    
    # Atmosphere
    atmosphere: Optional[str] = None  # e.g., "professional", "casual", "dramatic", "romantic"
    mood: Optional[str] = None  # e.g., "energetic", "calm", "mysterious", "joyful"
    
    def to_dict(self):
        return {
            "lighting_type": self.lighting_type.value,
            "light_direction": self.light_direction,
            "light_intensity": self.light_intensity,
            "light_color_temp": self.light_color_temp,
            "shadows": self.shadows,
            "background_type": self.background_type.value,
            "background_color": self.background_color,
            "background_blur": self.background_blur,
            "setting_type": self.setting_type,
            "space_scale": self.space_scale,
            "visible_elements": list(self.visible_elements),
            "camera_angle": self.camera_angle,
            "camera_distance": self.camera_distance,
            "camera_movement": self.camera_movement,
            "frame_composition": self.frame_composition,
            "dominant_colors": list(self.dominant_colors),
            "color_harmony": self.color_harmony,
            "saturation": self.saturation,
            "atmosphere": self.atmosphere,
            "mood": self.mood,
        }


# ─────────────────────────────────────────────────────────────────
# SCENE NODE (combines all attributes)
# ─────────────────────────────────────────────────────────────────

@dataclass
class SceneNode:
    """Complete scene node with all semantic attributes."""
    
    scene_id: str
    timestamp: float
    
    # Character info
    character: CharacterAttributes
    
    # Outfit info
    outfit: OutfitAttributes
    
    # Pose/motion info
    pose: PoseAttributes
    
    # Environment info
    environment: EnvironmentAttributes
    
    # Scene-level metadata
    prompt: str
    scene_type: Optional[str] = None  # e.g., "establishing", "dialogue", "action", "transition"
    sequence_number: int = 0
    
    def to_dict(self):
        return {
            "scene_id": self.scene_id,
            "timestamp": self.timestamp,
            "character": self.character.to_dict(),
            "outfit": self.outfit.to_dict(),
            "pose": self.pose.to_dict(),
            "environment": self.environment.to_dict(),
            "prompt": self.prompt,
            "scene_type": self.scene_type,
            "sequence_number": self.sequence_number,
        }


# ─────────────────────────────────────────────────────────────────
# EDGE/RELATIONSHIP SCHEMA
# ─────────────────────────────────────────────────────────────────

@dataclass
class SceneEdge:
    """Relationship between two scenes."""
    
    source_scene: str
    target_scene: str
    
    # Relationship type
    relation_type: str  # e.g., "continues", "matches", "contrasts", "introduces", "resolves"
    
    # Continuity metadata
    character_consistent: bool = True
    outfit_consistent: bool = True
    pose_continuous: bool = True
    environment_consistent: bool = True
    
    # Transition details
    transition_type: Optional[str] = None  # e.g., "cut", "dissolve", "fade", "wipe"
    temporal_gap: Optional[str] = None  # e.g., "immediate", "minutes", "hours", "days"
    
    # Confidence scores
    confidence: float = 1.0
    
    def to_dict(self):
        return {
            "source": self.source_scene,
            "target": self.target_scene,
            "relation_type": self.relation_type,
            "character_consistent": self.character_consistent,
            "outfit_consistent": self.outfit_consistent,
            "pose_continuous": self.pose_continuous,
            "environment_consistent": self.environment_consistent,
            "transition_type": self.transition_type,
            "temporal_gap": self.temporal_gap,
            "confidence": self.confidence,
        }


# ─────────────────────────────────────────────────────────────────
# STORY STATE
# ─────────────────────────────────────────────────────────────────

@dataclass
class StoryState:
    """Complete story knowledge graph state."""
    
    story_id: str
    created_at: float
    updated_at: float
    
    # Nodes
    scenes: Dict[str, SceneNode] = field(default_factory=dict)
    
    # Edges
    transitions: Dict[str, SceneEdge] = field(default_factory=dict)  # Key: "source->target"
    
    # Story-level metadata
    title: Optional[str] = None
    description: Optional[str] = None
    generation_order: List[str] = field(default_factory=list)
    
    # Global consistency tracking
    global_character_consistency: float = 1.0
    global_continuity_score: float = 1.0
    
    def to_dict(self):
        return {
            "story_id": self.story_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "title": self.title,
            "description": self.description,
            "scenes": {k: v.to_dict() for k, v in self.scenes.items()},
            "transitions": {k: v.to_dict() for k, v in self.transitions.items()},
            "generation_order": self.generation_order,
            "global_character_consistency": self.global_character_consistency,
            "global_continuity_score": self.global_continuity_score,
        }
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)
