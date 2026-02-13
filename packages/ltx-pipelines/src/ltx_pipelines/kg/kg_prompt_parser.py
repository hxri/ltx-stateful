"""
Parse scene prompts to extract rich semantic attributes for KG nodes.
Uses pattern matching and keyword detection to populate scene graph.
Enhanced to extract more attributes and handle continuity better.
"""

import re
from typing import Optional, Set, Tuple
import logging

from .kg_schema import (
    CharacterAttributes, OutfitAttributes, PoseAttributes, EnvironmentAttributes,
    Gender, Ethnicity, AgeGroup, Lighting, Background, MotionType, SceneNode
)

logger = logging.getLogger(__name__)


class PromptParser:
    """Extract semantic attributes from natural language prompts."""
    
    # Gender keywords
    GENDER_MALE = {"man", "male", "boy", "gentleman", "guy", "his", "he", "him", "model"}
    GENDER_FEMALE = {"woman", "female", "girl", "lady", "woman", "her", "she"}
    
    # Ethnicity keywords
    ETHNICITY_KEYWORDS = {
        "caucasian": {"white", "caucasian", "european", "fair-skinned"},
        "african": {"black", "african", "african-american", "dark-skinned"},
        "asian": {"asian", "chinese", "japanese", "korean", "indian", "thai", "vietnamese", "east asian"},
        "middle_eastern": {"middle eastern", "arabic", "persian", "turkish", "middle eastern"},
        "latin": {"latin", "hispanic", "mexican", "brazilian", "latino"},
    }
    
    # Age keywords
    AGE_KEYWORDS = {
        "child": {"child", "kid", "young", "toddler", "infant"},
        "teenager": {"teenager", "teen", "adolescent", "youth"},
        "young_adult": {"young", "20s", "30s", "early", "youth"},
        "adult": {"adult", "man", "woman", "professional", "model"},
        "middle_aged": {"middle-aged", "40s", "50s", "mature"},
        "senior": {"elderly", "senior", "old", "aged"},
    }
    
    # Height keywords
    HEIGHT_KEYWORDS = {
        "tall": {"tall", "towering", "statuesque", "lengthy", "high", "long-legged"},
        "average": {"average", "medium", "normal", "standard", "regular"},
        "petite": {"petite", "short", "small", "diminutive", "compact"},
    }
    
    # Body type keywords
    BODY_TYPE_KEYWORDS = {
        "athletic": {"athletic", "muscular", "toned", "fit", "buff", "lean", "defined"},
        "slim": {"slim", "thin", "slender", "lean", "skinny", "lithe"},
        "curvy": {"curvy", "voluptuous", "curvaceous", "full", "rounded"},
        "stocky": {"stocky", "sturdy", "broad", "compact", "robust"},
    }
    
    # Hair keywords
    HAIR_COLOR_KEYWORDS = {
        "blonde": {"blonde", "blond", "fair", "light", "golden", "sandy"},
        "brown": {"brown", "brunette", "chestnut", "chocolate", "caramel"},
        "black": {"black", "dark", "ebony", "jet-black", "raven"},
        "red": {"red", "auburn", "copper", "ginger", "russet"},
        "grey": {"grey", "gray", "silver", "white", "salt-and-pepper"},
    }
    
    HAIR_STYLE_KEYWORDS = {
        "long": {"long", "flowing", "cascading", "waist-length", "shoulder-length"},
        "short": {"short", "cropped", "pixie", "bob", "close-cropped"},
        "curly": {"curly", "wavy", "coily", "curled", "waves", "ringlets"},
        "straight": {"straight", "sleek", "smooth", "silky"},
        "braided": {"braid", "braided", "cornrow", "plaits"},
    }
    
    # Clothing style keywords
    CLOTHING_STYLE_KEYWORDS = {
        "casual": {"casual", "relaxed", "informal", "everyday", "laid-back"},
        "formal": {"formal", "business", "professional", "dress", "suit", "tuxedo", "elegant"},
        "sporty": {"sporty", "athletic", "workout", "gym", "active"},
        "vintage": {"vintage", "retro", "classic", "timeless", "old-school"},
        "modern": {"modern", "contemporary", "sleek", "cutting-edge", "minimalist"},
        "elegant": {"elegant", "sophisticated", "refined", "chic", "upscale"},
    }
    
    # Outfit color keywords
    COLOR_KEYWORDS = {
        "black": {"black", "ebony", "noir"},
        "white": {"white", "cream", "ivory", "off-white", "pearl"},
        "red": {"red", "crimson", "scarlet", "burgundy", "wine"},
        "blue": {"blue", "navy", "cobalt", "azure", "cyan", "sapphire"},
        "green": {"green", "emerald", "olive", "sage", "forest"},
        "yellow": {"yellow", "gold", "golden", "mustard"},
        "pink": {"pink", "rose", "blush", "magenta", "coral"},
        "purple": {"purple", "violet", "lavender", "plum"},
        "brown": {"brown", "tan", "beige", "taupe", "bronze"},
        "grey": {"grey", "gray", "silver", "charcoal"},
    }
    
    # Motion keywords
    MOTION_KEYWORDS = {
        "static": {"standing", "sitting", "idle", "still", "stationary", "pose", "holding"},
        "walking": {"walk", "walking", "stroll", "striding", "stride", "steps forward"},
        "running": {"run", "running", "sprint", "sprinting", "jog"},
        "turning": {"turn", "turning", "rotate", "rotation", "spins", "pivots"},
        "dancing": {"dance", "dancing", "ballet", "waltz", "moves"},
        "gesture": {"gesture", "gesturing", "wave", "point", "gestures"},
        "squatting": {"squat", "squatting", "squats", "crouches", "crouch", "crouching", "kneels", "kneel", "kneeling", "lowers body"},
        "raising_arms": {"raises arms", "raising arms", "arms up", "hands up", "lifts arms", "lifting arms", "reaches up", "reaching up", "arms overhead", "stretches up"},
        "looking_around": {"looks up", "looking up", "looks down", "looking down", "tilts head", "head tilt", "glances", "gazes up", "gazes down", "chin up", "chin down"},
        "leg_raise": {"leg up", "raises leg", "lifting leg", "one leg", "leg raise", "knee up", "lifts knee", "kicks"},
        "bending": {"bends", "bending", "leans", "leaning", "bows", "bowing", "hunches"},
        "waving": {"waves", "waving", "hand wave", "waves hand"},
        "stepping": {"steps", "stepping", "sidestep", "side step", "shifts weight"},
    }
    
    # Pose/head angle keywords
    HEAD_ANGLE_KEYWORDS = {
        "frontal": {"facing camera", "front", "straight", "forward", "directly", "face forward"},
        "profile_left": {"left profile", "profile left", "turned left", "looking left"},
        "profile_right": {"right profile", "profile right", "turned right", "looking right"},
        "3_4_view": {"3/4", "three quarter", "angled", "quarter view"},
    }
    
    # Lighting keywords
    LIGHTING_KEYWORDS = {
        "natural": {"natural", "sunlight", "daylight", "natural light", "sun"},
        "studio": {"studio", "studio lighting", "professional", "controlled"},
        "soft": {"soft", "diffused", "gentle", "flattering", "warm"},
        "harsh": {"harsh", "bright", "high contrast", "direct"},
        "backlit": {"backlit", "back light", "rim light", "rimlit"},
        "sidelit": {"sidelit", "side light", "dramatic", "side-lit"},
    }
    
    # Background keywords
    BACKGROUND_KEYWORDS = {
        "studio": {"studio", "backdrop", "neutral background", "plain background"},
        "indoor": {"indoor", "inside", "room", "interior", "hall"},
        "outdoor": {"outdoor", "outside", "nature", "park", "street", "garden"},
        "abstract": {"abstract", "geometric", "minimal", "minimalist"},
        "blurred": {"blurred", "bokeh", "out of focus", "soft focus"},
        "plain": {"plain", "simple", "blank", "solid", "monochrome"},
    }
    
    # Expression keywords
    EXPRESSION_KEYWORDS = {
        "neutral": {"neutral", "expressionless", "blank", "calm"},
        "smiling": {"smile", "smiling", "grinning", "grin", "happy"},
        "serious": {"serious", "stern", "grim", "solemn", "focused"},
        "happy": {"happy", "joyful", "cheerful", "bright", "radiant"},
        "sad": {"sad", "sorrowful", "melancholy", "pensive"},
        "confident": {"confident", "self-assured", "assured", "proud"},
    }
    
    @staticmethod
    def _match_keywords(text: str, keyword_dict: dict) -> Optional[str]:
        """Match text against keyword dictionary, return first match."""
        text_lower = text.lower()
        for key, keywords in keyword_dict.items():
            if any(kw in text_lower for kw in keywords):
                return key
        return None
    
    @staticmethod
    def _extract_set(text: str, keyword_dict: dict) -> Set[str]:
        """Extract all matching keywords from text."""
        text_lower = text.lower()
        matches = set()
        for key, keywords in keyword_dict.items():
            if any(kw in text_lower for kw in keywords):
                matches.add(key)
        return matches
    
    @staticmethod
    def _extract_colors_from_text(text: str) -> Set[str]:
        """Extract all colors mentioned in text."""
        text_lower = text.lower()
        colors = set()
        for color, keywords in PromptParser.COLOR_KEYWORDS.items():
            if any(kw in text_lower for kw in keywords):
                colors.add(color)
        return colors
    
    @staticmethod
    def parse_character(prompt: str, char_id: str = "char_0", prev_char: Optional[CharacterAttributes] = None) -> CharacterAttributes:
        """Extract character attributes from prompt, maintaining consistency with previous scenes."""
        
        character = CharacterAttributes(character_id=char_id)
        
        # Inherit from previous scene if available
        if prev_char:
            character.gender = prev_char.gender
            character.ethnicity = prev_char.ethnicity
            character.age_group = prev_char.age_group
            character.height = prev_char.height
            character.body_type = prev_char.body_type
            character.hair_color = prev_char.hair_color
            character.hair_style = prev_char.hair_style
            character.eye_color = prev_char.eye_color
        
        # Override with new values if found
        if any(kw in prompt.lower() for kw in PromptParser.GENDER_MALE):
            character.gender = Gender.MALE
        elif any(kw in prompt.lower() for kw in PromptParser.GENDER_FEMALE):
            character.gender = Gender.FEMALE
        
        # Ethnicity - only override if explicitly mentioned
        eth = PromptParser._match_keywords(prompt, PromptParser.ETHNICITY_KEYWORDS)
        if eth:
            character.ethnicity = Ethnicity(eth)
        
        # Age - only override if explicitly mentioned
        age = PromptParser._match_keywords(prompt, PromptParser.AGE_KEYWORDS)
        if age:
            character.age_group = AgeGroup(age)
        
        # Height
        character.height = PromptParser._match_keywords(prompt, PromptParser.HEIGHT_KEYWORDS) or character.height
        
        # Body type
        character.body_type = PromptParser._match_keywords(prompt, PromptParser.BODY_TYPE_KEYWORDS) or character.body_type
        
        # Hair
        character.hair_color = PromptParser._match_keywords(prompt, PromptParser.HAIR_COLOR_KEYWORDS) or character.hair_color
        character.hair_style = PromptParser._match_keywords(prompt, PromptParser.HAIR_STYLE_KEYWORDS) or character.hair_style
        
        # Expression
        character.expression = PromptParser._match_keywords(prompt, PromptParser.EXPRESSION_KEYWORDS) or character.expression
        
        logger.info(f"   👤 Character: {character.gender.value}, {character.age_group.value}, {character.ethnicity.value}")
        return character
    
    @staticmethod
    def parse_outfit(prompt: str, outfit_id: str = "outfit_0", prev_outfit: Optional[OutfitAttributes] = None) -> OutfitAttributes:
        """Extract outfit attributes from prompt, maintaining consistency with previous scenes."""
        
        outfit = OutfitAttributes(outfit_id=outfit_id)
        
        # Inherit from previous scene if available (for continuity)
        if prev_outfit:
            outfit.style = prev_outfit.style
            outfit.formality = prev_outfit.formality
            outfit.top_type = prev_outfit.top_type
            outfit.top_color = prev_outfit.top_color
            outfit.bottom_type = prev_outfit.bottom_type
            outfit.bottom_color = prev_outfit.bottom_color
            outfit.shoes = prev_outfit.shoes
            outfit.shoe_color = prev_outfit.shoe_color
            outfit.accessories = prev_outfit.accessories.copy() if prev_outfit.accessories else set()
        
        # Override with new values if explicitly mentioned
        style = PromptParser._match_keywords(prompt, PromptParser.CLOTHING_STYLE_KEYWORDS)
        if style:
            outfit.style = style
        
        # Extract colors
        colors = PromptParser._extract_colors_from_text(prompt)
        if colors and not prev_outfit:
            # Only set if not inherited
            color_list = list(colors)
            outfit.top_color = color_list[0] if color_list else None
            outfit.bottom_color = color_list[-1] if len(color_list) > 1 else outfit.top_color
        
        # Clothing items
        if "dress" in prompt.lower():
            outfit.bottom_type = "dress"
        elif "pants" in prompt.lower() or "trousers" in prompt.lower():
            outfit.bottom_type = "pants"
        elif "skirt" in prompt.lower():
            outfit.bottom_type = "skirt"
        elif "shorts" in prompt.lower():
            outfit.bottom_type = "shorts"
        
        if "shirt" in prompt.lower() or "blouse" in prompt.lower():
            outfit.top_type = "shirt"
        elif "sweater" in prompt.lower():
            outfit.top_type = "sweater"
        elif "jacket" in prompt.lower():
            outfit.top_type = "jacket"
        elif "top" in prompt.lower():
            outfit.top_type = "top"
        
        if "heels" in prompt.lower():
            outfit.shoes = "heels"
        elif "boots" in prompt.lower():
            outfit.shoes = "boots"
        elif "sneakers" in prompt.lower():
            outfit.shoes = "sneakers"
        elif "shoes" in prompt.lower():
            outfit.shoes = "shoes"
        
        # Accessories
        if "hat" in prompt.lower():
            outfit.accessories.add("hat")
        if "scarf" in prompt.lower():
            outfit.accessories.add("scarf")
        if "necklace" in prompt.lower() or "jewelry" in prompt.lower():
            outfit.accessories.add("jewelry")
        if "bag" in prompt.lower() or "purse" in prompt.lower():
            outfit.accessories.add("bag")
        if "belt" in prompt.lower():
            outfit.accessories.add("belt")
        
        # Formality
        if "formal" in prompt.lower() or "tuxedo" in prompt.lower() or "gown" in prompt.lower() or "elegant" in prompt.lower():
            outfit.formality = "formal"
        elif "casual" in prompt.lower():
            outfit.formality = "casual"
        else:
            outfit.formality = outfit.formality or "semi-formal"
        
        logger.info(f"   👗 Outfit: {outfit.style}, {outfit.formality}, top={outfit.top_color}, bottom={outfit.bottom_color}")
        return outfit
    
    @staticmethod
    def parse_pose(prompt: str, pose_id: str = "pose_0", prev_pose: Optional[PoseAttributes] = None) -> PoseAttributes:
        """Extract pose and motion attributes from prompt."""
        
        pose = PoseAttributes(pose_id=pose_id)
        
        # Inherit from previous if available (for continuity)
        if prev_pose:
            pose.head_angle = prev_pose.head_angle
            pose.torso_angle = prev_pose.torso_angle
        
        # Motion type
        motion = PromptParser._match_keywords(prompt, PromptParser.MOTION_KEYWORDS)
        if motion:
            pose.motion_type = MotionType(motion)
        
        # Head angle
        head = PromptParser._match_keywords(prompt, PromptParser.HEAD_ANGLE_KEYWORDS)
        if head:
            pose.head_angle = head
        
        # Motion flags
        prompt_lower = prompt.lower()
        pose.is_walking = "walk" in prompt_lower
        pose.is_turning = "turn" in prompt_lower or "rotate" in prompt_lower or "spins" in prompt_lower
        pose.is_gesturing = "gesture" in prompt_lower or "wave" in prompt_lower or "point" in prompt_lower
        
        # Motion details
        if "slow" in prompt_lower:
            pose.motion_speed = "slow"
        elif "fast" in prompt_lower or "quick" in prompt_lower or "quickly" in prompt_lower:
            pose.motion_speed = "fast"
        else:
            pose.motion_speed = "moderate"
        
        logger.info(f"   🎬 Pose: {pose.motion_type.value}, head={pose.head_angle}, speed={pose.motion_speed}")
        return pose
    
    @staticmethod
    def parse_environment(prompt: str, prev_env: Optional[EnvironmentAttributes] = None) -> EnvironmentAttributes:
        """Extract environment and camera attributes from prompt."""
        
        env = EnvironmentAttributes()
        
        # Inherit from previous if available
        if prev_env:
            env.lighting_type = prev_env.lighting_type
            env.background_type = prev_env.background_type
            env.setting_type = prev_env.setting_type
            env.camera_distance = prev_env.camera_distance
            env.camera_angle = prev_env.camera_angle
        
        # Lighting
        lighting = PromptParser._match_keywords(prompt, PromptParser.LIGHTING_KEYWORDS)
        if lighting:
            env.lighting_type = Lighting(lighting)
        
        if "bright" in prompt.lower():
            env.light_intensity = "bright"
        elif "dim" in prompt.lower():
            env.light_intensity = "dim"
        else:
            env.light_intensity = "moderate"
        
        # Background
        background = PromptParser._match_keywords(prompt, PromptParser.BACKGROUND_KEYWORDS)
        if background:
            env.background_type = Background(background)
        
        # Setting
        if "studio" in prompt.lower():
            env.setting_type = "studio"
        elif "outdoor" in prompt.lower() or "outside" in prompt.lower():
            env.setting_type = "outdoor"
        elif "indoor" in prompt.lower() or "inside" in prompt.lower():
            env.setting_type = "indoor"
        
        # Camera
        if "close-up" in prompt.lower() or "close up" in prompt.lower():
            env.camera_distance = "close_up"
        elif "wide" in prompt.lower():
            env.camera_distance = "wide"
        else:
            env.camera_distance = "medium"
        
        if "eye level" in prompt.lower():
            env.camera_angle = "eye_level"
        elif "high angle" in prompt.lower():
            env.camera_angle = "high_angle"
        elif "low angle" in prompt.lower():
            env.camera_angle = "low_angle"
        
        logger.info(f"   🎥 Environment: {env.lighting_type.value}, {env.background_type.value}, camera={env.camera_distance}")
        return env
    
    @staticmethod
    def parse_scene(prompt: str, scene_id: str, timestamp: float, sequence_num: int = 0, prev_scene: Optional[SceneNode] = None) -> SceneNode:
        """Parse complete scene from prompt, maintaining consistency with previous scene."""
        
        logger.info(f"🔍 Parsing prompt for {scene_id}:")
        logger.info(f"   📝 Prompt: {prompt[:100]}...")
        
        character = PromptParser.parse_character(prompt, f"char_0", prev_scene.character if prev_scene else None)
        outfit = PromptParser.parse_outfit(prompt, f"outfit_0", prev_scene.outfit if prev_scene else None)
        pose = PromptParser.parse_pose(prompt, f"pose_0", prev_scene.pose if prev_scene else None)
        environment = PromptParser.parse_environment(prompt, prev_scene.environment if prev_scene else None)
        
        scene = SceneNode(
            scene_id=scene_id,
            timestamp=timestamp,
            character=character,
            outfit=outfit,
            pose=pose,
            environment=environment,
            prompt=prompt,
            sequence_number=sequence_num,
        )
        
        logger.info(f"✅ Scene parsed successfully")
        return scene