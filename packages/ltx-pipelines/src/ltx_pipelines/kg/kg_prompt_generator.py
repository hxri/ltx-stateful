"""
Generate detailed, semantically-rich prompts from scene nodes.
MOTION-FIRST VERSION: Leads with motion verbs, not static descriptions.
Video diffusion models respond best to action-first prompts.
"""

import logging
from typing import Optional, Set
from .kg_schema import SceneNode, CharacterAttributes, OutfitAttributes, PoseAttributes, EnvironmentAttributes

logger = logging.getLogger(__name__)


class DetailedPromptGenerator:
    """Generate rich, detailed prompts from scene semantic data.
    
    KEY INSIGHT: Video diffusion models generate motion from VERBS at the 
    start of prompts. Static descriptions (appearance, clothing) should come 
    AFTER the motion description. The model pays most attention to the first 
    ~30 tokens.
    """

    @staticmethod
    def _is_valid(value: Optional[str]) -> bool:
        """Check if value is valid and not 'unknown'."""
        return value is not None and str(value).lower() not in ("unknown", "none", "")

    @staticmethod
    def generate_motion_lead(pose: PoseAttributes) -> str:
        """
        Generate the MOTION LEAD — this goes FIRST in the prompt.
        Each motion type gets a unique, concrete, verb-heavy description.
        No generic "rotating" — every motion is physically specific.
        """
        if not pose.motion_type or pose.motion_type.value in ("static", "unknown"):
            return ""

        # Concrete, physically-specific motion descriptions
        # Each one describes WHAT the body parts actually do
        motion_sentences = {
            "turning": (
                "The person slowly turns their body to the side, "
                "shifting weight from one foot to the other, "
                "shoulders and hips rotating together"
            ),
            "walking": (
                "The person is walking forward with natural strides, "
                "legs moving step by step, arms swinging naturally. "
                "Continuous walking motion"
            ),
            "running": (
                "The person is running with energetic motion, "
                "arms pumping, legs in full stride. "
                "Fast dynamic running movement"
            ),
            "dancing": (
                "The person is dancing with fluid body movements, "
                "swaying and stepping rhythmically. "
                "Expressive dance motion throughout"
            ),
            "gesture": (
                "The person is actively gesturing with their hands and arms, "
                "making expressive movements. "
                "Dynamic hand and arm motion"
            ),
            "squatting": (
                "The person bends their knees and lowers their body down into a squat, "
                "hips dropping toward the ground, thighs bending deeply, "
                "then slowly rising back up to standing position. "
                "The whole body moves downward and then upward"
            ),
            "raising_arms": (
                "The person lifts both arms upward from their sides, "
                "raising their hands higher and higher above their head, "
                "arms fully extended overhead, fingers reaching toward the sky. "
                "The arms move visibly from low to high"
            ),
            "looking_around": (
                "The person tilts their head upward looking at the ceiling, "
                "then slowly lowers their chin downward looking at the floor, "
                "the head moves up and down in a slow deliberate motion. "
                "The neck and head change angle visibly"
            ),
            "leg_raise": (
                "The person lifts one leg off the ground, bending the knee upward, "
                "raising the thigh until it is parallel to the floor, "
                "balancing on one foot with the other leg held in the air. "
                "One leg visibly lifts up while standing on the other"
            ),
            "bending": (
                "The person bends forward at the waist, "
                "lowering their upper body toward the ground, "
                "then slowly straightens back up to an upright position. "
                "The torso tilts forward and then returns"
            ),
            "waving": (
                "The person raises one hand and waves it side to side, "
                "the hand moving left and right repeatedly in a greeting gesture, "
                "fingers spread open, arm extended outward. "
                "The hand and arm move back and forth visibly"
            ),
            "stepping": (
                "The person takes a deliberate step to the side, "
                "shifting their weight and moving their feet apart, "
                "then stepping back to the original position. "
                "The feet and legs move laterally"
            ),
            "complex": (
                "The person performs a sequence of movements, "
                "transitioning between multiple poses fluidly. "
                "Continuous dynamic motion"
            ),
        }

        base = motion_sentences.get(
            pose.motion_type.value,
            f"The person is {pose.motion_type.value} with visible body movement"
        )

        # Add speed modifier
        if DetailedPromptGenerator._is_valid(pose.motion_speed):
            speed_map = {
                "slow": " slowly and deliberately",
                "moderate": " at a natural steady pace",
                "fast": " quickly with energetic speed",
            }
            base += speed_map.get(pose.motion_speed, "")

        # Add direction only if relevant (not for squatting, arm raising, etc.)
        directional_motions = {"turning", "walking", "running", "stepping"}
        if (DetailedPromptGenerator._is_valid(pose.motion_direction) 
            and pose.motion_type.value in directional_motions):
            direction_map = {
                "forward": ", moving forward",
                "backward": ", stepping backward",
                "left": ", toward the left",
                "right": ", toward the right",
                "turning": ", pivoting in place",
            }
            base += direction_map.get(pose.motion_direction, "")

        return base + "."

    @staticmethod
    def generate_character_brief(character: CharacterAttributes) -> str:
        """Generate a SHORT character description. Keep it brief — motion matters more."""
        parts = []

        gender = ""
        if character.gender and character.gender.value != "unknown":
            gender = character.gender.value

        age = ""
        if character.age_group and character.age_group.value != "unknown":
            age = character.age_group.value.replace("_", " ")

        if gender or age:
            parts.append(f"{'A ' + age + ' ' + gender if age else 'A ' + gender}".strip())
        else:
            parts.append("A person")

        # One line of physical traits max
        traits = []
        if DetailedPromptGenerator._is_valid(character.body_type):
            traits.append(character.body_type)
        if DetailedPromptGenerator._is_valid(character.hair_color):
            traits.append(f"{character.hair_color} hair")

        if traits:
            parts.append(f" with {', '.join(traits)}")

        return "".join(parts)

    @staticmethod
    def generate_outfit_brief(outfit: OutfitAttributes) -> str:
        """Generate SHORT outfit description."""
        pieces = []

        if DetailedPromptGenerator._is_valid(outfit.formality):
            pieces.append(outfit.formality)

        if DetailedPromptGenerator._is_valid(outfit.top_color):
            pieces.append(outfit.top_color)

        if DetailedPromptGenerator._is_valid(outfit.top_type):
            pieces.append(outfit.top_type)
        
        if DetailedPromptGenerator._is_valid(outfit.bottom_type):
            if DetailedPromptGenerator._is_valid(outfit.bottom_color) and outfit.bottom_color != outfit.top_color:
                pieces.append(f"{outfit.bottom_color} {outfit.bottom_type}")
            else:
                pieces.append(outfit.bottom_type)

        if not pieces:
            return ""

        return "wearing " + " ".join(pieces)

    @staticmethod
    def generate_pose_details(pose: PoseAttributes) -> str:
        """Generate secondary pose details (head angle, gaze, etc.)."""
        parts = []

        if DetailedPromptGenerator._is_valid(pose.head_angle):
            angle_map = {
                "frontal": "facing the camera",
                "profile_left": "turned to show left profile",
                "profile_right": "turned to show right profile",
                "3_4_view": "at a three-quarter angle",
                "looking_up": "head tilted upward",
                "looking_down": "head tilted downward",
            }
            parts.append(angle_map.get(pose.head_angle, ""))

        return ", ".join(p for p in parts if p)

    @staticmethod
    def generate_environment_brief(environment: EnvironmentAttributes) -> str:
        """Generate SHORT environment description."""
        parts = []

        if environment.lighting_type and environment.lighting_type.value != "unknown":
            lighting_map = {
                "natural": "Natural lighting",
                "studio": "Studio lighting",
                "soft": "Soft diffused light",
                "harsh": "Bright dramatic light",
                "backlit": "Backlit",
                "sidelit": "Side-lit",
            }
            parts.append(lighting_map.get(environment.lighting_type.value, ""))

        if environment.background_type and environment.background_type.value != "unknown":
            bg_map = {
                "studio": "studio backdrop",
                "indoor": "indoor setting",
                "outdoor": "outdoor setting",
                "abstract": "abstract background",
                "blurred": "blurred background",
                "plain": "plain background",
            }
            bg = bg_map.get(environment.background_type.value, "")
            if bg:
                parts.append(bg)

        if DetailedPromptGenerator._is_valid(environment.camera_distance):
            cam_map = {
                "close_up": "Close-up shot",
                "medium": "Medium shot",
                "wide": "Wide shot, full body visible",
                "extreme_wide": "Extreme wide shot",
            }
            parts.append(cam_map.get(environment.camera_distance, ""))

        return ". ".join(p for p in parts if p)

    @staticmethod
    def generate_detailed_prompt(scene_node: SceneNode, include_context: bool = True) -> str:
        """
        Generate prompt optimized for VIDEO GENERATION with actual motion.
        
        STRUCTURE (order matters!):
        1. MOTION LEAD — Action verbs first (most important for video models)
        2. Character brief — Who is doing it
        3. Outfit brief — What they're wearing
        4. Pose details — Head angle, gaze
        5. Environment brief — Where
        6. Quality tags — Technical quality
        """
        prompt_sections = []

        # ═══════════════════════════════════════════════════════
        # 1. MOTION LEAD (FIRST — MOST IMPORTANT)
        # ═══════════════════════════════════════════════════════
        motion_lead = DetailedPromptGenerator.generate_motion_lead(scene_node.pose)
        
        # ═══════════════════════════════════════════════════════
        # 2. CHARACTER BRIEF (SHORT)
        # ═══════════════════════════════════════════════════════
        char_brief = DetailedPromptGenerator.generate_character_brief(scene_node.character)

        # ═══════════════════════════════════════════════════════
        # 3. OUTFIT BRIEF (SHORT)
        # ═══════════════════════════════════════════════════════
        outfit_brief = DetailedPromptGenerator.generate_outfit_brief(scene_node.outfit)

        # ═══════════════════════════════════════════════════════
        # 4. POSE DETAILS (HEAD ANGLE, GAZE)
        # ═══════════════════════════════════════════════════════
        pose_details = DetailedPromptGenerator.generate_pose_details(scene_node.pose)

        # ═══════════════════════════════════════════════════════
        # 5. BUILD THE PROMPT — MOTION FIRST
        # ═══════════════════════════════════════════════════════
        if motion_lead:
            # MOTION SCENES: Lead with action
            prompt_sections.append(motion_lead)
            
            subject = char_brief
            if outfit_brief:
                subject += " " + outfit_brief
            if pose_details:
                subject += ", " + pose_details
            prompt_sections.append(subject + ".")
        else:
            # STATIC SCENES: Standard description
            subject = char_brief
            if outfit_brief:
                subject += " " + outfit_brief
            
            static_desc = "standing still, stationary, holding a steady pose"
            if pose_details:
                static_desc += ", " + pose_details
            
            prompt_sections.append(subject + " " + static_desc + ".")

        # ═══════════════════════════════════════════════════════
        # 6. ENVIRONMENT (BRIEF)
        # ═══════════════════════════════════════════════════════
        env_brief = DetailedPromptGenerator.generate_environment_brief(scene_node.environment)
        if env_brief:
            prompt_sections.append(env_brief + ".")

        # ═══════════════════════════════════════════════════════
        # 7. QUALITY TAGS (WITH MOTION EMPHASIS)
        # ═══════════════════════════════════════════════════════
        if include_context:
            is_motion = (scene_node.pose.motion_type 
                        and scene_node.pose.motion_type.value not in ("static", "unknown"))
            
            if is_motion:
                quality = (
                    "Smooth continuous motion. "
                    "Fluid natural body movement throughout the entire video. "
                    "High quality cinematic video. "
                    "The body moves and changes position over time."
                )
            else:
                quality = (
                    "High quality cinematic video. "
                    "Photorealistic. Professional photography."
                )
            prompt_sections.append(quality)

        detailed_prompt = " ".join(prompt_sections)

        # Clean up
        while "  " in detailed_prompt:
            detailed_prompt = detailed_prompt.replace("  ", " ")
        detailed_prompt = detailed_prompt.replace(". .", ".").replace("..", ".")

        logger.info(f"   📝 Prompt ({len(detailed_prompt)} chars): {detailed_prompt[:200]}...")
        return detailed_prompt.strip()


# Example usage
if __name__ == "__main__":
    import time
    from kg_schema import (
        CharacterAttributes, OutfitAttributes, PoseAttributes, EnvironmentAttributes, SceneNode,
        Gender, Ethnicity, AgeGroup, Lighting, Background, MotionType
    )

    # Create sample scene
    char = CharacterAttributes(
        character_id="char_1",
        gender=Gender.FEMALE,
        ethnicity=Ethnicity.CAUCASIAN,
        age_group=AgeGroup.YOUNG_ADULT,
        height="tall",
        body_type="athletic",
        hair_color="blonde",
        hair_style="long",
        expression="confident"
    )

    outfit = OutfitAttributes(
        outfit_id="outfit_1",
        style="formal",
        formality="formal",
        top_type="blouse",
        top_color="black",
        bottom_type="pants",
        bottom_color="black",
        shoes="heels",
        shoe_color="black",
        accessories={"jewelry", "belt"},
        overall_color_scheme="monochrome"
    )

    pose = PoseAttributes(
        pose_id="pose_1",
        motion_type=MotionType.TURNING,
        motion_speed="moderate",
        motion_direction="turning",
        head_angle="3_4_view",
        torso_angle="upright",
        is_turning=True
    )

    environment = EnvironmentAttributes(
        lighting_type=Lighting.STUDIO,
        light_intensity="bright",
        background_type=Background.PLAIN,
        camera_distance="medium",
        camera_angle="eye_level",
        atmosphere="professional"
    )

    scene = SceneNode(
        scene_id="test_scene",
        timestamp=time.time(),
        character=char,
        outfit=outfit,
        pose=pose,
        environment=environment,
        prompt="Test"
    )

    detailed = DetailedPromptGenerator.generate_detailed_prompt(scene)
    print("Generated detailed prompt:")
    print(detailed)