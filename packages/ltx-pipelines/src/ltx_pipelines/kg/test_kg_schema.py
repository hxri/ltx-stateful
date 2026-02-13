"""
Tests for KG schema, semantic encoder (CLIP + DINOv2), KG encoder,
latent observer, dynamic graph, and graph store.
"""

import json
import time
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch

# ─── Schema imports ──────────────────────────────────────────────
from .kg_schema import (
    Gender,
    Ethnicity,
    AgeGroup,
    Lighting,
    Background,
    MotionType,
    CharacterAttributes,
    OutfitAttributes,
    PoseAttributes,
    EnvironmentAttributes,
    SceneNode,
    SceneEdge,
    StoryState,
)

# ─── Component imports ───────────────────────────────────────────
from .dynamic_graph import DynamicSceneGraph, SceneNode as DynSceneNode
from .graph_store import GraphStore
from .kg_encoder import KGConditionEncoder
from .kg_semantic_encoder import (
    scene_node_to_text,
    CLIPSceneEncoder,
    DINOFrameEncoder,
    KGConsistencyScorer,
    KGConditioner,
)
from .latent_observer import LatentObserver


# ═════════════════════════════════════════════════════════════════
# FIXTURES
# ═════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_character():
    return CharacterAttributes(
        character_id="char_1",
        name="Alice",
        gender=Gender.FEMALE,
        ethnicity=Ethnicity.CAUCASIAN,
        age_group=AgeGroup.YOUNG_ADULT,
        height="tall",
        body_type="athletic",
        hair_color="blonde",
        hair_style="long",
        expression="confident",
    )


@pytest.fixture
def sample_outfit():
    return OutfitAttributes(
        outfit_id="outfit_1",
        style="casual",
        top_type="blouse",
        top_color="white",
        bottom_type="jeans",
        bottom_color="blue",
        shoes="sneakers",
        accessories={"watch", "earrings"},
    )


@pytest.fixture
def sample_pose():
    return PoseAttributes(
        pose_id="pose_1",
        motion_type=MotionType.WALKING,
        motion_speed="moderate",
        motion_direction="forward",
        head_angle="frontal",
        gaze_direction="forward",
        is_walking=True,
    )


@pytest.fixture
def sample_environment():
    return EnvironmentAttributes(
        lighting_type=Lighting.NATURAL,
        light_direction="frontal",
        background_type=Background.OUTDOOR,
        setting_type="park",
        camera_distance="medium",
        atmosphere="calm",
        dominant_colors={"green", "blue"},
    )


@pytest.fixture
def sample_scene_node(sample_character, sample_outfit, sample_pose, sample_environment):
    return SceneNode(
        scene_id="scene_001",
        timestamp=time.time(),
        character=sample_character,
        outfit=sample_outfit,
        pose=sample_pose,
        environment=sample_environment,
        prompt="A young woman walking through a park in casual clothes",
        scene_type="establishing",
        sequence_number=0,
    )


@pytest.fixture
def sample_scene_edge():
    return SceneEdge(
        source_scene="scene_001",
        target_scene="scene_002",
        relation_type="continues",
        character_consistent=True,
        outfit_consistent=True,
        transition_type="cut",
        temporal_gap="immediate",
        confidence=0.95,
    )


@pytest.fixture
def sample_story_state(sample_scene_node, sample_scene_edge):
    return StoryState(
        story_id="story_1",
        created_at=time.time(),
        updated_at=time.time(),
        scenes={"scene_001": sample_scene_node},
        transitions={"scene_001->scene_002": sample_scene_edge},
        title="Test Story",
        generation_order=["scene_001"],
    )


@pytest.fixture
def dummy_frame():
    """Random RGB uint8 frame (224x224)."""
    return np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)


@pytest.fixture
def dummy_frame_pil(dummy_frame):
    """PIL Image version of dummy frame."""
    from PIL import Image
    return Image.fromarray(dummy_frame)


# ═════════════════════════════════════════════════════════════════
# TESTS: ENUMS
# ═════════════════════════════════════════════════════════════════

class TestEnums:
    def test_gender_values(self):
        assert Gender.MALE.value == "male"
        assert Gender.FEMALE.value == "female"
        assert Gender.NEUTRAL.value == "neutral"
        assert Gender.UNKNOWN.value == "unknown"

    def test_gender_is_string(self):
        assert isinstance(Gender.MALE, str)
        assert Gender.MALE == "male"

    def test_ethnicity_values(self):
        assert len(Ethnicity) == 7
        assert Ethnicity.ASIAN.value == "asian"

    def test_age_group_values(self):
        assert AgeGroup.YOUNG_ADULT.value == "young_adult"
        assert AgeGroup.SENIOR.value == "senior"

    def test_lighting_values(self):
        assert Lighting.STUDIO.value == "studio"
        assert Lighting.BACKLIT.value == "backlit"

    def test_background_values(self):
        assert Background.OUTDOOR.value == "outdoor"
        assert Background.STUDIO.value == "studio"

    def test_motion_type_values(self):
        assert MotionType.WALKING.value == "walking"
        assert MotionType.SQUATTING.value == "squatting"
        assert MotionType.RAISING_ARMS.value == "raising_arms"
        assert MotionType.LEG_RAISE.value == "leg_raise"

    def test_all_motion_types_present(self):
        expected = {
            "static", "walking", "running", "turning", "dancing",
            "gesture", "squatting", "raising_arms", "looking_around",
            "leg_raise", "bending", "waving", "stepping", "complex", "unknown",
        }
        actual = {m.value for m in MotionType}
        assert actual == expected


# ═════════════════════════════════════════════════════════════════
# TESTS: CHARACTER ATTRIBUTES
# ═════════════════════════════════════════════════════════════════

class TestCharacterAttributes:
    def test_creation(self, sample_character):
        assert sample_character.character_id == "char_1"
        assert sample_character.gender == Gender.FEMALE
        assert sample_character.hair_color == "blonde"

    def test_defaults(self):
        char = CharacterAttributes(character_id="c1")
        assert char.gender == Gender.UNKNOWN
        assert char.ethnicity == Ethnicity.UNKNOWN
        assert char.age_group == AgeGroup.UNKNOWN
        assert char.hair_color is None
        assert char.facial_features == set()
        assert char.distinguishing_marks == set()

    def test_to_dict(self, sample_character):
        d = sample_character.to_dict()
        assert isinstance(d, dict)
        assert d["character_id"] == "char_1"
        assert d["gender"] == "female"
        assert d["ethnicity"] == "caucasian"
        assert d["hair_color"] == "blonde"
        assert isinstance(d["facial_features"], list)
        assert isinstance(d["distinguishing_marks"], list)

    def test_set_fields_serialized(self):
        char = CharacterAttributes(
            character_id="c2",
            facial_features={"beard", "glasses"},
            distinguishing_marks={"tattoo"},
        )
        d = char.to_dict()
        assert set(d["facial_features"]) == {"beard", "glasses"}
        assert d["distinguishing_marks"] == ["tattoo"]


# ═════════════════════════════════════════════════════════════════
# TESTS: OUTFIT ATTRIBUTES
# ═════════════════════════════════════════════════════════════════

class TestOutfitAttributes:
    def test_creation(self, sample_outfit):
        assert sample_outfit.outfit_id == "outfit_1"
        assert sample_outfit.top_color == "white"
        assert "watch" in sample_outfit.accessories

    def test_defaults(self):
        outfit = OutfitAttributes(outfit_id="o1")
        assert outfit.style is None
        assert outfit.accessories == set()
        assert outfit.layers is None

    def test_to_dict(self, sample_outfit):
        d = sample_outfit.to_dict()
        assert d["outfit_id"] == "outfit_1"
        assert d["top_type"] == "blouse"
        assert set(d["accessories"]) == {"watch", "earrings"}

    def test_to_dict_is_json_serializable(self, sample_outfit):
        d = sample_outfit.to_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)


# ═════════════════════════════════════════════════════════════════
# TESTS: POSE ATTRIBUTES
# ═════════════════════════════════════════════════════════════════

class TestPoseAttributes:
    def test_creation(self, sample_pose):
        assert sample_pose.motion_type == MotionType.WALKING
        assert sample_pose.is_walking is True
        assert sample_pose.motion_direction == "forward"

    def test_defaults(self):
        pose = PoseAttributes(pose_id="p1")
        assert pose.motion_type == MotionType.UNKNOWN
        assert pose.is_walking is False
        assert pose.is_turning is False
        assert pose.is_gesturing is False

    def test_to_dict(self, sample_pose):
        d = sample_pose.to_dict()
        assert d["motion_type"] == "walking"
        assert d["is_walking"] is True
        assert d["pose_id"] == "pose_1"


# ═════════════════════════════════════════════════════════════════
# TESTS: ENVIRONMENT ATTRIBUTES
# ═════════════════════════════════════════════════════════════════

class TestEnvironmentAttributes:
    def test_creation(self, sample_environment):
        assert sample_environment.lighting_type == Lighting.NATURAL
        assert sample_environment.background_type == Background.OUTDOOR
        assert "green" in sample_environment.dominant_colors

    def test_defaults(self):
        env = EnvironmentAttributes()
        assert env.lighting_type == Lighting.UNKNOWN
        assert env.background_type == Background.UNKNOWN
        assert env.dominant_colors == set()
        assert env.visible_elements == set()

    def test_to_dict(self, sample_environment):
        d = sample_environment.to_dict()
        assert d["lighting_type"] == "natural"
        assert d["background_type"] == "outdoor"
        assert set(d["dominant_colors"]) == {"green", "blue"}


# ═════════════════════════════════════════════════════════════════
# TESTS: SCENE NODE
# ═════════════════════════════════════════════════════════════════

class TestSceneNode:
    def test_creation(self, sample_scene_node):
        assert sample_scene_node.scene_id == "scene_001"
        assert sample_scene_node.scene_type == "establishing"
        assert sample_scene_node.sequence_number == 0

    def test_nested_to_dict(self, sample_scene_node):
        d = sample_scene_node.to_dict()
        assert "character" in d
        assert "outfit" in d
        assert "pose" in d
        assert "environment" in d
        assert d["character"]["gender"] == "female"
        assert d["outfit"]["top_type"] == "blouse"
        assert d["pose"]["motion_type"] == "walking"
        assert d["environment"]["lighting_type"] == "natural"

    def test_json_serializable(self, sample_scene_node):
        d = sample_scene_node.to_dict()
        s = json.dumps(d)
        parsed = json.loads(s)
        assert parsed["scene_id"] == "scene_001"

    def test_prompt_preserved(self, sample_scene_node):
        d = sample_scene_node.to_dict()
        assert "walking through a park" in d["prompt"]


# ═════════════════════════════════════════════════════════════════
# TESTS: SCENE EDGE
# ═════════════════════════════════════════════════════════════════

class TestSceneEdge:
    def test_creation(self, sample_scene_edge):
        assert sample_scene_edge.source_scene == "scene_001"
        assert sample_scene_edge.target_scene == "scene_002"
        assert sample_scene_edge.confidence == 0.95

    def test_defaults(self):
        edge = SceneEdge(source_scene="a", target_scene="b", relation_type="continues")
        assert edge.character_consistent is True
        assert edge.outfit_consistent is True
        assert edge.confidence == 1.0
        assert edge.transition_type is None

    def test_to_dict(self, sample_scene_edge):
        d = sample_scene_edge.to_dict()
        assert d["source"] == "scene_001"
        assert d["target"] == "scene_002"
        assert d["relation_type"] == "continues"
        assert d["confidence"] == 0.95


# ═════════════════════════════════════════════════════════════════
# TESTS: STORY STATE
# ═════════════════════════════════════════════════════════════════

class TestStoryState:
    def test_creation(self, sample_story_state):
        assert sample_story_state.story_id == "story_1"
        assert len(sample_story_state.scenes) == 1
        assert len(sample_story_state.transitions) == 1

    def test_to_dict(self, sample_story_state):
        d = sample_story_state.to_dict()
        assert d["story_id"] == "story_1"
        assert "scene_001" in d["scenes"]
        assert d["scenes"]["scene_001"]["character"]["gender"] == "female"

    def test_to_json(self, sample_story_state):
        j = sample_story_state.to_json()
        parsed = json.loads(j)
        assert parsed["title"] == "Test Story"
        assert parsed["generation_order"] == ["scene_001"]

    def test_defaults(self):
        state = StoryState(
            story_id="s1",
            created_at=0.0,
            updated_at=0.0,
        )
        assert state.scenes == {}
        assert state.transitions == {}
        assert state.generation_order == []
        assert state.global_character_consistency == 1.0

    def test_round_trip_json(self, sample_story_state):
        """to_json → parse → verify all nested structures survive."""
        j = sample_story_state.to_json()
        parsed = json.loads(j)
        scene = parsed["scenes"]["scene_001"]
        assert scene["outfit"]["top_color"] == "white"
        assert scene["pose"]["is_walking"] is True
        edge = parsed["transitions"]["scene_001->scene_002"]
        assert edge["confidence"] == 0.95


# ═════════════════════════════════════════════════════════════════
# TESTS: DYNAMIC GRAPH
# ═════════════════════════════════════════════════════════════════

class TestDynamicSceneGraph:
    def test_init(self):
        g = DynamicSceneGraph(dim=128)
        assert g.dim == 128
        assert g.global_state.shape == (128,)
        assert len(g.nodes) == 0

    def test_add_node(self):
        g = DynamicSceneGraph(dim=64)
        emb = torch.randn(64)
        g.add_or_update_node("n1", emb)
        assert "n1" in g.nodes
        assert g.nodes["n1"].embedding.shape == (64,)

    def test_update_existing_node(self):
        g = DynamicSceneGraph(dim=64)
        emb1 = torch.ones(64)
        g.add_or_update_node("n1", emb1)
        old = g.nodes["n1"].embedding.clone()
        emb2 = torch.zeros(64)
        g.add_or_update_node("n1", emb2)
        new = g.nodes["n1"].embedding
        # Should be EMA: 0.9 * old + 0.1 * new
        expected = 0.9 * old + 0.1 * emb2
        assert torch.allclose(new, expected, atol=1e-5)

    def test_update_global(self):
        g = DynamicSceneGraph(dim=64)
        obs = torch.ones(64)
        g.update_global(obs)
        expected = 0.98 * torch.zeros(64) + 0.02 * obs
        assert torch.allclose(g.global_state, expected, atol=1e-5)

    def test_readout(self):
        g = DynamicSceneGraph(dim=32)
        obs = torch.randn(32)
        g.update_global(obs)
        ro = g.readout()
        assert torch.allclose(ro, g.global_state)
        # readout is a clone, not a reference
        ro[0] = 999.0
        assert g.global_state[0] != 999.0

    def test_serialize_roundtrip(self):
        g = DynamicSceneGraph(dim=32)
        g.update_global(torch.randn(32))
        g.add_or_update_node("a", torch.randn(32))
        g.add_or_update_node("b", torch.randn(32))

        state = g.serialize()
        g2 = DynamicSceneGraph(dim=32)
        g2.load(state)
        assert torch.allclose(g.global_state, g2.global_state)
        assert set(g.nodes.keys()) == set(g2.nodes.keys())


# ═════════════════════════════════════════════════════════════════
# TESTS: GRAPH STORE
# ═════════════════════════════════════════════════════════════════

class TestGraphStore:
    def test_save_and_load(self, tmp_path):
        path = tmp_path / "test_kg.pt"
        store = GraphStore(path=str(path))
        g = DynamicSceneGraph(dim=16)
        g.update_global(torch.randn(16))
        g.add_or_update_node("x", torch.randn(16))

        store.save(g)
        assert path.exists()

        g2 = DynamicSceneGraph(dim=16)
        store.load(g2)
        assert torch.allclose(g.global_state, g2.global_state)

    def test_load_nonexistent(self, tmp_path):
        path = tmp_path / "nonexistent.pt"
        store = GraphStore(path=str(path))
        g = DynamicSceneGraph(dim=16)
        # Should not crash
        store.load(g)
        assert g.global_state.sum() == 0.0


# ═════════════════════════════════════════════════════════════════
# TESTS: KG CONDITION ENCODER
# ═════════════════════════════════════════════════════════════════

class TestKGConditionEncoder:
    def test_output_shape(self):
        enc = KGConditionEncoder(input_dim=768, model_dim=4096)
        x = torch.randn(1, 768)
        out = enc(x)
        assert out.shape == (1, 4096)

    def test_batched(self):
        enc = KGConditionEncoder(input_dim=768, model_dim=2048)
        x = torch.randn(4, 768)
        out = enc(x)
        assert out.shape == (4, 2048)

    def test_custom_dims(self):
        enc = KGConditionEncoder(input_dim=512, model_dim=1024)
        x = torch.randn(2, 512)
        out = enc(x)
        assert out.shape == (2, 1024)

    def test_gradients_flow(self):
        enc = KGConditionEncoder(input_dim=768, model_dim=4096)
        x = torch.randn(1, 768, requires_grad=True)
        out = enc(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


# ═════════════════════════════════════════════════════════════════
# TESTS: scene_node_to_text
# ═════════════════════════════════════════════════════════════════

class TestSceneNodeToText:
    def test_basic_conversion(self, sample_scene_node):
        text = scene_node_to_text(sample_scene_node)
        assert isinstance(text, str)
        assert len(text) > 10
        assert text.endswith(".")

    def test_contains_character_info(self, sample_scene_node):
        text = scene_node_to_text(sample_scene_node)
        text_lower = text.lower()
        assert "female" in text_lower or "young adult" in text_lower

    def test_contains_outfit_info(self, sample_scene_node):
        text = scene_node_to_text(sample_scene_node)
        text_lower = text.lower()
        assert "blouse" in text_lower or "white" in text_lower

    def test_contains_motion(self, sample_scene_node):
        text = scene_node_to_text(sample_scene_node)
        assert "walking" in text.lower()

    def test_contains_environment(self, sample_scene_node):
        text = scene_node_to_text(sample_scene_node)
        assert "outdoor" in text.lower() or "natural" in text.lower()

    def test_dict_input(self, sample_scene_node):
        d = sample_scene_node.to_dict()
        text = scene_node_to_text(d)
        assert isinstance(text, str)
        assert len(text) > 10

    def test_unknown_fields_handled(self):
        """Minimal scene node with all unknowns should still produce text."""
        char = CharacterAttributes(character_id="c1")
        outfit = OutfitAttributes(outfit_id="o1")
        pose = PoseAttributes(pose_id="p1")
        env = EnvironmentAttributes()
        node = SceneNode(
            scene_id="s1",
            timestamp=0.0,
            character=char,
            outfit=outfit,
            pose=pose,
            environment=env,
            prompt="test",
        )
        text = scene_node_to_text(node)
        assert "person" in text.lower()


# ═════════════════════════════════════════════════════════════════
# TESTS: CLIP + DINO ENCODERS (with mocking)
#
# These tests mock transformers to avoid downloading models in CI.
# Integration tests below (marked slow) actually load models.
# ═════════════════════════════════════════════════════════════════

class TestCLIPSceneEncoderMocked:
    """Unit tests with mocked CLIP model."""

    def _make_encoder(self):
        """Create encoder with mocked internals."""
        import ltx_pipelines.kg.kg_semantic_encoder as mod

        mock_model = MagicMock()
        mock_model.get_text_features.return_value = torch.randn(1, 768)
        mock_model.get_image_features.return_value = torch.randn(1, 768)

        mock_processor = MagicMock()
        mock_processor.return_value = {"input_ids": torch.zeros(1, 10, dtype=torch.long)}

        mod._clip_model = mock_model
        mod._clip_processor = mock_processor

        enc = CLIPSceneEncoder(device=torch.device("cpu"))
        enc._loaded = True
        return enc, mod

    def test_text_embedding_shape(self):
        enc, mod = self._make_encoder()
        try:
            emb = enc("a test sentence")
            assert emb.shape == (1, 768)
        finally:
            mod._clip_model = None
            mod._clip_processor = None

    def test_text_embedding_normalized(self):
        enc, mod = self._make_encoder()
        try:
            emb = enc("test")
            norm = emb.norm(dim=-1)
            assert torch.allclose(norm, torch.ones(1), atol=1e-4)
        finally:
            mod._clip_model = None
            mod._clip_processor = None


class TestDINOFrameEncoderMocked:
    """Unit tests with mocked DINOv2 model."""

    def _make_encoder(self):
        import ltx_pipelines.kg.kg_semantic_encoder as mod

        mock_outputs = MagicMock()
        mock_outputs.last_hidden_state = torch.randn(1, 197, 768)

        mock_model = MagicMock()
        mock_model.return_value = mock_outputs

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mod._dino_model = mock_model
        mod._dino_processor = mock_processor

        enc = DINOFrameEncoder(device=torch.device("cpu"))
        enc._loaded = True
        return enc, mod

    def test_frame_embedding_shape(self, dummy_frame):
        enc, mod = self._make_encoder()
        try:
            emb = enc(dummy_frame)
            assert emb.shape == (1, 768)
        finally:
            mod._dino_model = None
            mod._dino_processor = None

    def test_frame_embedding_normalized(self, dummy_frame):
        enc, mod = self._make_encoder()
        try:
            emb = enc(dummy_frame)
            norm = emb.norm(dim=-1)
            assert torch.allclose(norm, torch.ones(1), atol=1e-4)
        finally:
            mod._dino_model = None
            mod._dino_processor = None


# ═════════════════════════════════════════════════════════════════
# TESTS: KG CONSISTENCY SCORER (mocked)
# ═════════════════════════════════════════════════════════════════

class TestKGConsistencyScorerMocked:
    def _setup_mocks(self):
        import ltx_pipelines.kg.kg_semantic_encoder as mod

        # CLIP
        mock_clip = MagicMock()
        mock_clip.get_text_features.return_value = torch.randn(1, 768)
        mock_clip.get_image_features.return_value = torch.randn(1, 768)
        mock_clip_proc = MagicMock()
        mock_clip_proc.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        # DINO
        mock_dino_out = MagicMock()
        mock_dino_out.last_hidden_state = torch.randn(1, 197, 768)
        mock_dino = MagicMock()
        mock_dino.return_value = mock_dino_out
        mock_dino_proc = MagicMock()
        mock_dino_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mod._clip_model = mock_clip
        mod._clip_processor = mock_clip_proc
        mod._dino_model = mock_dino
        mod._dino_processor = mock_dino_proc
        return mod

    def _teardown_mocks(self, mod):
        mod._clip_model = None
        mod._clip_processor = None
        mod._dino_model = None
        mod._dino_processor = None

    def test_score_text_image_returns_float(self, dummy_frame):
        mod = self._setup_mocks()
        try:
            scorer = KGConsistencyScorer(device=torch.device("cpu"))
            scorer.clip_encoder._loaded = True
            scorer.dino_encoder._loaded = True
            score = scorer.score_text_image("a person walking", dummy_frame)
            assert isinstance(score, float)
            assert -1.0 <= score <= 1.0
        finally:
            self._teardown_mocks(mod)

    def test_score_scene_image(self, sample_scene_node, dummy_frame):
        mod = self._setup_mocks()
        try:
            scorer = KGConsistencyScorer(device=torch.device("cpu"))
            scorer.clip_encoder._loaded = True
            scorer.dino_encoder._loaded = True
            score = scorer.score_scene_image(sample_scene_node, dummy_frame)
            assert isinstance(score, float)
        finally:
            self._teardown_mocks(mod)

    def test_score_frame_pair(self, dummy_frame):
        mod = self._setup_mocks()
        try:
            scorer = KGConsistencyScorer(device=torch.device("cpu"))
            scorer.clip_encoder._loaded = True
            scorer.dino_encoder._loaded = True
            score = scorer.score_frame_pair(dummy_frame, dummy_frame)
            assert isinstance(score, float)
        finally:
            self._teardown_mocks(mod)


# ═════════════════════════════════════════════════════════════════
# TESTS: KG CONDITIONER (mocked)
# ═════════════════════════════════════════════════════════════════

class TestKGConditionerMocked:
    def _setup(self):
        import ltx_pipelines.kg.kg_semantic_encoder as mod

        mock_clip = MagicMock()
        mock_clip.get_text_features.return_value = torch.randn(1, 768)
        mock_clip.get_image_features.return_value = torch.randn(1, 768)
        mock_clip_proc = MagicMock()
        mock_clip_proc.return_value = {
            "input_ids": torch.zeros(1, 10, dtype=torch.long),
            "pixel_values": torch.randn(1, 3, 224, 224),
        }

        mock_dino_out = MagicMock()
        mock_dino_out.last_hidden_state = torch.randn(1, 197, 768)
        mock_dino = MagicMock()
        mock_dino.return_value = mock_dino_out
        mock_dino_proc = MagicMock()
        mock_dino_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

        mod._clip_model = mock_clip
        mod._clip_processor = mock_clip_proc
        mod._dino_model = mock_dino
        mod._dino_processor = mock_dino_proc
        return mod

    def _teardown(self, mod):
        mod._clip_model = None
        mod._clip_processor = None
        mod._dino_model = None
        mod._dino_processor = None

    def test_encode_scene_node(self, sample_scene_node):
        mod = self._setup()
        try:
            cond = KGConditioner(device=torch.device("cpu"))
            cond.clip_encoder._loaded = True
            emb = cond.encode_scene_node(sample_scene_node)
            assert emb.shape == (1, 768)
        finally:
            self._teardown(mod)

    def test_encode_frame(self, dummy_frame):
        mod = self._setup()
        try:
            cond = KGConditioner(device=torch.device("cpu"))
            cond.dino_encoder._loaded = True
            emb = cond.encode_frame(dummy_frame)
            assert emb.shape == (1, 768)
        finally:
            self._teardown(mod)

    def test_score(self, sample_scene_node, dummy_frame):
        mod = self._setup()
        try:
            cond = KGConditioner(device=torch.device("cpu"))
            cond.clip_encoder._loaded = True
            cond.scorer.clip_encoder._loaded = True
            cond.scorer.dino_encoder._loaded = True
            s = cond.score(sample_scene_node, dummy_frame)
            assert isinstance(s, float)
        finally:
            self._teardown(mod)

    def test_unload(self):
        import ltx_pipelines.kg.kg_semantic_encoder as mod
        mod._clip_model = "placeholder"
        mod._dino_model = "placeholder"
        cond = KGConditioner()
        cond.unload()
        assert mod._clip_model is None
        assert mod._dino_model is None


# ═════════════════════════════════════════════════════════════════
# TESTS: LATENT OBSERVER (mocked)
# ═════════════════════════════════════════════════════════════════

class TestLatentObserverMocked:
    def _setup(self):
        import ltx_pipelines.kg.kg_semantic_encoder as mod
        mock_dino_out = MagicMock()
        mock_dino_out.last_hidden_state = torch.randn(1, 197, 768)
        mock_dino = MagicMock()
        mock_dino.return_value = mock_dino_out
        mock_dino_proc = MagicMock()
        mock_dino_proc.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}
        mod._dino_model = mock_dino
        mod._dino_processor = mock_dino_proc
        return mod

    def _teardown(self, mod):
        mod._dino_model = None
        mod._dino_processor = None

    def test_forward(self, dummy_frame):
        mod = self._setup()
        try:
            obs = LatentObserver(device=torch.device("cpu"))
            emb = obs(dummy_frame)
            assert emb.shape == (1, 768)
        finally:
            self._teardown(mod)

    def test_observe_batch(self, dummy_frame):
        mod = self._setup()
        try:
            obs = LatentObserver(device=torch.device("cpu"))
            frames = [dummy_frame, dummy_frame, dummy_frame]
            emb = obs.observe_batch(frames)
            assert emb.shape == (1, 768)
            norm = emb.norm(dim=-1)
            assert torch.allclose(norm, torch.ones(1), atol=1e-4)
        finally:
            self._teardown(mod)


# ═════════════════════════════════════════════════════════════════
# INTEGRATION TESTS (require model downloads — skip in CI)
# ═════════════════════════════════════════════════════════════════

@pytest.mark.slow
class TestCLIPIntegration:
    """These tests actually load CLIP. Mark with `pytest -m slow` to run."""

    def test_clip_text_encode(self):
        enc = CLIPSceneEncoder(device=torch.device("cpu"))
        emb = enc("a person in a red dress walking outside")
        assert emb.shape == (1, 768)
        norm = emb.norm(dim=-1)
        assert torch.allclose(norm, torch.ones(1), atol=1e-4)

    def test_clip_different_texts_different_embeddings(self):
        enc = CLIPSceneEncoder(device=torch.device("cpu"))
        e1 = enc("a cat sitting on a mat")
        e2 = enc("a rocket launching into space")
        sim = (e1 @ e2.T).item()
        assert sim < 0.95  # Should not be identical


@pytest.mark.slow
class TestDINOIntegration:
    def test_dino_frame_encode(self, dummy_frame):
        enc = DINOFrameEncoder(device=torch.device("cpu"))
        emb = enc(dummy_frame)
        assert emb.shape == (1, 768)

    def test_dino_pil_input(self, dummy_frame_pil):
        enc = DINOFrameEncoder(device=torch.device("cpu"))
        emb = enc(dummy_frame_pil)
        assert emb.shape == (1, 768)


@pytest.mark.slow
class TestEndToEndScoring:
    def test_conditioner_full_pipeline(self, sample_scene_node, dummy_frame):
        cond = KGConditioner(device=torch.device("cpu"))
        scene_emb = cond.encode_scene_node(sample_scene_node)
        frame_emb = cond.encode_frame(dummy_frame)
        score = cond.score(sample_scene_node, dummy_frame)
        assert scene_emb.shape == (1, 768)
        assert frame_emb.shape == (1, 768)
        assert isinstance(score, float)
        cond.unload()