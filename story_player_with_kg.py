"""
🎬 Story Player with Live 3D Knowledge Graph
- Shows all scene videos in a grid (click to play)
- Selecting a scene shows the cumulative KG up to that scene
- 3D interactive Plotly graph updates when scene selection changes
"""

import json
import time
import os
from pathlib import Path
from collections import defaultdict

import streamlit as st
import numpy as np
import networkx as nx
import plotly.graph_objects as go


# =============================
# CONFIG
# =============================
JSON_PATH = "story_outputs_backup/final_story_kg.json"
VIDEO_DIR = "story_outputs_backup"

# Node colors by type
NODE_COLORS = {
    "story": "#FF6B6B",       # Red
    "scene": "#4ECDC4",       # Teal
    "character": "#45B7D1",   # Blue
    "outfit": "#96CEB4",      # Green
    "motion": "#FFEAA7",      # Yellow
    "environment": "#DDA0DD", # Plum
    "attribute": "#FFB347",   # Orange
    "shared": "#FF69B4",      # Hot pink — shared attributes
}

NODE_SIZES = {
    "story": 18,
    "scene": 14,
    "character": 10,
    "outfit": 10,
    "motion": 12,
    "environment": 10,
    "attribute": 7,
    "shared": 9,
}


# =============================
# LOAD STORY
# =============================
@st.cache_data
def load_story():
    with open(JSON_PATH, "r") as f:
        return json.load(f)


# =============================
# EXTRACT SCENE ATTRIBUTES
# =============================
def extract_scene_attributes(scene_data):
    """Extract meaningful attributes from a scene for KG nodes."""
    attrs = {}

    # Character attributes
    char = scene_data.get("character", {})
    if char.get("gender") and char["gender"] != "unknown":
        attrs["gender"] = char["gender"]
    if char.get("age_group") and char["age_group"] != "unknown":
        attrs["age_group"] = char["age_group"]
    if char.get("body_type"):
        attrs["body_type"] = char["body_type"]
    if char.get("hair_color"):
        attrs["hair_color"] = char["hair_color"]
    if char.get("hair_style") and char["hair_style"] != "unknown":
        attrs["hair_style"] = char["hair_style"]

    # Outfit attributes
    outfit = scene_data.get("outfit", {})
    if outfit.get("style"):
        attrs["style"] = outfit["style"]
    if outfit.get("formality"):
        attrs["formality"] = outfit["formality"]
    if outfit.get("top_color"):
        attrs["top_color"] = outfit["top_color"]
    if outfit.get("top_type"):
        attrs["top_type"] = outfit["top_type"]
    if outfit.get("bottom_type"):
        attrs["bottom_type"] = outfit["bottom_type"]

    # Pose/motion attributes
    pose = scene_data.get("pose", {})
    if pose.get("motion_type") and pose["motion_type"] != "unknown":
        attrs["motion"] = pose["motion_type"]
    if pose.get("motion_speed"):
        attrs["speed"] = pose["motion_speed"]

    # Environment attributes
    env = scene_data.get("environment", {})
    if env.get("lighting_type") and env["lighting_type"] != "unknown":
        attrs["lighting"] = env["lighting_type"]
    if env.get("background_type") and env["background_type"] != "unknown":
        attrs["background"] = env["background_type"]
    if env.get("camera_distance"):
        attrs["camera"] = env["camera_distance"]

    return attrs


def categorize_attribute(attr_key):
    """Return node type for an attribute key."""
    if attr_key in ("gender", "age_group", "body_type", "hair_color", "hair_style"):
        return "character"
    elif attr_key in ("style", "formality", "top_color", "top_type", "bottom_type"):
        return "outfit"
    elif attr_key in ("motion", "speed"):
        return "motion"
    elif attr_key in ("lighting", "background", "camera"):
        return "environment"
    return "attribute"


# =============================
# BUILD RICH KG FOR SCENES
# =============================
def build_rich_graph(scenes_data, scene_order, up_to_scene_idx):
    """Build cumulative knowledge graph up to scene_order[up_to_scene_idx]."""
    G = nx.DiGraph()
    G.add_node("📖 Story", node_type="story", label="📖 Story")

    # Track which attribute values map to which scenes
    # attr_value → list of scene_ids
    attr_to_scenes = defaultdict(list)

    active_scenes = scene_order[:up_to_scene_idx + 1]

    for idx, scene_name in enumerate(active_scenes):
        scene = scenes_data.get(scene_name, {})
        attrs = extract_scene_attributes(scene)

        # Scene label
        motion_type = attrs.get("motion", "static")
        scene_label = f"🎬 {idx}: {motion_type}"
        G.add_node(scene_label, node_type="scene", label=scene_label, scene_idx=idx)

        # Story → Scene
        G.add_edge("📖 Story", scene_label, relation="contains", edge_type="structural")

        # Temporal edge to previous scene
        if idx > 0:
            prev_scene_name = active_scenes[idx - 1]
            prev_motion = extract_scene_attributes(scenes_data.get(prev_scene_name, {})).get("motion", "static")
            prev_label = f"🎬 {idx-1}: {prev_motion}"
            G.add_edge(prev_label, scene_label, relation="→ next", edge_type="temporal")

        # Add attribute nodes
        for attr_key, attr_val in attrs.items():
            if attr_val is None:
                continue

            category = categorize_attribute(attr_key)

            # Unique node per attribute value (shared across scenes)
            attr_node_id = f"{attr_key}:{attr_val}"

            # Emoji per category
            emoji_map = {
                "character": "👤",
                "outfit": "👗",
                "motion": "🏃",
                "environment": "🌍",
                "attribute": "📌",
            }
            emoji = emoji_map.get(category, "📌")
            attr_label = f"{emoji} {attr_key}={attr_val}"

            # Track for shared detection
            attr_to_scenes[attr_node_id].append(scene_label)

            if not G.has_node(attr_node_id):
                # First time seeing this attribute value
                G.add_node(
                    attr_node_id,
                    node_type=category,
                    label=attr_label,
                    attr_key=attr_key,
                    attr_val=attr_val,
                )

            # Scene → Attribute
            G.add_edge(scene_label, attr_node_id, relation=f"has_{attr_key}", edge_type="attribute")

    # Mark shared attributes (connected to multiple scenes)
    for attr_node_id, connected_scenes in attr_to_scenes.items():
        if len(connected_scenes) > 1:
            # This attribute is shared — mark it
            G.nodes[attr_node_id]["node_type"] = "shared"
            G.nodes[attr_node_id]["shared_count"] = len(connected_scenes)

    return G


# =============================
# 3D GRAPH RENDER
# =============================
def graph_to_plotly_3d(G, active_scene_idx=0):
    """Render graph as interactive 3D Plotly figure."""

    if len(G.nodes) == 0:
        fig = go.Figure()
        fig.add_annotation(text="Select a scene to view the Knowledge Graph",
                           showarrow=False, font=dict(size=18, color="white"))
        fig.update_layout(height=650, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    # Layout with fixed seed for stability
    pos = nx.spring_layout(G, dim=3, seed=42, k=2.0, iterations=80)

    # Separate nodes by type for different traces
    node_groups = defaultdict(lambda: {"x": [], "y": [], "z": [], "text": [], "hover": [], "size": []})

    for node in G.nodes():
        data = G.nodes[node]
        x, y, z = pos[node]
        node_type = data.get("node_type", "attribute")
        label = data.get("label", str(node))

        # Hover info
        hover_parts = [f"<b>{label}</b>"]
        if data.get("shared_count"):
            hover_parts.append(f"Shared across {data['shared_count']} scenes")
        if data.get("scene_idx") is not None:
            hover_parts.append(f"Scene #{data['scene_idx']}")

        node_groups[node_type]["x"].append(x)
        node_groups[node_type]["y"].append(y)
        node_groups[node_type]["z"].append(z)
        node_groups[node_type]["text"].append(label)
        node_groups[node_type]["hover"].append("<br>".join(hover_parts))
        node_groups[node_type]["size"].append(NODE_SIZES.get(node_type, 8))

    # Edge traces — separate by type
    edge_traces = []

    # Temporal edges (thick, dashed)
    temporal_x, temporal_y, temporal_z = [], [], []
    attribute_x, attribute_y, attribute_z = [], [], []
    shared_x, shared_y, shared_z = [], [], []

    for u, v, data in G.edges(data=True):
        x0, y0, z0 = pos[u]
        x1, y1, z1 = pos[v]
        edge_type = data.get("edge_type", "attribute")

        # Check if this edge connects to a shared node
        u_shared = G.nodes[u].get("node_type") == "shared"
        v_shared = G.nodes[v].get("node_type") == "shared"

        if edge_type == "temporal":
            temporal_x += [x0, x1, None]
            temporal_y += [y0, y1, None]
            temporal_z += [z0, z1, None]
        elif u_shared or v_shared:
            shared_x += [x0, x1, None]
            shared_y += [y0, y1, None]
            shared_z += [z0, z1, None]
        else:
            attribute_x += [x0, x1, None]
            attribute_y += [y0, y1, None]
            attribute_z += [z0, z1, None]

    traces = []

    # Attribute edges (thin, grey)
    if attribute_x:
        traces.append(go.Scatter3d(
            x=attribute_x, y=attribute_y, z=attribute_z,
            mode="lines", line=dict(width=1, color="rgba(150,150,150,0.3)"),
            hoverinfo="none", showlegend=False,
        ))

    # Shared attribute edges (medium, pink)
    if shared_x:
        traces.append(go.Scatter3d(
            x=shared_x, y=shared_y, z=shared_z,
            mode="lines", line=dict(width=3, color="#FF69B4"),
            hoverinfo="none", name="🔗 Shared Attributes",
        ))

    # Temporal edges (thick, white)
    if temporal_x:
        traces.append(go.Scatter3d(
            x=temporal_x, y=temporal_y, z=temporal_z,
            mode="lines", line=dict(width=5, color="white"),
            hoverinfo="none", name="⏩ Scene Flow",
        ))

    # Node traces by type
    for node_type, group in node_groups.items():
        color = NODE_COLORS.get(node_type, "#CCCCCC")
        name_map = {
            "story": "📖 Story",
            "scene": "🎬 Scene",
            "character": "👤 Character",
            "outfit": "👗 Outfit",
            "motion": "🏃 Motion",
            "environment": "🌍 Environment",
            "shared": "🔗 Shared",
            "attribute": "📌 Attribute",
        }
        traces.append(go.Scatter3d(
            x=group["x"], y=group["y"], z=group["z"],
            mode="markers+text",
            text=group["text"], hovertext=group["hover"], hoverinfo="text",
            textposition="top center", textfont=dict(size=9, color="white"),
            marker=dict(size=group["size"], color=color, opacity=0.9,
                        line=dict(width=1, color="white")),
            name=name_map.get(node_type, node_type),
        ))

    fig = go.Figure(data=traces)
    fig.update_layout(
        height=650,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        scene=dict(
            xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False),
            bgcolor="rgba(17,17,17,1)",
        ),
        legend=dict(font=dict(color="white", size=11), bgcolor="rgba(30,30,30,0.8)",
                    bordercolor="rgba(100,100,100,0.5)", borderwidth=1),
        title=dict(
            text=f"Knowledge Graph — Scenes 1 → {active_scene_idx + 1}",
            font=dict(color="white", size=16),
        ),
    )
    return fig


# =============================
# STREAMLIT UI
# =============================
st.set_page_config(layout="wide", page_title="🎬 Story KG Player")

st.markdown("""
<style>
    .stApp { background-color: #111111; }
    h1, h2, h3, p, span, label, .stMarkdown { color: white !important; }
    .scene-card {
        background: rgba(30,30,30,0.95);
        border: 2px solid #333;
        border-radius: 10px;
        padding: 10px;
        margin: 4px 0;
        text-align: center;
    }
    .scene-card-active {
        background: rgba(30,30,30,0.95);
        border: 2px solid #4ECDC4;
        border-radius: 10px;
        padding: 10px;
        margin: 4px 0;
        text-align: center;
        box-shadow: 0 0 15px rgba(78, 205, 196, 0.3);
    }
    .motion-tag {
        display: inline-block;
        background: rgba(78, 205, 196, 0.2);
        border: 1px solid #4ECDC4;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8em;
        margin: 2px;
    }
    .shared-tag {
        display: inline-block;
        background: rgba(255, 105, 180, 0.2);
        border: 1px solid #FF69B4;
        border-radius: 4px;
        padding: 2px 8px;
        font-size: 0.8em;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎬 Story Player — Select Scene → View Cumulative KG")

story = load_story()
scenes = story.get("scenes", {})
order = story.get("generation_order", [])

if not order:
    st.error("No scenes found in story JSON")
    st.stop()

# =============================
# SESSION STATE
# =============================
if "selected_scene_idx" not in st.session_state:
    st.session_state.selected_scene_idx = 0

# =============================
# SIDEBAR — SCENE LIST
# =============================
with st.sidebar:
    st.header("📋 Scene Overview")
    st.caption(f"**Story:** {story.get('title', 'Untitled')}")
    st.caption(f"**Scenes:** {len(order)}")
    st.divider()

    for idx, scene_name in enumerate(order):
        scene = scenes.get(scene_name, {})
        pose = scene.get("pose", {})
        motion = pose.get("motion_type", "unknown")
        emoji_map = {
            "static": "🧍", "squatting": "🏋️", "raising_arms": "🙌",
            "looking_around": "👀", "leg_raise": "🦵", "waving": "👋",
            "bending": "🙇", "walking": "🚶", "turning": "🔄",
            "dancing": "💃", "running": "🏃",
        }
        emoji = emoji_map.get(motion, "🎬")

        is_selected = (idx == st.session_state.selected_scene_idx)
        prefix = "▶️ " if is_selected else "   "

        if st.button(f"{prefix}{idx + 1}. {emoji} {motion}", key=f"sidebar_{idx}",
                      use_container_width=True,
                      type="primary" if is_selected else "secondary"):
            st.session_state.selected_scene_idx = idx
            st.rerun()

# =============================
# TOP: VIDEO GRID (all 8 scenes)
# =============================
st.subheader("🎥 All Scenes — Click any video to play")

# Determine grid layout: 4 columns
num_cols = 4
rows = (len(order) + num_cols - 1) // num_cols

for row_idx in range(rows):
    cols = st.columns(num_cols)
    for col_idx in range(num_cols):
        scene_idx = row_idx * num_cols + col_idx
        if scene_idx >= len(order):
            break

        scene_name = order[scene_idx]
        scene = scenes.get(scene_name, {})
        pose = scene.get("pose", {})
        motion = pose.get("motion_type", "unknown")

        emoji_map = {
            "static": "🧍", "squatting": "🏋️", "raising_arms": "🙌",
            "looking_around": "👀", "leg_raise": "🦵", "waving": "👋",
            "bending": "🙇", "walking": "🚶", "turning": "🔄",
            "dancing": "💃", "running": "🏃",
        }
        emoji = emoji_map.get(motion, "🎬")

        is_selected = (scene_idx == st.session_state.selected_scene_idx)

        with cols[col_idx]:
            # Scene selection button
            btn_label = f"{'▶️ ' if is_selected else ''}{emoji} Scene {scene_idx + 1}: {motion}"
            if st.button(btn_label, key=f"grid_btn_{scene_idx}",
                         use_container_width=True,
                         type="primary" if is_selected else "secondary"):
                st.session_state.selected_scene_idx = scene_idx
                st.rerun()

            # Video player
            video_path = os.path.join(VIDEO_DIR, f"{scene_name}.mp4")
            if os.path.exists(video_path):
                st.video(video_path)
            else:
                st.warning(f"Not found")

st.divider()

# =============================
# BOTTOM: KG + SCENE DETAILS
# =============================
selected_idx = st.session_state.selected_scene_idx
selected_name = order[selected_idx]
selected_scene = scenes.get(selected_name, {})
selected_attrs = extract_scene_attributes(selected_scene)
selected_motion = selected_attrs.get("motion", "static")

left, right = st.columns([1, 1])

# ─── LEFT: Knowledge Graph ───
with left:
    st.subheader(f"🧠 Cumulative KG — Scenes 1 → {selected_idx + 1}")

    G = build_rich_graph(scenes, order, selected_idx)
    fig = graph_to_plotly_3d(G, active_scene_idx=selected_idx)
    st.plotly_chart(fig, use_container_width=True, key=f"kg_main_{selected_idx}")

    num_shared = sum(1 for n in G.nodes() if G.nodes[n].get("node_type") == "shared")
    st.markdown(
        f"**Nodes:** {len(G.nodes)} · **Edges:** {len(G.edges)} · "
        f"**🔗 Shared attributes:** {num_shared}"
    )

# ─── RIGHT: Scene Details ───
with right:
    st.subheader(f"📋 Scene {selected_idx + 1} Details")

    # Motion type
    emoji_map = {
        "static": "🧍", "squatting": "🏋️", "raising_arms": "🙌",
        "looking_around": "👀", "leg_raise": "🦵", "waving": "👋",
        "bending": "🙇", "walking": "🚶", "turning": "🔄",
        "dancing": "💃", "running": "🏃",
    }
    motion_emoji = emoji_map.get(selected_motion, "🎬")
    st.markdown(f"### {motion_emoji} {selected_motion}")
    st.caption(f"`{selected_name}`")

    # Attributes grouped by category
    st.markdown("#### 👤 Character")
    char_attrs = {k: v for k, v in selected_attrs.items() if categorize_attribute(k) == "character"}
    if char_attrs:
        for k, v in char_attrs.items():
            st.markdown(f"- **{k}:** `{v}`")
    else:
        st.caption("No character attributes")

    st.markdown("#### 👗 Outfit")
    outfit_attrs = {k: v for k, v in selected_attrs.items() if categorize_attribute(k) == "outfit"}
    if outfit_attrs:
        for k, v in outfit_attrs.items():
            st.markdown(f"- **{k}:** `{v}`")
    else:
        st.caption("No outfit attributes")

    st.markdown("#### 🏃 Motion")
    motion_attrs = {k: v for k, v in selected_attrs.items() if categorize_attribute(k) == "motion"}
    if motion_attrs:
        for k, v in motion_attrs.items():
            st.markdown(f"- **{k}:** `{v}`")
    else:
        st.caption("No motion attributes")

    st.markdown("#### 🌍 Environment")
    env_attrs = {k: v for k, v in selected_attrs.items() if categorize_attribute(k) == "environment"}
    if env_attrs:
        for k, v in env_attrs.items():
            st.markdown(f"- **{k}:** `{v}`")
    else:
        st.caption("No environment attributes")

    # Detailed prompt
    detailed = selected_scene.get("detailed_prompt", "")
    if detailed:
        st.markdown("#### 📝 Generation Prompt")
        st.code(detailed, language=None)

    # Shared attributes with other scenes
    st.markdown("#### 🔗 Shared With Other Scenes")
    shared_with = []
    for other_idx, other_name in enumerate(order):
        if other_idx == selected_idx:
            continue
        if other_idx > selected_idx:
            break
        other_scene = scenes.get(other_name, {})
        other_attrs = extract_scene_attributes(other_scene)
        common = set(selected_attrs.items()) & set(other_attrs.items())
        if common:
            common_strs = [f"`{k}={v}`" for k, v in common]
            shared_with.append(f"- **Scene {other_idx + 1}** ({other_attrs.get('motion', '?')}): {', '.join(common_strs)}")

    if shared_with:
        for s in shared_with:
            st.markdown(s)
    else:
        st.caption("No shared attributes yet (first scene)")

# =============================
# BOTTOM: Full attribute table
# =============================
st.divider()
st.subheader("📊 Scene Attribute Comparison")

table_data = []
for idx, scene_name in enumerate(order):
    scene = scenes.get(scene_name, {})
    attrs = extract_scene_attributes(scene)
    row = {"#": idx + 1, "Scene": scene_name}
    row.update(attrs)
    table_data.append(row)

st.dataframe(table_data, use_container_width=True)
