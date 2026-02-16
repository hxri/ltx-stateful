# LTX-Stateful: Knowledge-Graph Integrated Video Generation

This repository extends the **LTX-2** Audio-Video generation framework with **stateful storytelling capabilities**. By integrating a dynamic Knowledge Graph (KG), Semantic Encoders (CLIP/DINOv2), and a Latent Observer, this project transforms LTX from a single-shot generator into a continuous video storytelling agent.

It solves the "amnesia" problem in generative video by persisting character attributes, enhancing temporal consistency via latents, and enforcing semantic alignment across sequential scenes.

---

## 🚀 Key Features

*   **Stateful Generation**: Maintains a `SimpleSceneGraph` or `DynamicSceneGraph` to track generated assets, scene order, and transition logic.
*   **Visual Continuity**: Automatically passes the last frame of Scene $N$ as the conditioning latent for Scene $N+1$ to ensure smooth cuts.
*   **Semantic Scoring**: Uses CLIP (Text $\leftrightarrow$ Image) and DINOv2 (Frame $\leftrightarrow$ Frame) to score generations on consistency and temporal flow.
*   **Latent Observation**: Extracts and stores embeddings of generated content back into the KG, preparing datasets for future LoRA fine-tuning or Reinforcement Learning (RL).
*   **Live Visualization**: Real-time rendering of the story graph as generation proceeds.
*   **Interactive UI**: A Streamlit-based interface for interactive, node-by-node story generation.

---

## 📂 File Structure

The workspace is organized into three main packages, with the primary stateful innovations located in `ltx-pipelines`.

```text
.
├── packages/
│   ├── ltx-pipelines/           # INFERENCE & KG LOGIC
│   │   ├── src/ltx_pipelines/
│   │   │   ├── distilled_kg.py          # Base pipeline with visual state tracking
│   │   │   ├── distilled_kg_semantic.py # Extended pipeline with CLIP/DINO scoring
│   │   │   ├── story_driver_semantic.py # Orchestrator for multi-scene stories
│   │   │   ├── streamlit_demo.py        # Interactive Web UI
│   │   │   └── kg/                      # KNOWLEDGE GRAPH MODULES
│   │   │       ├── kg_schema.py         # Data classes (SceneNode, StoryState)
│   │   │       ├── kg_semantic_encoder.py # CLIP/DINOv2 wrappers
│   │   │       ├── latent_observer.py   # Extracts embeddings from outputs
│   │   │       ├── dynamic_graph.py     # Graph logic with memory decay
│   │   │       ├── kg_prompt_generator.py # Formats prompts (Motion -> Actor -> Env)
│   │   │       └── kg_live_visualizer.py # Real-time graph renderer
│   │
│   ├── ltx-trainer/             # TRAINING TOOLS
│   │   ├── scripts/             # Data processing, captioning, and training scripts
│   │   └── ...
│   │
│   └── ltx-core/                # BASE MODELS
│       └── src/ltx_core/model/  # Transformer, VAE, Tokenizer definitions
│
├── run_kg.sh                    # Script: Run a single stateful generation
├── run_story_kg.sh              # Script: Run a full story sequence + Visualizer
├── run_streamlit_app.sh         # Script: Launch interactive UI
└── vram_log.csv                 # GPU memory profiling logs
```

---

## 🏗️ Architecture Overview

### 1. The Distilled KG Pipeline
Located in `packages/ltx-pipelines/src/ltx_pipelines/distilled_kg.py`, this class subclass `DistilledPipeline`.
*   **Logic**: Instead of just returning a video, it saves the video artifacts and updates an internal graph with the path to the *last generated frame*.
*   **Statefulness**: When `generate_and_track` is called for the next scene, it automatically retrieves that last frame, encodes it, and injects it as a "Guiding Latent" (STG - Spatio-Temporal Guidance) into the diffusion process.

### 2. Semantic Observer & Scoring
Located in `packages/ltx-pipelines/src/ltx_pipelines/distilled_kg_semantic.py`.
*   **Conditioning**: Uses `KGConditioner` to encode prompts via CLIP.
*   **Observation**: After generation, `LatentObserver` runs DINOv2 on the output.
*   **Feedback**: Consistency scores are written to `kg_state.json`. High consistency scores indicate the model adhered to the "Character Attributes" defined in the KG.

### 3. Knowledge Graph Schema
Located in `packages/ltx-pipelines/src/ltx_pipelines/kg/kg_schema.py`.
*   **Nodes**: `SceneNode` containing `character`, `environment`, `pose`, and `weather`.
*   **Edges**: `SceneEdge` representing transitions.
*   **State**: `StoryState` serializes the entire session to JSON, allowing pause/resume functionality.

---

## 🏃 Usage & Scripts

Ensure you have the LTX-2 model checkpoint and Gemma tokenizer downloaded.

### 1. Run Interactive UI
The best way to explore the stateful capabilities is via the Streamlit app.

```bash
./run_streamlit_app.sh
```
*   **Features**: Edit scene nodes visually, generate scenes one by one, view consistency scores, and see the graph build up in real-time.

### 2. Run Autonomous Story
To run a pre-defined sequence of scenes (defined in the driver script) with the live graph visualizer running in the background:

```bash
./run_story_kg.sh
```
*   **Output**: Generates MP4s in `story_outputs/` and updates `final_story_kg.json`.
*   **Visualization**: A window will pop up showing the directed graph of scenes.

### 3. Run Single KG Generation
To test specific prompt engineering or continuity settings manually:

```bash
./run_kg.sh
```

---

## 🛠️ Technical Details & Customization

### Prompt Engineering
The system uses `DetailedPromptGenerator` (`packages/ltx-pipelines/src/ltx_pipelines/kg/kg_prompt_generator.py`) to structure prompts specifically for LTX-2.
*   **Logic**: LTX-2 follows instructions better when motion verbs come first.
*   **Format**: `[Motion Description] + [Character Visuals] + [Environment] + [Lighting]`
*   **Customization**: Modify `_format_pose` or `_format_character` in this file to change how the KG attributes are converted to text.

### Training Data Generation
This framework effectively functions as a data engine for **IC-LoRA (In-Context LoRA)** training.
1.  Run a story using `run_story_kg.sh`.
2.  The system saves:
    *   The video (`.mp4`)
    *   The text prompt
    *   The embedding of the generated result
    *   The score
3.  Use `packages/ltx-trainer/scripts/process_dataset.py` pointing to the generated `final_story_kg.json` to prepare a dataset for fine-tuning the model on specific characters.

### Memory Management
The project includes VRAM protection:
*   `run_streamlit_app.sh` sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`.
*   Pipelines explicitly call `cleanup_memory()` between heavy encoding/decoding stages.
*   Adjust `height` and `width` in the driver scripts if OOM errors occur (default 768x512).

---

## 📦 Requirements

*   **Python**: 3.10+
*   **CUDA**: 12.1+ (Recommended for FP8 Flash Attention)
*   **Dependencies**:
    *   `torch`, `diffusers`, `transformers`
    *   `sentence-transformers` (for CLIP/DINO)
    *   `networkx`, `matplotlib`, `opencv-python` (for Visualizer)
    *   `streamlit`, `streamlit-agraph` (for UI)
