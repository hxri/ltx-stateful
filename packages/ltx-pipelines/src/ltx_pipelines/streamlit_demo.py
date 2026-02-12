############################################################
# MUST BE FIRST — BEFORE torch import
############################################################
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:512"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

############################################################
# SAFE IMPORTS
############################################################
import streamlit as st
import gc
import logging
from pathlib import Path
from datetime import datetime

############################################################
# STREAMLIT CONFIG
############################################################
st.set_page_config(page_title="KG Video Generation", layout="wide")
st.title("🎬 KG Continual Video Generation (Safe Mode)")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("KG_STREAMLIT")

############################################################
# SESSION STATE
############################################################
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if "pipeline_loading" not in st.session_state:
    st.session_state.pipeline_loading = False

if "history" not in st.session_state:
    st.session_state.history = []

if "output_dir" not in st.session_state:
    st.session_state.output_dir = Path("./kg_outputs")
    st.session_state.output_dir.mkdir(exist_ok=True, parents=True)

############################################################
# GPU SAFE IMPORT FUNCTION
############################################################
def lazy_import_pipeline():
    logger.info("Lazy importing pipeline...")
    from ltx_pipelines.distilled_kg import DistilledPipelineKG
    return DistilledPipelineKG


############################################################
# SAFE PIPELINE CREATION
############################################################
def get_or_create_pipeline():

    if st.session_state.pipeline is not None:
        return st.session_state.pipeline

    if st.session_state.pipeline_loading:
        raise RuntimeError("Pipeline already initializing")

    st.session_state.pipeline_loading = True

    import torch

    try:
        logger.info("Initializing CUDA context once...")
        if torch.cuda.is_available():
            torch.cuda.init()

        logger.info("Importing pipeline class...")
        PipelineClass = lazy_import_pipeline()

        logger.info("Constructing pipeline instance...")
        pipe = PipelineClass(
            checkpoint_path="ltx-2-19b-distilled-fp8.safetensors",
            gemma_root="gemma-3-12b-it-qat-q4_0-unquantized",
            spatial_upsampler_path="ltx-2-spatial-upscaler-x2-1.0.safetensors",
            loras=[],
        )

        st.session_state.pipeline = pipe

        logger.info("Pipeline ready.")
        return pipe

    finally:
        st.session_state.pipeline_loading = False


############################################################
# GENERATION FUNCTION
############################################################
def generate_video(prompt, seed, height, width, frames, scene_id):

    import torch

    pipe = get_or_create_pipeline()

    output_path = st.session_state.output_dir / f"{scene_id}.mp4"

    # ← NEW: Use last frame from previous scene if available
    use_prev_frame = len(st.session_state.history) > 0

    pipe.generate_and_track(
        scene_id=scene_id,
        prompt=prompt,
        seed=seed,
        height=height,
        width=width,
        num_frames=frames,
        frame_rate=30.0,
        images=[],
        output_path=str(output_path),
        use_previous_frame=use_prev_frame,  # ← NEW
    )

    st.session_state.history.append({
        "scene": scene_id,
        "prompt": prompt,
        "path": str(output_path),
        "time": datetime.now().isoformat(),
    })

    return output_path


############################################################
# UI
############################################################
col1, col2 = st.columns([2, 1])

with col1:

    st.header("Scene Generation")

    scene_id = st.text_input("Scene ID", f"scene_{len(st.session_state.history)}")

    prompt = st.text_area(
        "Prompt",
        "A person walking and turning around smoothly",
        height=120
    )

    seed = st.number_input("Seed", 0, 999999, 1337)
    frames = st.number_input("Frames", 9, 257, 25, step=8)

    height = st.number_input("Height", 64, 2048, 512, step=64)
    width = st.number_input("Width", 64, 2048, 512, step=64)

    if st.button("🎬 Generate", use_container_width=True):

        if height % 64 != 0 or width % 64 != 0:
            st.error("Resolution must be divisible by 64")

        elif (frames - 1) % 8 != 0:
            st.error("(frames - 1) must be divisible by 8")

        else:
            with st.spinner("Generating video..."):
                try:
                    out = generate_video(prompt, seed, height, width, frames, scene_id)
                    st.success("Generation complete")
                    st.video(str(out))

                except Exception as e:
                    st.error(str(e))
                    logger.exception(e)


############################################################
# HISTORY PANEL
############################################################
with col2:

    st.header("History")

    for h in reversed(st.session_state.history[-10:]):
        st.caption(h["scene"])
        try:
            st.video(h["path"])
        except:
            pass


############################################################
# GPU STATUS (SAFE — NO INIT)
############################################################
st.divider()
st.subheader("GPU Status")

try:
    import torch

    if torch.cuda.is_available():
        alloc = torch.cuda.memory_allocated()/1024**3
        res = torch.cuda.memory_reserved()/1024**3
        total = torch.cuda.get_device_properties(0).total_memory/1024**3

        st.write(f"Allocated: {alloc:.2f} GB")
        st.write(f"Reserved: {res:.2f} GB")
        st.write(f"Total: {total:.2f} GB")

except:
    st.caption("GPU info unavailable")


############################################################
# RESET
############################################################
st.divider()

if st.button("🧹 Reset GPU + Pipeline"):

    import torch

    st.session_state.pipeline = None

    gc.collect()

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    st.success("Reset complete")
