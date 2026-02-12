#!/bin/bash
# filepath: /workspace/LTX-2/run_streamlit_app.sh

# CRITICAL: Set BEFORE anything else
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Install dependencies
pip install -q -r streamlit_requirements.txt

echo "============================================"
echo "  Stateful KG Video Generation"
echo "============================================"
echo ""
echo "Arch: Pipeline cached once, reused per generation"
echo "Run: streamlit run ..streamlit_demo.py"
echo ""

# Point to the correct streamlit_demo.py file
streamlit run packages/ltx-pipelines/src/ltx_pipelines/streamlit_demo.py \
    --logger.level=info \
    --theme.base="dark" \
    --client.showErrorDetails=true