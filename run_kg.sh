#!/bin/bash

LOG_FILE=vram_log_kg.csv
SEED=13370

echo "Starting VRAM logging..."

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > $LOG_FILE &

SMI_PID=$!

python -m ltx_pipelines.distilled_kg \
    --checkpoint-path ltx-2-19b-distilled-fp8.safetensors \
    --gemma-root gemma-3-12b-it-qat-q4_0-unquantized \
    --spatial-upsampler-path ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --prompt "A man walks and remembers previous scene context" \
    --height 1280 \
    --width 768 \
    --num-frames 96 \
    --output-path kg_output.mp4

kill $SMI_PID
