#!/bin/bash

LOG_FILE=vram_log.csv
SEED=13370

echo "Starting VRAM logging..."

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > $LOG_FILE &

SMI_PID=$!

python -m ltx_pipelines.distilled \
    --checkpoint-path ltx-2-19b-distilled-fp8.safetensors \
    --gemma-root gemma-3-12b-it-qat-q4_0-unquantized \
    --spatial-upsampler-path ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --prompt "Person imedeatly turns completely around facing back standing at the same place in a smooth manner. The person is strictly not talking or making any gestures or hand movements." \
    --image assets/img_1770309598_0.png 0 1.0 \
    --height 1280 \
    --width 768 \
    --num-frames 96 \
    --output-path img_1770309598_0.mp4

python -m ltx_pipelines.distilled \
    --checkpoint-path ltx-2-19b-distilled-fp8.safetensors \
    --gemma-root gemma-3-12b-it-qat-q4_0-unquantized \
    --spatial-upsampler-path ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --prompt "Person imedeatly turns completely around facing back standing at the same place in a smooth manner. The person is strictly not talking or making any gestures or hand movements." \
    --image assets/img_1770309772_0.png 0 1.0 \
    --height 1280 \
    --width 768 \
    --num-frames 96 \
    --output-path img_1770309772_0.mp4

python -m ltx_pipelines.distilled \
    --checkpoint-path ltx-2-19b-distilled-fp8.safetensors \
    --gemma-root gemma-3-12b-it-qat-q4_0-unquantized \
    --spatial-upsampler-path ltx-2-spatial-upscaler-x2-1.0.safetensors \
    --prompt "Person imedeatly turns completely around facing back standing at the same place in a smooth manner. The person is strictly not talking or making any gestures or hand movements." \
    --image assets/img_1770309817_0.png 0 1.0 \
    --height 1280 \
    --width 768 \
    --num-frames 96 \
    --output-path img_1770309817_0.mp4

kill $SMI_PID

echo "VRAM log saved to $LOG_FILE"