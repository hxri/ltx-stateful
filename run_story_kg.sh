#!/bin/bash

LOG_FILE=vram_log.csv

echo "Starting VRAM logging..."

nvidia-smi \
  --query-gpu=timestamp,memory.used,memory.total,utilization.gpu \
  --format=csv -l 1 > $LOG_FILE &

SMI_PID=$!

echo "Starting KG Visualizer..."
python -m ltx_pipelines.kg.kg_live_visualizer &
VIS_PID=$!

echo "Running Story + Generations..."
python packages/ltx-pipelines/src/ltx_pipelines/story_driver.py

kill $VIS_PID
kill $SMI_PID

echo "Done."
