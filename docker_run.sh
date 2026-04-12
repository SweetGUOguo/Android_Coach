#!/usr/bin/env bash
# run.sh

IMAGE="androidcoach:latest"
DATA_DIR="${1:-./data}"
OUTPUT_DIR="${2:-./results}"

mkdir -p "$DATA_DIR" "$OUTPUT_DIR"

docker run --rm -it \
  --gpus all \
  --shm-size 8G \
  -v "$(pwd)/$DATA_DIR:/data:ro" \
  -v "$(pwd)/$OUTPUT_DIR:/output" \
  -v "$(pwd):/workspace" \
  -w /workspace \
  --user $(id -u):$(id -g) \
  "$IMAGE" \
  /bin/bash -c "cd /workspace && exec bash"
