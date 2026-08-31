#!/bin/bash
# Run the Escha W2 x LowGPU monolithic hybrid with the CUDA build.
set -e
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
cd "$(dirname "$0")/.."
MODEL="${ESCHA_MODEL:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf}"
exec ./build-cuda/bin/llama-cli \
    -m "$MODEL" \
    -ngl 99 \
    --temp 0 --seed 42 \
    "$@"
