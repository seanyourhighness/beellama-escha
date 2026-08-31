#!/bin/bash
# Generation smoke test for the Escha W2 x LowGPU monolithic hybrid (CUDA build).
# CUDA graphs are disabled: graph capture of the escha ops currently hangs.
set -e
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH}
cd "$(dirname "$0")/.."
# external data dir overrideable via env; documented default reflects local layout
MODEL="${ESCHA_MODEL:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf}"
if [ "$1" = "--no-graphs" ]; then
    shift
    export GGML_CUDA_DISABLE_GRAPHS=1
fi
exec ./build-cuda/bin/llama-cli \
    -m "$MODEL" \
    -ngl 99 -c 512 \
    --temp 0 --seed 42 \
    "$@"
