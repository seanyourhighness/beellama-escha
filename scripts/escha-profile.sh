#!/bin/bash
# Profile the CUDA activity of a short generation run to find the hang.
cd "$(dirname "$0")/.."
rm -f /tmp/prof.nsys-rep
timeout 150 /usr/local/cuda/bin/nsys profile -o /tmp/prof --force-overwrite true -t cuda --duration 100 \
  ./build-cuda/bin/llama-cli -m "${ESCHA_MODEL:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf}" \
  -ngl 99 -c 512 -p 'Hello' -n 4 --temp 0 --seed 42 --no-display-prompt > /tmp/prof.log 2>&1
echo "nsys exit: $?"
