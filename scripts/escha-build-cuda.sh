#!/bin/bash
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
# Build the target Ada path (RTX 4070 Ti, SM89) alongside the local Blackwell
# validation path.  Override this for a smaller, single-architecture build.
: "${ESCHA_CUDA_ARCHITECTURES:=89;120}"
cmake -S . -B build-cuda -G Ninja \
    -DGGML_CUDA=ON \
    "-DCMAKE_CUDA_ARCHITECTURES=${ESCHA_CUDA_ARCHITECTURES}" \
    -DGGML_NATIVE=OFF \
    -DCMAKE_BUILD_TYPE=Release
cmake --build build-cuda -j 12 --target llama-cli llama-server
