#!/bin/bash
set -e
export PATH=/usr/local/cuda/bin:$PATH
cd "$(dirname "$0")/.."
cmake --build build-cuda -j 12 --target llama-cli llama-server
