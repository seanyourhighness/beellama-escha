#!/usr/bin/env bash
# P-ARCH-17 route-only control capture (profiling is not used for timing).
# Usage: escha-parch17-route.sh <label> <model.gguf> <evidence-dir>
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD='build-cuda-parch10-async'
BIN="$REPO/$BUILD/bin/llama-bench"
IDS="${ESCHA_IDS:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids}"
LABEL="$1"
MODEL="$2"
OUT="$3"

mkdir -p "$OUT"
export ESCHA_PROFILE=1
export GGML_CUDA_DISABLE_GRAPHS=1
unset ESCHA_CUBLAS_PREFILL
unset ESCHA_WMMA_PREFILL
unset ESCHA_NO_MMA
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

{
    echo "run_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gate=P-ARCH-17"
    echo "purpose=route-only; do not use profile timings for throughput"
    echo "label=$LABEL"
    echo "gpu_before=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')"
    echo "llama_bench_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "model_sha256=$(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "ids_sha256=$(sha256sum "$IDS" | cut -d' ' -f1)"
    echo "ids_count=$(wc -w < "$IDS")"
    echo "commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "command=ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 $BIN -m $MODEL --prompt-tokens-file $IDS -p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json"
} > "$OUT/run-manifest.txt"

"$BIN" -m "$MODEL" --prompt-tokens-file "$IDS" \
    -p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on \
    -r 1 -o json -oe json > "$OUT/stdout.json" 2> "$OUT/stderr.log"
RC=$?
echo "exit_code=$RC" >> "$OUT/run-manifest.txt"
echo "gpu_after=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')" >> "$OUT/run-manifest.txt"
exit "$RC"
