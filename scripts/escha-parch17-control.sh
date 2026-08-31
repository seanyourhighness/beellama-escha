#!/usr/bin/env bash
# P-ARCH-17 original-Escha-W2 vs LowGPU-hybrid Bee control.
#
# Usage: escha-parch17-control.sh <label> <model.gguf> <evidence-dir> [reps]
# Runs the immutable shared 2,048-ID prefill with the P-ARCH-15 candidate
# binary.  It writes reproducibility metadata and the unmodified bench streams.
set -u

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD='build-cuda-parch10-async'
BIN="$REPO/$BUILD/bin/llama-bench"
IDS="${ESCHA_IDS:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids}"

LABEL="$1"
MODEL="$2"
OUT="$3"
REPS="${4:-4}"

mkdir -p "$OUT"
unset ESCHA_PROFILE
unset GGML_CUDA_DISABLE_GRAPHS
unset ESCHA_CUBLAS_PREFILL
unset ESCHA_WMMA_PREFILL
unset ESCHA_NO_MMA
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

{
    echo "run_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "gate=P-ARCH-17"
    echo "label=$LABEL"
    echo "build_dir=$BUILD"
    echo "repetitions=$REPS"
    echo "gpu_before=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')"
    echo "llama_bench_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "libggml_cuda_sha256=$(sha256sum "$REPO/$BUILD/bin/libggml-cuda.so.0.19.0" | cut -d' ' -f1)"
    echo "model_path=$MODEL"
    echo "model_sha256=$(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "ids_path=$IDS"
    echo "ids_sha256=$(sha256sum "$IDS" | cut -d' ' -f1)"
    echo "ids_count=$(wc -w < "$IDS")"
    echo "commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "cuda_flags=$(grep '^CMAKE_CUDA_FLAGS:STRING=' "$REPO/$BUILD/CMakeCache.txt" | sed 's/^CMAKE_CUDA_FLAGS:STRING=//')"
    echo "env_ESCHA_PROFILE=${ESCHA_PROFILE:-}"
    echo "env_GGML_CUDA_DISABLE_GRAPHS=${GGML_CUDA_DISABLE_GRAPHS:-}"
    echo "command=$BIN -m $MODEL --prompt-tokens-file $IDS -p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r $REPS -o json -oe json"
} > "$OUT/run-manifest.txt"

"$BIN" \
    -m "$MODEL" \
    --prompt-tokens-file "$IDS" \
    -p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on \
    -r "$REPS" -o json -oe json \
    > "$OUT/stdout.json" 2> "$OUT/stderr.json"
RC=$?

echo "exit_code=$RC" >> "$OUT/run-manifest.txt"
echo "gpu_after=$(nvidia-smi --query-gpu=name,driver_version,utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')" >> "$OUT/run-manifest.txt"
exit "$RC"
