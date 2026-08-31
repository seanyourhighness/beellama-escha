#!/bin/bash
# P-ARCH-13 symmetric K2 geometry measurement helper.
#
# Usage:
#   escha-parch13-measure.sh <build-dir> <outdir> <profile|plain> [reps]
#
#   build-dir  path under the repo root, e.g. build-cuda-parch10-async
#   outdir     evidence output directory (created if missing)
#   mode       profile = repaired ESCHA_PROFILE capture (graphs disabled)
#              plain   = uninstrumented production-default benchmark
#   reps       llama-bench -r value (default 1)
#
# Writes stdout.log, stderr.log and run-manifest.txt into outdir, then exits
# with llama-bench's exit code.
set -u

# Repo root derived from this script's location; external data dirs overrideable
# via env (documented defaults reflect the standard local layout).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${ESCHA_MODEL:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf}"
IDS="${ESCHA_IDS:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids}"

BUILD="$1"
OUT="$2"
MODE="$3"
REPS="${4:-1}"

mkdir -p "$OUT"
BIN="$REPO/$BUILD/bin/llama-bench"

if [ "$MODE" = "profile" ]; then
    export ESCHA_PROFILE=1
    export GGML_CUDA_DISABLE_GRAPHS=1
else
    unset ESCHA_PROFILE
    unset GGML_CUDA_DISABLE_GRAPHS
fi
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"

{
    echo "run_time=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "build_dir=$BUILD"
    echo "mode=$MODE"
    echo "reps=$REPS"
    echo "gpu_before=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')"
    echo "llama_bench_sha256=$(sha256sum "$BIN" | cut -d' ' -f1)"
    echo "libggml_cuda_sha256=$(sha256sum "$REPO/$BUILD/bin/libggml-cuda.so.0.19.0" | cut -d' ' -f1)"
    echo "model_sha256=$(sha256sum "$MODEL" | cut -d' ' -f1)"
    echo "ids_sha256=$(sha256sum "$IDS" | cut -d' ' -f1)"
    echo "commit=$(git -C "$REPO" rev-parse HEAD)"
    echo "cuda_flags=$(grep '^CMAKE_CUDA_FLAGS:STRING=' "$REPO/$BUILD/CMakeCache.txt" | sed 's/^CMAKE_CUDA_FLAGS:STRING=//')"
    echo "escha_moe_cu_sha256=$(sha256sum "$REPO/ggml/src/ggml-cuda/escha-moe.cu" | cut -d' ' -f1)"
    echo "env_ESCHA_PROFILE=${ESCHA_PROFILE:-}"
    echo "env_GGML_CUDA_DISABLE_GRAPHS=${GGML_CUDA_DISABLE_GRAPHS:-}"
    echo "command=$BIN -m $MODEL --prompt-tokens-file $IDS -p 2048 -n 0 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on -r $REPS -o json -oe json"
} > "$OUT/run-manifest.txt"

"$BIN" \
    -m "$MODEL" \
    --prompt-tokens-file "$IDS" \
    -p 2048 -n 0 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on \
    -r "$REPS" -o json -oe json \
    > "$OUT/stdout.log" 2> "$OUT/stderr.log"
RC=$?

echo "exit_code=$RC" >> "$OUT/run-manifest.txt"
echo "gpu_after=$(nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader | tr -d '\n')" >> "$OUT/run-manifest.txt"

exit $RC
