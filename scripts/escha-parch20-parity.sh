#!/usr/bin/env bash
# P-ARCH-20 deterministic prefix parity gate.
# Runs llama-cli on the immutable shared 2048-ID prompt with the same binary args
# as prior parity gates; compares generated token sequence + asserts exit 0.
set -u
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL="${ESCHA_MODEL:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf}"
PROMPT="${ESCHA_PROMPT:-/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.txt}"
BUILD="$1"
OUT="$2"
export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
unset ESCHA_PROFILE GGML_CUDA_DISABLE_GRAPHS ESCHA_CUBLAS_PREFILL ESCHA_WMMA_PREFILL ESCHA_NO_MMA

mkdir -p "$OUT"
"$REPO/$BUILD/bin/llama-cli" \
    -m "$MODEL" \
    -f "$PROMPT" \
    -n 16 --temp 0 --seed 42 --no-display-prompt \
    -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on \
    > "$OUT/stdout.txt" 2> "$OUT/stderr.txt"
RC=$?
echo "exit_code=$RC"
echo "generated_lines=$(grep -cE '^[0-9a-f]{8}: ' "$OUT/stdout.txt" || true)"
tail -20 "$OUT/stdout.txt"
exit "$RC"
