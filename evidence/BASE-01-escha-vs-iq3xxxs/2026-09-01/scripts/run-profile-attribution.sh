#!/usr/bin/env bash
# BASE-01 Phase 3: symmetric graphs-off attribution profiles.
# ESCHA: ESCHA_PROFILE=1 (CUDA-event per-stage lines). IQ3: nsys kernel trace.
# Both arms: graphs OFF, same -p 2048 -n 0 contract. Attribution only.
set -uo pipefail
REPO="/mnt/d/CODEX WORKSPACE/beellama-escha"
BENCH="$REPO/build-cuda-base01/bin/llama-bench"
IDS="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids"
MODEL_A="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
MODEL_B="/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf"
OUT="${1:-$REPO/evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/profile}"
mkdir -p "$OUT"
export PATH="/usr/lib/wsl/lib:$PATH"
ARGS=(-p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json)

echo "=== A: ESCHA_PROFILE graphs-off (3 runs) ==="
for i in 1 2 3; do
  ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 "$BENCH" -m "$MODEL_A" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
    >"$OUT/A-escha-profile-$i.json" 2>"$OUT/A-escha-profile-$i.stderr"
done

echo "=== A: ESCHA graphs-off total (3 runs) ==="
for i in 1 2 3; do
  GGML_CUDA_DISABLE_GRAPHS=1 "$BENCH" -m "$MODEL_A" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
    >"$OUT/A-escha-graphsoff-$i.json" 2>"$OUT/A-escha-graphsoff-$i.stderr"
done

echo "=== B: IQ3 nsys trace graphs-off (3 runs) ==="
for i in 1 2 3; do
  GGML_CUDA_DISABLE_GRAPHS=1 nsys profile -o "$OUT/B-iq3-nsys-$i" --force-overwrite true \
    "$BENCH" -m "$MODEL_B" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
    >"$OUT/B-iq3-nsys-$i.json" 2>"$OUT/B-iq3-nsys-$i.stderr"
done

echo "=== B: IQ3 graphs-off total (3 runs) ==="
for i in 1 2 3; do
  GGML_CUDA_DISABLE_GRAPHS=1 "$BENCH" -m "$MODEL_B" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
    >"$OUT/B-iq3-graphsoff-$i.json" 2>"$OUT/B-iq3-graphsoff-$i.stderr"
done

echo "=== nsys stats per run ==="
for i in 1 2 3; do
  nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/B-iq3-nsys-$i.nsys-rep" > "$OUT/B-iq3-nsys-$i.kernels.csv" 2>/dev/null || \
  nsys stats --report cuda_gpu_kern_sum --format csv "$OUT/B-iq3-nsys-$i.nsys-rep" > "$OUT/B-iq3-nsys-$i.kernels.csv" 2>&1 || true
done
echo "PROFILE PHASE COMPLETE"
