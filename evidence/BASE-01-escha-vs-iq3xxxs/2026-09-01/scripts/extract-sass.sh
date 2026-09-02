#!/usr/bin/env bash
# BASE-01 Phase 5: focused SASS + resource extraction for dominant ESCHA symbols
# and the IQ3 quantized GEMM symbols. Uses cuobjdump from CUDA 13.0.
set -uo pipefail
REPO="/mnt/d/CODEX WORKSPACE/beellama-escha"
LIB="$REPO/build-cuda-base01/bin/libggml-cuda.so"
OUT="${1:-$REPO/evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/sass}"
mkdir -p "$OUT"
CUOBJDUMP=/usr/local/cuda-13.0/bin/cuobjdump

echo "=== symbols of interest ==="
$CUOBJDUMP --list-elf "$LIB" | grep -iE "escha_matmul|escha.*k2|escha.*k3|ggml_cuda.*mmq|iq3|dequant" | head -40 > "$OUT/symbols.txt" || true
cat "$OUT/symbols.txt" | head -40

echo "=== resource usage (full) ==="
$CUOBJDUMP --dump-resource-usage "$LIB" > "$OUT/resource-usage-full.txt" 2>&1 || true
grep -E "Function|REG|STACK|SHARED|LOCAL" "$OUT/resource-usage-full.txt" | grep -iE "escha|mmq|iq3" | head -60 > "$OUT/resource-escha-mmq.txt" || true

echo "=== focused SASS: escha matmul symbols ==="
$CUOBJDUMP --dump-sass "$LIB" > "$OUT/sass-full.txt" 2>&1 || true
grep -c "HMMA" "$OUT/sass-full.txt" || true

echo "=== extract per-symbol SASS for dominant kernels ==="
python3 - "$OUT" <<'PY'
import re, os, sys
out = sys.argv[1]
txt = open(f"{out}/sass-full.txt", errors="replace").read()
# cuobjdump sections: "Function : <name>" followed by SASS lines
sections = re.split(r"(?=Function : )", txt)
want = ("escha_matmul_dense_tiled_mma", "mmq", "iq3", "dequant")
selected = [s for s in sections if any(w in s[:300] for w in want)]
with open(f"{out}/sass-focused.txt", "w") as f:
    for s in selected:
        f.write(s[:6000] + "\n\n")
print(f"wrote {len(selected)} focused sections")
PY
echo "SASS PHASE COMPLETE"
