#!/usr/bin/env bash
# EXP-08 canonical matched 9-pair campaign: control (build-cuda-base01) vs
# candidate fused-finalize (build-cuda-exp08-fusedfin). AB BA BA AB AB BA BA AB AB,
# fresh process per trial, one unrecorded warmup per arm. Graphs ON.
set -uo pipefail
REPO="/mnt/d/CODEX WORKSPACE/beellama-escha"
CTL="$REPO/build-cuda-base01/bin/llama-bench"
CAND="$REPO/build-cuda-exp08-fusedfin/bin/llama-bench"
IDS="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids"
MODEL="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
OUT="${1:-$REPO/evidence/EXP-08-fusedfinalize/2026-09-01/bench}"
mkdir -p "$OUT/noise-run"
export PATH="/usr/lib/wsl/lib:$PATH"
ARGS=(-p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json)
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }
run_trial() {
  local arm="$1" bin="$2" tag="$3"
  log "TRIAL $tag arm=$arm"
  "$bin" -m "$MODEL" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
    >"$OUT/noise-run/$tag-$arm.json" 2>"$OUT/noise-run/$tag-$arm.stderr" || { log "TRIAL $tag FAILED"; return 1; }
  python3 - "$OUT/noise-run/$tag-$arm.json" "$arm" "$tag" >>"$OUT/run.log" <<'PY'
import json,sys
fn,arm,tag = sys.argv[1],sys.argv[2],sys.argv[3]
try:
    d=json.load(open(fn)); r=d[0]
    print(f"trial {tag} arm={arm} tokps={r['avg_ts']:.2f} ns={int(r['avg_ns'])}")
except Exception as e: print(f"trial {tag} arm={arm} PARSE_ERROR {e}")
PY
}
# warmups (unrecorded)
log "WARMUP control"; "$CTL" -m "$MODEL" --prompt-tokens-file "$IDS" "${ARGS[@]}" >/dev/null 2>"$OUT/noise-run/warmup-ctl.stderr"
log "WARMUP candidate"; "$CAND" -m "$MODEL" --prompt-tokens-file "$IDS" "${ARGS[@]}" >/dev/null 2>"$OUT/noise-run/warmup-cand.stderr"
ORDER="AB BA BA AB AB BA BA AB AB"
i=0
for pair in $ORDER; do
  i=$((i+1)); a=$(echo "$pair" | cut -c1); b=$(echo "$pair" | cut -c2)
  log "PAIR $i order=$pair"
  if [ "$a" = A ]; then run_trial control "$CTL" "p${i}t1" && run_trial candidate "$CAND" "p${i}t2"; else run_trial candidate "$CAND" "p${i}t1" && run_trial control "$CTL" "p${i}t2"; fi
done
log "CAMPAIGN COMPLETE"
