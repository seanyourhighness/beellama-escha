#!/usr/bin/env bash
# BASE-01 Phase 2: canonical matched 9-pair campaign (AB BA BA AB AB BA BA AB AB)
# One frozen binary (build-cuda-base01) for both arms. Fresh process per trial,
# one unrecorded warm-up per arm. Throughput = 2048 / measured_prompt_seconds.
set -uo pipefail

REPO="/mnt/d/CODEX WORKSPACE/beellama-escha"
BENCH="$REPO/build-cuda-base01/bin/llama-bench"
IDS="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids"
MODEL_A="/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf"
MODEL_B="/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf"
OUT="${1:-/mnt/d/CODEX WORKSPACE/beellama-escha/evidence/BASE-01-escha-vs-iq3xxxs/2026-09-01/bench}"
mkdir -p "$OUT/noise-run"

export PATH="/usr/lib/wsl/lib:$PATH"
ARGS=(-p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json)

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$OUT/run.log"; }

nvidia_telemetry() {
  nvidia-smi --query-gpu=clocks.sm,clocks.mem,power.draw,temperature.gpu,memory.used,memory.total --format=csv,noheader,nounits | tr -d ' ' | tr '\n' ' '
}

run_trial() {
  local arm="$1" model="$2" tag="$3"
  local before after
  before=$(nvidia_telemetry)
  log "TRIAL $tag arm=$arm BEFORE telemetry: $before"
  "$BENCH" -m "$model" --prompt-tokens-file "$IDS" "${ARGS[@]}" \
    >"$OUT/noise-run/$tag-$arm$SUF.json" 2>"$OUT/noise-run/$tag-$arm$SUF.stderr"
  local rc=$?
  after=$(nvidia_telemetry)
  log "TRIAL $tag arm=$arm exit=$rc AFTER telemetry: $after"
  if [ $rc -ne 0 ]; then
    log "TRIAL $tag arm=$arm FAILED rc=$rc"
    return $rc
  fi
  # extract throughput from JSON
  python3 - "$OUT/noise-run/$tag-$arm$SUF.json" "$arm" "$tag" "$BLOCK" >> "$OUT/run.log" <<'PY'
import json, sys
fn, arm, tag, block = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
try:
    d = json.load(open(fn))
    r = d[0]
    ns = r["avg_ns"]; ts = r["avg_ts"]
    print(f"trial {tag} block={block} arm={arm} tokps={ts:.2f} ns={int(ns)} n_prompt={r['n_prompt']} n_gen={r['n_gen']}")
except Exception as e:
    print(f"trial {tag} block={block} arm={arm} PARSE_ERROR {e}")
PY
}

# ---- residency proofs (Sol REVISE item 1): -v load, per-layer offload lines ----
BLOCK="${BLOCK:-1}"
SUF=""
if [ "$BLOCK" != "1" ]; then SUF="-$BLOCK"; fi
log "RESIDENCY A (ESCHA) -v load"
"$BENCH" -v -m "$MODEL_A" --prompt-tokens-file "$IDS" "${ARGS[@]}" >"$OUT/residency-A$SUF.json" 2>"$OUT/residency-A$SUF.stderr"
log "RESIDENCY B (IQ3) -v load"
"$BENCH" -v -m "$MODEL_B" --prompt-tokens-file "$IDS" "${ARGS[@]}" >"$OUT/residency-B$SUF.json" 2>"$OUT/residency-B$SUF.stderr"
python3 - "$OUT" "$SUF" <<'PY'
import re, sys, collections, json
out, suf = sys.argv[1], sys.argv[2]
rows = []
for arm in ("A", "B"):
    txt = open(f"{out}/residency-{arm}{suf}.stderr", errors="replace").read()
    layers = re.findall(r"load_tensors: layer\s+(\d+) assigned to device (\S+)", txt)
    dev = collections.Counter(d for _, d in layers)
    cpu_mapped = [l.strip() for l in txt.splitlines()
                  if "CPU_Mapped" in l or ("CPU" in l and "buffer" in l)]
    vram = {}
    try:
        j = json.load(open(f"{out}/residency-{arm}{suf}.json"))
        r = j[0]
        for k in ("cuda_used_model_bytes", "cuda_used_context_bytes", "cuda_used_peak_bytes",
                  "cuda_total_bytes", "cuda_context_buffer_bytes", "cuda_compute_buffer_bytes"):
            vram[k] = r.get(k)
    except Exception as e:
        vram["error"] = str(e)
    rows.append({"arm": arm, "layer_count": len(layers), "devices": dict(dev),
                 "cpu_mapped_lines": cpu_mapped, "vram_bytes": vram})
    print(f"residency {arm}{suf}: {len(layers)} layer lines, devices={dict(dev)}")
    for line in cpu_mapped[:6]:
        print(f"  CPU-related: {line[:160]}")
with open(f"{out}/residency{suf}.md", "w") as f:
    f.write(f"# BASE-01 residency proof (block {suf or 1})\n\n")
    for row in rows:
        f.write(f"## Arm {row['arm']}\n- layer lines: {row['layer_count']}\n- devices: {row['devices']}\n")
        f.write("- CPU_Mapped/CPU buffer lines:\n")
        for line in row["cpu_mapped_lines"]:
            f.write(f"  - `{line}`\n")
        f.write(f"- VRAM bytes: `{row['vram_bytes']}`\n\n")
print("WROTE", f"{out}/residency{suf}.md")
PY

# ---- warm-ups (unrecorded) ----
log "WARMUP A (ESCHA) unrecorded"
"$BENCH" -m "$MODEL_A" --prompt-tokens-file "$IDS" "${ARGS[@]}" >/dev/null 2>"$OUT/noise-run/warmup-A$SUF.stderr"
log "WARMUP B (IQ3) unrecorded"
"$BENCH" -m "$MODEL_B" --prompt-tokens-file "$IDS" "${ARGS[@]}" >/dev/null 2>"$OUT/noise-run/warmup-B$SUF.stderr"

# ---- matched pairs, pre-registered order ----
# AB BA BA AB AB BA BA AB AB
ORDER="AB BA BA AB AB BA BA AB AB"
i=0
for pair in $ORDER; do
  i=$((i+1))
  a=$(echo "$pair" | cut -c1)
  b=$(echo "$pair" | cut -c2)
  log "PAIR $i block=$BLOCK order=$pair"
  if [ "$a" = "A" ]; then
    run_trial A "$MODEL_A" "p${i}t1" || exit 1
    run_trial B "$MODEL_B" "p${i}t2" || exit 1
  else
    run_trial B "$MODEL_B" "p${i}t1" || exit 1
    run_trial A "$MODEL_A" "p${i}t2" || exit 1
  fi
done
log "BLOCK $BLOCK COMPLETE"
