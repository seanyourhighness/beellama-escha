# ESCHA-W2 PREFILL — next-phase plan

Scope: BeeLlama `f61c9cc2f`, canonical full-Escha model
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf`
(2,058 tensors; SHA-256 `e307007f...`), RTX 5090/SM120. Preserve the GGUF;
gate every Escha-only runtime path on `escha_version != 0`; standard Qwen must
continue through stock paths. No filename/model-name/benchmark conditionals.

## 1. Bottleneck synthesis

- P011 measured an Escha projection at 512 rows as rotate 0.057 ms, matmul
  1.657 ms, epilogue 0.025 ms: **95.2% matmul**. Optimize the packed GEMM body
  before transforms, output fusion, graph launch overhead, or vocabulary work.
- P014/P-ARCH-03 proved `mma-fp16` completes, preserves P1/P2/P5/P6/P7, and
  was 1.897x the old
  tiled-FMA route at controlled 2k. P-ARCH-10/11 further proved SM120 async A
  staging is safe and large. Current source still puts that overlap behind
  `ESCHA_MMA_SM120_ASYNC_EXPERIMENT`; `build-cuda-p20-control` (the source of
  the ~2311 tok/s base result) has it; the certified gated build does not.
  Consolidating this is dispatch debt, not a
  new gain relative to the 2311 tok/s historical async baseline.
- After async overlap, the remaining credible kernel gap is packed decode/GEMM
  structure: Bee materializes B in shared memory, barriers, reloads it with
  `ldmatrix`, and carries high register pressure.
  P-ARCH-14 still found the MMA body dominant. This is the primary new speed
  target.
- P-ARCH-23I's ~3300 tok/s replaced the Escha body with standard GGML tensors.
  It is a runtime-path ceiling, but an **artifact
  substitution result**, not evidence that full-Escha packed execution was
  improved.
- Closed controls narrow the search: FP16 accumulation alone regressed 44.2%
  (P-ARCH-20); 128x64 alone regressed 0.78% (P-ARCH-13); finalize fusion was
  neutral (P-ARCH-14). Native Escha's speed requires the surrounding schedule,
  not any one of those isolated changes.

## 2. Runtime opportunities vs artifact opportunities

**In scope (runtime):** SM120 dispatch; packed K2/K3 fragment decode/MMA;
CTA/warp ownership, pipeline depth, register/shared-memory lifetime and launch
selection; shared input rotation; CUDA graphs after compute is fixed. Keep the
existing sidecars/shared LUT and GGUF hash unchanged.

**Out of scope for this phase:** replacing FFN/attention/GDN weights with
standard GGML tensors, body reconstruction/requantization, vocabulary changes,
or a new model. P-ARCH substitutions remain attribution/ceiling controls. An
artifact experiment requires a separate phase.

## 3. Ranked bottlenecks and smallest discriminating experiments

All candidates are compile-time or runtime opt-in until promoted. One variable
per experiment; use fresh build directories. Timed runs have `ESCHA_PROFILE`
unset. `CONTROL` and `CANDIDATE` below are absolute build directories.

### 1 — SM120 route consolidation (high confidence, highest immediate value)

Hypothesis: the qualified async A-stage route should be the normal SM120
full-Escha prefill route; the plain fresh build currently leaves proven speed
unused. Implement only the dispatch/pipeline selection, gated by CUDA arch and
the active Escha operator; retain `ESCHA_NO_MMA`/a new explicit sync fallback.

Probe: build clean sync control and candidate from one commit. Run one route
proof each (section 4 command, add `ESCHA_PROFILE=1
-r 1`, redirect stderr, require all true-prefill records to say
`route=mma-fp16`; never use this run for throughput). Then run the section-4
2k and decode commands on both.

Keep if candidate prefill is >=25% over the fresh sync control, is within 2% of
the retained async baseline when hardware/run contract match, P2/P7 smoke
passes, and decode is no worse than 2%. Reject if the intended route is not
proved. Roll back on CUDA error/hang/nonfinite output, >2% decode loss, or
failed smoke. Do not report this as uplift over the ~2311 tok/s base: it makes
that proven path the reproducible default.

### 2 — direct-fragment packed GEMM, one shape (highest new upside)

Hypothesis: for K2 `M=2048, IC=5120, OC=17408`, decode packed codes into
warp-owned MMA B fragments (or equivalent) to remove shared-B stores/barrier
and B `ldmatrix` reloads. Start FP32-accumulating and retain async A staging,
current partial
layout/finalize, and artifact bytes. Do **not** combine accumulator, BN, fusion,
or graph changes.

Probe: add one compile guard plus an exact shape predicate and a distinct
profile tag such as `mma-directfrag-fp32`. With graphs disabled, run a 512-row
profile (`-p 512 -b 2048 -ub 512 -r 1`) and aggregate only matching K2
5120->17408 `matmul_ms`; then run the exact 2k development gate. Confirm the
resources with `cuobjdump --dump-resource-usage
"$CANDIDATE/bin/libggml-cuda.so"`: target fewer than
128 regs/thread and no shared-B allocation; resources do not replace timing.

Keep the prototype if matching-shape median matmul falls >=10%, full 2k rises
>=5%, at least 4/5 timed samples beat the control median, CV is <=2%, P2/P7
passes, and decode changes <=2%. If full-wall gain is <3% despite a kernel win,
retain only as diagnostic and re-attribute call coverage. Roll back on any
wrong/tail output, fallback, spill, CUDA or quality failure. Only after success
extend one family at a time (remaining K2 short-IC, then K3), repeating gates.

### 3 — packed GEMM tiling/scheduling after direct-fragment proof (medium-high)

Hypothesis: residual time is occupancy/issue limited by warp ownership,
register lifetime, or insufficient inter-tile overlap—not raw K2 format.
Sweep only one axis per build around the winning fragment design: warp bands,
CTA M/N ownership, pipeline stages, then slice/grid policy. Include production
shapes; never tune only 5120->17408.

Probe each: run `-p 512 -ub 512` profile once, then section-4 2k. Keep only if
aggregate Escha matmul improves >=5% and full prefill >=3% versus the preceding
candidate, CV <=2%, no family regresses >5%, smoke passes, and decode <=2%.
Rollback otherwise. Do not retry the already-rejected standalone 128x64
geometry; a width change is admissible only as part of a proven direct-fragment
schedule, labeled as a new coupled experiment.

### 4 — shared-input transform / projection fusion (medium-low)

Hypothesis: `build_layer_ffn` up/gate and `build_qkvz` qkv/z share inputs. A
grouped Escha op could rotate once for two packed GEMMs. Activate only when both
are Escha sidecars; leave standard Qwen graphs unchanged.

Probe before implementation: use one graph-disabled `ESCHA_PROFILE=1 -p 512
-ub 512 -r 1` capture and sum duplicate rotate+host gap for those pairs. If the
measured upper bound is <3% of 2k wall, stop. Otherwise prototype **FFN up+gate
only**, keep outputs/layout identical, and run section 4. Keep for >=3% full
prefill, P2/P7 pass, decode <=2%, and no workspace increase >128 MiB; rollback
on graph/aliasing changes, tails, or smaller gain. Add qkv+z separately.

### 5 — CUDA-graph prefill capture/replay (low)

Historical graph-enabled and graph-disabled base controls were near-neutral;
graphs cannot remove the measured GEMM body. Revisit only after items 1-3.
Probe with the same candidate binary and exact IDs in alternating cold-process
A/B/B/A order: section-4 command with `GGML_CUDA_DISABLE_GRAPHS=1`, versus both
graph variables unset (and `ESCHA_ALLOW_CUDA_GRAPHS=1` only if the current
runtime still requires it). Require debug logs to show successful capture/replay.
Keep only for >=3% median prefill improvement across two independent 5-sample
sets, no capture failures, smoke pass, and decode <=2%; otherwise graphs remain
a deployment choice, not a prefill fix.

### 6 — LUT/dependency access and residual microbatch policy (lowest)

MMA computes `escha_codebook_h()`/`escha_dep_pi()`; `lut`/`dep` are unused in
the hot kernel, so shared-LUT traffic is unproven. Reopen only if SASS
or a matched microprobe shows codec dependency latency after direct-fragment
work. A candidate must improve matching-shape matmul >=5% and full wall >=3%
with exact output/smoke/decode gates.

Microbatch is already controlled at `-b 2048 -ub 2048`; P-ARCH-07 showed small
ubatches multiply projection calls. After kernel work, sweep `-ub 512,1024,2048`
with fixed IDs and otherwise identical commands. Change defaults only if a
setting wins >=3% at 2k, does not regress 128/512/1544/4096 prompts, and stays
inside the VRAM envelope. Never special-case benchmark tokens.

## 4. Lightweight development gate

Use the same persisted 2,048-token IDs for every arm. Default warmup stays on;
JSON retains all five samples. Profiling and graphs are disabled:

```bash
MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
GGML_CUDA_DISABLE_GRAPHS=1 "$BUILD/bin/llama-bench" -m "$MODEL" \
  --prompt-tokens-file "$IDS" -p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 \
  -ctk f16 -ctv f16 -fa on -r 5 -o json -oe json
```

Report every sample, median, mean, CV, commit/binary/model hashes, GPU/driver,
route, peak VRAM and exit code. Compare same-session
fresh control/candidate; if the result is within 3%, reverse order and repeat.

Decode guard (same graph mode and five samples):

```bash
GGML_CUDA_DISABLE_GRAPHS=1 "$BUILD/bin/llama-bench" -m "$MODEL" \
  -p 0 -n 64 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on \
  -r 5 -o json -oe json
```

Tiny deterministic smoke (Python needs `transformers` and `requests`):

```bash
ESCHA_SERVER_BIN="$BUILD/bin/llama-server" python3 scripts/escha-compare/run_compare.py \
  --model "$MODEL" --only P2,P7 --max-new-tokens 16 --ctx-size 4096 \
  --outdir "/tmp/escha-prefill/$EXP/smoke"
```

Require exit 0, finite output, and no loss from control's 16-token agreement.
For identical arithmetic require exact control output. Arithmetic changes need
a new quality contract; none is authorized here.

## 5. Promotion and full-certification trigger

Before changing a production default: fresh build; route proof; 128/512/2048/
4096 plus row-tail stability; P1/P2/P5/P6/P7 at 16 tokens (P5 >=1544 prompt
tokens); decode <=2%; VRAM; standard-Qwen smoke on the same binary; raw logs and
hashes added to the experiment ledger/current-state update.

Run the full club-3090 medium 5-pack at a **milestone**, not each iteration:
(a) the first runtime/kernel candidate is promoted as default; (b) cumulative
full-Escha prefill improves >=10% or reaches >=3000 tok/s; (c) arithmetic,
qwen35 graph semantics, tensor interpretation, or artifact changes; or (d) any
small gate shows unexplained quality movement. Run the full 8-pack for a release
candidate, an artifact change, or a medium-suite discrepancy—not for routine
kernel iterations.

## 6. Explicitly avoid without new contradictory evidence

- vocabulary-size caps, embedding/vocab representation reopening (P-ARCH-22);
- gate-bias theory or substituting the incompatible `attn_gate.weight` source;
- whole-model reconstruction/requantization or P-ARCH-23I as final architecture;
- FP16 accumulator-only (P-ARCH-20), 128x64-only (P-ARCH-13), or finalize-only
  fusion (P-ARCH-14);
- legacy WMMA paths, full per-call dequant+cuBLAS, or TF32 as a production fix;
- scheduler work premised on prefill having only 2-4 rows (true prefill was
  route-proved at 512 rows); arbitrary tile sweeps without route/resource proof;
- shared-LUT work before evidence (hot kernel computes codec/dependency mapping);
- filename/artifact-name gates, benchmark-specific dispatch, stale builds,
  best-of-N reporting, or comparing server TTFT directly with `llama-bench`.

Evidence: prefill ledger, runtime audit, PREFILL-PARITY-REVIEW, and
P-ARCH-03/09/10/11/13/14/19/20 manifests. GBrain corroboration: “BeeLlama
PREFILL BREAKTHROUGH SPRINT — P-ARCH-18B → 20” and “Escha P-ARCH-10 takeover”;
repo manifests/ledger take precedence.
