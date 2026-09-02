# EXP-11 funded program plan — load-time transcode cache / new sidecar representation

Date: 2026-09-02  
Program owner and gate issuer: Sol  
Review: Terra math + gate review, then Sean authorization  
Phase: **PLAN GATE 1 only** — no implementation, build, benchmark, or commit was performed

## 1. Gate-1 decision

**PLAN VERDICT: READY FOR TERRA REVIEW.**

Attempt 1 should be an opt-in, persistent **standard-tensor overlay GGUF**
generated before `llama_model_loader` performs tensor planning. A separate
NumPy-authoritative cache generator reads the canonical packed GGUF, reconstructs
the selected projections with the same arithmetic and iteration order as
`convert_escha_to_gguf.py`, quantizes them to standard GGML types, and atomically
writes an accelerator-only overlay. The canonical GGUF remains unchanged and
continues to provide all metadata, tokenizer data, recurrent-state semantics,
the LowGPU LM head, and every tensor not shadowed by the overlay.

The loader mounts the overlay only after validating its source hash, recipe,
format version, tensor allowlist, shapes, types, sizes, and content hashes. In
`qwen35.cpp`, a validated overlay `.weight` wins over its packed projection;
otherwise the existing packed route wins; a native standard weight remains the
last fallback. Consequently, a cached FFN uses stock `MUL_MAT` through the
existing `build_ffn` path and never reaches `ggml_cuda_op_escha_mul_mat`.

Do **not** implement the transcode itself inside `qwen35.cpp`. At that point the
main loader has already indexed the canonical GGUF, and model memory planning is
about to begin. Putting file generation there would couple model semantics,
filesystem policy, Python/NumPy working memory, and backend allocation. The
qwen35 file should contain only eligibility and route selection. Cache
preparation belongs before construction of `llama_model_loader`; cache mounting,
masking, and data-source selection belong in `llama-model-loader`.

## 2. Evidence basis and an important limitation

The plan is based on these established facts:

- BASE-01 assigns 213.8 ms, 94.6% of the positive packed-versus-LowGPU gap, to
  FFN gate/up/down.
- P-ARCH-21A replaced 192 packed FFN projections with standard tensors and
  measured 727.724 ms / 2814.31 tok/s, with coherent output and decode at
  30.44 tok/s versus 23.69 tok/s for its production control.
- P-ARCH-23 added 48 correctly reconstructed Q2_K GDN gates and measured
  697.032 ms / 2938.27 tok/s. Q4_K was slower and Q6_K tied while larger.
- P-ARCH-23G changed the embedding to standard Q4_K `GET_ROWS` and measured
  665.188 ms / 3078.96 tok/s.
- P-ARCH-23I put the entire hot body on stock GGML and measured roughly
  619–621 ms / 3300 tok/s, within about 1.2% of the LowGPU reference.
- EXP-02/07/09/10 and the retrospective close incremental work on the current
  packed BM128xBN128 kernel: the remaining wall is representation, not a missing
  local scheduling edit.

There is one critical distinction. P-ARCH-21A and P-ARCH-23I **copied** the 192
FFN tensors, and P-ARCH-23I also copied the 64 full-attention projections, from
the compatible LowGPU artifact. They did not reconstruct those families from
the canonical Escha GGUF. The 48 GDN gates and 96 linear-attention QKV/SSM
projections were genuinely reconstructed from Escha sidecars. Therefore
2814/3300 tok/s are strong stock-route performance targets, but not bit oracles
for a canonical-source FFN transcode. EXP-11 must establish its own oracle and
quality evidence for reconstructed FFN/full-attention weights. Donor bytes are
not permitted in the cache.

## 3. Frozen invariants

1. The canonical `escha-w2-lowgpu-mono-parity.gguf` is the sole source of
   truth. Its recorded SHA-256 is
   `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d`.
   It is never rewritten, appended to, or renamed by EXP-11.
2. Eligibility is metadata-only: `general.architecture == qwen35` and
   `qwen35.escha.version == 1`, followed by structural validation of the needed
   sidecars. No filename, model-name, benchmark, or prompt condition is allowed.
3. `qwen35.escha.version` remains sourced from the canonical GGUF and continues
   to gate raw `A_log -> -exp(A_log)` and grouped/repeat-interleaved GDN Q/K head
   order exactly once, even when the body projection is standard.
4. `qwen35.lowgpu.version` also remains canonical. Attempt 1 leaves both vocab
   sides alone; later body recipes may use a standard Q4_K embedding, but the
   packed LowGPU LM head and its three sidecars remain authoritative.
5. A cache may replace execution representation, never semantic metadata.
   Overlay KV keys cannot override canonical model KV keys.
6. A projection is shadowed only by a complete, validated recipe entry. For
   FFN, gate/up/down are atomic per layer because `build_layer_ffn` selects one
   branch for all three.
7. Invalid, partial, stale, corrupt, or unknown caches never feed inference.
   `auto` rebuilds or falls back to the packed canonical path; `required` aborts
   with an actionable error. Neither mode accepts suspect weights.
8. Standard Qwen3.x GGUF behavior must be byte/route identical with cache
   support compiled in but ineligible.

## 4. Target inventory and staged recipes

The canonical body has 400 packed projection roles: 192 FFNs, 64 projections
across 16 full-attention layers, and 144 projections across 48 recurrent layers
(48 GDN gates, 48 QKV, 48 SSM output). The P-ARCH-23I target inventory is:

| Family | Count | Target representation | Proven reference | Cache payload estimate |
| --- | ---: | --- | --- | ---: |
| FFN gate/up/down | 192 | reconstructed Q2_K in EXP-11; stock `MUL_MAT` | 21A used donor mixed-IQ bytes, 2814 class | 5.615 GB |
| Recurrent GDN gate | 48 | reconstructed Q2_K | P-ARCH-23 | 0.495 GB |
| Recurrent QKV + SSM out | 96 | reconstructed Q2_K | P-ARCH-23I | 1.321 GB |
| Full-attention Q/K/V/O | 64 | reconstructed Q2_K for canonical-source purity | 23I copied donor mixed-IQ bytes | 0.551 GB |
| Token embedding | 1 | reconstructed/dequantized then Q4_K, chunked | P-ARCH-23G | 0.715 GB |
| LM head | 3 sidecars | unchanged LowGPU packed representation | P-ARCH-23I | no cache payload |
| Markers/shared codec | 3 tables + metadata | retained from canonical | P-ARCH-23I | negligible |

The Q2_K sizes above follow the exact GGML ratio of 84 bytes per 256 values.
The full-body overlay would be about 8.70 GB. It is not Attempt 1's initial
scope. The family order after an Attempt-1 FFN pass is GDN gate, embedding,
full attention, then recurrent QKV/SSM. Each extension is a new recipe revision
under the same attempt/architecture and must pass its own oracle and smoke gate.
The performance waypoints are approximately 2814, 2938, 3079, 3180, and 3300
tok/s; because FFN/full-attention will be reconstructed rather than donor-copied,
these are expectations, not claims.

## 5. Exact reconstruction and quantization contract

### 5.1 Input contract

The runtime cache builder reads the arrays from the **canonical GGUF**, not the
original safetensors and not a donor GGUF:

- `blk.{L}.{role}.escha_code`, with `K = ne[0] / 16` and K in {2, 3};
- `blk.{L}.{role}.escha_rin` and `.escha_rout`;
- the canonical shared `escha_lut` / dependency tables for validation;
- the role dimensions derived from GGUF metadata and tensor shapes.

The converter folded checkpoint `s_in` and `s_out` into the GGUF rin/rout when
it created the canonical artifact. Reconstruction from the GGUF must therefore
use those stored vectors directly and must not apply `s_in`/`s_out` a second
time. The fp32 bias-correction tensor remains ignored, matching the current
packed runtime and the converter's validated derived artifacts.

### 5.2 Math contract

The oracle is the exact `escham_cpu.reconstruct_deploy_weight` sequence already
used by `convert_escha_to_gguf.py`:

1. Decode each packed 16x16 tile with the cbA codebook. The cbA transform uses
   uint32 multiplication by `3417055213`, then
   `(x & 0x8FFF8FFF) ^ 0x3B603B60`; interpret the low/high halves as fp16,
   sum them in fp32, and cast the LUT result to fp16.
2. Assemble `W0` as `(IC, OC)` fp16, then cast to fp32.
3. Apply the normalized Sylvester Hadamard-128 along IC exactly as
   `escha_t128(W0.T.copy()).T.copy()`; normalization is `1/sqrt(128)`.
4. Multiply by stored `rin.astype(float32)[:, None]`.
5. Apply the same Hadamard-128 along OC with `escha_t128(W)`.
6. Multiply by stored `rout.astype(float32)[None, :]`.
7. Transpose to contiguous `(OC, IC)` fp32, because GGUF reverses NumPy dims and
   stock `MUL_MAT` expects `ne = (IC, OC)`.
8. Run the existing pure-NumPy `quantize_q2_k`, `quantize_q4_k`, or
   `quantize_q6_k` without changing block order, batch order, fp16 round-trips,
   ties-to-even behavior, or reduction dtype.

The implementation should first refactor these functions without semantic
changes into a shared Python module used by both `convert_escha_to_gguf.py` and
the cache generator. Pin the NumPy major/minor version in the cache recipe.
Changing NumPy, the oracle module digest, quantizer ABI, codebook constants,
batch block size, or floating-point policy invalidates the cache recipe.

### 5.3 Required oracle evidence

- Two independent builds of the same recipe produce identical overlay and
  per-tensor SHA-256 values.
- Every cached tensor's payload is byte-equal to the shared NumPy oracle output.
- Dequantized MAE is reported, but MAE never substitutes for byte equality.
- K2 and K3 golden vectors are compared with the existing reference decoder;
  Q2_K/Q4_K/Q6_K golden blocks remain byte-checked against GGML layout.
- The loaded cached model is token-equal to a separately generated offline
  derived-oracle model for deterministic P2/P7. Equality to donor-based 21A/23I
  is neither required nor expected.

## 6. Cache format, location, keying, and lifecycle

### 6.1 Location and naming

Default root:

`<llama cache>/escha-transcode/v1/`

where `<llama cache>` uses the existing `LLAMA_CACHE` / XDG / platform cache
resolution. An explicit `--escha-transcode-cache-dir PATH` overrides it. A
recipe lives at:

`<root>/<source-sha256>/<recipe-id>/`

with:

- `overlay.gguf` — replacement tensors only;
- `manifest.json` — canonical identity, recipe, entry map, hashes, timing, and
  resource accounting;
- `overlay.sha256` — whole-overlay digest;
- `complete` — zero-length commit marker created last;
- `.lock` and temporary files only while generating.

`recipe-id` is a SHA-256 over a canonical JSON recipe, not a friendly label.
The JSON includes format version, source SHA, architecture, escha version,
scope, ordered role list, target quant per role, oracle-module SHA, NumPy
version, quantizer ABI, endianness, and tensor-layout version.

### 6.2 Per-entry key

Each tensor manifest entry is keyed by:

`(source_gguf_sha256, layer, tensor_role, source_code_sha256,
 source_rin_sha256, source_rout_sha256, K, IC, OC, target_quant,
 oracle_abi, layout_version)`.

It records the output standard tensor name, GGML type, shape, byte count, data
offset, and payload SHA-256. The source file SHA is mandatory even though the
three sidecar hashes are redundant; the latter make diagnostics local.

### 6.3 Atomicity and concurrency

Generation takes an exclusive recipe lock, writes `overlay.gguf.tmp.<pid>` and
`manifest.json.tmp.<pid>`, closes and fsyncs both, validates them as a new
reader, then atomically renames them and creates `complete` last. A second
process waits with progress reporting or uses the last valid complete cache.
It never reads a `.tmp` file. An interrupted process leaves no selectable
cache. Rebuilds preserve the old complete cache until the new one validates.

### 6.4 Validation and failure modes

Before mounting an overlay, validate in this order:

1. eligibility from canonical metadata;
2. canonical full SHA-256 and file size;
3. exact recipe and ABI;
4. `complete`, manifest schema, and whole-overlay SHA-256;
5. allowlisted tensor names only; no KV override and no duplicate output name;
6. exact family completeness, shape, GGML type, byte size, and per-entry hash;
7. source component hashes for every replacement;
8. loader-side bounds and row-data validation.

On mismatch, `auto` takes the lock and re-transcodes. If regeneration fails,
it logs a single structured reason and uses the untouched packed path. In
`required` mode it aborts. `off` performs no source hash and no cache lookup.
`rebuild` ignores a valid cache but retains it until the replacement validates.
There is no mode that loads a mismatched overlay.

Source and overlay hashing may add I/O. Cached-load acceptance is no transcode,
no materialized fp32 weight, and no more than 10 seconds or 10% (whichever is
larger) over cache-off model load on the target host. If double hashing misses
that gate, integrate overlay hashing with the loader's upload/read pass, but do
not weaken content validation.

## 7. Loader and dispatch architecture

### 7.1 User/API controls

Planned controls:

- `--escha-transcode-cache off|auto|required|rebuild` (rollout default `off`);
- `--escha-transcode-scope ffn|body` (Attempt 1 exposes `ffn`; later recipes
  add explicit family sets rather than hidden behavior);
- `--escha-transcode-cache-dir PATH`;
- `--escha-transcode-quant ROLE=TYPE` for explicit, validated recipes only;
- `LLAMA_ARG_ESCHA_TRANSCODE_CACHE*` environment equivalents.

Add corresponding fields to `common_params`, translate them in
`common_model_params_to_llama`, and add a stable public model-parameter/API
representation if non-common callers are to mount an overlay. A file-pointer
load with no stable source path cannot auto-build in Attempt 1; it either uses
an explicitly supplied, hash-valid overlay or stays packed.

### 7.2 Preparation point

The common frontend invokes the out-of-process generator synchronously in
`auto/required/rebuild` **before** `llama_model_load_from_file()` and before GPU
buffers exist. The child process exits before the model loader starts, so its
1–2.5 GB NumPy working set cannot overlap model VRAM/host allocation. The core
library independently revalidates the resulting cache; it never trusts the
generator's exit code alone.

`src/llama.cpp` resolves and validates the overlay path before constructing
`llama_model_loader`. `qwen35.cpp` is not the generator.

### 7.3 Overlay mounting and logical masking

`llama_model_loader` gains an accelerator-overlay source distinct from GGUF
split files. It opens a separate GGUF context/file and builds an **active**
weight map. Overlay names may not collide with canonical tensor names. Each
replacement standard `.weight` masks only the manifest-listed canonical
`.escha_code`, `.escha_rin`, `.escha_rout`, and ignored `.bias` entries. Shared
codec tables and all non-replaced canonical tensors remain active.

`n_tensors`, `n_created`, `size_data`, mmap ranges, progress, and
`done_getting_tensors()` operate on the active map. Masked source entries are
not allocated, uploaded, or counted as missing. The original canonical map is
retained for source validation and diagnostics. Overlay and canonical file
indexes remain unambiguous in `load_data_for()` and `load_all_data()`.

### 7.4 qwen35 route selection

For every cacheable projection, use this precedence:

1. validated overlay standard weight for this recipe entry;
2. canonical `*.escha_code` plus rin/rout;
3. canonical native standard `.weight` (supports existing hybrid artifacts);
4. required-tensor error.

For FFN, first assert a complete layer triplet and then populate either all
three standard pointers or all three `llm_escha_exps`. `build_layer_ffn()` needs
no new math: standard pointers already select `build_ffn`, which emits ordinary
`MUL_MAT`; packed pointers retain `build_escha_mm`. The same `.active()` split
already exists for full attention, GDN gate/QKV, and SSM output.

No new dispatch branch is needed in `ggml/src/ggml-cuda/escha-moe.cu` for
Attempts 1–2. Cached tensors never reach it. It remains the packed fallback and
must continue to profile the remaining 208 projections. Add route proof at the
qwen/graph or graph-inspection layer so both stock and Escha routes are counted;
`ESCHA_PROFILE` alone sees only the packed half.

### 7.5 Expected Attempt-1 route proof

One full model traversal has 400 cacheable projections. The fixed route audit
uses two traversals and must report exactly **800/800** decisions:

- 384 `stock-mul-mat/cache` decisions: 192 FFN roles x 2;
- 416 `escha-packed` decisions: 208 remaining body roles x 2;
- zero `stock-canonical`, unexpected fallback, missing, partial-triplet, or
  unknown decisions.

A later all-body recipe expects 800 stock-cache and zero packed projection
decisions. Shared tables are not projection calls.

## 8. Memory, VRAM, disk, and latency budgets

### 8.1 Attempt 1

- Output cache: about 5.615 GB for 192 Q2_K FFNs, versus 5.587 GB of donor
  mixed-quant FFN bytes in P-ARCH-21A.
- Expected resident model: about 8.5 GB, under the established 10 GB cap.
- Packed FFN source sidecars must not be uploaded alongside replacements.
- Largest FFN tensor: 89,128,960 values; approximately 178 MB as fp16 or
  357 MB as fp32. Generate one tensor at a time with one worker initially.
- Incremental generator RSS: target <=2.5 GB, hard gate 3.0 GB. No GPU
  allocation during transcode.
- First cache creation on the target machine: target 30–120 seconds, hard gate
  120 seconds. Measure one layer first and project all 64 before launching the
  full build. A slower but correct prototype can establish semantics, but it
  fails the operational Attempt-1 gate and triggers Attempt 2.
- Valid cached-load overhead: <=10 seconds or <=10% over cache-off load.
- Free disk: at least 6 GB for first creation and 12 GB for a replacement while
  preserving the last valid overlay; fail before work if unavailable.

The existing NumPy reconstruction makes multiple full-matrix temporaries, so
the RSS gate is a real stop condition, not bookkeeping. Chunking must preserve
the exact block/order contract. Running multiple FFN workers is forbidden until
one-worker peak is measured.

### 8.2 Later body scopes

Additional Q2_K GDN + recurrent linear + full-attention payload is about
2.37 GB; Q4_K embedding is about 0.715 GB. The complete overlay is about
8.70 GB, while the expected active model remains in the P-ARCH-23I 9.35 GB
class because masked packed tensors are not resident. Full-body resident VRAM
must remain <=10.0 GB and transient GPU overhead <=0.5 GB above final.

The full-body monolithic cache needs roughly 9 GB for first creation and 18 GB
for safe replacement. If this is operationally poor, that is an Attempt-2
sharding trigger; it is not permission to skip atomic validation.

## 9. Decode-path contract

P-ARCH-21A decode was 30.44 tok/s versus 23.69 tok/s for its production
control because the standard FFN weights use stock quantized matmul for every
row count. The packed implementation instead detects small `n_rows` in
`ggml_cuda_op_escha_mul_mat` and enters its custom generation split-K/warp-GEMV
path, including packed decode and rotation/finalize work. Moving FFN weights to
standard tensors removes those custom generation calls too; it does not merely
accelerate prefill.

EXP-11 must not add a prefill-only switch that sends the same tensor back to
packed code during decode. A cached standard projection is the sole active
representation for both phases. The packed source stays on disk as fallback,
not in VRAM. Acceptance is candidate decode no worse than 2% relative to the
same binary/model with cache off, at `-p 0 -n 64`, plus no >2% regression in
the recorded depth/context decode matrix. An improvement is expected but is not
credited against failures in correctness or quality.

## 10. The three-attempt funded program

The budget is **three total architecture attempts**, not three attempts plus a
fourth. Ordinary fixes found by code review before a candidate enters its gate
do not consume another attempt. Once a distinct architecture reaches its
correctness/route gate, the resulting confirm/reject decision consumes that
attempt. Every transition requires a written Sol revision that cites the new
evidence and freezes the next recipe before implementation.

### Attempt 1 — NumPy-authoritative FFN overlay cache

**Architecture.** The design in Sections 5–8: one atomic overlay GGUF, built by
an out-of-process Python generator from canonical GGUF sidecars, 192 FFN
standard Q2_K tensors, loader masking, and stock GGML dispatch.

**Touch points.** Refactor converter primitives into a shared conversion module;
add the generator; add common/API cache controls; resolve the overlay before
loader construction; add loader overlay source + active-map masking; change
qwen35 per-family precedence; add cache/oracle/route/isolation tests. No Escha
CUDA kernel change is required except optional non-semantic profiling support.

**Expected gain.** 2750–2900 tok/s, with 2814 tok/s the primary sanity target.
Relative to the promoted 2319.22 control, 2814.31 is +21.3%.

**Attempt-1 success gate.** All correctness, cache, route, P2/P7, isolation,
memory, and decode gates pass; matched canonical 2K median is at least
2667.10 tok/s (+15%), at least 4/5 initial pairs favor the cache, no family
regresses >5%, and candidate CV/noise handling satisfies the existing paired
protocol. The frozen program floor is 2435.18 tok/s (+5%); a result between
+5% and +15% is bankable evidence but is **REVISE**, not an Attempt-1 success,
because it misses the representation result predicted by BASE-01/21A.

After an Attempt-1 pass, add the remaining families in the inventory order
under the same overlay architecture, with a gate after each. A >=3200 tok/s
full-body result is the parity-class program outcome; the contractual minimum
remains the frozen +5% floor. The full 75-case quality run occurs only at the
chosen milestone scope, not at every family iteration.

### Attempt 2 — native streaming, layer-sharded, resumable cache

Attempt 2 is not a second monolithic Python implementation. It is the concrete
revision for an operational wall discovered by Attempt 1.

**Architecture.** Implement a native, deterministic streaming transcode module
that emits 64 independently atomic layer-overlay GGUF shards plus one signed
manifest. It reconstructs and quantizes one fixed row/block window at a time,
writes output blocks at deterministic offsets, and releases the window before
continuing. Independent output blocks may be parallelized, but reductions and
writes inside a block retain fixed order. The loader mounts the complete shard
set as one logical overlay and can resume generation at the first absent layer.
An explicit `prepare` mode supports deployment prewarming; `auto` remains
synchronous when no prepared cache exists.

**Why it is different.** It removes Python packaging/startup from the serving
path, avoids full-matrix NumPy copies and monolithic rewrite amplification,
bounds RSS, permits resumable/pre-provisioned generation, and lets validation
run alongside each shard's read/upload. It must still be byte-equal to the
NumPy oracle; "close enough" is not acceptable.

**Roadblock-specific recipe.** For latency/RSS/disk/package failures, retain
Q2_K for all FFNs so steady-state performance is comparable to Attempt 1. If
the new evidence is specifically a Q2_K quality failure concentrated in the
long-IC/K3 down family, freeze the Attempt-2 recipe as Q2_K gate/up plus Q4_K
down. That overlay is about 6.95 GB and is predicted to keep the active model
near, but below, 10 GB; measure rather than assume. Do not use Q6_K absent a
new quality result—P-ARCH-23 showed it tied Q2_K at substantially larger size.

**Touch points.** New native transcode library/tool and tests; loader support
for an ordered shard set and active-map aggregation; common CLI `prepare` mode;
the same qwen35 route selection. The stock and packed CUDA operators remain
unchanged.

**Expected gain.** Same steady-state class as the equivalent Attempt-1 recipe
(within 2%); operational targets are <=120 seconds cold preparation, <=1.0 GB
incremental RSS, <=5 seconds/10% cached overhead, resumability after any layer,
and no whole-cache rewrite when one shard is invalid.

**Attempt-2 success gate.** Byte equality for every block/tensor, determinism,
all cache and route gates, steady-state throughput within 2% of or better than
the valid Attempt-1 measurement, >=15% over control, decode <=2% regression,
active VRAM <=10 GB, and the operational targets above. If Attempt 1 failed
despite correct 384/416 routing and acceptable resources because stock Q2_K
itself gained <15%, skip Attempt 2 and go directly to Attempt 3; changing the
builder cannot repair a steady-state representation miss.

### Attempt 3 — MMA-ready Escha sidecar representation

**Architecture.** Create `escha-mma-cache-v1`, an accelerator overlay derived
from the canonical code stream. It preserves the compressed symbols but reorders
each 16x16 tile record into the actual CUDA launch order:

`[projection][output-CTA][K-stage][64-col band][warp publication record]`.

The record contains contiguous K2/K3 payload words plus compact, pre-resolved
lane descriptors for ring-word pair, shift, fragment row/column, and dependency
selection. Descriptors shared by all tiles/K values live once per overlay, not
once per weight. Do not materialize a 16-bit LUT index per weight: that would
expand W2 by roughly 8x and violate the program. The builder is deterministic,
content-addressed, and uses the same atomic cache contract.

Add a versioned tensor suffix such as `*.escha_mma_code` and a distinct
`GGML_OP_ESCHA_MMA_MUL_MAT`. The qwen35 loader selects it only when both
`qwen35.escha.version == 1` and a validated `escha-mma-cache-v1` overlay entry
exist. A new CUDA file or clearly separated section of `escha-moe.cu` consumes
the fragment-ordered stream with cooperative decode/publication. The new
representation is the sole active projection payload; the original packed
tensor is not duplicated in VRAM. Decode must read the same reordered records
through a qualified small-row specialization. Unsupported devices ignore the
optional overlay and use the canonical packed representation.

**Discriminating first slice.** Repack exactly one K2 5120->17408 FFN
projection and build a direct-op harness before full-model integration.

**Mechanism gate.** Continue only if representation growth is <=25% over that
projection's canonical packed code+rotation payload, fp16 register count is
<=104 and fp32 <=128, stack/local/spills are zero, decode/address ALU drops at
least 30% from the promoted packed kernel, two-CTA residency remains possible,
and direct-op matmul is at least 15% faster at M=2048. These are fail-fast
requirements before model timing.

**Full Attempt-3 success gate.** Exact reconstructed-weight equivalence to the
canonical packed representation; full route proof; P2/P7 and quality pass;
matched 2K >=2435.18 tok/s (+5%) with at least 4/5 pairs faster; decode <=2%
regression; no depth/family regression >5%; active VRAM <=10 GB; cache format
and fallback work on supported/unsupported backends as specified. The expected
ceiling is less certain than stock Q2_K; this attempt exists only because the
first two representation-conversion approaches were exhausted.

If Attempt 3 fails its mechanism or full gate, Sol's final program verdict is
**CONFIRM-REJECT**: keep the promoted packed control and any independently
accepted cache scope, record the evidence, and do not authorize Attempt 4 under
EXP-11.

## 11. Revision and pivot triggers

| Observed roadblock | Required diagnosis | Next action |
| --- | --- | --- |
| Oracle bytes differ, shape/orientation wrong, or cache is non-deterministic | Stop before route/perf; distinguish implementation bug from NumPy/version contract | Fix within the same attempt until the frozen algorithm is implemented; never relax to MAE |
| FFN triplet partially present, overlay masks wrong source tensors, or 800 route count is wrong | Loader/manifest defect, not performance evidence | Fix within Attempt 1; do not benchmark or consume a new approach |
| Attempt-1 result is correct and >=15%, but cold time >120 s, RSS >3 GB, Python/package dependency is unacceptable, cached overhead >10 s/10%, or monolithic disk/atomic rewrite is unacceptable | Operational load-lifecycle wall | Written revision -> Attempt 2 native streaming/layer shards |
| Attempt-1 Q2_K quality is below smoke/offline-oracle expectations while routes and bytes are correct | Identify whether loss concentrates in K3 `ffn_down`; report per-role quant error | Attempt 2 with frozen Q2 gate/up + Q4 down if predicted active VRAM <=10 GB; otherwise Attempt 3 |
| Attempt 1 has exact bytes, 384/416 route proof, acceptable resources, but <15% full-wall gain | Stock-Q2 representation did not reproduce the 21A mechanism strongly enough; builder changes cannot help | Skip Attempt 2 and pivot directly to Attempt 3 |
| Attempt-1 full-body extension fails only on one family | Keep last passing recipe; inspect source/orientation/quant evidence for that family | Revise that family within Attempt 1; do not contaminate passed families |
| Attempt 2 cannot equal the NumPy oracle | Native floating-point/order contract is not portable enough | Reject Attempt 2 and pivot to Attempt 3; do not ship approximate native bytes |
| Attempt 2 passes operational gates but misses the same steady-state perf gate | Load lifecycle is solved; standard transcode is the remaining limit | Attempt 3 |
| Attempt-3 one-projection size/resource/ALU/direct-op gate fails | New representation cannot preserve compression and economical decode | Final reject; no full-model implementation |
| Any attempt causes >2% decode loss, standard-Qwen route change, metadata leakage, or medium quality below control | Semantic/product gate failure | Reject or revise before performance promotion; no compensating speed credit |

The revision document must name the failed gate, attach raw evidence, state why
the next architecture addresses that failure, freeze the next cache recipe and
budgets, and explicitly say how many attempts remain.

## 12. Planned implementation sequence and review seams

1. Freeze the cache schema, canonical JSON recipe encoding, role inventory,
   masking rules, and CLI/API names.
2. Refactor converter primitives with golden-vector tests; prove the existing
   converter produces unchanged bytes.
3. Build a one-layer/three-FFN overlay generator. Validate orientation, Q2_K
   bytes, atomic publication, corrupt-cache rejection, and projected time/RSS.
4. Add loader overlay mounting and active-map accounting. Prove cache-off loads
   the canonical model identically before changing qwen route selection.
5. Add qwen35 overlay precedence and FFN triplet atomicity. Prove one layer
   stock and 63 packed, then all 64 stock FFNs.
6. Generate the full 192-tensor Attempt-1 cache. Run oracle and route gates.
7. Run P2/P7, decode, then the matched 2K campaign. Only after performance
   acceptance run the milestone medium suite.
8. Issue Sol Attempt-1 gate. On a trigger, write the required revision before
   any Attempt-2/3 implementation.

Expected implementation files/directories (names may be refined in Terra
review, but responsibilities may not be collapsed):

- `conversion/escha/transcode_oracle.py` — shared deterministic math;
- `tools/escha-transcode-cache.py` or installed `llama-escha-transcode` —
  cache prepare/verify/inspect;
- `include/llama.h`, `common/common.h`, `common/arg.cpp`, `common/common.cpp` —
  public/common controls and cache-root policy;
- `src/llama.cpp` — pre-loader resolution/validation handoff;
- `src/llama-model-loader.h/.cpp` — overlay file contexts, active map, masking,
  mmap/data source, accounting;
- `src/models/qwen35.cpp` — metadata-gated overlay precedence and atomic family
  loading;
- tests for oracle bytes, manifest corruption/invalidation/concurrency,
  loader masking, standard-Qwen isolation, and route counts;
- Attempt 3 only: versioned tensor/KV names, op definition/backend dispatch,
  new repack tool, CUDA kernel, and direct-op harness.

## 13. Definition of Done and exact gate commands

These commands are the executable contract for the later implementation. They
are recorded now and were **not run in this planning phase**. Planned scripts
must accept exactly these interfaces or the plan must be revised before data.

### 13.1 Frozen variables and provenance

```bash
EXP11_REPO='/mnt/d/CODEX WORKSPACE/beellama-escha'
EXP11_MODEL='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf'
EXP11_LOWG='/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf'
EXP11_IDS='/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids'
EXP11_BUILD="$EXP11_REPO/build-cuda-exp11-a1"
EXP11_OUT="$EXP11_REPO/evidence/EXP-11-transcode-cache/2026-09-02/attempt-1"
EXP11_CACHE="$EXP11_OUT/cache"

sha256sum "$EXP11_MODEL" "$EXP11_IDS" "$EXP11_BUILD/bin/llama-bench" "$EXP11_BUILD/bin/llama-server"
git -C "$EXP11_REPO" rev-parse HEAD
git -C "$EXP11_REPO" diff --stat
```

Required source/ID hashes are the canonical model hash above and
`695c3609bc35a32003a23be3ba1fbacc16cc94955548c2e855e91661c3f62350`
for `shared-2048.ids`.

### 13.2 Prepare twice, determinism, and deep validation

```bash
python3 "$EXP11_REPO/tools/escha-transcode-cache.py" prepare \
  --source "$EXP11_MODEL" --cache-dir "$EXP11_CACHE/run-a" \
  --scope ffn --quant 'ffn_gate=q2_k,ffn_up=q2_k,ffn_down=q2_k' \
  --jobs 1 --report "$EXP11_OUT/prepare-a.json"

python3 "$EXP11_REPO/tools/escha-transcode-cache.py" prepare \
  --source "$EXP11_MODEL" --cache-dir "$EXP11_CACHE/run-b" \
  --scope ffn --quant 'ffn_gate=q2_k,ffn_up=q2_k,ffn_down=q2_k' \
  --jobs 1 --report "$EXP11_OUT/prepare-b.json"

python3 "$EXP11_REPO/tools/escha-transcode-cache.py" compare \
  --cache-a "$EXP11_CACHE/run-a" --cache-b "$EXP11_CACHE/run-b" \
  --require-overlay-sha-equal --require-entry-sha-equal \
  --report "$EXP11_OUT/determinism.json"

python3 "$EXP11_REPO/tools/escha-transcode-cache.py" verify \
  --source "$EXP11_MODEL" --cache-dir "$EXP11_CACHE/run-a" \
  --scope ffn --deep --require-complete --report "$EXP11_OUT/cache-verify.json"
```

### 13.3 Oracle equality and invalidation/fail-closed tests

```bash
python3 "$EXP11_REPO/tests/test-escha-transcode-oracle.py" \
  --source "$EXP11_MODEL" --cache-dir "$EXP11_CACHE/run-a" \
  --scope ffn --quant q2_k --require-bitwise \
  --report "$EXP11_OUT/oracle-equality.json"

ctest --test-dir "$EXP11_BUILD" --output-on-failure \
  -R 'escha_transcode_(golden|manifest|corruption|invalidation|concurrency|isolation)'
```

The test set must corrupt one payload byte, change source SHA/mtime/path, remove
one FFN role, change qtype/shape, truncate an overlay, leave only temp files,
race two builders, and present a valid-looking overlay to an unmarked standard
Qwen model. Expected results are rebuild/fallback or required-mode abort, never
successful inference from the invalid overlay.

### 13.4 Cold/cached load and memory accounting

Use a new empty `cold` recipe directory; do not delete a valid cache to create
the test.

```bash
/usr/bin/time -v python3 "$EXP11_REPO/tools/escha-transcode-cache.py" prepare \
  --source "$EXP11_MODEL" --cache-dir "$EXP11_CACHE/cold" \
  --scope ffn --quant 'ffn_gate=q2_k,ffn_up=q2_k,ffn_down=q2_k' \
  --jobs 1 --report "$EXP11_OUT/cold-prepare.json" \
  2> "$EXP11_OUT/cold-prepare.time"

/usr/bin/time -v "$EXP11_BUILD/bin/llama-bench" \
  -m "$EXP11_MODEL" --escha-transcode-cache auto \
  --escha-transcode-cache-dir "$EXP11_CACHE/cold" \
  --prompt-tokens-file "$EXP11_IDS" -p 128 -n 0 -ngl 99 \
  -b 128 -ub 128 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
  > "$EXP11_OUT/cached-load.json" 2> "$EXP11_OUT/cached-load.time"

/usr/bin/time -v "$EXP11_BUILD/bin/llama-bench" \
  -m "$EXP11_MODEL" --escha-transcode-cache off \
  --prompt-tokens-file "$EXP11_IDS" -p 128 -n 0 -ngl 99 \
  -b 128 -ub 128 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
  > "$EXP11_OUT/cache-off-load.json" 2> "$EXP11_OUT/cache-off-load.time"
```

Record generator wall time/max RSS, overlay bytes, free disk before/after,
loader wall time, host max RSS, `nvidia-smi` peak/steady VRAM, active/masked
tensor bytes, and whether any packed FFN tensor was uploaded.

### 13.5 Exact 800/800 route proof

```bash
ESCHA_ROUTE_TRACE=1 GGML_CUDA_DISABLE_GRAPHS=1 \
  "$EXP11_BUILD/bin/llama-bench" \
  -m "$EXP11_MODEL" --escha-transcode-cache required \
  --escha-transcode-cache-dir "$EXP11_CACHE/run-a" \
  --prompt-tokens-file "$EXP11_IDS" -p 2048 -n 0 -ngl 99 \
  -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json \
  > "$EXP11_OUT/route-proof.json" 2> "$EXP11_OUT/route-proof.stderr"

python3 "$EXP11_REPO/tools/escha-route-audit.py" \
  --input "$EXP11_OUT/route-proof.stderr" --expect-total 800 \
  --expect-stock-cache 384 --expect-escha-packed 416 \
  --expect-stock-canonical 0 --expect-unexpected 0 \
  --report "$EXP11_OUT/route-audit.json"
```

Also inspect the generated graph/operator counts to prove every cached FFN is
`MUL_MAT` and no cached tensor reaches `GGML_OP_ESCHA_MUL_MAT`.

### 13.6 P2/P7 deterministic smoke and offline-derived parity

```bash
bash "$EXP11_REPO/scripts/escha-exp11-parity.sh" \
  --binary "$EXP11_BUILD/bin/llama-server" --model "$EXP11_MODEL" \
  --cache-dir "$EXP11_CACHE/run-a" --cache-mode required \
  --oracle-model "$EXP11_OUT/offline-oracle-ffn.gguf" \
  --prompts 'P2-factual,P7-tool-call' --tokens 16 --temperature 0 --seed 42 \
  --output "$EXP11_OUT/p2-p7"
```

PASS requires candidate run 1 == candidate run 2 == offline-derived-oracle for
all 16 tokens on P2 and P7, coherent completion, finite logits, and no server
errors. Divergence from the packed control is recorded but is not by itself a
failure because Q2_K is a representation/quantization change.

### 13.7 Decode guard

```bash
"$EXP11_BUILD/bin/llama-bench" -m "$EXP11_MODEL" \
  --escha-transcode-cache off -p 0 -n 64 -ngl 99 \
  -b 512 -ub 512 -ctk f16 -ctv f16 -fa on -r 5 -o json -oe json \
  > "$EXP11_OUT/decode-control.json" 2> "$EXP11_OUT/decode-control.stderr"

"$EXP11_BUILD/bin/llama-bench" -m "$EXP11_MODEL" \
  --escha-transcode-cache required --escha-transcode-cache-dir "$EXP11_CACHE/run-a" \
  -p 0 -n 64 -ngl 99 -b 512 -ub 512 -ctk f16 -ctv f16 -fa on \
  -r 5 -o json -oe json \
  > "$EXP11_OUT/decode-candidate.json" 2> "$EXP11_OUT/decode-candidate.stderr"

python3 "$EXP11_REPO/tools/escha-exp11-analyze.py" decode \
  --control "$EXP11_OUT/decode-control.json" \
  --candidate "$EXP11_OUT/decode-candidate.json" \
  --max-regression-pct 2 --report "$EXP11_OUT/decode-report.json"
```

Repeat with the established depth/context matrix before milestone promotion.

### 13.8 Canonical matched 2K campaign

```bash
bash "$EXP11_REPO/scripts/escha-exp11-campaign.sh" \
  --binary "$EXP11_BUILD/bin/llama-bench" --model "$EXP11_MODEL" \
  --ids "$EXP11_IDS" --control-cache off \
  --candidate-cache required --cache-dir "$EXP11_CACHE/run-a" \
  --pairs 'AB BA BA AB AB BA BA AB AB' --warmups 1 \
  --args '-p 2048 -n 0 -ngl 99 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json' \
  --output "$EXP11_OUT/bench"

python3 "$EXP11_REPO/tools/escha-exp11-analyze.py" campaign \
  --input "$EXP11_OUT/bench" --control-tokps 2319.22 \
  --program-floor-pct 5 --attempt1-gate-pct 15 --min-winning-pairs 4/5 \
  --max-family-regression-pct 5 --report "$EXP11_OUT/campaign-report.json"
```

Use all nine pairs for inference/noise reporting; `4/5` is the minimum early
continuation check, not permission to discard later pairs. Report median,
geometric paired ratio, confidence interval, per-arm CV, wins, raw values,
clocks, temperature, and process exit status. Keep graphs on for the primary
campaign; route/profile runs remain separate.

### 13.9 Standard-Qwen isolation

```bash
bash "$EXP11_REPO/scripts/escha-exp11-parity.sh" \
  --binary "$EXP11_BUILD/bin/llama-server" --model "$EXP11_LOWG" \
  --cache-dir "$EXP11_CACHE/run-a" --cache-mode auto \
  --compare-cache-mode off --prompts 'P2-factual,P7-tool-call' \
  --tokens 16 --temperature 0 --seed 42 \
  --require-log 'ineligible-standard-qwen' \
  --output "$EXP11_OUT/standard-qwen-isolation"
```

PASS requires cache-on/off token and route equality, no cache generation or
source hash for the unmarked model, and unchanged normal qwen35 semantics.

### 13.10 Milestone medium quality suite

Start the candidate server with the same certified configuration:

```bash
"$EXP11_BUILD/bin/llama-server" -m "$EXP11_MODEL" --alias exp11-a1 \
  --escha-transcode-cache required --escha-transcode-cache-dir "$EXP11_CACHE/run-a" \
  --host 127.0.0.1 --port 18121 -ngl all -c 32768 -np 1 \
  -b 2048 -ub 512 -fa on -ctk f16 -ctv f16 --jinja \
  --reasoning-budget 0 --chat-template-kwargs '{"enable_thinking":false}' \
  > "$EXP11_OUT/quality.server.log" 2>&1

benchlocal-cli run --medium --endpoint http://127.0.0.1:18121/v1 \
  --model exp11-a1 --no-thinking --no-retry --no-sandboxed-packs \
  --output json --save-json "$EXP11_OUT/quality-medium.json" \
  --incremental --progress > "$EXP11_OUT/quality.stdout.json" \
  2> "$EXP11_OUT/quality.stderr.log"
```

PASS requires exactly the five 15-case packs (75 total), thinking off, chat
parsing enabled, no Docker packs, no HTTP/server errors, total >=65/75 (no worse
than canonical control), and no unexplained pack-level collapse. Also run the
same suite on the standard LowGPU artifact at the final program milestone if
the loader/isolation diff changed after its smoke gate; its reference is 66/75.

## 14. Final acceptance matrix

An attempt cannot be promoted unless all applicable rows pass:

| Gate | Requirement |
| --- | --- |
| Source | canonical SHA, no mutation, no donor input |
| Oracle | every output byte equal; two builds deterministic |
| Cache | corruption/invalidation/concurrency/fail-closed tests pass |
| Inventory | exact expected entries; no extra KV/tensors; masked source not resident |
| Route | exact 800/800 expected split, zero unexpected fallback |
| Semantics | Escha marker behavior preserved; LowGPU head preserved |
| P2/P7 | deterministic equality to offline derived oracle, coherent output |
| Standard Qwen | cache support ineligible and behavior unchanged |
| Memory | active VRAM <=10 GB; attempt-specific RSS/transient limits |
| Decode | <=2% regression and no depth/context regression |
| Prefill | frozen floor >=5%; attempt-specific gate as defined above |
| Quality | milestone >=65/75 with no unexplained pack collapse |
| Evidence | raw JSON/logs, hashes, runtime/commit, cache recipe, launch config, resource and timing report retained |

Sol issues one of:

- **CONFIRM-PROMOTE** — all gates pass; name exact recipe/scope promoted.
- **CONFIRM-BANK** — a safe improvement clears the frozen +5% floor but misses
  the current attempt's higher gate; retain evidence/optional recipe and revise.
- **REVISE** — a mapped roadblock has a funded next approach; attach revision.
- **CONFIRM-REJECT** — correctness/product gate fails, or Attempt 3 exhausts the
  program; canonical packed control remains default.

The final program gate must state attempts consumed, last passing recipe,
source/cache/binary hashes, cold and cached load cost, active VRAM, matched 2K
result, decode result, quality score, standard-Qwen isolation result, rollback
(`--escha-transcode-cache off` plus removal of only the named cache recipe), and
whether the 3200 tok/s parity-class objective or only the +5% contract was met.

## 15. Risks and watch-outs

- **FFN evidence is not an identity proof.** P-ARCH-21A used donor weights with
  ~0.835 historical correlation. Canonical reconstruction must earn new quality
  evidence; never copy donor data into EXP-11.
- **Orientation and scaling.** The required order is decode -> H(IC) -> rin ->
  H(OC) -> rout -> transpose -> quantize. GGUF rin/rout already include s_in/out.
  Applying them twice, omitting the transpose, or applying the ignored bias
  recreates the P-ARCH-21B class of semantic failure.
- **Quant choice.** Q2_K was sufficient for reconstructed GDN/linear families;
  it is not yet certified for reconstructed FFNs. Q4/Q6 are not automatic
  rescues: P-ARCH-23 found Q4 slower and Q6 tied/larger. Any quant change is a
  new recipe and must fit the 10 GB cap.
- **Loader accounting.** Merely adding `.weight` entries is insufficient.
  Masked packed tensors must be excluded from active counts, mmap/upload, and
  `done_getting_tensors`, while still available for source validation.
- **Partial FFN recipes.** The existing graph branches on `ffn_gate_escha` but
  then assumes up/down match. Manifest and loader must reject partial triplets.
- **Hash cost.** Strong full-file and overlay hashes may add several seconds of
  disk I/O. Optimize by integrating verification with required reads, never by
  trusting path/mtime alone.
- **Disk amplification.** Canonical + 5.6 GB FFN cache is intentional. Atomic
  rebuild briefly needs another overlay. Attempt 2 exists if this is not viable.
- **Python/NumPy deployment.** Attempt 1 chooses oracle certainty over packaging
  elegance. Missing Python is a packed fallback in `auto`, an error in
  `required`, and an explicit Attempt-2 trigger if product deployment rejects it.
- **First-load latency.** The 120-second all-FFN target may be aggressive for
  17.1B reconstructed values. Measure one layer and project before full work.
- **Mmap and multiple files.** Overlay contexts are not model splits. Never
  reuse split-count metadata or permit duplicate canonical names.
- **LowGPU head.** `lowgpu.version=1` and output sidecars must continue to select
  `LOWGPU_MUL_MAT`. Body cache state must not cause a standard `output.weight`
  fallback.
- **Standard-Qwen isolation.** No cache lookup/hash, Escha semantics, or route
  preference may run for `escha_version==0`. The 5/75 -> 66/75 recovery is the
  warning against broad qwen35 changes.
- **Backend portability.** Attempts 1–2 store standard GGML types and should use
  normal backend support. Attempt 3 is a CUDA accelerator artifact and must
  fall back to canonical packed data on unsupported devices.
- **No performance-only exception.** Benchmark speed cannot override an oracle,
  route, decode, quality, source-integrity, or standard-Qwen failure.

## 16. Terra review flags

1. **Confirm the canonical-input oracle contract.** The current converter's
   P-ARCH-23I path reconstructs from checkpoint arrays, while EXP-11 reads
   folded rin/rout from the canonical GGUF. Terra should confirm that using the
   stored vectors once is the correct runtime-equivalent oracle and identify a
   golden tensor/hash.
2. **Confirm FFN Q2_K is the intended Attempt-1 quant.** It matches the requested
   tier and size/performance class, but 21A's donor mixed-IQ FFNs are not direct
   quality evidence. If the frozen intent was to reproduce donor bytes, that
   conflicts with canonical-source truth and must be resolved before code.
3. **Review the 800 count arithmetic.** The plan expects two traversals: 384
   cached FFN + 416 remaining packed = 800. The route harness must prove its
   traversal count rather than normalize logs after the fact.
4. **Challenge the 120-second cold target.** One-layer projection must be used
   to decide whether Attempt 1 can plausibly meet it; otherwise pre-authorize
   the Attempt-2 operational pivot without weakening the oracle.
5. **Review whole-cache hashing strategy.** Validation must remain cryptographic,
   but a second 5.6–8.7 GB read on every startup may miss cached-load targets.
6. **Confirm masking of ignored bias sidecars.** Standard reconstructed weights
   must not accidentally apply the checkpoint bias-correction that the current
   Escha runtime intentionally ignores.
7. **Confirm full-body family order and source purity.** P-ARCH-23I's copied full
   attention cannot be used as the canonical oracle; reconstructing it to Q2_K
   is a new quality variable even though the stock route is proven.
8. **Decide the milestone quality floor wording.** This plan uses >=65/75 to
   implement "no worse than control"; the retrospective's >=64/75 suggestion
   would be a deliberate one-case tolerance and requires Sean/Terra approval.
9. **Review Attempt-3 size accounting before funding it.** Per-weight resolved
   LUT indices are explicitly prohibited; only compact shared/per-tile
   descriptors can plausibly meet <=25% representation growth.
10. **Confirm rollout default.** The plan keeps the accelerator opt-in (`off`)
    until certification. Promotion to `auto` should be a separate final-gate
    decision, not implicit in implementation.
