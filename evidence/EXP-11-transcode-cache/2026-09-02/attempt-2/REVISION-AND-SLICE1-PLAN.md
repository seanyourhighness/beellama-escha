# EXP-11 required revision — Attempt 1 operational reject and Attempt 2 Slice 1

Date: 2026-09-02  
Issuer: Sol  
Decision: **REVISE — consume Attempt 1 and authorize Attempt 2 Slice 1**

## 1. Failed gate and controlling evidence

Attempt 1 passed its Slice-1 semantic/correctness gates but failed the frozen
cold-preparation wall gate in PROGRAM-PLAN §§8.1 and 11.

- Run A projected all 64 layers at **6,662.683 s**.
- Run B projected all 64 layers at **6,860.000 s**.
- The frozen cold-preparation budget is **<=120 s**.
- The projection is therefore **55.5–57.2x over budget** and both reports set
  `wall_budget_holds=false` and `attempt2_trigger=true`.
- Observed one-layer wall was **169.48 s** externally for run A and **150.66 s**
  for run B. This includes the one-time SHA-256 pass over the 8,619,127,360-byte
  source: 58.159 s in A and 35.788 s in B.
- Peak RSS was 2,786,504 KiB (A) and 2,785,928 KiB (B), approximately
  **2.66 GiB**. That is below the 3.0 GiB hard stop but above the 2.5 GiB
  Attempt-1 target. The cold wall alone is sufficient to fire the pivot.

Raw evidence:

- `../attempt-1/SLICE1-RESULT.md`
- `../attempt-1/prepare-a.json`
- `../attempt-1/prepare-a.time.txt`
- `../attempt-1/prepare-b.json`
- `../attempt-1/prepare-b.time.txt`
- `../attempt-1/oracle-equality.json`
- `../attempt-1/determinism.json`
- `../attempt-1/cache-verify.json`
- `../attempt-1/cache/run-a/e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d/a6063882922cda881fd77353f76951afb8b377c38cc6284b23e3c739598f0cd7/manifest.json`

The correctness result is retained unchanged as the Attempt-2 contract:

- Oracle payload byte equality: PASS for layer-0 gate/up/down.
- Converter regression: `byte_equal=true`.
- Two prepares: identical overlay SHA-256
  `f3911881dc4c9ec971eef00a3a2c33f52f64c0c252602c0b324ebd72dc52847b`
  and identical entry hashes.
- Deep verification: `status=ok`, no errors.
- Dequantized MAE: 0.002174–0.002335. MAE remains diagnostic only; it does not
  replace byte equality.

Per PROGRAM-PLAN §11 row 3 and Terra review flag 4, this is the pre-authorized
operational load-lifecycle pivot. No full 64-layer Attempt-1 build is allowed.

## 2. Root cause and quantization-hotspot finding

The decisive cost is the pure-NumPy Q2_K quantizer, not Escha reconstruction.
Run A's scalable one-layer work was 103.101 s:

| Phase | Three tensors | Share of scalable layer time |
| --- | ---: | ---: |
| Q2_K quantization | 84.023 s | 81.5% |
| Escha reconstruction | 15.744 s | 15.3% |
| Dequant/MAE validation | 1.879 s | 1.8% |
| Overlay write/hash/validate | 1.455 s | 1.4% |

Each FFN tensor has 89,128,960 values, or 348,160 Q2_K blocks of 256 values.
`quantize_q2_k` does **not** run a Python loop once per Q2_K block. With the
frozen `batch_blocks=4096`, it executes only 85 outer batch iterations per
tensor. Each batch reshapes 4,096 blocks into 65,536 16-value subgroups and
then calls vectorized helpers.

The expensive helper is `make_qkx3_quants`. It performs an initial fit plus
37 trial fits. Every trial launches multiple whole-batch NumPy operations:
level calculation/clipping, float conversions, several weighted reductions,
candidate scale/min solves, reconstructed-difference calculation, loss
reduction, and conditional copies. `make_qp_quants` then fits the 16 group
scales and minima twice, including trial and coordinate-refinement loops.
These are Python-controlled loops over large vectorized kernels; their cost is
repeated array scans, ufunc launches, reductions, and temporary allocation.
It is not 348,160 iterations of scalar Python math.

### Controlled timing breakdown

Environment: Python 3.14.3, NumPy 2.5.1, deterministic synthetic fp32 input;
timings used `time.perf_counter()` after one warm-up.

For one 4,096-block batch (1,048,576 values):

| Component | Seconds | Percent of `quantize_q2_k` |
| --- | ---: | ---: |
| `make_qkx3_quants` (one call, 37 trials) | 0.39137 | 94.35% |
| `make_qp_quants` (two calls) | 0.01261 | 3.04% |
| Packing, requantization, outer function | 0.01084 | 2.61% |
| Total | 0.41482 | 100% |

Small-block timing demonstrates interpreter/setup amortization but does not
explain the production wall: 1 block cost 2,187 us/block, 2 blocks 1,203
us/block, 256 blocks 138 us/block, and 4,096 blocks 106 us/block. On a larger
34,816-block input, changing only the NumPy batch size from the current 4,096
to the best observed 1,024 reduced median time from 3.700 s to 2.879 s
(22.2%). That is useful tuning, but it is two orders of magnitude short of the
55.5–57.2x end-to-end reduction demanded by the cold gate.

**Finding:** the 28 s/tensor is not dominated by a per-256-block Python loop.
It is dominated by the Q2_K search algorithm expressed as 37 repeated,
memory-heavy NumPy passes in `make_qkx3_quants` (94% in the controlled batch),
plus associated temporary arrays and kernel-launch boundaries. The correct
fix is a compiled, fused Q2_K reference loop with bounded work buffers and
native parallelism. Merely changing NumPy batching or wrapping the current
full-matrix function in more Python processes cannot close the gate.

## 3. Why the Attempt-2 architecture addresses this failure

Attempt 2 replaces full-matrix NumPy materialization and quantization with a
compiled native streaming pipeline. It reconstructs only a fixed block-aligned
window, immediately quantizes it with the exact Q2_K reference arithmetic,
writes its bytes to a predetermined GGUF offset, and releases the window.

The separable Hadamard-128 transforms and Q2_K's 256-value row blocks permit a
bounded work unit aligned to 128 output rows and 256 input values. The precise
window may be reduced during implementation, but it may not cross or reorder
the oracle's Hadamard or Q2_K block boundaries. A fixed-size buffer pool bounds
RSS independently of tensor size. Independent output Q2_K blocks and layer
shards can run on native worker threads; every block retains the oracle's fixed
operation and packing order and is written with positional I/O to its canonical
offset. Thread scheduling therefore cannot change payload bytes.

This directly addresses the observed wall:

- It fuses the 37 Q2_K trials inside compiled loops instead of allocating and
  rescanning large NumPy temporaries.
- It parallelizes independent blocks/layers without multiplying 357 MB fp32
  matrices or 2.66 GiB process working sets.
- It never retains a full reconstructed tensor or a monolithic 192-tensor
  product list.
- A completed layer is independently durable, so interruption resumes at the
  first absent or invalid layer.
- Replacing one bad layer rewrites approximately one layer, not the 5.615 GB
  cache.
- Publication and byte validation remain strict; the architecture does not
  weaken the proven oracle or source hashing.

## 4. Frozen Attempt-2 recipe and budgets

The roadblock is latency/RSS/package lifecycle, not quality. PROGRAM-PLAN §10
therefore freezes the comparable all-Q2_K recipe; the Q4-down branch is not
authorized.

### Recipe

- Eligibility gate: `general.architecture=qwen35` and
  `qwen35.escha.version=1`; never filename or artifact-name behavior.
- Source for certification: canonical SHA-256
  `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d`.
- Scope: all 64 layers, exactly 192 FFN tensors.
- Per-layer atomic family/order: `ffn_gate`, `ffn_up`, `ffn_down`.
- Target type: standard GGML `Q2_K` for gate/up/down.
- Reconstruction ABI: `escha-reconstruct-cba-h128-fp32-v1`.
- Tensor layout: contiguous standard `(OC, IC)`,
  `gguf-standard-weight-oc-ic-v1`.
- Correctness oracle: `conversion/escha/transcode_oracle.py`; every native
  block and tensor must be byte-equal. Native code must use IEEE binary32,
  round-to-nearest-even, little-endian fp16 fields, fixed reduction order, and
  no fast-math/FMA contraction that changes oracle bytes.
- Output: 64 independently atomic layer GGUF shards plus one canonical ordered
  manifest. The complete cache is mounted only when all required shards pass.
- Output size class: approximately 5.615 GB, unchanged from Attempt 1.

### Frozen operational and product budgets

- Full cold preparation, including source identity verification: **<=120 s**.
- Incremental transcoder RSS: **<=1.0 GB**; also report absolute peak RSS and
  the idle source-reader baseline used for the increment.
- Cached-load overhead: **<=5 s and <=10%** over cache-off load.
- Resume after any completed layer; an interrupted run must not recompute a
  valid published layer.
- Invalidating one layer must not rewrite unaffected shards.
- No GPU allocation during transcode.
- Active model VRAM: **<=10.0 GB**.
- Steady-state throughput: within 2% of or better than the byte-valid Attempt-1
  representation measurement and >=15% over the 2319.22 tok/s control.
- Decode regression: <=2% and no prohibited depth/context regression.
- Complete cache determinism, corruption/invalidation/fail-closed behavior,
  exact routes, P2/P7 parity, standard-Qwen isolation, and milestone quality
  gates remain as frozen in PROGRAM-PLAN §§10 and 13.
- Keep the full source SHA-256. It may be pipelined with temporary shard work
  if measurement shows a win, but no shard set or manifest may publish until
  the source digest matches.

## 5. Attempt-2 Slice-1 architecture decision

### Options considered

| Option | Oracle equality | Operational assessment | Decision |
| --- | --- | --- | --- |
| Further NumPy vectorization/batch tuning | Preserved readily | Current code is already vectorized across 4,096 blocks; measured best batch-only gain was 22.2%, versus a required 55–57x end-to-end gain. Full-matrix RSS remains. | Reject for Attempt 2 |
| Python multiprocessing over layer shards | Each process can preserve bytes | Resumable, but duplicates the approximately 2.66 GiB Attempt-1 working set per active process, violates <=1.0 GB incremental RSS, increases source/I/O contention, and needs implausible process scaling to close the wall. | Reject for Attempt 2 |
| Compiled C++ native module/library with a standalone prepare CLI, fixed windows, native worker pool, and layer shards | Preserved as a hard byte gate using the exact reference arithmetic | Removes NumPy temporaries and Python kernel boundaries, permits many small independent workers under one bounded buffer pool, and provides native shard publication/resume. This is the only option with a credible path to both <=120 s and <=1.0 GB. | **Selected** |

The selected “C++ extension” option means a reusable native transcode library
and thin native CLI, not a Python multiprocessing wrapper and not a new llama
decode operator. It must reuse or exactly port the canonical GGML Q2_K
reference quantizer and must contain a deterministic native implementation of
the reconstruction ABI. The stock and packed CUDA operators remain untouched.

### Shard and manifest contract for Slice 1

1. Precompute the GGUF header, tensor order, logical shapes, Q2_K sizes, and
   aligned data offsets for one layer. A shard contains exactly three tensors
   in gate/up/down order.
2. Create `layers/blk.NNN.gguf.tmp.<pid>`, pre-size it, and write each completed
   Q2_K block/window to its deterministic offset with positional I/O.
3. Use a bounded native buffer pool. Worker count is chosen from measured
   throughput subject to the <=1.0 GB aggregate incremental-RSS gate; there is
   no full `(OC, IC)` fp32 allocation.
4. Close and fsync the temporary shard, reopen it as a new reader, verify its
   GGUF allowlist/types/shapes, compare every tensor payload to the NumPy
   oracle for Slice 1, compute tensor and whole-shard SHA-256 values, then
   atomically rename it. A layer receipt containing the recipe/source/layer and
   hashes is written atomically so resume can validate it without trusting a
   temporary file.
5. After all 64 receipts validate, emit one canonical JSON manifest ordered by
   layer and role. It contains source SHA, recipe ID, oracle/quantizer ABI,
   build identity, every shard digest, every tensor digest/shape/offset, and
   masking rules. Publish its detached SHA-256 integrity signature
   (`manifest.sha256`) and the zero-length `complete` marker last. Here
   “signed” means the frozen digest-signed integrity envelope; it is not a
   claim of public-key provenance.
6. Loader eligibility is all-or-nothing for the complete manifest, while shard
   preparation is resumable. A corrupt/absent shard is rebuilt alone; partial
   shards never enter the logical overlay.

### Slice-1 implementation and gates

Slice 1 is one native layer shard for layer 0, followed by a scalability probe;
it is not a 64-layer launch.

1. Implement one fixed-window native reconstruction path and the compiled
   Q2_K reference path for layer-0 gate/up/down.
2. Establish golden equality at three levels: existing small Q2_K golden
   vector, selected first/middle/last windows, then the complete three-tensor
   shard. The required complete payload hashes are:
   `ea4cb733...` (gate), `041c0045...` (up), and `e5a94a62...` (down).
3. Run twice with different supported worker counts/schedules. Tensor bytes,
   shard bytes, receipts, and canonical manifest material must be identical.
4. Measure phase wall, aggregate values/s, source-hash wall, disk write wall,
   absolute/incremental RSS, and per-worker buffer bytes. Project the complete
   17,112,760,320-value/64-layer job using measured aggregate throughput and
   retained source-hash cost. Continue to full implementation only if the
   projection is <=120 s and incremental RSS <=1.0 GB.
5. Kill after publishing layer 0, restart, and prove layer 0 is validated and
   skipped. Corrupt its payload, restart, and prove only layer 0 is rejected
   and rebuilt. Temp files must never count as resumable output.
6. If the native path differs by one byte, stop before timing/promotion and fix
   the implementation. If exact native arithmetic cannot be made portable,
   PROGRAM-PLAN §11 requires rejection of Attempt 2 and pivot to Attempt 3;
   approximate equality is forbidden.

No C++ implementation, loader integration, or llama binary execution is part
of this written-revision dispatch.

## 6. Attempt accounting

- Attempt 1 — NumPy-authoritative monolithic overlay: **consumed/exhausted** on
  the frozen operational cold-wall gate; its correctness oracle is banked.
- Attempt 2 — native streaming, layer-sharded, resumable cache: **authorized
  now**, beginning with the Slice-1 mechanism/scalability gate above.
- Attempt 3 — MMA-ready sidecar representation: **reserved**.

At issuance, two funded architectures remain available including the active
Attempt 2. Once Attempt 2 reaches and consumes its architecture gate, exactly
one reserve attempt remains: Attempt 3. There is no Attempt 4.
