# EXP-06 Phase 1 — token accounting, official BM semantics, and revalidation

Date: 2026-09-01  
Starting control: `4bc1afc1d` (promoted EXP-04 Stage 2 mixed-accumulator path)  
Target: K3 `IC=17408 → OC=5120`, effective direct-op `M=2048`, FP32 accumulator.

This is **read-only evidence**. No BeeLlama kernel source has changed.

## A. Token-accounting correction

### Raw request facts (five sequential requests)
Source: `token-accounting/p2a-boundary-*.json`; canonical IDs from
`…/P-ARCH-05/…/shared-2048.ids`.

| condition | logical request tokens | server `meta_info.prompt_tokens` | server e2e latency | client wall | 2048 / client wall | 2048 / server e2e |
|---|---:|---:|---:|---:|---:|---:|
| cold | 2048 | 2048 | 785.185 ms | 800.7 ms | 2557.8 tok/s | 2608.3 tok/s |
| warm median (4) | 2048 | 2048 | **619.678 ms** | **634.9 ms** | **3225.7 tok/s** | **3304.9 tok/s** |

- **Proven:** logical prompt length is exactly **2048**, not ~1890; no cached
  prompt tokens; `M=2048` agrees with the logical request count for these runs.
- **Measured:** local client serialization/HTTP/response overhead is about
  `634.9 - 619.678 = 15.22 ms` on the warm median.
- **Measured:** server scheduler log for the same warm 2048-token requests
  reports `input throughput` 2499.54 → 3010.58 → 3023.16 → 3012.77 tok/s.
- **Unknown:** the server's exact internal numerator/denominator and whether it
  includes a scheduler accounting interval not identical to response e2e. The
  logs show `#new-token: 2048`, so 3029 tok/s is *not* a 2048/server-e2e
  calculation (2048 / 0.619678 = 3304.9).
- **Profiler-covered server prefill duration:** unknown in this environment.
  Server-side nsys CUDA capture was unreliable under WSL; direct-op profiler
  durations are kernel-op evidence, not a model-forward/e2e substitute.
- **Graph behavior:** config requests graphs; the scheduler log labels these
  prefill batches `cuda graph: False`. No claim that the observed e2e uses a
  captured prefill CUDA graph is made.

### Canonical boundary for EXP-06 full-runtime comparisons
Use **logical prompt tokens / client wall time**, because that is directly
observable and includes the end-to-end request boundary. State it explicitly as
`2048 / client_wall_seconds`. Keep `server_e2e_latency` separately as a
secondary server boundary. Do **not** use scheduler `input throughput` as the
numerator/denominator for this comparison until the runtime identifies it.

### Historical reconciliation
P-ARCH-19's `623.380 ms` and `3285.31 tok/s` are mutually consistent under
`2048 / 0.623380 = 3285.31`. The previous EXP-05 `623.8 ms` plus `3029 tok/s`
combined two different timing/accounting boundaries and must not be read as one
calculation. Preserve both values historically; the corrected interpretation is
above.

## B. Official BM64 revalidation

### Controlled fresh-process protocol
- Synthetic direct-op tensors, fixed seed `20260901`; shape M=2048, K=3,
  IC=17408, OC=5120, acc_mode=0.
- Fixed across arms: `ESCHAM_GEMM_BK=64`, `ESCHAM_GEMM_FEPI=1`,
  `ESCHAM_GEMM_WIDE_HAD=1`; only `ESCHAM_GEMM_BM=128|64` changes.
- Five alternating fresh-process pairs: BM128 then BM64, each with 8 warmup
  calls and a CUDA-event timed batch of 50 direct-op calls.
- GPU clocks/memory clocks/power/temperature snapshots retained with each trial.
- Raw files + exact script: `official-revalidation/`.

| arm | selected official symbol | grid / block | regs/thread | smem | blocks/SM est. | median direct-op ms | CV |
|---|---|---|---:|---:|---:|---:|---:|
| BM128 | `<1,3,128,64,2,false,true>` | [40,16,1] / [256,1,1] | 122 | 45,056 B | 3.765 | **4.170747** | 0.189% |
| BM64 | `<1,3,64,64,3,false,true>` | [40,32,1] / [256,1,1] | 65 | 35,840 B | 7.529 | **2.295105** | 0.178% |

Paired BM128/BM64 ratios: 1.8115, 1.8131, 1.8250, 1.8216, 1.8203.
Geometric ratio: **1.8183×**; median-latency reduction: **44.97%**.

Numerical direct-op comparison (same seeded inputs) is byte-identical:
relative RMS 0, max abs 0, exact equality true, NaN/Inf 0/0. (The initial
float32 cosine calculation was >1 from round-off; exact equality is decisive.)

This reproduces a large official synthetic-shape causal effect. It is **not** a
real-model absolute-performance result and does not prove BeeLlama transfer.

## C. Official BM meaning and BeeLlama mapping

| Concept | Official BM128 | Official BM64 | BeeLlama equivalent |
|---|---|---|---|
| Logical dimension controlled | CTA row/M coverage = 128 | CTA row/M coverage = 64 | `ESCHA_MMA_BM`: 128 → **64** |
| Output coverage / CTA | 128 columns: two 64-column bands | same 128 columns: two 64-column bands | retain `ESCHA_MMA_BN=128`, `WN=2` |
| Grid at M=2048, OC=5120 | x=40 output CTAs, y=16 row CTAs | x=40, y=32 | Bee expected grid row dimension doubles (subject to existing split-K z) |
| Threads | 256 | 256 | retain 256 |
| FP accumulator policy | FP32 (`false`) | FP32 (`false`) | retain promoted long-IC FP32 path |
| Template’s fifth int | 2 | 3 | **not mapped**: selection is coupled official codegen behavior, not a proven Bee parameter; do not change Bee K/split policy |
| Registers / smem | 122 / 45,056 B | 65 / 35,840 B | target ≤128 regs, zero spills; resource effect to measure after compile |
| A-stage | unknown from source (binary-only) | unknown from source (binary-only) | existing Bee double-buffered `[2][BM][16]`; at BM64 it mechanically becomes `[2][64][16]` |
| B/decode lifetime | not proven in this phase | not proven in this phase | unchanged existing shared decoded-B `s_w[BN][16]` |

### Classification
- **Proven by Bee source:** Bee `row0 = blockIdx.x*BM`, `oc0 = blockIdx.y*BN`;
  `WM=4`, `WN=2`, `MT=BM/16/WM`, and `BN=128`. Thus BM is the row coverage and
  BM64 gives MT=1 while preserving two 64-column warp groups (`WN=2`).
- **Measured:** official grid y doubles while grid x stays 40; resource/timing
  changes in the preceding table.
- **Inferred:** official template `BN=64` denotes each output band because
  grid.x=40 × 128 output columns = OC 5120 although the symbol prints 64.
- **Unknown:** official per-warp store address ownership and its full shared
  allocation decomposition; these are not prerequisites for this isolated
  Bee row-tile test and must not be claimed as proven direct-fragment behavior.

## D. Proposed single-variable candidate for Sol Gate 1

**Candidate variable:** guarded K3 down-projection row tile geometry in the
existing Bee MMA kernel: `ESCHA_MMA_BM 128 → 64` only for non-generation,
prefill MMA calls where `K=3`, `IC=17408`, `OC=5120`, at row ranges Sol
approves. Keep total CTA N coverage at 128, WN=2, FP32 accumulation, packed
code, decoded-B shared layout, transforms/finalize, and split-K policy
unchanged. Repair only the BM-dependent activation-copy staging invariant
(CPB 16→8 or logically equivalent safe indexing) required for BM64.

**Expected geometry:** CTA threads 256; Bee row grid count doubles at a fixed
n_rows; MT=2→1; accumulator footprint per thread halves. No claim is made that
this reproduces official’s coupled fifth template parameter.

**Falsification / rollback:** reject before full certification if target-family
median gain <15%, any family regresses >5%, regs >128/spills/local stack occur,
numerics/P2/P7/decode fail, or full 2K gain <5%. Revert the isolated operator
commit and prove `git diff 4bc1afc1d -- ggml/src/ggml-cuda/escha-moe.cu` empty.
