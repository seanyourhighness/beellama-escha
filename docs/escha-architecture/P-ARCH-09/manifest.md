# P-ARCH-09 — matched W2 per-kernel efficiency divergence

**Status:** `COMPLETE — SERIAL SHARED-MEMORY DECODE/STAGING IS THE FIRST MATERIAL DIVERGENCE`

## Matched representative call

The representative is the K=2 wide FFN projection `IC=5120`, `OC=17408`,
`M=2048`, on the P-ARCH-05 persisted ID stream and RTX 5090 / SM120.

| Field | Bee | Escha |
|---|---|---|
| Kernel | `escha_matmul_dense_tiled_mma<2,128,128>` | `escham_code_gemm_kernel<1,2,128,64,2,true,true>` |
| Grid / block | `(16,136,1)` / 256 threads (source-derived) | `(136,16,1)` / 256 threads (trace) |
| Effective output coverage | 128 rows × 128 cols / CTA | same 2,176 CTAs; template has two 64-col bands |
| Measured MMA/GEMM duration | **3.812 ms/call** (128 measured calls) | **1.203 ms/call** (matching 136×16 trace event) |
| Effective tensor throughput | **95.78 TFLOP/s** | **303.48 TFLOP/s** |
| Registers / thread | **136** in the SM120 cubin | 80 in the GPU trace |
| Shared memory | 13,824 B dynamic (Bee launch formula) | 45,056 B in the GPU trace |
| Accumulation | fp32 MMA accumulator | fp32 for this `true,true` fused variant |

The 365,072,220,160 FLOP count is `2*M*IC*OC`.  The representative kernel
is therefore 3.17× slower in Bee before Bee's separate rotation/finalization
is included.  Bee's whole op for the same call is 4.049 ms (6.328 µs rotate,
3.812 ms MMA, 24.098 µs epilogue); those separate stages are real but cannot
explain the MMA-body gap.

## First proven architectural divergence

Bee uses a serialized, shared-memory materialization path for every reduction
tile:

1. On SM120 it synchronously copies a 128×16 activation tile global→shared;
   the otherwise asynchronous `cp.async` path is explicitly disabled because
   of the WSL/Blackwell pending-kernel issue (`escha-moe.cu:1041-1063`).
2. It publishes payload/activation state with a CTA-wide barrier
   (`:1070-1083`).
3. Each of 256 threads decodes eight scalar weights, writes the 16×128
   decoded B tile to shared, and reaches another CTA-wide barrier
   (`:1096-1105`).
4. All warps reload B with `ldmatrix`, issue HMMA, then reach a third
   CTA-wide barrier before the next activation copy can be issued
   (`:1107-1140`).

For IC=5120 this loop has 320 reduction tiles, so Bee executes **960
CTA-wide barriers per CTA**, including 320 barriers whose only purpose is to
make the fully materialized B tile visible.  The next activation tile is
issued only after the current tile's HMMA/barrier, so SM120 has no cross-tile
copy/compute overlap.

By contrast, the Escha fused prefill wrapper documents that
`escham_code_gemm` **decodes code straight into MMA B fragments**, with no
fp16 decoded-weight round-trip; it also performs WHT/rout/s_out in its fused
epilogue (wheel `sglang/srt/layers/quantization/escha.py:201-207`).  This
direct-fragment path removes Bee's scalar shared-B stores, B `ldmatrix`
reloads, and their publication barrier from the critical loop.  It is the
first source-visible difference that sits inside the measured 3.17×
representative MMA-body ratio.

This conclusion does not rely on launch count: P-ARCH-07 already matched it
at 400 calls and the current per-call comparison has matching row/output
coverage.  It also does not attribute the gap to output fusion: Bee's full
separate epilogue was only 24.098 µs/call here, while the MMA-body gap is
2.609 ms/call.

## Secondary factors, deliberately not promoted to root cause

- Bee's SM120 image uses 136 registers/thread versus the traced Escha
  kernel's 80.  This is consistent with lower schedulable warp residency, but
  no Bee occupancy counter is available under WSL, so residency is evidence,
  not a quantified attribution.
- Bee's explicit fp32 accumulator tiles (`tile_c`) and its 8-warp,
  128-column CTA shape differ from Escha's internal implementation.  The
  Escha trace proves the kernel geometry/resources but not its complete SASS
  issue schedule.
- Bee's independent rotation/finalize are worth preserving for P-ARCH-10,
  but together they are far smaller than the MMA-body divergence.

## Measurement limits

Nsight Compute counters remain unavailable (`ERR_NVGPUCTRPERM`), so HMMA
issue counts and global-memory transactions cannot be recovered dynamically.
The resource figures above come from Bee's SM120 cubin
(`cuobjdump --dump-resource-usage`) and Escha's recorded GPU trace.  No
unmeasured occupancy, bandwidth, or HMMA count is invented.

## Gate

P-ARCH-09 closes with the first material per-kernel divergence: **Bee's
SM120 serial shared-memory B decode/staging loop, including a CTA barrier
before every MMA tile, versus Escha's direct-to-MMA-fragment fused decode.**

P-ARCH-10 should make one isolated change only: restore safe cross-tile
activation overlap or remove the shared-B materialization from the MMA path,
then remeasure this exact representative call before considering rotation or
epilogue fusion.
