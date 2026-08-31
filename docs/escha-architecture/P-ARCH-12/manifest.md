# P-ARCH-12 — K2 W2 residual root cause

**Status:** `COMPLETE — DIAGNOSTIC ONLY; NO KERNEL OR DEFAULT CHANGE`
**NEXT_GATE:** `P-ARCH-13 — validate one isolated K2 geometry/codegen correction`

## Scope and evidence discipline

P-ARCH-01 through P-ARCH-11 remain closed. This gate made no source or binary change. The canonical path named in the request, `docs/escha-architecture/PROTOCOL.md`, is absent in this checkout; the retained P-ARCH-11 manifest, architecture-diff ledger, exact shared-token configuration, and immutable P-ARCH-06/P-ARCH-11 profiler artifacts are the protocol evidence used here.

Hermes submissions for the requested DeepSeek V4 Flash lanes all failed before execution with `permission_denied`: the OAuth client lacks the `agent` scope. That is environment evidence only. Read-only secondary audits were checked against the primary parsing below; no worker conclusion is used without this independent verification.

## Phase 1 — decomposition of the matched K2 residual

The comparison is at the full logical 2,048-token prefill boundary. Bee's P-ARCH-11 profile is four 512-row launches per logical Escha trace launch; the ratio is exactly four for every row below. Escha's output-width grid uniquely identifies the shape (and the 1024 output uses its retained 32x32 specialization). Thus the table compares aggregate work, not falsely matched individual launch durations. Bee values are complete K2 stage values (rotate + MMA + epilogue), as is the already-established 442.862 ms P-ARCH-11 residual boundary.

| K2 shape (IC→OC) | Bee calls / Escha events | Bee total ms | Escha total ms | Delta ms | Delta % |
|---|---:|---:|---:|---:|---:|
| 5120→17408 | 256 / 64 | 184.532 | 77.704 | **106.828** | **41.55%** |
| 6144→5120 | 256 / 64 | 71.552 | 29.297 | **42.255** | **16.43%** |
| 5120→10240 | 192 / 48 | 76.493 | 36.338 | **40.155** | **15.62%** |
| 5120→6144 | 192 / 48 | 58.990 | 23.220 | **35.770** | **13.91%** |
| 5120→12288 | 64 / 16 | 37.838 | 14.896 | **22.942** | **8.92%** |
| 5120→1024 | 128 / 32 | 13.458 | 4.270 | **9.188** | **3.57%** |
| **K2 total** | **1088 / 272** | **442.862** | **185.725** | **257.137** | **100.00%** |

The residual is a family-wide issue, not a one-shape anomaly. The dominant 5120→17408 row is 41.55%, while the next three rows bring the cumulative share to 87.51%. The row deltas reconcile to the aggregate to rounding.

The same re-sum found a clerical count error in P-ARCH-11: its six persisted K2 rows total 1,088 (not 1,472), and all W2 rows total 1,600 (not 1,984). The P-ARCH-11 timing, correctness, and decision are unchanged; its manifest and ledger now carry this narrow correction.

## Phase 2 — matched dominant shape

For logical `M=2048, K=2, IC=5120, OC=17408`, Bee executes four M=512 profile calls, totaling 184.532 ms for the complete K2 stage (165.519 ms inside its MMA body). Its selected kernel is `escha_matmul_dense_tiled_mma<2,128,128>`: 256 threads/CTA, 128 registers/thread, 13,824 B dynamic shared memory, and unchanged CTA barriers. Its launch form is `grid=(ceil(M/128), OC/128, n_slices)`.

The matched Escha trace executes 64 fused `escham_code_gemm_kernel<1,2,128,64,2,true,true>` events, totaling 77.704 ms: 256 threads/CTA, 80 registers/thread, 45,056 B shared memory, 33% estimated achieved occupancy, grid `(136,16,1)`. It is an HMMA kernel. This establishes the logical-work boundary and the material implementation difference, while preserving the fact that the runtimes deliberately use different launch granularity.

Effective TFLOP/s, instruction mix, HMMA issue rate, global/shared transaction counts, and stall reasons were **not** claimed: the retained Nsight Compute lane is blocked by `ERR_NVGPUCTRPERM`, and a full cubin SASS dump was terminated as a nonproductive local diagnostic after it grew beyond 5 GB. The temporary file was moved to system Trash; no repository evidence was altered.

## Phase 3/4 — first proven K2-specific divergence

Bee dispatches both K2 and K3 through one template, `escha_matmul_dense_tiled_mma<K,128,128>`. The K-dependent work is limited to compile-time packed-word constants (`NWD=8*K`, `NB=32*NWD`, `n_wd=8*K`) and payload addressing. Each thread decodes the same eight weights per tile in both K2 and K3 (`DPT=8`); activation staging, shared-B materialization, `ldmatrix`, HMMA fragment construction, and three CTA synchronization points per reduction tile are structurally shared. There is therefore no source proof of a Bee-only K2 decode branch, extra fragment construction, or K2-only barrier.

The first material divergence actually demonstrated at the K2 execution boundary is instead **H/F/C — K2-specialized reference codegen/launch geometry versus Bee's generic K2 instantiation**:

| Property | Bee K2 | Escha K2 |
|---|---|---|
| MMA tile | 128x128 | 128x64 |
| register allocation | 128/thread | 80/thread |
| B/shared staging allocation | 13,824 B dynamic | 45,056 B |
| operator boundary | separate rotate → MMA partial → finalize | fused `code_gemm` launch |

This difference is present across all six rows, while Bee K3 uses the same generic 128x128/128-register implementation and is already within 25.931 ms of Escha. It bounds the affected K2 residual to **257.137 ms (100.00%)**, with the MMA-body portion **199.162 ms (77.45%)** and Bee's separately timed rotate + epilogue portion **57.975 ms (22.55%)**. These are measured boundary contributions, not a causal assertion that registers alone explain 199.162 ms. No counter evidence supports classifying decode, barriers, or shared-B as the first K2 root cause.

## Decision

P-ARCH-12 closes as **mixed codegen/fragment geometry/register-lifetime divergence**, with the smallest plausible correction identified but deliberately unimplemented: a K2-only diagnostic/prototype of the reference's 128x64, lower-register fused-boundary geometry, retaining the already-qualified async A-stage overlap and leaving K3/default dispatch untouched. P-ARCH-13 must implement and validate only that isolated correction; it must measure the geometry and any fusion contribution separately before combining them.

Evidence: P-ARCH-11 paired profile `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-11/2026-08-29/validation-001/async-2048-ub512-k-profile.stderr`; P-ARCH-06 Escha trace `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-06/2026-08-29/trace-001/escha-prefill_batch1_input2048_output1_prefill.trace.json.gz`.
