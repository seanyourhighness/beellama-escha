# P-ARCH-13 — K2-only 128x64 geometry validation

**Status:** `COMPLETE — CASE C (geometry is not the K2 residual driver)`
**NEXT_GATE:** `P-ARCH-14 — isolate Bee rotate -> MMA partial -> finalize versus Escha fused K2 code-GEMM`

## Final symmetric result (2026-08-29/30)

The repaired profiler was validated at the timing boundary: the fresh 128x128
control K2 stage median is 442.713 ms, giving a same-instrumentation residual of
256.988 ms versus Escha — matching the retained historical 257.137 ms boundary
to 0.06%. Four clean cold captures per geometry (all exit 0, 3,200/3,200
records, every row `mma-fp16`) produce:

| K2 family | 128x128 median ms | K2 128x64 median ms | Escha ms | Delta removed |
|---|---:|---:|---:|---:|
| 5120->1024 | 14.394 | 15.642 | 4.270 | -1.248 |
| 5120->6144 | 59.032 | 58.257 | 23.220 | +0.776 |
| 5120->10240 | 76.661 | 88.543 | 36.338 | -11.882 |
| 5120->12288 | 37.909 | 36.456 | 14.896 | +1.453 |
| 5120->17408 | 184.267 | 192.531 | 77.704 | -8.264 |
| 6144->5120 | 70.450 | 74.254 | 29.297 | -3.804 |
| **K2 total** | **442.713** | **465.683** | **185.725** | **-22.969** |

- Geometry explained fraction of the 257.137 ms residual: **-8.93%** (the
  geometry change adds 22.969 ms of residual; it explains none of it).
- MMA-body component: +18.713 ms additional (the 199.162 ms historical MMA-body
  residual is not reduced); rotate+epilogue: +3.546 ms additional.
- Uninstrumented A/B (profiler OFF, graphs ON, 10 samples each): 128x128 median
  1729.79 tok/s vs 128x64 median 1716.34 tok/s (-0.78%).
- Deterministic gate: both current binaries P1/P2/P5/P6/P7 16/16.
- Cubin check: experimental contains `<2,128,64>` + unchanged `<3,128,128>`;
  control contains `<2,128,128>`; no fallback (`mma-fp16` 3,200/3,200).

**Decision: CASE C.** K2 128x64 geometry is rejected as non-dominant; no source
or production-default change is retained. The 257.137 ms residual remains
attributable to the Bee rotate -> MMA partial -> finalize execution structure
versus Escha's fused `escham_code_gemm`, which P-ARCH-14 must isolate.

Exit evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-13/2026-08-29/`
— `profiler-reuse-003/` (4 cold captures), `uninstrumented-ab-001/` (4 A/B
processes), `parity-reuse-001/` (both builds), `analysis-001/` (family table,
aggregates, run-to-run values, Hermes attempt note).

Rollback: drop `ESCHA_MMA_SM120_K2_BN64_EXPERIMENT=1` from the build flags;
the profiler-only event-reuse mitigation (`escha-moe.cu:1564-1601`) is
measurement-only and optional.

## One-variable source diff, before measurement

The experimental compile flag is `ESCHA_MMA_SM120_K2_BN64_EXPERIMENT`, used only
with the pre-existing `ESCHA_MMA_SM120_ASYNC_EXPERIMENT`. K3 retains its existing
`<3,128,128>` instantiation. The current decode representation, shared-B
materialization, `ldmatrix` B loads, asynchronous A-stage pipeline, split-K
selection, rotate kernel, partial layout, and finalize kernel are unchanged.

| Changed source region | Classification | Necessity |
|---|---|---|
| K2-only template instantiation `<2,128,64>` | TILE GEOMETRY | Creates the isolated output tile. |
| K2-only `OC/64` CTA-grid y dimension | GRID/INDEXING | Doubles K2 output CTAs so all columns remain covered. |
| K2-only payload/shared-B dynamic-memory calculation (`NTJ=4`, `BN=64`) | TILE GEOMETRY | Matches the template's 64-column shared layout. |
| `OC % 64` assertion | OUTPUT COVERAGE | Rejects a shape whose columns cannot be exactly partitioned. All six measured K2 shapes satisfy it. |
| compile-time flag guard | COMPILATION | Leaves existing binary/default behavior byte-for-byte on the normal build path. |

No other source line belongs to the geometry experiment. In particular,
`n_slices` remains derived from the existing scheduler calculation; changing it
would add a reduction/scheduling variable outside this geometry gate. A later,
separately classified profiler-only event-reuse mitigation is documented below;
it does not alter either geometry path or kernel execution.

## Expected geometry invariants

| Quantity | 128x128 baseline | K2 128x64 experiment |
|---|---:|---:|
| CTA output tile | 128x128 | 128x64 |
| threads/CTA | 256 | 256 |
| warp-row accumulator tiles (`MT`) | 2 | 2 |
| warp-column accumulator tiles (`NTT`) | 8 | 4 |
| payload tiles (`NTJ`) | 8 | 4 |
| decoded values/thread/CTA (`DPT`) | 8 | 4 |
| B fragments/thread | 8 | 4 |
| accumulator fragments/thread | 16 | 8 |
| K2 CTA grid-y | `OC/128` | `OC/64` |

The four 64-column CTAs per 256 output columns preserve total K2 decode/output
work. Indexing remains `oc0=blockIdx.y*BN`, with `n=0..BN-1`, so stores occupy
`[oc0, oc0+BN)` exactly once. No fusion is introduced.

## Handoff — ESCHA_PROFILE abort diagnosis (2026-08-29)

**Gate remains OPEN. No K2 geometry-performance conclusion is authorized.**

Three retained ordinary profiler attempts on the experimental build emitted
exactly 401 valid records and exited 134. The last valid record was K2
`5120→10240`; record 402 would have been K2 `5120→6144`. Disassembly of the
retained backtrace resolves `ggml_cuda_op_escha_mul_mat+0x3e2e` to the
`CUDA_CHECK(cudaEventSynchronize(profile_stop))` call (old source line 1582).
The exit is therefore `ggml_cuda_error → GGML_ABORT → abort()`, not `assert()`,
an allocator failure, or a profile-record container overflow. The retained
logger omitted the CUDA status string, so it is not proven whether the event
operation itself failed or surfaced an earlier asynchronous CUDA error.

The first 400 records are one complete M=512 ubatch. Record schema and order
match the stable 128×128 capture through record 401. Geometry doubles only K2
MMA grid-y/CTA count; it does not change host operator launches or profile
records. Expected full capture remains 3,200 records: warm-up + measured passes,
each containing 1,088 K2 and 512 K3 records. `ESCHA_PROFILE` has no counter,
vector, ring, fixed buffer, map, aggregation key, or file writer. Its only
instrumentation state was four local CUDA events created and destroyed for
every W2 call (12,800 event create/destroy operations per full capture).

Retained controls prevent classifying this as geometry/indexing failure:

- unprofiled exact-2k run completed in 1.180773217 s;
- Compute Sanitizer completed all 3,200 profiled records with `ERROR SUMMARY: 0 errors`;
- P1/P2/P5/P6/P7 deterministic prefixes are 16/16 exact;
- `CUDA_LAUNCH_BLOCKING=1` still failed at the same 401-record boundary;
- the stable P-ARCH-11 K2/K3 128×128 profile contains all 3,200 records.

The requested Hermes DeepSeek V4 Flash worker route was attempted. The decisive
submission error was `permission_denied: submit_agent requires OAuth client
with agent scope`; another Hermes provider attempt logged HTTP 401 invalid/
blocked/out-of-funds credentials. Local support-worker results were treated as
non-decisive and independently checked.

## Minimal profiler-only repair under validation

`ESCHA_PROFILE` now lazily creates four `thread_local` events per CUDA device and
reuses them after the existing stop-event synchronization. This removes event
object churn while preserving the exact start/rotate/MMA/finalize boundaries,
record schema, stream, kernel launches, and output path. Events intentionally
live until process/context teardown; the production path is unchanged when the
environment variable is absent. Source region: `escha-moe.cu:1564-1601`.

Builds succeeded for both matched geometries:

| Build | CUDA flags | llama-bench SHA-256 | libggml-cuda SHA-256 |
|---|---|---|---|
| 128×128 control | `ESCHA_MMA_SM120_ASYNC_EXPERIMENT=1` | `32911ef90000dfc31d7149d5cf9897e7b087547f7ac87ff348d8ebbb97865edc` | `3178e5ad91bc58594212a29a19cbee081624cf86e3201face523944bd928a614` |
| K2 128×64 | above + `ESCHA_MMA_SM120_K2_BN64_EXPERIMENT=1` | `17e057c19df94ee0dbd2c826609831a2157f43479a7c6d996e79f238ae556e70` | `7c36c7040c7c613bce0fbbffbfce1e697d4bde7fab4e55391e1f529c62d8b416` |

Commit remains `0b035b3a26f1a71edbd1b1ff3bef2654c1a2257d`; source is an uncommitted
experimental worktree. Current `escha-moe.cu` SHA-256 is
`f956cadc00e7f957d9b334230c790bed88b1bbc51fc929ee4707760768f39401`.

Four cold repaired-profiler processes completed: two 128×128 and two K2 128×64.
Each exited 0, emitted 3,200/3,200 records, selected `mma-fp16` for every row,
and had exact expected family counts. The fresh repaired 128×128 K2 MMA total
was 388.171 ms versus 384.888 ms in the retained old-profiler capture (+0.85%),
supporting boundary equivalence. This does **not** prove that event churn caused
the old abort: immediately before the patch, fresh unmodified profiler runs also
completed, so the failure is intermittent/state-sensitive and the reuse change
is a bounded mitigation.

Instrumentation materially perturbs benchmark wall time and must not be used as
the performance result. Repaired K2 128×64 profiled runs were 1.320161/1.343476 s;
the fresh uninstrumented three-sample control averaged 1.199498 s
(`1.223433/1.190275/1.184785 s`), roughly +11% profiled elapsed. The original
retained uninstrumented sample was 1.180773 s. CUDA-event stage totals remain the
candidate matched measurement; all Bee geometry comparisons must use the same
repaired instrumentation on both sides.

## Exact run recipe and evidence

Common command (substitute the build directory):

```bash
ESCHA_PROFILE=1 GGML_CUDA_DISABLE_GRAPHS=1 \
  BUILD/bin/llama-bench \
  -m '/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf' \
  --prompt-tokens-file '/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-05/2026-08-29/escha-controlled-server-002/shared-2048.ids' \
  -p 2048 -n 0 -b 2048 -ub 512 -ctk f16 -ctv f16 -fa on -r 1 -o json -oe json
```

Build directories are `build-cuda-parch10-async` (128×128) and
`build-cuda-parch13-k2bn64` (K2 128×64). New raw logs are under
`/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-13/2026-08-29/profiler-reuse-002/`.
Original failures, unprofiled run, memcheck, and parity evidence remain under
the sibling `geometry-001/` directory. Post-run GPU state was idle, 0–1%
utilization, approximately 1,499–1,500 MiB used, with no compute process.

## Next operator checklist — do not skip

1. Run a third cold repaired-profiler capture for each geometry; require exit 0,
   3,200 records, all `mma-fp16`, and exact K2 measured-pass counts
   `128/192/192/64/256/256` by family.
2. Aggregate only the last 1,600 records (measured pass). Use medians across the
   three cold captures for each family and keep MMA, rotate, and epilogue columns
   separate.
3. Quantify A/B/C throughput with matched commands: no instrumentation, retained
   old stable profiler where applicable, and repaired profiler. Do not infer
   geometry speed from throughput.
4. Re-run the deterministic prefix gate after the profiler-only source change
   and confirm the experimental cubin still contains K2 `<2,128,64>` and K3
   `<3,128,128>` with no fallback.
5. Only after those checks fill the required family table against retained Escha
   totals `77.704/29.297/36.338/23.220/14.896/4.270 ms`, calculate the fraction
   of the old 257.137 ms residual removed. Until then, leave P-ARCH-13 open and
   make no geometry-performance claim.

Rollback of the profiler-only mitigation is limited to restoring per-call local
event handles/create/destroy calls in `escha-moe.cu:1564-1601`; it does not touch
the K2 experimental geometry block.
