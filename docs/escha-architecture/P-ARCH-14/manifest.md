# P-ARCH-14 — remaining K2 execution structure (rotate / MMA partial / finalize)

**Status:** `COMPLETE — BOUNDARY FUSION MEASURED NEUTRAL; MMA BODY REMAINS DOMINANT`
**NEXT_GATE:** `P-ARCH-15 — production-candidate prefill regression matrix`

## Question

With P-ARCH-13 having rejected the K2 128x64 geometry change (CASE C), what
does the remaining K2 execution structure look like, and does the smallest
isolated fusion/lifetime correction — fusing the single-slice finalize into the
MMA kernel — reduce the 256.988 ms same-instrumentation residual?

## Boundary decomposition (measured, 128x128 control medians of 4 runs)

| stage | K2 total ms | share of residual |
|---|---:|---:|
| rotate (input transform) | 31.949 | 12.4% |
| MMA body (partial write included) | 385.378 | 77.7% |
| finalize (Hadamard-128 + rout) | 25.565 | 9.9% |
| **K2 stage** | **442.892** | — |

vs Escha K2 185.725 ms -> residual 256.988 ms. n_slices per family (M=512):
5120->1024:16, 5120->6144:2, 5120->10240:1, 5120->12288:1, 5120->17408:1,
6144->5120:3. The fp32 partial write+read round trip is ~64 GB per measured
pass; the three n_slices==1 families carry 29.6 GB of it (512 of 1,088 calls).
Full table: `evidence/P-ARCH-14/2026-08-29/boundary-001/decomposition.md`.

## One-variable experiment — fused single-slice finalize

`escha_matmul_dense_tiled_mma_ff<K,128,128>` (new kernel, flag
`ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT=1`): for n_slices==1 the MMA kernel runs
the Hadamard-128 + rout epilogue on its own accumulator tile (16-row chunks,
8 KB shared staging, same stage order and FP operations as
`escha_finalize_dense`) and writes dst directly, skipping p_buf and the
finalize launch. Decode, A-stage overlap, shared-B, ldmatrix, HMMA, rotation,
and geometry are byte-for-byte unchanged; the partial/finalize path remains for
n_slices>1 and for the K2 128x64 experiment build.

Build: `build-cuda-parch14-fusedfin`; llama-bench
`2edc01535edc442f6a78fb48d96ffe51c7f3e3c924511c7044089df92298fef9`;
libggml-cuda `94e8928c1e854e021c39a43cbefb3040217a3c1d3e18b93eb20a5a7425f59e62`;
cubin contains `_ff<2,128,128>`/`_ff<3,128,128>` (128 regs).

## Result (symmetric repaired-profiler medians, 4 cold runs each)

| K2 family | control ms | fused ms | delta |
|---|---:|---:|---:|
| 5120->1024 | 14.394 | 15.031 | -0.637 |
| 5120->6144 | 59.032 | 58.909 | +0.124 |
| 5120->10240 | 76.661 | 75.817 | +0.844 |
| 5120->12288 | 37.909 | 38.018 | -0.109 |
| 5120->17408 | 184.267 | 185.233 | -0.966 |
| 6144->5120 | 70.450 | 71.043 | -0.594 |
| **K2 total** | **442.713** | **444.051** | **-1.338** |

The fusion is neutral: ~11.8 ms of finalize work moves into the MMA stage and
~10.9 ms of separate rotate+epilogue time disappears, net -0.52% of the
residual (within run-to-run noise). The in-kernel Hadamard epilogue (72 extra
CTA barriers and 64 KB shared staging per CTA) costs almost exactly the
separate finalize launch + fp32 partial read it saves. Uninstrumented A/B
(graphs ON, 10 samples each): fused median 1746.55 vs control 1729.79 tok/s
(+0.97%, same-session noise; CUDA-event totals are the decisive comparison).

## Correctness and safety

- Deterministic parity gate: P1/P2/P5/P6/P7 16/16 on the fused binary
  (`parity-fusedfin-001/compare-report.json`) — the fused epilogue is
  output-identical to the partial+finalize path at the token gate.
- Compute Sanitizer memcheck: full 3,200-record workload, exit 0,
  `ERROR SUMMARY: 0 errors` (`fusedfin-memcheck-001/`).
- All four fused profile captures: exit 0, 3,200/3,200 records, every row
  `mma-fp16`; no fallback.

## Decision

The smallest isolated fusion/lifetime correction consistent with the gate's
constraints (no decode/staging/overlap/geometry change) is **measured neutral
and rejected for production assembly**. The rotate/finalize boundary
(57.5 ms, 22.3% of the residual) is bounded and not the K2 divergence; the
dominant item remains the MMA-body execution (~199.7 ms residual), which
P-ARCH-13 (geometry) and P-ARCH-14 (boundary fusion) both leave unmoved. The
remaining untested Escha-side differences are the fused input transform and
decode/staging structure (direct-fragment decode without shared-B + ldmatrix,
45 KB shared footprint), out of scope for this gate. Hermes worker lanes were
attempted again; the first DeepSeek V4 Flash lane succeeded (arithmetic audit)
and two heavier lanes hit the per-lane timeout — non-decisive, all decisive
claims verified by the primary agent.

Exit evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-14/2026-08-29/`
(`boundary-001/`, `fusedfin-profile-001/`, `parity-fusedfin-001/`,
`fusedfin-memcheck-001/`, `uninstrumented-ab-001/`).

Rollback: rebuild without `-DESCHA_MMA_FUSED_FINALIZE_EXPERIMENT=1`; the
control binaries are untouched. The rejected kernel remains flag-gated in the
source and is not on the production path.
