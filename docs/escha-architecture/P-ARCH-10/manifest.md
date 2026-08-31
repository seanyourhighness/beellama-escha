# P-ARCH-10 — SM120 K2 cross-tile activation overlap

**Status:** `COMPLETE — CASE A: CROSS-TILE OVERLAP IS SAFE AND LARGE`  
**NEXT_GATE:** `P-ARCH-11`

## Isolated correction and representation

Only `escha_matmul_dense_tiled_mma<K,128,128>` changes.  Defining
`ESCHA_MMA_SM120_ASYNC_EXPERIMENT=1` selects its existing double-buffered
`cp.async` activation-copy path on SM120 rather than the synchronous `uint4`
fallback.  K, BM/BN, dispatch, MMA instruction, epilogue, and production
defaults are unchanged.

The K2 B representation is unchanged from P-ARCH-01: decoded weights are still
materialized as `[BN][16]` FP16 `s_w`, then loaded as
`tile<8,8,half2>` `ldmatrix` B fragments.  This is overlap only, not direct
fragment decode.

## Measurement table

RTX 5090 / SM120; K2 M=2048, IC=5120, OC=17408; persisted P-ARCH-05 IDs;
`-b 2048 -ub 512`, F16 KV, FA on.  CUDA-event runs disable graphs and contain
512 measured K2 records after warm-up.

| Variant | Kernel ms | TFLOP/s | Regs/thread | Shared mem | Barriers/CTA | Correctness | Stable |
|---|---:|---:|---:|---:|---:|---|---|
| Bee baseline | 3.812000 | 95.78 | 136 | 13,824 B dynamic | 960 | prior qualified | yes |
| Escha reference | 1.203000 | 303.48 | 80 | 45,056 B | not recovered | reference | yes |
| Bee SM120 async overlap | **0.651196** | **560.62** | **128** | 13,824 B dynamic | 960 | P1/P2/P5: 16/16 prefix | 3/3 complete |

Measured-run means: `0.649255`, `0.652628`, `0.651705` ms.  The candidate
keeps all three barriers for each of 320 reduction tiles (960/CTA), including
the shared-B publication barrier.  It overlaps the next global-to-shared A copy
with current-tile compute; it does not remove shared-B materialization.

`fraction of gap explained = (3.812 - 0.651196) / 2.609 = 1.211500`
(121.15%).  The value is intentionally not clipped: the candidate is faster
than the historical Escha event-time reference in this configuration.  P-ARCH-11
must reproduce this magnitude across K2/K3, prompts, ubatches, workspace, and
numerical regression before any default promotion.

## Safety and correctness gate

- Isolated build, three profiled 2k runs, and full-prefill run all exited 0
  under hard timeouts; GPU was idle after each command.
- All representative records selected `route=mma-fp16`; no fallback, CUDA
  error, illegal access, or launch failure appeared in evidence.
- Deterministic e3 reference-prefix validation at temperature 0/seed 42:
  P1 conversation, P2 factual, and P5 long context (1,544 prompt tokens) each
  matched 16/16 prefix tokens.  This is the established output/logit proxy,
  not a claim of full-continuation byte equality.
- Candidate CUDA library SHA-256:
  `9acc7f0241f1ddad84b9eae8b002a9ff780045e1f315859a051b2c9eebf7a27b`.

## Full 2,048-token prefill impact

Uninstrumented, graph-enabled `llama-bench -r 3` completed with exit 0:
`1734.51 / 1751.83 / 1798.80 tok/s`, mean **1761.71 tok/s** (1.162779 s).
Against the P-ARCH-04 official 1230.03 tok/s baseline (1.665000 s), this is
**1.43225x** / **+43.23%**, a **502.221 ms** reduction.

## Decision

**CASE A.** Preserve this compile-time correction for P-ARCH-11 only; do not
promote it to production and do not combine it with direct-B-fragment work.
The remaining dominant divergence is whether this unexpectedly large overlap
result survives the P-ARCH-11 cross-shape/stability envelope.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/P-ARCH-10/2026-08-29/async-overlap-001/`.
