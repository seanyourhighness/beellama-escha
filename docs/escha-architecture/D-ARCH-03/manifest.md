# D-ARCH-03 — decode W2 path mapping

**Status:** `COMPLETE — FIRST MATERIAL DIVERGENCE IS THE DECODE W2 KERNEL FAMILY`
**NEXT_GATE:** `D-ARCH-04 — launch/graph/fusion comparison`

## Answer

Bee decode W2 = `escha_matmul_dense<K,1>` (`gen-splitk-fp32`, R=1, 64/55 regs,
128 threads, split-K up to 18) in a rotate->matmul->finalize triple per
projection (400 triples/step, per-call floor ~0.047 ms for K2 5120->17408).
Escha decode W2 = fused `escham_gemv_bw<1,K>` (256 threads, 64 regs, 2,048 B
shared, 67% occupancy) + `escham_mma_gemv<1,K>` for large shapes, with
in-kernel `had_in`/`had_epilogue` and multi-shard fusion (per-projection
equivalent ~0.024 ms). The first material divergence is the decode W2 kernel
family and launch structure, not registers per se.

Evidence: `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/evidence/D-ARCH-03/2026-08-29/kernel-map.md`
(cubin resource dumps from `build-cuda-parch10-async` and the retained
Escha decode trace; per-call floors from `scripts/escha-decode-w2-floor.py`).

An existing opt-in experimental Bee decode path (`ESCHA_WARP_GEMV=1`,
`escha_matmul_dense_warp<K>`, 33/37 regs) writes the identical partial layout
and is the candidate for D-ARCH-05.
