# D-ARCH-07 — final decode attribution

**Status:** `COMPLETE — W2 DECODE GEMM STILL DOMINANT; REMAINING FIX IS A NEW KERNEL`
**NEXT_GATE:** `D-ARCH-08 — production regression of the combined candidate`

## Answer

After the D-ARCH-05 correction, Bee decode per-step is ~19.7-21.5 ms at c=1
(best corrected path ~19.6 ms) vs Escha ~11.6 ms — a remaining ~1.7x step
deficit. W2 decode GEMM remains the dominant stage (>=82% of step) on both
runtimes. At c=8 the aggregate deficit is ~8x (Bee ~62 vs Escha ~496 tok/s)
because Escha's CUDA-graph batches decode streams while Bee does not batch.

| metric | Bee (best corrected) | Escha | remaining deficit |
|---|---:|---:|---:|
| step ms (c=1) | 19.6 | 11.6 | 1.69x |
| aggregate tok/s (c=8) | 62.4 | 496.4 | 7.95x |
| dominant stage | W2 decode GEMM (>=82%) | W2 decode GEMM (83.4%) | — |

The remaining W2 gap is not addressable by the bounded corrections available
(geometry, boundary fusion, flag-switch decode paths were all measured);
closing it requires a dedicated Blackwell batched-GEMV decode kernel in Bee's
style of `escham_gemv_bw`. That is beyond a bounded experiment, so decode
optimization stops at this clean point and D-ARCH-08 assembles the production
regression.

Evidence: D-ARCH-01/02/03/05 manifests and their evidence directories.
