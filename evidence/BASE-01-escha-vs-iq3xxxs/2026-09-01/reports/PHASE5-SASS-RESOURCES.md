# BASE-01 Phase 5 — Resource & SASS Comparison (2026-09-01)

Source: `cuobjdump --dump-resource-usage` on the frozen `build-cuda-base01`
libggml-cuda.so (canonical binary; the profile build is byte-different by design
but the kernels are identical apart from the env-gated dispatch hook).
Raw: `sass/resource-usage-full.txt`, `sass/RESOURCES.json`.

## Dominant ESCHA kernels (the packed projection path)

| kernel | regs | stack | local | smem | notes |
|---|---|---|---|---|---|
| escha_matmul_dense_tiled_mma<K2,128,128,fp32> | 128 | 0 | 0 | 1024 | FP32 MMA acc (HMMA.16816.F32) |
| escha_matmul_dense_tiled_mma<K2,128,128,fp16> | 97 | 0 | 0 | 1024 | FP16 MMA acc (HMMA.16816.F16) |
| escha_matmul_dense_tiled_mma<K3,128,128,fp32> | 128 | 0 | 0 | 1024 | |
| escha_matmul_dense_tiled_mma<K3,128,128,fp16> | 97 | 0 | 0 | 1024 | |
| escha_rotate_in_dense | 20 | 0 | 0 | 9216 | input Hadamard/rotation |
| escha_finalize_dense | 40 | 0 | 0 | 1536 | output Hadamard/scale/store |
| escha_matmul_partial<K2/K3> | 40 | 0 | 0 | 1024 | split-K partial writer |

These match the promoted Stage 2 certification (fp32 128 regs, fp16 97 regs,
no spills, smem 1024/13824-class) — no new resource surprise.

## Dominant IQ3 kernels (standard GGML MMQ path)

| quant type | fp16acc | BK=128 regs | stack | smem |
|---|---|---|---|---|
| IQ3_XXS (18) | 0 | 255 | 0 | 1024 |
| IQ3_XXS (18) | 1 | 252 | 0 | 1024 |
| IQ1_S (19) | 0 | 242 | 0 | 1024 |
| IQ3_S (21) | 0 | 252 | 0 | 1024 |
| IQ4_XS (23) | 1 | 255 | 32 | 1024 |
| Q2_K (10) | 1 | 255 | 64 | 1024 |

The MMQ kernels are register-heavy (255/thread at BK=128) but use standard
decode-once-then-MMA structure with no packed-code dependency tables, no
separate rotate/finalize passes, and no per-K-tile barrier triple.

## SASS classification (per mission rules)

- **measured:** per-family operator times (Phase 3), whole-run graphs-off totals.
- **proven from source/SASS:** ESCHA executes rotate -> packed-code GEMM
  (decode + dep-table traversal + shared-B + HMMA) -> finalize as separate
  passes with u_buf/p_buf partial buffers; IQ3 executes a single standard MMQ
  quantized GEMM kernel per projection (decode inside mainloop, no separate
  rotate/finalize); register counts from cuobjdump; no spills either side.
- **inferred:** the per-instruction decode-cost delta (SASS disassembly not
  fully extracted; full cuobjdump SASS dump exceeded host memory at 6 GB and
  was aborted). Classified as inferred, not measured.
- **unknown:** occupancy counters (WSL `ERR_NVGPUCTRPERM`), clock-accurate
  HMMA throughput per instruction.

## Normalization note
Per-CTA/per-invocation comparison is not attempted beyond the resource table;
the decisive evidence is the per-family wall-time delta (Phase 3) and the
M-scaling (Phase 4). Registers alone are not a controlled performance
variable here (documented P-ARCH-13: fewer registers made K2 worse).
