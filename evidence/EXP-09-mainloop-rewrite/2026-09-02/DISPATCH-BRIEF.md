# EXP-09 — Mainloop Rewrite Dispatch Brief (Sol PLAN Gate 1)

Date: 2026-09-02 · Branch: `escha-w2-prefill` · HEAD: `1c193ad4c` (pushed)
Repo: `/mnt/d/CODEX WORKSPACE/beellama-escha`
Sol = Codex CLI `gpt-5.6-sol` v0.151.0. Terra (Hermes) supervises math + gates.

## Decision (Sean, 2026-09-02)

Closure of EXP-07/08 pushed. Next: **full mainloop rewrite** of Bee's promoted
Stage 2 ESCHA-W2 prefill MMA kernel to mirror the official `escham_code_gemm`
warp-local two-band structure — adapted to Bee's partial→finalize contract —
with Sol planning the entire rewrite, then implementing it under tight
supervision (no errors in math or execution).

## Context chain (read first)

- `AGENTS.md` (repo conventions; context-budget rules apply)
- `docs/current-state.md` §EXP-07/EXP-08 + Series status (2026-09-02)
- `docs/escha-prefill-experiment-ledger.md` (EXP-04..08 rows)
- `evidence/EXP-05-audit/2026-09-01/AUDIT.md` — the reference audit
- `evidence/EXP-07-mainloop-coopdecode/2026-09-01/` — register-B negatives
- `evidence/EXP-08-fusedfinalize/2026-09-01/` — fused-finalize negatives
- Kernel + dispatch: `ggml/src/ggml-cuda/escha-moe.cu`
  (`escha_matmul_dense_tiled_mma` line 958; host `ggml_cuda_op_escha_mul_mat`
  line 1840; promoted mixed-acc policy at line 2003–2007)

## Established facts that bound the design

1. **Control (promoted Stage 2, EXP-04 Phase 2, +9.31%):** BM128×BN128 CTA,
   256 threads, WN=2/WM=4, MT=2/NTT=8 fp16 or fp32 accumulator fragments per
   warp (`FP16_ACC` when IC ≤ 6144). Shared-B round trip: decode into
   `s_w[BN][16]`, CTA barrier, `ldmatrix` B fragments, HMMA; 3 CTA barriers per
   16-wide K tile; cp.async double-buffered A. **Regs 97 (fp16) / 128 (fp32),
   no spills, 13,824 B smem, 2 CTA/SM occupancy.**
2. **Official hot kernel** (wheel sha `735f4b7a…`, sm_120a):
   `escham_code_gemm_kernel<1,K,128,64,2,FP16ACC,FEPI>` for gate/up/qkv —
   CTA covers 128 rows × two independently-owned 64-col bands (grid.x=OC/128),
   256 threads, **80 regs (fp16), 45,056 B smem**, warp-local decoded B
   (**0 STS.U16**), in-kernel Hadamard (80 SHFL.BFLY), 40+40
   WARPSYNC/ENDCOLLECTIVE, 136 LDS. down_proj uses fp32 acc (122 regs), and
   official BM=128/BK=2 is a trap there (BM=64 halves it).
3. **EXP-07 negative (register-B):** removing the shared-B round trip by
   warp-collective register decode exceeded ceilings (124–154 regs vs 97/128)
   → occupancy loss. Any register-B design MUST stay ≤ control ceilings and
   prove no spills, or it fails pre-timing.
4. **EXP-08 negative (fused finalize):** warp-pair-owned epilogue with 8
   batches × 2 named barriers serialized (−18.29%). Finalize fusion is only
   viable as a **full-CTA-parallel** design; otherwise keep the separate
   partial→finalize contract (which is the promoted default and NOT the
   deficit — the deficit is the packed-GEMM body).
5. **Mixed-acc policy is a +9.31% promoted win — preserve it.** FP16 acc for
   IC ≤ 6144, fp32 above; per-family route tags must stay unambiguous
   (`mma-fp16-mixedacc` / `mma-fp32-mixedacc`).
6. **A double-buffer (EXP-01) is banked.** Keep cp.async overlap; do not
   regress to sync A staging.
7. **Partial-store seam** (Stage 2 fp16 lines 1153–1192): fp16 `tile_ah`
   fragments pack 2 f16 lanes per 32-bit reg — mapping x[0].x/x[0].y/x[1].x/
   x[1].y to fp32 fragment coords l=0..3 is the correct store math. Preserve
   it exactly for any fp16-acc partial store.
8. **Benchmark contract:** canonical full-Escha control artifact
   `escha-w2-lowgpu-mono-parity.gguf`, 2048-token matched prefill, graphs ON,
   9-pair ABBA, per-arm CV, paired-log G + CI. Gates: ≥5% tok/s gain to
   classify SMALLER POSITIVE; ≤2% decode regression; P2/P7 16/16 parity;
   route proof 800/800 0-mismatch; resource gates REG ≤97 fp16 / ≤128 fp32,
   no spills, smem within budget.

## Deliverable (PLAN phase — no source edits)

Write `evidence/EXP-09-mainloop-rewrite/2026-09-02/REWRITE-PLAN.md` containing:

1. **Design** — exact CTA tile/geometry choice, band ownership, warp/thread
   mapping (MT/NTT per warp), and where it diverges from EXP-07's failed
   register-B layout (why this one stays ≤97/128 regs).
2. **Decode math** — per-thread codebook-word assignment, funnelshift
   positions, dw0/dw1/dsh mapping, r/column invariance proofs, the exact
   register-level B fragment build (ldmatrix vs direct), and the barrier
   scheme (count + kind per K tile vs control's 3 CTA barriers).
3. **Accumulator + partial-store math** — fp16/fp32 fragment layout, tile
   indices, store seams, boundary guards (row < n_rows), split-K slice lo/hi
   handling, and why results remain bit-compatible with the current decode
   (same codebook/LUT/dep semantics).
4. **Implementation steps** — compile-guarded candidate block + new route tag
   (e.g. `ESCHA_MMA_SM120_MAINLOOP_REWRITE_EXPERIMENT`, `mma-fp16-mlw` /
   `mma-fp32-mlw`), revertible by removing the block; default path untouched.
5. **DoD + verification gates** — exact commands for compile, cuobjdump
   resource check, ESCHA_PROFILE route proof, parity, decode, matched bench.
6. **Risks/watch-outs** — register ceiling, barrier count, smem, tail rows,
   fp16 store seam, split-K fallback, what NOT to touch (finalize kernel,
   standard-Qwen path, artifact side).

Do NOT modify any source file. Do NOT commit. Do NOT run judge/gbrain.
Write ONLY the plan file, then report the design's key numbers (regs/smem/
barrier count/HMMA per tile predicted) in your final summary.
