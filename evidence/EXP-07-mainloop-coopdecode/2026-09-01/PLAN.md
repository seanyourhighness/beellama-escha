# EXP-07 — Warp-collective register-decoded B (eliminate shared-B round trip) — PLAN v2

Supersedes PLAN v1 (which Sol REVISE'd: "CTA-cooperative decode-once" restated
the control — the control already decodes into shared `s_w` once per CTA).

## Mechanism evidence (Sol REVISE item 3 — pre-implementation probe)

SASS counts from EXP-05 retained extracts (`EXP-05-reference-mainloop-audit/
2026-09-01/*sass.txt`), sm_120a:

| symbol | STS for B | LDS for B | HMMA.16816 | WARPSYNC/ENDCOLL | SHFL.BFLY | total |
|---|---|---|---|---|---|---|
| Bee K2 fp16 control (`escha_matmul_dense_tiled_mma<K2,128,128,fp16>`) | 8 STS.U16 | 8 LDS.64 | 16 | 0 | 0 | 625 |
| Official hot K2 (`escham_code_gemm_kernel<1,2,128,64,2,T,T>`) | 0 (B stays in regs) | 0 | 64 | 40 ENDCOLLECTIVE | 80 | 1231 |

Official keeps decoded B in registers (warp-local, no STS.U16, no LDS.64 for B),
exchanges fragments warp-collectively, and HMMAs 4x per K-tile (64 vs 16 is a
per-symbol specialization difference; the structural point is 0 shared-B
round trip). EXP-05 measured official faster on gate/up/qkv short-IC
(0.78–0.90x). EXP-02 (Bee direct-fragment) failed at 176 regs because it
duplicated decode across all four M-warps. The un-tested middle path is a
**warp-collective register decode with disjoint slices + shuffle exchange**,
which removes the shared-B STS+LDS+2-barrier round trip per 16-wide K tile
without per-M-warp duplication.

## ONE variable (this experiment)

Replace the shared-B materialization for the B operand in the promoted tiled
MMA kernel with **warp-collective register decode**:

- `ESCHA_MMA_REGISTERB_EXPERIMENT=1` compile gate; new guarded specialization
  `escha_matmul_dense_tiled_mma_regb<K,BM,BN,FP16_ACC>`.
- Decode ownership: each of the WN=2 column-warp groups decodes a DISJOINT
  slice of the 16-wide x BN tile directly into its MMA B fragments; fragments
  are exchanged across the WN boundary only if the MMA mapping requires it
  (same fragment layout as ldmatrix would produce). All M-warps in a column
  group share the same decoded B fragments by construction — NO per-M-warp
  duplicate decode (this is the EXP-02 fix).
- Removed per K-tile: `STS.U16` B stores, `LDS.64` B reloads, and one
  `__syncthreads()` barrier (the decode-publish barrier). A-stage cp.async
  double-buffer and the payload-ahead (`s_pay`) mechanism are UNCHANGED.
- Unchanged (frozen): BM=BN=128, grid, split-K, accumulator policy
  (IC<=6144 fp16 else fp32), input rotation, partial/finalize path, route
  dispatch. Candidate route tags: `mma-fp16-regb` / `mma-fp32-regb`.
- Applies to K2 and K3 prefill MMA only (the FFN + attn_qkv families that
  carry the gap); generation/decode/fallbacks untouched.

## Pre-implementation gates (Sol REVISE item 3)

Before timing, demonstrate from the compiled candidate SASS (vs control):
1. STS.U16 count for B == 0 and LDS.64 B-reload == 0 in the new symbol
   (A-stage STS/LDGSTS and payload writes may remain).
2. Register count <= control (fp16 <= 97, fp32 <= 128), STACK/LOCAL 0, no
   spill loads/stores.
3. Barrier count per K-tile reduced by >=1 relative to control (compile-time
   from source structure; SASS BAR.SYNC count reduced).
4. HMMA count >= control for the same tile (no lost MMA work).
5. Route proof: profile records on new tags for the intended families with
   zero fallback/predicate mismatch; unchanged families keep prior tags.
6. Per-shape K2/K3 FFN direct measurements (M=512/1024/2048) vs control using
   the same profile harness (attribution only); candidate must show the
   mechanism removes measurable matmul time before the matched campaign.

## Route arithmetic (corrected — Sol REVISE item 4)

Prior 800/800 route records = full prefill projection set (400 projections x2
K2/K3 = 800 calls at M=2048, one pass). The three FFN families are 128 calls
each per pass (ffn_gate K2 128, ffn_up K3 128, ffn_down K3 128) = 384 FFN
records; attn_qkv 96, attn_gate 96, ssm_out 96, attn_q 32, attn_kv 64, plus
attn_output 16 etc. Expected candidate-tag counts after the change:
- `mma-fp16-regb`: every IC<=6144 K2/K3 prefill call (gate/up/qkv/attn_gate/
  attn_q/attn_kv/short-IC), unchanged count vs control's fp16 route.
- `mma-fp32-regb`: IC>6144 calls (ffn_down 17408->5120 + others).
- Unchanged routes: generation/ragged/cublas/wmma tags stay identical.
State exact tag counts in the route-proof file; zero mismatch.

## Gates (family-regression basis explicit — Sol REVISE item 5)

- Family regression gate uses **total projection time per family**
  (rotate+matmul+epilogue from ESCHA_PROFILE, matching BASE-01 convention),
  ratio always candidate/control; no family >5% slower (candidate worse).
- Keep: FFN-family matmul >=10% faster (candidate/control <0.90), aggregate
  projection >=7% faster, full-2K wall >=5% faster (median).
- Resource/SASS/numeric/parity/decode/9-pair gates as v1 (P2/P7 16/16,
  decode <=2%, CV<=2% else paired-log G>=1.05 CI>1.0 >=8/9, FP32 families
  bitwise, FP16 rel-RMS <=1.5e-3).

## Risks / rollback

Risk: fragment layout mismatch vs ldmatrix (must produce identical MMA A/B
fragment geometry), cross-warp exchange correctness, register growth if the
exchange is over-buffered, barrier removal safety (s_pay publish ordering).
Rollback: remove the guarded specialization + candidate build dir only;
promoted default unchanged; control build SHAs unchanged.

## Evidence dir

`evidence/EXP-07-mainloop-coopdecode/2026-09-01/` (renamed content: v2 plan,
mechanism-probe SASS counts above).
