# EXP-10 — Cooperative BK32 double-buffered B (Decision B)

Date: 2026-09-02 · Branch `escha-w2-prefill` · HEAD `969db62df` (clean)
Authorizer: Sean (Decision B). Sol = Codex CLI `gpt-5.6-sol`, high reasoning.
Terra (Hermes) supervises math + gates.

## Decision

Authorize the **single** cooperative BK32 double-buffered-B architecture
experiment for exact packed Escha-W2 execution (Sol retrospective direction 3,
Decision B). This is the one authorized packed-runtime kernel experiment; per
Decision B, if its gate fails, incremental kernel work stops and the remaining
paths are the load-time transcode cache or a new sidecar representation/kernel
project.

## Why this design (Sol retrospective, direction 3)

EXP-02/07/09 removed shared-B and each paid 4× decode duplication or all-row
ownership (124–176 regs vs 97/128 → 1 CTA/SM). EXP-03 changed CTA aspect and
was neutral. The untested variable: **temporal pipelining of the decode that
Bee already does cooperatively** — decode tile N+1 while MMA runs on tile N,
instead of serializing decode→barrier→MMA per 16-wide K tile.

## Design constraints (frozen)

- **Preserve the broadcast**: CTA-cooperative decode + shared publication. Do
  NOT duplicate row-warp decode; do NOT make B warp-private.
- **Current control**: `escha_matmul_dense_tiled_mma` (line 958, promoted
  Stage 2, mixed acc, BM128×BN128, 256 threads). Three CTA barriers per 16-wide
  K tile today. cp.async double-buffered A already banked (EXP-01).
- **Candidate**: double-buffer decoded B; stage BK32 (two 16-wide K tiles) or
  BK64 (four) so decode of the next K tile overlaps MMA of the current one.
  Producer/consumer scoped barriers. Dedicated producer warps only if consumer
  MMA coverage remains sufficient. K-tile order, partial layout, split-K
  lo/hi, store seams, and `escha_finalize_dense` unchanged.
- **Register gates** (measured, decisive): FP16 ≤97–104, FP32 ≤128, no spills,
  no STACK/LOCAL, ≥2 resident CTAs/SM by resource count.
- **Route**: `mma-fp16-bk32` / `mma-fp32-bk32` tags (or similar), macro-guarded
  `ESCHA_MMA_COOP_BK32_EXPERIMENT`, default path byte-identical with macro off.
- **Bench gates** (Decision B): target-family (K2 5120→17408 M2048) matmul
  ≥10% faster; full 2K ≥5% faster with ≥4/5 samples over control median; no
  family or depth (128→4096) regression >5%; decode ≤2% regression; P2/P7
  16/16 parity; route proof 800/800 0-mismatch. <2% or resource failure =
  REJECT + revert + stop incremental kernel work per Decision B.

## Gate sequence (frozen)

1. Sol Gate 1 PLAN (this dispatch) → Terra math audit.
2. Sol implementation (guarded, revertible) → Terra diff review → Sol code review.
3. Build control + candidate; cuobjdump resource/SASS gate (regs, smem, spills,
   BAR count per K tile vs control's 3, STS.U16, HMMA preserved, 2-CTA
   residency).
4. Route proof 800/800; family smoke; fragment/partial/tails/split-K gates;
   compute-sanitizer smallest tails.
5. P2/P7 parity + decode bench.
6. Canonical matched 9-pair ABBA campaign (graphs ON) per EXP-08/09 protocol.
7. Sol final gate verdict (CONFIRM-promote / CONFIRM-REJECT) → docs/ledger +
   commit/push + GBrain.

Evidence dir: `evidence/EXP-10-coop-bk32/2026-09-02/`.
