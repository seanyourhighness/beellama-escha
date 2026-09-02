# EXP-11 — Funded load-time transcode cache / new sidecar representation

Date: 2026-09-02 · Branch `escha-w2-prefill` · HEAD `b1734d0b8` (clean)
Authorizer: Sean — funded path after EXP-10 closed the packed-runtime kernel
line (Decision B). Sol = Codex CLI `gpt-5.6-sol`, high reasoning, for PLAN,
verification, and FINAL gates. Terra (Hermes) supervises math + gates.

## Decision

Execute the load-time transcode cache / new sidecar representation program
(Sol retrospective directions 2 + 5). Sean grants a **3-retry budget with
different approaches**; Sol plans the tier structure, verifies each attempt,
revises the plan when new information blocks an approach, and issues the final
gate.

## Why this path (evidence)

- BASE-01: 94.6% of the prefill gap lives in the FFN projections; the packed
  sidecar representation is the structural blocker, not the kernel schedule.
- P-ARCH-21A (FFN→standard, offline): 2300 → 2814 tok/s, quality PASS.
  P-ARCH-23I (full-body standard, offline): ~619–621 ms / ~3300 tok/s.
  These prove the representation fix works; they are offline artifact rewrites.
- EXP-11 moves the same proven reconstruction/quantization into the RUNTIME as
  a load-time, cached transcode — preserving the canonical packed GGUF as the
  source of truth while executing the hot body on stock GGML `MUL_MAT`.
- Converter `convert_escha_to_gguf.py` already contains the validated,
  cross-checked primitives: `reconstruct_escha` (Hadamard-128 + rin/rout),
  `reconstruct_gdn_gate`, `quantize_q2_k/q4_k/q6_k` (byte-verified vs
  ggml-quants.c, MAE 0.0035), and the exact tensor-name/write paths that
  produced P-ARCH-23I's inventory (856 tensors, escha_version markers kept).

## Guiding requirements (frozen)

1. **Canonical artifact is the source of truth.** The packed
   `escha-w2-lowgpu-mono-parity.gguf` remains the interchange file. Nothing
   rewrites or mutates it.
2. **Runtime = stock GGML on the hot body.** After load-time transcode, FFN
   (+ gate/linear-attention as tiered) projections execute as ordinary
   `MUL_MAT` tensors with standard quant types, exactly like P-ARCH-23I.
   Decode path must not regress.
3. **Determinism + cache.** Transcode is deterministic (fixed seed-free numpy
   path as the converter). A persistent disk cache keyed by (source GGUF SHA,
   layer, tensor role, target quant) must make repeated loads ~free. Cache
   corruption/mismatch fails closed to re-transcode, never to wrong weights.
4. **Correctness gate = oracle.** The runtime-transcoded weights must be
   numerically identical to the offline converter's reconstruction for the
   same inputs (bit/MAE parity vs `reconstruct_escha` oracle), and the loaded
   model must reproduce P-ARCH-21A/23I quality behavior. No silent semantic
   drift (P-ARCH-21B lesson: names/shapes do not establish weight identity).
5. **Escha semantics preserved where required.** qwen35.escha.version gating
   (A_log → -exp, GDN head order) must still be honored for whatever remains
   packed (LM head / codec tables) exactly as 23I does.
6. **Bench goal:** canonical 2K prefill ≥ 5% over the 2319-control toward the
   P-ARCH-23I ~3300 tok/s class, decode ≤2% regression, standard-Qwen
   isolation, quality no worse than control on the medium suite.

## 3-retry tier structure (Sol to refine in Gate 1)

- **Attempt 1 (direction 2a):** load-time transcode cache — FFN-only first
  (192 tensors; P-ARCH-21A's proven scope), reconstruct→quantize→cache→dispatch
  stock MUL_MAT. Expected ~2814 tok/s class; if clean, extend tiers toward
  gate/linear-attention (23I scope).
- **Attempt 2 (direction 2b):** if FFN-only cache hits a loader/dispatch or
  memory wall, revise with new information (e.g., transcode granularity,
  in-memory-only cache, tensor-layout, async load, split across layers) —
  Sol defines the concrete pivot.
- **Attempt 3 (direction 5):** if load-time transcode cannot reach the goal
  (e.g., load latency or VRAM unacceptable), pivot to the new MMA-ready
  sidecar representation (repack the code stream into fragment-ordered,
  pre-resolved-index form) as a funded kernel+format project.
- A roadblock triggers Sol revision BEFORE the next attempt; each attempt is
  gated (correctness oracle → route proof → quality smoke → 2K bench) before
  it can be called successful or exhausted.

## Gate sequence (frozen)

1. Sol Gate 1 PLAN (tiers, architecture, loader/dispatch touch points, cache
   format, DoD, gates) → Terra review → Sean funds attempt 1.
2. Sol implementation (guarded, revertible) → Terra diff review → Sol code review.
3. Correctness oracle vs offline converter; route proof (800/800 target
   shapes on stock MUL_MAT + remaining escha routes); P2/P7; decode.
4. Canonical matched 2K campaign vs promoted Stage 2 control.
5. Sol verification gate per attempt; on failure/roadblock → Sol revision →
   next attempt (budget 3 total).
6. Sol FINAL gate → docs/ledger/GBrain + commit/push.

Evidence dir: `evidence/EXP-11-transcode-cache/2026-09-02/`.
