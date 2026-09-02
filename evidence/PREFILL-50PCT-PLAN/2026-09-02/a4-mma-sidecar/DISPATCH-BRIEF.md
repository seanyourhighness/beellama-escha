# A4 / EXP-11 Attempt 3 — MMA-ready sidecar: dispatch brief

Date: 2026-09-02 · Priority per Sean: spend the Codex cap (resets ~12:20 PDT)
here. Authorizer: Sean. Sol plans/verifies/final-gates; Terra supervises.

## Goal

`escha-mma-cache-v1`: repack the canonical Escha code stream into a
fragment-ordered, pre-resolved-index accelerator overlay so warp-local decode
becomes cheap enough to run the PACKED artifact at/above standard-GGML speed.
This is the ONLY path that keeps exact packed execution AND has a +50% prefill
ceiling (canonical control 2319 → target 3479 tok/s; LowGPU parity is 3339).

## Frozen spec (from EXP-11 PROGRAM-PLAN §10 Attempt 3)

- **Order:** `[projection][output-CTA][K-stage][64-col band][warp publication
  record]`; record = contiguous K2/K3 payload words + compact pre-resolved lane
  descriptors (ring-word pair, shift, fragment row/col, dependency selection).
  Shared descriptors live ONCE per overlay. NO per-weight 16-bit LUT index
  (would expand ~8×).
- Versioned tensor suffix `*.escha_mma_code`; distinct `GGML_OP_ESCHA_MMA_MUL_MAT`
  consumed by a new/separated CUDA path. Loader picks it only when
  escha.version==1 AND a validated overlay entry exists. Canonical packed data
  is the fallback on unsupported devices; never duplicated in VRAM.
- Deterministic, content-addressed builder with the same atomic cache contract.

## Discriminating first slice (this dispatch's deliverable)

Repack exactly **one K2 5120→17408 FFN projection** (blk.0 ffn_gate or a
representative K2 gate) and build a **direct-op harness** before any full-model
work.

## Mechanism gate (fail-fast; do NOT proceed past it without all PASS)

1. Representation growth ≤25% over that projection's canonical packed
   code+rotation payload.
2. fp16 registers ≤104, fp32 ≤128, STACK/LOCAL/spills = 0.
3. Decode/address ALU drops ≥30% from the promoted packed kernel
   (`escha_matmul_dense_tiled_mma` SASS decode-ALU baseline: IMAD+LOP3+IADD3
   ≈ 170 fp16 / 165 fp32 on K2 per earlier EXP-10 SASS counts; recompute on
   the actual control symbol).
4. Two-CTA residency remains possible.
5. Direct-op matmul ≥15% faster at M=2048 than the packed control.

## Critical design tension to solve (why EXP-09/10 failed — must differ)

EXP-09 died at 145–176 regs from 4× row-warp decode duplication + per-weight
index math. EXP-10 died on accumulator stack-homing. Attempt-3's entire thesis
is that PRE-RESOLVING the index chain (dw0/dw1/dsh/fragment mapping ONCE, in
the overlay) removes the per-weight ALU that killed EXP-09, so decode becomes
cheap loads/shifts from contiguous records. The plan must show the register
accounting that proves this stays ≤104/≤128 — same discipline as EXP-10's
Terra audit. If the repack cannot cut decode ALU ≥30% at ≤25% growth, the
mechanism gate fails and Attempt 3 (and the packed-exact line) closes.

## Why this could beat LowGPU (the +50% case)

Standard Q2_K/IQ3 GGML kernels dequantize then MMA. A fragment-ordered packed
stream at ~2.5 bpw with pre-resolved decode could beat both: smaller memory
footprint than LowGPU (9.57 GB) AND cheaper decode than the current packed
kernel. That combination is the only measured-plausible route past 3479.

## Evidence to read

- `evidence/EXP-11-transcode-cache/2026-09-02/PROGRAM-PLAN.md` §10 Attempt 3
- `evidence/EXP-09-mainloop-rewrite/2026-09-02/` (REWRITE-PLAN, REJECTION,
  SASS) — the decode-ALU failure to beat
- `evidence/EXP-10-coop-bk32/2026-09-02/` (plan/audit/rejection) — the
  register/spill discipline to respect
- `evidence/EXP-05-audit/2026-09-01/AUDIT.md` — official kernel structure
- `ggml/src/ggml-cuda/escha-moe.cu` — control decode formula (lines ~1015-1110),
  host dispatch, the escha_dep_pi/funnelshift/codebook chain to pre-resolve

Evidence dir: `evidence/PREFILL-50PCT-PLAN/2026-09-02/a4-mma-sidecar/`.
