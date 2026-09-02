# EXP-11 — Terra review of Sol PROGRAM-PLAN.md (Gate 1)

Date: 2026-09-02 · Plan: `PROGRAM-PLAN.md` (Sol, 48,934 B, 899 lines).

## Independent verification

- **Oracle math** verified line-by-line against the reference
  `/home/sean/research/escha-refs/yaniss/tools/escha/escham_cpu.py`
  `reconstruct_deploy_weight`: decode → fp32 → `escha_t128(w.T).T` (IC) →
  `w * rin[:,None]` → `escha_t128(w)` (OC) → `w * rout[None,:]` → (IC,OC).
  Plan §5.2 steps 1–7 reproduce this exactly, plus the GGUF (OC,IC) transpose
  and quantize. **CONFIRMED.**
- **Canonical GGUF rin/rout are the folded vectors**: the converter reads
  `.escha_rin/.escha_rout` from the GGUF and calls the same oracle with no
  extra `s_in/s_out` application — identical to what EXP-11 reads. No double
  scaling. **CONFIRMED.**
- **800 count arithmetic**: 400 cacheable projection roles × 2 traversals =
  800; Attempt 1 = 192 FFN roles × 2 = 384 stock-cache + 208 × 2 = 416 packed.
  384 + 416 = 800. **CONFIRMED.**
- **Target sizes** use the Q2_K 84 B/256-value ratio; FFN 5.615 GB,
  full-body ~8.70 GB overlay, active model in the 23I 9.35 GB class.
  **CONFIRMED by arithmetic** (measured at implementation).
- **Decode mechanism** (why decode improves on stock FFN: small-row gen path
  bypasses the packed custom generation split-K/warp-GEMV) matches
  P-ARCH-21A's 30.44 vs 23.69 tok/s observation. **CONFIRMED.**

## Review flags (Sol §16) — decisions

1. **Canonical-input oracle: CONFIRMED** (see above). Golden tensor = layer 0
   `ffn_gate` reconstructed; byte-compare against the shared oracle module.
2. **FFN Q2_K as Attempt-1 quant: APPROVED.** 21A's donor mixed-IQ bytes are
   NOT the oracle — reconstruction is the contract. Q2_K matches the funded
   size class (~8.5 GB resident) and P-ARCH-23's proven quant choice
   (Q4 slower, Q6 tied). Expectation ≠ claim: quality earned at the milestone.
3. **800 count: CONFIRMED** (arithmetic above). Harness must prove traversal
   count, not normalize.
4. **120 s cold target: ACCEPT as a projection gate.** Measure one layer
   (3 FFN tensors) and project ×64 before the full build; if the projection
   exceeds ~120 s or RSS >3 GB, that IS the documented Attempt-2 operational
   trigger — pre-authorized without weakening the oracle.
5. **Whole-cache hashing: ACCEPT the risk, keep cryptography.** If the
   double-read misses ≤10 s / ≤10% cached-load, integrate with the loader
   read/upload pass; never weaken validation.
6. **Masked bias sidecars: CONFIRMED as required.** The runtime intentionally
   ignores the checkpoint bias-correction; overlay masking must exclude
   `.bias` entries so no accidental correction is applied.
7. **Full-body purity: CONFIRMED.** 23I's donor-copied full-attention bytes
   are not an oracle; reconstructing to Q2_K is a NEW quality variable —
   family order and per-family gates in the plan are correct.
8. **Quality floor: use ≥65/75** ("no worse than canonical control"), NOT
   64/75. The retrospective's 64/75 was a loose paraphrase; 65/75 is the
   correct no-regression reading of the certified 65/75 control.
9. **Attempt-3 size accounting: APPROVED** — per-weight LUT indices
   prohibited; only compact shared/per-tile descriptors meet ≤25%.
10. **Rollout default off → promote to auto only at a separate final gate:
    CONFIRMED.**

## Verdict

**PLAN=READY for Attempt 1 funding.** Sol's tier structure, cache contract,
loader architecture, revision triggers, and gate commands are sound. Sean has
already funded the program and granted 3 attempts; this review authorizes
implementation of Attempt 1 per §12 sequence: (1) freeze schema, (2) refactor
converter primitives with golden tests proving the existing converter's bytes
are unchanged, (3) one-layer/3-FFN overlay generator + validation + time/RSS
projection, (4) loader mounting, (5) qwen35 precedence, (6) full 192-tensor
cache + oracle/route gates, (7) P2/P7/decode/2K, (8) Sol Attempt-1 gate.

Non-negotiables: canonical GGUF never mutated; donor bytes never enter the
cache; oracle byte-equality is the only correctness bar (MAE never
substitutes); fail-closed cache; standard-Qwen isolation; decode ≤2%;
rollout stays opt-in.
