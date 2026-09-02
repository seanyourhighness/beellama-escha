# EXP-09 — Mechanism-gate REJECTION (pre-timing)

Date: 2026-09-02 · Branch `escha-w2-prefill` · Control HEAD `1c193ad4c`
Candidate: `ESCHA_MMA_SM120_MAINLOOP_REWRITE_EXPERIMENT` guarded
`escha_matmul_dense_tiled_mma_mlw` (macro-on build `build-cuda-exp09-mlw`).

## Verdict

**REJECTED + REVERTED at the pre-timing resource gate.** No benchmark or
correctness campaign was run, per the approved fail-fast rule (plan §7.2:
"Any result over 97/128 … rejects the candidate before timing").

## What was built (faithful to the approved plan)

- 128×128 CTA expressed as two independent 128×64 bands (band=warp&1,
  row_group=warp>>1); per-warp footprint 32 rows × 64 cols = control's.
- Direct warp-local `tile_b` construction from the codebook payload
  (`escha_decode_mma_mlw_weight`), one 8-column B fragment live at a time;
  no decoded-B shared store, no B `ldmatrix`.
- Exactly **two CTA barriers per K tile** (vs control's three).
- cp.async double-buffered A, split-K partials, FP16/FP32 store seams copied
  literally from control. Smem 9,728 B (static_asserted). Route tags
  `mma-fp16-mlw`/`mma-fp32-mlw`; macro-off path byte-identical.
- Terra code review of the diff: decode math (sp/dw0/dsh/funnelshift/codebook),
  geometry, store seams, host dispatch all verified correct before build.

## Resource gate (cuobjdump, sm_120a)

| symbol | REG | SHARED | STACK | LOCAL |
|---|---:|---:|---:|---:|
| control fp16 (K2/K3) | 97 | 1,024 (static; dyn 13,824) | 0 | 0 |
| control fp32 (K2/K3) | 128 | 1,024 | 0 | 0 |
| **mlw fp16 K2 / K3** | **145 / 150** | 1,024 (dyn 9,728) | 0 | 0 |
| **mlw fp32 K2 / K3** | **176 / 176** | 1,024 | 0 | 0 |

Ceilings: fp16 ≤97, fp32 ≤128 → **FAIL** both modes, both K. No spills
(STACK=LOCAL=0) — the pressure is live registers, not a spill defect.

## SASS analysis (K2 symbols, static occurrences)

| metric | control fp16 | mlw fp16 | control fp32 | mlw fp32 |
|---|---:|---:|---:|---:|
| BAR | 3 | **2** | 6 | **4** |
| HMMA.16816 | 16 | 16 | 32 | 32 |
| STS | 10 | 2 | 20 | 4 |
| LDSM | 10 | 2 | 20 | 4 |
| LDS | 8 | 32 | 16 | 64 |
| decode ALU (IMAD+LOP3+IADD3) | 170 | 310 | 165 | 450 |

- Structural goals met: barrier count halved (3→2 / 6→4), decoded-B STS and
  B-ldmatrix eliminated, HMMA/output coverage preserved.
- **Failure mechanism:** the approved 4-row-warp decode duplication (plan §9
  flag 1) forces every row warp to recompute the full payload→weight index
  chain (`escha_dep_pi`, mod-NB, funnel-shift address, codebook) for its own
  fragment pieces. LDS 8→32 (fp16) / 16→64 (fp32) and decode ALU ~2×–2.7×
  show the duplication cost. The extra index temporaries lift registers to
  145–176 → **1 CTA/SM** (145×512 > 65,536), the same occupancy loss that
  killed EXP-07 (124–154) and EXP-02 (176).
- Unlike EXP-07 (all-row/full-tile ownership), the failure here is the
  register cost of per-weight index arithmetic multiplied by 4× decode
  redundancy — not accumulator/B-tile lifetime. A bounded fix would need a
  different payload organization (e.g., warp-collective staged decode +
  fragment exchange), which is a new design, not a patch to this one.

## Series implication

T1 (register-B), T2 (finalize fusion), and now the official-style warp-local
two-band mainloop are all negative at BM128×BN128/M2048 on SM120. Each removal
of Bee's shared-B round trip has cost more in registers than it saves:
EXP-07 (124–154), EXP-09 (145–176) vs the 97/128 shared-B control. The
promoted Stage 2 shared-B/ldmatrix mainloop remains the default. Remaining
un-tried levers: T3 fused input rotation (~4.6%, below standalone gate),
T4 shape-specific ffn_down decode structure, or artifact-side substitution
(P-ARCH-23I already demonstrated ~3300 tok/s outside the runtime-only path).

## Files

- Source: `ggml/src/ggml-cuda/escha-moe.cu` — **reverted** (git checkout
  HEAD; macro-off diff zero).
- Builds: `build-cuda-exp09-control` (kept, clean HEAD control),
  `build-cuda-exp09-mlw` (candidate; remove with the guarded block).
- Evidence: `control-resources.txt`, `candidate-resources.txt`,
  `ctl-fp16-k2.sass`, `ctl-fp32-k2.sass`, `cand-fp16-k2.sass`,
  `cand-fp32-k2.sass` (all in this dir).
