# EXP-10 — cooperative BK32 final-gate REJECTION (pre-timing)

Date: 2026-09-02 · Branch `escha-w2-prefill` · Control HEAD `969db62df`
Candidate: `ESCHA_MMA_COOP_BK32_EXPERIMENT` guarded
`escha_matmul_dense_tiled_mma_coop_bk32` (build `build-cuda-exp10-bk32`).

## Verdict

**REJECTED + REVERTED at the pre-timing resource gate. Sol FINAL GATE =
CONFIRM-REJECT.** No benchmark or correctness campaign was run, per the frozen
plan §7/§9.2 and Decision B ("a resource failure ends incremental work on this
kernel"). No further kernel variants are authorized.

## What was built (faithful to the approved plan)

- Cooperative BK32 two-slot B ring: decode B[n+1] into `s_w[(phase^1)]` while
  MMA(n) consumes `s_w[phase]`; even/odd warp symmetric phasing (decode-next →
  MMA-current vs MMA-current → decode-next); P(n+1)/D(n+1) full-CTA barriers
  (2 per 16-wide tile vs control 3); cp.async A retained; decode formula,
  ldmatrix B, store seams, split-K lo/hi all literal from control; 17,920 B
  dynamic smem (static_asserted); tags `mma-fp16-bk32`/`mma-fp32-bk32`.
- Terra math audit PASSED; Sol code review CONFIRM (no defects). Both builds
  clean; macro-off path byte-identical (357 pure insertions).

## Resource gate (cuobjdump, sm_120a) — the failure

| symbol | REG | STACK | LOCAL | spills (SASS) |
|---|---:|---:|---:|---|
| control fp16 (K2/K3) | 97 | **0** | 0 | none |
| control fp32 (K2/K3) | 128 | **0** | 0 | none |
| **coop fp16 K2/K3** | **64** | **128** | 0 | 40 LDL + 8 value STL |
| **coop fp32 K2/K3** | **82** | **256** | 0 | 40 LDL + 8 value STL |

- Register counts are far BELOW ceilings (64/82 vs 97/128) — occupancy is not
  the problem. **STACK is nonzero (control 0)** and SASS shows real spill
  traffic: ~12 LDL/STL between BAR.SYNC boundaries (main loop) and ~36 after
  the last barrier (epilogue/store region), in both parity paths.
- Sol diagnosis: STACK sizes exactly equal the full accumulator arrays —
  FP16 2×8×2×4 = 128 B, FP32 2×8×4×4 = 256 B. ptxas homes the accumulators to
  local memory because `acc[i][j]`/`acc16[i][j]` are runtime-indexed under the
  `#pragma unroll 1` MMA/decode helpers; the store epilogue reloads them from
  stack. This is accumulator homing, not transient pressure.
- Sol verdict on fixes: full unrolling is the only plausible structural remedy
  but cannot be proven to keep FP32 ≤128 (an 82-reg kernel promoting a
  256-byte stack starts at the edge); branch scoping / reference movement /
  launch-bounds hints provide no proof of STACK=0 and launch-bounds departs
  from the frozen measurement contract. No bounded fix is admissible.

## Interpretation

The cooperative BK32 *schedule itself* was never invalidated — the design
(P/D barriers, B-ring, even/odd phasing) compiled and the register counts prove
the overlap hypothesis is not register-blocked. The failure is a ptxas
allocation artifact of the unroll-1 helper structure forcing accumulator
homing. Under Decision B's one-experiment/frozen-gate rule this still closes
the experiment as negative; retrying the same schedule with different unroll
or helper inlining would be a second kernel variant, which Decision B
explicitly forbids without a new Sean decision.

## Series status (updated)

T1 (register-B), T2 (finalize fusion), T3-mainloop-rewrite (EXP-09 warp-local
two-band), and now the cooperative BK32 temporal pipeline (EXP-10) are all
negative. The packed-runtime incremental kernel program is **STOPPED per
Decision B**. Promoted Stage 2 remains the default. Remaining paths:
Decision A (certify P-ARCH-23I, ~3300 tok/s demonstrated), load-time transcode
cache, or a new sidecar representation/kernel project. The exact-packed-
execution question is now a funding/architecture decision, not an
incremental-kernel decision.

## Files

- Source: `ggml/src/ggml-cuda/escha-moe.cu` — **reverted** (git checkout
  HEAD; macro-off diff zero).
- Builds: `build-cuda-exp10-control` (kept, clean HEAD control),
  `build-cuda-exp10-bk32` (remove with the guarded block).
- Evidence: `control-resources.txt`, `candidate-resources.txt`,
  `ctl-fp16-k2.sass`, `cand-coop-fp16-k2.sass`, `cand-coop-fp32-k2.sass`,
  `DISPATCH-BRIEF.md`, `REWRITE-PLAN.md`, `TERRA-AUDIT.md` (all in this dir).
- Sol final-gate transcript: process log `proc_d53b6162c65c`.
