# EXP-05 — ESCHA-W2 prefill continuation closure report (2026-09-01)

## Starting and ending commits
- Starting local == remote: `b81a6ed7d` (prior phase closure)
- Ending local == remote: **`43c14ca8c`** (4 commits added: `e09ed6855` audit,
  `f70e451fe` REVISE fixes + raw sweeps, `43c14ca8c` docs)
- Branch `escha-w2-prefill`; worktree clean (untracked = local-aid symlinks).

## Phase 0 — Repository reconciled ✅
- Local/remote HEAD equality verified at start and end.
- SHAs: Stage 2 promotion `ace024e72`, Stage 3 impl `03f648a3b` (reverted),
  protected rollback `4501b3ee1`. All artifacts confirmed (P-ARCH-20
  correction, Stage 2 evidence + Sol CONFIRM, promotion, Stage 3 impl +
  Sol reviews).
- Worker preflight: **BLOCKED (HTTP 401)**, recorded once, no retries.

## Phase 1 — EXP-04 Stage 3 closure ✅ (already closed)
- Bounded-K FP16 for 17408→5120: Sol PLAN → impl → code review CONFIRM →
  REJECTED (+2.76% < 5%) → reverted → Sol VERIFY CONFIRM. Promoted Stage 2
  retained as default. No new action required.

## Phase 2 — EXP-05 reference mainloop audit ✅ (NO-GO)
- Official wheel sha `735f4b7a…` verified; runtime reproduced at M=2048
  (TTFT 623.8 ms / 3029 tok/s vs Bee-Stage2 ~817–870 ms).
- **Template space mapped definitively:** `escham_code_gemm_kernel
  <A,K,BM,BN,BK,FP16ACC,FEPI>` (56 sm_120 symbols); param 6 = fp16-acc,
  param 7 = FEPI; env ESCHAM_GEMM_BM/BK/FEPI/WIDE_HAD mapped via fresh-process
  one-factor sweeps (raw traces retained in `raw-sweeps/`).
- Per-family default kernels + grid/block captured (hot short-IC
  `<1,K,128,64,2,true,true>` = two 64-col bands, 80 regs, 45 KiB smem, fp16
  acc; down_proj fp32 122 regs).
- SASS side-by-side: official = warp-collective two-band, warp-local decoded-B
  (0 STS.U16), in-kernel Hadamard (80 SHFL.BFLY), 40+40 WARPSYNC/ENDCOLLECTIVE,
  136 LDS; Bee = shared-B round trip (16 STS.U16 + 16 LDSM), separate
  finalize, no collectives, 13,824 B smem.
- **Attribution:** official faster short-IC (gate 0.78×, qkv 0.80×, up 0.90×),
  slower down_proj (1.78×; official's own BM=128/BK=2 default is a trap —
  BM=64 → 1.99 ms vs 3.96 default).
- **Sol review: VERDICT=REVISE → Phase 3 gate NO-GO.** Measured short-IC
  coverage = 91.1 ms = **10.96%** of aggregate matmul (< 15% bar). Per goal:
  **no speculative implementation; EXP-05 closes as negative evidence.**
- History corrections appended (P-ARCH-09 short-IC = FP16 acc; `-ub 2048`
  supersedes `-ub 512`).
- **Next targets ranked:** (1) fuse output rotation/scale into GEMM epilogue
  (finalize 6.7%, Sol PLAN READY at
  `evidence/EXP-04-nextvar/2026-09-01/NEXTVAR-PLAN.md`), (2) down_proj
  BM=64-style tile sweep (separate single variable), (3) fuse input rotation
  (4.6%), (4) MLP up+gate sharing only if fresh ≥5% bound.

## Phase 3 — NOT IMPLEMENTED (Sol NO-GO)
- No BeeLlama kernel changes were made in this phase (read-only audit).

## Verdicts
- EXP-04 Stage 3: Sol VERIFY **CONFIRM** (reject+revert) — prior phase.
- EXP-05 Phase 2 audit: Sol review **REVISE** (provenance) → resolved;
  Phase 3 gate **NO-GO** (budget < 15%).

## Evidence / provenance
- `evidence/EXP-05-audit/2026-09-01/`: AUDIT.md, provenance.manifest,
  per_family_gridblock.json, direct_gemm_results.json, code_gemm_resources.txt,
  hot-k2/off-down/bee-k2-k3 SASS files, raw-sweeps/ (18 fresh-process traces).
- Ledger + current-state updated. GBrain sync in progress.

## SHA equality / worktree
- local == remote `43c14ca8c` (verify after push). Worktree clean.
