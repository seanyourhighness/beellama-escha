# ESCHA-W2 PREFILL — Optimization Series Plan (post-BASE-01, 2026-09-01)

## Context and goal

BASE-01 (canonical matched campaign, frozen binary HEAD be6bf478d, graphs ON):
- ESCHA 2326.77 tok/s (880.19 ms) vs LowGPU IQ3 3212.63 tok/s (637.48 ms);
  paired-log G=1.3863 CI[1.3717,1.4010] 9/9; gap 242.7 ms.
- FFN block (ffn_up+ffn_down+ffn_gate) = +213.8 ms = 94.6% of the positive gap,
  on SAME-weights families (corr 0.83–0.87) -> operator/quant-path deficit.
- Projection families carry the full graphs-off gap; graph/orchestration delta
  only +16.8 ms (~7%). Classified: ESCHA operator architecture deficit.
- Depth matrix: per-call ffn_down ratio ~1.7–2.1x flat across M=128..4096 ->
  broad mainloop deficit, not tiling.

Prior evidence: EXP-01 async A-stage promoted (+58.8%); EXP-04 Stage 2 mixed
accumulator promoted (+9.31%); EXP-02 direct-fragment REJECTED (-3.9%, 176
regs); EXP-03 256x64 tile REJECTED (neutral); EXP-04 Stage 3 bounded-K FP16
REJECTED (+2.76%); EXP-06 BM64 down_proj REJECTED (+46.95% worse); EXP-05
official-mainloop audit NO-GO at 10.96% short-IC coverage < 15% bar.

## Series targets (ranked, each ONE variable, isolated, Sol-gated)

### T1 (EXP-07) — Warp-collective register-decoded B (remove shared-B round trip)
Highest value. Detail in `evidence/EXP-07-mainloop-coopdecode/2026-09-01/PLAN.md`
(v2, revised after Sol REVISE). Mechanism: keep decoded B in registers via
disjoint column-warp slices + fragment exchange, removing STS.U16/LDS.64 and
one barrier per 16-wide K tile; no per-M-warp duplicate decode (EXP-02 fix).
Pre-implementation SASS gate: 0 STS for B, regs <= control, fewer barriers,
>= control HMMA. Covers the FFN mainloop (94.6% of gap).

### T2 (EXP-08) — Warp-owned output-finalize fusion (Sol PLAN already READY)
From `evidence/EXP-04-nextvar/2026-09-01/NEXTVAR-PLAN.md`. Fuses Hadamard-128 +
normalize + rout into the GEMM epilogue for n_slices==1; removes p_buf partial
write+read; ~6.7% finalize bound -> credible +2–4%. Rebase the READY plan to
current HEAD. Independent of T1. Gate: >=5% smaller positive, resources/smem
(22,016 B/CTA), P2/P7, decode <=2%.

### T3 — Fused input rotation (shared up+gate transform) 
Rotate once, launch two unchanged packed GEMMs for FFN up+gate (and qkv+z).
Bound ~4.6–5.4% (BASE-01 rotate 5.4%). Keep only if measured >=3% wall after
T1/T2; do not run solely for the historical bound.

### T4 — Shape-specific ffn_down decode structure (after T1)
If T1 leaves ffn_down (17408->5120, per-call 1.81x) dominant, a shape-specific
variant of the new mainloop for that single shape is the highest-value narrow
target (EXP-06 proved tile aspect is the wrong variable; decode structure is
the lever).

### T5 — Split-K/grid policy on short-output K2 families
Small (0–4% expected); only after kernel wins; low priority.

### Explicitly NOT in this series (per BASE-01 / prior gates)
- Artifact substitution (P-ARCH-23I ceiling is not the goal architecture).
- Graph/runtime-only work (measured +16.8 ms; revisit only after kernel wins).
- Rotate/finalize fusion alone as a breakthrough gate.
- Combined multi-variable patches; any new variable requires its own Sol PLAN.

## Parity accounting (880.19 ms baseline, graphs ON)

| target | latency needed | remove | required |
|---|---|---|---|
| 3000 tok/s | 682.7 ms | 197.5 ms | -22.4% |
| 3212 (LowGPU this binary) | 637.5 ms | 242.7 ms | -27.6% |
| 3339 (historical LowGPU) | 613.4 ms | 266.8 ms | -30.3% |

A credible path: T1 recovers a fraction of the FFN mainloop deficit (~10-20%
of 880 = 88-176 ms if FFN halves its per-call ratio), T2 adds ~10-15 ms,
T3/T4 add ~5-15 ms. Full LowGPU parity is aggressive on the current packed
format; the series is run one variable at a time with honest gates.

## Organization (unchanged from prior experiments)

- One plan per experiment under `evidence/EXP-<n>-*/2026-09-01/`.
- Compile-gated, route-tagged; frozen control build; shared-2048 IDs.
- Sol PLAN -> implement (one variable) -> Sol code review -> route/SASS/
  resource/numeric proof -> parity P2/P7 -> decode gate -> matched 9-pair
  noise protocol -> promote or REJECT+revert -> ledger/current-state/GBrain.
- Commits: docs+evidence only (no binaries/weights/GGUF).
