# EXP-04 Stage 3 — attribution for the 17408→5120 family

Date: 2026-09-01. Derived from raw ESCHA_PROFILE evidence:
- `evidence/EXP-04-stage1/2026-09-01/stage1-profile.stderr` (fp32 baseline)
- `evidence/EXP-04-stage2/2026-09-01/mixedacc-profile.stderr` (Stage 2 cand)

## Family: k=3 ic=17408 oc=5120 (the only IC>6144 prefill family)

| metric | value |
| --- | --- |
| Calls per 2k prefill | 128 (127 steady-state after excluding first cold call) |
| Profile-mode family matmul | 282.9 ms (Stage 1 fp32) |
| Share of all steady-state matmul | **23.3%** |
| Share of measured projection time | 22.3% (profile-mode, sync-inflated) |
| Estimated wall share | ~209 ms of ~898 ms control wall (2281 tok/s) |
| Split-K / partial overhead | family is M=2048 rows, IC=17408, K=3; grid 16×1, split-K target 512 CTAs → naturally unsplit (single slice) |

## Projected full matched-2K gains (family-level improvement scenarios)

| family matmul improvement | wall saved | new wall | tok/s | full-2K gain |
| --- | --- | --- | --- | --- |
| +18% (low end of Stage 2 family gains) | 38 ms | 860 ms | 2381 | **+4.4%** |
| +22% | 46 ms | 852 ms | 2404 | **+5.4%** |
| +26% (high end of Stage 2 family gains) | 54 ms | 843 ms | 2428 | **+6.4%** |

## Decision per goal directive

"Only implement Stage 3 if its measured upper bound can plausibly produce at
least a 5% full matched-2K improvement."

- The credible upper bound (family ≈ +22–26%, the same range Stage 2 measured
  for IC≤6144 families) projects **+5.4% to +6.4% full matched-2K**, i.e. it
  can plausibly exceed the 5% bar, but only marginally.
- The low end (+18%) projects +4.4% (<5%), so the outcome is sensitive to how
  much of the fp16-acc win transfers to the bounded-K design for IC=17408.
- **Proceed to Sol PLAN gate** for the bounded-K design. Sol must approve the
  segment size, split-K mapping, accumulation-depth safety argument, FP32
  partial/finalize behavior, register/smem budget, correctness harness, and
  rollback before any code is written. If Sol's plan review concludes the
  credible bound is <5%, stop and choose the largest rotate/finalize fusion
  opportunity instead (per the goal's fallback).

## Design constraint (from goal)

- Divide IC=17408 into numerically bounded, aligned segments; fp16 MMA within
  each segment; convert each segment (or split-K partial) to FP32; combine in
  the existing FP32 finalize/reduction path.
- Do NOT raise the IC≤6144 threshold (that would exceed the proven-safe
  accumulation depth). Do NOT change tile shape, codec decode, async-A
  staging, projection fusion, or artifact data.
- Distinct guarded route `mma-fp16-boundedk`, restricted initially to the
  exact 17408→5120 family.
