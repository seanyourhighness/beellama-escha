# EXP-04 Stage 3 — bounded-K FP16 accumulation for 17408→5120 — REJECTED

Date: 2026-09-01. Plan: `/tmp/exp04-stage3-plan.md` (Sol-approved, STAGE3 PLAN
READY). Implementation commits `03f648a3b` + `12110c78a` (FP32 twin toggle) +
`536075307` (env-gated dst capture). Sol code review: **VERDICT=CONFIRM**,
Implementation gate PASS (`/tmp/escha-exp04-stage3-implreview.md`).

## What was tested

- ONE variable: for the only IC>6144 prefill family (k=3 ic=17408 oc=5120),
  force `n_slices=4` (272 tiles/slice = IC 4352, < the Stage-2-proven-safe 384)
  and use FP16 MMA accumulation within each slice; FP32 partials per slice;
  unchanged `escha_finalize_dense` sums slices in FP32. Route `mma-fp16-boundedk`.
  Stage 2 IC≤6144 dispatch unchanged; no global threshold raise.
- Three arms: (1) promoted Stage 2 1-slice FP32 control, (2) 4-slice FP32
  topology twin (`ESCHA_BOUNDEDK_FP32_TWIN=1`, route `mma-fp32-boundedk`),
  (3) 4-slice FP16 candidate.

## Route proof (ESCHA_PROFILE, graphs off)

- Arm 3: 800/800 tagged — 128× `mma-fp16-boundedk` (target), 672× Stage 2
  tags; **0 predicate mismatches**.
- Arm 2: 128× `mma-fp32-boundedk`; 672× Stage 2 tags; 0 mismatches.
- All rows=2048 gen=0.

## SASS / resource proof

- Bounded-K K3 FP16 kernel: 16× `HMMA.16816.F16`, 0 F32; **REG 97**,
  STACK/LOCAL 0, no spills, static SHARED 1024, dynamic 13824 (unchanged).
- FP32 twins unchanged (32× `HMMA.16816.F32`, REG 128).

## Numerical comparison (identical-input dst capture, 2048×5120 fp32)

| pair | max-abs | rel-RMS | cosine | NaN/Inf |
| --- | --- | --- | --- | --- |
| 4-slice FP32 twin vs 1-slice FP32 (reassociation+split) | 3.89e-5 | 1.73e-5 | 0.99999988 | 0 |
| 4-slice FP16 vs 4-slice FP32 (fp16 rounding, same topology) | 1.09e-3 | 1.08e-3 | 0.99999958 | 0 |
| 4-slice FP16 vs 1-slice FP32 (total) | 1.11e-3 | 1.08e-3 | 0.99999982 | 0 |

All outputs finite, no NaN/Inf. FP32 reassociation cost is negligible
(rel-RMS 1.7e-5); within-segment fp16 rounding at 272-tile depth is benign
(rel-RMS ~1.1e-3, cosine >0.9999995). Note: larger than Stage 2's ~2.1e-4
expected scale because 272 fp16 steps accumulate over 4 segments then FP32-sum.

## Correctness gates

- P2/P7: candidate **16/16** both prompts (greedy, seed 42, 16 tokens).
- Decode r5: candidate median 42.51 vs Phase 2 control 43.08 → **−1.31%**
  (within ≤2%).
- Family regressions: target-only change; non-target families retain Stage 2
  routes/tags (route proof 0 mismatch).

## Performance gate (pre-approved noise protocol, frozen binaries)

Matched 9-pair ABBA run (control = promoted Stage 2, candidate = bounded-K;
GPU 2887 MHz constant, all exit=0, all n_prompt=2048):

- control: median 2353.6 (mean 2337.2, CV 2.45%)
- candidate: median 2418.5 (mean 2400.4, CV 1.73%)
- **median gain +2.76%**
- paired-log: **G=1.0272, 95% CI [1.0005, 1.0546], 8/9 faster**
- Primary CV≤2%: not met (control 2.45%). Fallback: **G=1.0272 < 1.05 → FAIL**

Attribution: 4-slice FP32 topology twin was −4% vs 1-slice control (split-K
+ 4× partial/finalize traffic), while FP16 at the same topology recovered
+8.7%. Net wall gain +2.76% — below the ≥5% gate.

## Classification

**REJECT (performance gate <5%).** Full matched-2K improvement +2.76%
(paired-log G=1.0272, CI lower bound 1.0005 barely above 1.0) is below the
SMALLER POSITIVE threshold. Correctness, route, SASS, numerical, parity, and
decode gates all pass — the design is sound but the split-K/finalize overhead
consumes most of the FP16 benefit, so the credible bound (~+5.4-6.4% upper
range) did not materialize.

## Action (per goal: "If Stage 3 is negative")

1. **Revert the guarded Stage 3 operator** (revert `03f648a3b`, `12110c78a`,
   `536075307` guarded blocks) — retain promoted Stage 2 as the default
   prefill control.
2. Next structural variable from the committed Stage 1 per-stage profile:
   rotate 4.6% / matmul 88.6% / epilogue 6.7% — the largest fuseable bound is
   epilogue (6.7%) then rotate (4.6%). Per the goal's fallback ordering:
   (a) fuse output rotation/scale into the GEMM epilogue if finalize is the
   largest fuseable bound; (b) fuse input rotation into the GEMM if rotate is
   largest; (c) share duplicated transformed activation across MLP up+gate
   only if its measured upper bound is ≥5%. Finalize (6.7%) > rotate (4.6%),
   so **fuse output rotation/scale into the GEMM epilogue** is the next
   candidate — subject to a Sol PLAN gate before implementation.

## Raw artifacts

- `evidence/EXP-04-stage3/2026-09-01/`:
  - ATTRIBUTION.md (23.3% matmul share; +4.4-6.4% projected)
  - boundedk-profile.stderr / boundedk-twin-profile.stderr (route proofs)
  - noise-run/trial-*.json + run.log (matched 9-pair, G=1.0272)
  - boundedk-bench.json / control-bench.json / twin-bench.json (r3 sanity)
  - boundedk-decode-r5.json, parity-boundedk/compare-report.json
  - num-1slice/dst.f32.bin, num-4slice-fp32/dst.f32.bin,
    num-4slice-fp16/dst.f32.bin (numerical capture)
- Frozen binaries: /tmp/exp04-stage3-freeze/{control,candidate}
- Sol plan: /tmp/exp04-stage3-plan.md; Sol impl review:
  /tmp/escha-exp04-stage3-implreview.md (CONFIRM)
