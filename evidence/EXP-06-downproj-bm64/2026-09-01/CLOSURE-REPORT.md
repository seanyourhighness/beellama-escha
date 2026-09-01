# EXP-06 closure report — Down-projection BM64-equivalent tile geometry

Date: 2026-09-01

## Decision
**REJECT + REVERT.** The candidate is not promoted. The current source is the
promoted EXP-04 Stage 2 mixed-accumulator control.

## Ancestry and checkpoints
- Protected ARCH-01 rollback: `4501b3ee1`
- Remote-clean preimplementation checkpoint: `1cf4f5d19`
- Guarded implementation: `cf53d803c`
- Sol Gate 2 review checkpoint: `604b4a7ca` (CONFIRM)
- Source rollback/evidence checkpoint: `eb6679159`
- Sol Gate 3: `VERIFY=CONFIRM` (this closure)

## Phase 1 — accounting and official BM evidence
- Corrected the incompatible old `623.8 ms / 3029 tok/s` pairing.
  Fresh server evidence: 2048 logical/server prompt tokens; warm median
  619.678 ms server e2e / 634.9 ms client wall; explicit throughput is
  3304.9 / 3225.7 tok/s respectively (`2048 / seconds`). Scheduler 3010–3023
  tok/s is a different internal accounting boundary and is not used canonically.
- Five alternating fresh-process direct-op pairs revalidated official
  17408→5120 BM128→BM64: 4.170747→2.295105 ms, paired geometric 1.8183×,
  byte-identical synthetic output. BM controls CTA row coverage (grid y 16→32),
  while output remains two 64-column bands / 128 columns total.

## Candidate and compiled proof
- Single guarded variable: Bee K3 17408→5120 M=2048 row tile BM128→64;
  FP32 accumulator, BN128/WN2, packing/decode/split-K/transforms/finalize all
  retained.
- Sol Gate 1: PLAN=READY. Sol Gate 2: VERDICT=CONFIRM.
- Frozen macro-off control versus macro-on candidate build: BM128 focused SASS
  byte-identical (SHA `ed103498…`). Candidate BM64: 92 regs, STACK=0, LOCAL=0,
  static shared 1024 B, planned dynamic shared 9728 B; 16 FP32 HMMA/CTA versus
  control 32, consistent with half row coverage. Static A-stage and FP32
  output-store maps have complete one-write coverage.

## Decisive performance failure
- Completed diagnostic profile target samples: 64 per arm (first 4 discarded
  per family). Control `mma-fp32-mixedacc` median 2.25125 ms; candidate
  `mma-down-bm64-exp` median 3.30810 ms: **candidate +46.95% slower**.
- Successful non-profile smoke agrees: 2275.93 candidate vs 2457.93 control
  tok/s (**−7.40%**).
- Therefore the pre-registered ≥15% target-family gain gate fails. No ABBA,
  depth/tails, P2/P7, decode, quality, or promotion campaign was run.
- `ESCHA_PROFILE` aborts after all 400 calls in both frozen control and
  candidate; all profile samples are explicitly diagnostic, not full-wall
  certification evidence.

## Rollback / source verification
- `git diff 4bc1afc1d -- ggml/src/ggml-cuda/escha-moe.cu` is empty.
- The exact guarded source hunk from `cf53d803c` was reversed; evidence and
  docs remain committed.
- Full all-fatbin cuobjdump is preserved outside git at
  `/tmp/escha-wheel/exp06-full-cuobjdump`; no weights/GGUFs/venvs/binaries/
  build directories/local aids were committed.

## Verdict and next target
Sol Gate 3 `VERIFY=CONFIRM`: reject+revert valid; candidate must not promote.
Next independent target: output-finalize fusion under the pre-existing plan at
`evidence/EXP-04-nextvar/2026-09-01/NEXTVAR-PLAN.md`.

## Evidence
`evidence/EXP-06-downproj-bm64/2026-09-01/`: Phase 0/1, raw official trials,
Sol plan/review/verify, static proof, focused SASS/resources/hashes, route logs,
profile summarizer, rejection report, and provenance manifest.
