# EXP-06 rejection report — Down-projection BM64-equivalent tile geometry

Date: 2026-09-01  
Starting promoted-control checkpoint: `4bc1afc1d`  
Implementation: `cf53d803c`  
Implementation review: Sol `VERDICT=CONFIRM` (`SOL-IMPLEMENTATION-REVIEW.md`)

## Decision: REJECT and REVERT

The candidate fails the decisive target-family performance gate before the
full-wall/quality certification stage. It will not be promoted.

## What was tested

One guarded structural variable only:
- prefill MMA, K3 `IC=17408`, `OC=5120`, `n_rows=2048`;
- `escha_matmul_dense_tiled_mma<3,64,128,false>` instead of the promoted
  `<3,128,128,false>` route;
- mixed accumulator policy retained (this family remains FP32);
- packed-code/decode, B shared layout, transforms/finalize, split-K and all
  other routes retained;
- candidate tag `mma-down-bm64-exp`.

Sol Gate 2 source review confirmed the scope/predicate/A-stage alignment and
FP32 fragment-store coverage prior to execution.

## Compiled proof

| property | Control BM128 | Candidate BM64 |
|---|---:|---:|
| template | `<3,128,128,false>` | `<3,64,128,false>` |
| static SASS regs | 128 | 92 |
| static stack / local | 0 / 0 | 0 / 0 |
| static shared | 1024 B | 1024 B |
| launch dynamic shared | 13,824 B | 9,728 B |
| HMMA.F32 / CTA | 32 | 16 |
| global stores / CTA | 64 | 32 |
| block | 256 threads | 256 threads |
| grid at M=2048 | [16,40,1] | [32,40,1] |

Macro-off control equivalence passed: focused BM128 SASS from frozen control
and macro-on candidate binary are byte-identical (SHA
`ed103498824d174d9f5360581af0e49289c95ec02ac41ccb2479bc9225b66e4d`,
zero diff). Candidate BM64 focused SASS SHA:
`a4e17ce72273428b4bcbdb9cd56ede46345a5028321de2fe829d599ad1f4d4c3`.

Static coverage proof passed: BM64 A-stage writes all 1024 `[64][16]` half
values exactly once; FP32 fragment mapping writes all 8192 `[64][128]` output
elements exactly once. This is a static proof only, not end-to-end numerical
certification.

## Route proof and target-family result

`ESCHA_PROFILE=1` completed 400 projection calls in each arm before an inherited
post-run CUDA error (also reproduces on frozen control):
- control has 64 target K3 17408→5120 calls tagged `mma-fp32-mixedacc`;
- candidate has 64 target calls tagged `mma-down-bm64-exp`;
- candidate target predicate is exact (no wrong-shape experimental tags);
- remaining 336 calls retain non-target promoted tags per run.

The profile harness abort occurs after all 400 calls in both arms, so its
per-call CUDA-event data is retained as diagnostic route/timing evidence, **not
a clean full-wall benchmark**.

From the 64 completed target calls per arm (first four discarded):

| target K3 17408→5120 M=2048 | control median matmul | candidate median matmul | candidate vs control |
|---|---:|---:|---:|
| profile diagnostic | 2.25125 ms | 3.30810 ms | **+46.95% slower** |

Required target-family gate is ≥15% improvement. This result fails it by a wide
margin.

Non-profile smoke run (successful, one run/arm; not ABBA):
- control: 2457.93 ± 83.78 tok/s
- candidate: 2275.93 ± 96.11 tok/s
- candidate: **−7.40%** vs control

This direction independently agrees with rejection.

## Not run after decisive failure

No nine-pair ABBA campaign, full-wall classification, depth/tail matrix, P2/P7,
decode guardrail, five-pack quality, or promotion review was run. Continuing
would violate the pre-registered fail-fast protocol. Dynamic runtime resource
residency/occupancy is not measured (ncu counter permission unavailable).

## Rollback

Revert only `cf53d803c`'s guarded source implementation, retain all evidence and
Sol review records, then prove:
```
git diff 4bc1afc1d -- ggml/src/ggml-cuda/escha-moe.cu
```
is empty. The promoted Stage 2 mixed-accumulator default remains the fallback.

## Next target

Return to the independent, Sol-planned output-finalize fusion experiment
(`evidence/EXP-04-nextvar/2026-09-01/NEXTVAR-PLAN.md`). Do not combine it with
this rejected BM64 geometry.
