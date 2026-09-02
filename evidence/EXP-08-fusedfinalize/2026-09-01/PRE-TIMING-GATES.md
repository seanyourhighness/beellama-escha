# EXP-08 — Implementation + Pre-timing Gates (2026-09-01)

## Sol code review
- Round 1: CODE_REVIEW=REVISE — accumulator-to-v[] transpose wrong (native
  8*j+2*lane_col+{0,1} vs assumed 4*q+lane_col); shuffle stages executed wrong
  lengths.
- Fix applied: explicit transpose into v[q] = column 4*q+lane_col via width-4
  shfl (src_lane_col=2*(q%2)+lane_col/2; col0/col1 select by lane_col%2), FP16
  seam preserved (x[0].{x,y}->l={0,1}, x[1].{x,y}->l={2,3}); butterfly order
  1,2,4,8,16,32,64 retained.
- Round 2: CODE_REVIEW=REVISE (summarized text omitted the post-read barrier).
- Round 3: CODE_REVIEW=CONFIRM — transpose correct, butterfly order/signs
  correct, pair slabs/barrier IDs disjoint, second barrier present (lines A/B
  in loop body) so no slab overwrite race; FP32 operation ordering supports
  bitwise equivalence.

## Resource gate (cuobjdump, gate-on build build-cuda-exp08-fusedfin)
| kernel | regs | control | stack | local | smem launch |
|---|---|---|---|---|---|
| fusedfin K2/K3 fp16 | 96 | 97 | 0 | 0 | 13,824 + 8,192 = 22,016 B (assert) |
| fusedfin K2/K3 fp32 | 128 | 128 | 0 | 0 | same |
No spills. smem bound exactly 22,016 B per plan contract.

## Route proof (ESCHA_PROFILE, graphs off, M=2048, 800 records)
- 608 mma-fp16-fusedfin + 128 mma-fp32-fusedfin = 736 fused (n_slices==1)
- 64 mma-fp16-mixedacc remain = split-K calls (fused only n_slices==1) — correct
- 0 fallback / predicate mismatch.

## Numeric / parity smoke (P2 + P7, greedy seed 42, 16 tokens, control vs candidate)
- P2-factual: control 16 tokens == candidate 16 tokens, agree=True
- P7-tool-call: control 16 tokens == candidate 16 tokens, agree=True
- Decisive: deterministic output identical after epilogue fusion.

## Status
Candidate implemented (guarded, gate ESCHA_MMA_FUSED_FINALIZE_EXPERIMENT=1),
Sol code-review CONFIRM, resource + route + parity gates PASS. Next: decode
gate (<=2%), then canonical matched 9-pair ABBA campaign.

Evidence: smoke/parity.json, smoke/*.server.log, /tmp/exp08-fixed.diff.
