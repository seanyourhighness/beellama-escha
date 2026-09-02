# EXP-07 — Mechanism-Gate Rejection (2026-09-01)

## Pre-implementation gate result: FAIL (stop before timing)

Gate (Sol PLAN v2, item 3): before timing, candidate SASS must show
1. 0 STS.U16 for B (and 0 LDS.64 B-reload) — the mechanism's whole point;
2. **register count <= control (fp16 <= 97, fp32 <= 128), STACK/LOCAL 0, no
   spills**;
3. barrier count per K tile reduced >=1;
4. HMMA count >= control;
5. route proof; 6. per-shape direct measurements.

## Measured resources (cuobjdump --dump-resource-usage, gate-on build
build-cuda-exp07-regb, sm_120)

| kernel | regs | control regs | stack | local | verdict |
|---|---|---|---|---|---|
| regb K2 fp16 (`<2,128,128,true>`) | 124 | 97 | 0 | 0 | **FAIL (+27 regs)** |
| regb K2 fp32 (`<2,128,128,false>`) | 154 | 128 | 0 | 0 | **FAIL (+26 regs)** |
| regb K3 fp16 (`<3,128,128,true>`) | 124 | 97 | 0 | 0 | **FAIL (+27 regs)** |
| regb K3 fp32 (`<3,128,128,false>`) | 154 | 128 | 0 | 0 | **FAIL (+26 regs)** |

Register math: fp32 154 regs x 256 threads = 39,424 regs/CTA -> **1 CTA/SM**
(SM120 64K regfile), control 128 regs = 32,768 -> 2 CTAs/SM. fp16 124 regs x
256 = 31,744 -> 2 CTAs/SM (same as control's 97-reg 2-CTA residency), but the
per-thread +27 register growth indicates the warp-ownership transpose
(MT=8 all-rows per warp + NTT=2 bands + streamed A) did not reduce live state
as designed.

## SASS (K2 fp16 candidate)

- No STS.U16 for B present (mechanism partially achieved: B stays in
  registers), 16 HMMA.16816.F16 (>= control), 8 LDS.64 (A-stage only).
- Total candidate body 1074 lines vs control 625-class — instruction growth
  from per-warp all-row A handling + decode addressing.

## Conclusion

The register-B mechanism is not register-neutral in this implementation. Per
the approved fail-fast rule ("Any spill or occupancy loss stops work before
timing"), EXP-07 as implemented is **REJECTED before benchmarking**. No
numerical/parity/timing campaign is run. Candidate source reverted.

## Options (for Sol)

- (a) Confirm rejection; pivot to T2 (fused finalize, Sol PLAN already READY).
- (b) Authorize a register-bounded EXP-07b revision ONLY if a concrete
  mechanism reduces live registers (e.g., split the all-rows ownership back
  to 2 row-warps per 16-col band so each warp owns MT=4 rows x 16 cols, or
  drop FP32 acc family from the candidate) and the SASS gate is re-checked
  before timing.
- (c) Close EXP-07 as negative evidence (shared-B round trip removal costs
  more registers than it saves on SM120), matching EXP-02/03/06 pattern.

Evidence: /tmp/exp07-resources.txt, /tmp/exp07-k2fp16.sass, git diff at
rejection time.
