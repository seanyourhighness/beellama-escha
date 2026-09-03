# V4-PIPE cooperative decode-once deep B-ring gate

Date: 2026-09-02  
Scope: `blk.0.ffn_gate`, K2 5120 -> 17408, direct op M=2048 on NVIDIA GeForce RTX 5090 (sm_120a; runtime CC 12.0).

## Verdict

**CONFIRM-REJECT**. Both depths are bit-exact, preserve control-class decode economics, remain spill-free at two CTAs/SM, and implement the planned band-opposed decode/MMA schedule. Neither beats the 1.591085 ms shared-B control. D=4 is faster by 1.15% between V4 variants, which is too small to make ring distance the missing factor.

## D=2 / BK32 superstage

| row | measured | threshold | result |
| --- | --- | --- | --- |
| correctness | 0 / 35,651,584 mismatches | 0 | PASS |
| representation | 0.001147%; byte-exact; descriptor-free | <=25%; exact | PASS |
| decode evals/CTA/superstage | 4,096 (2,048/K16); control 4,096; V3 16,384 | control-class, not 4x | PASS |
| resources | FP16/FP32 104/127 regs; stack/local/spills 0; 2/2 CTA/SM; 45,056 B | <=104/128; zero; >=2 | PASS |
| HMMA/superstage | 32 executed/warp; 64/64 static across two band branches | 32 executed/warp | PASS |
| barriers/superstage | 2 band-local (1.00/K16); control 3/K16; V3 1/BK64 | 2; record emitted sync | PASS |
| direct op: beat shared-B | 3.832224 ms; 2.409x fixed control; -58.48% throughput | <1.591085 ms | FAIL |
| direct op: breakthrough | 3.832224 ms | <=1.352422 ms | FAIL |

Fresh control median: 1.580851 ms. Candidate samples: 3.799488, 3.832224, 3.837030, 3.813677, 3.862528.

## D=4 / BK64 superstage

| row | measured | threshold | result |
| --- | --- | --- | --- |
| correctness | 0 / 35,651,584 mismatches | 0 | PASS |
| representation | 0.001147%; byte-exact; descriptor-free | <=25%; exact | PASS |
| decode evals/CTA/superstage | 8,192 (2,048/K16); control 8,192; V3 32,768 | control-class, not 4x | PASS |
| resources | FP16/FP32 103/127 regs; stack/local/spills 0; 2/2 CTA/SM; 45,056 B | <=104/128; zero; >=2 | PASS |
| HMMA/superstage | 64 executed/warp; 128/128 static across two band branches | 64 executed/warp | PASS |
| barriers/superstage | 2 band-local (0.50/K16); control 3/K16; V3 1/BK64 | 2; record emitted sync | PASS |
| direct op: beat shared-B | 3.787975 ms; 2.381x fixed control; -58.00% throughput | <1.591085 ms | FAIL |
| direct op: breakthrough | 3.787975 ms | <=1.352422 ms | FAIL |

Fresh control median: 1.645766 ms. Candidate samples: 3.780435, 3.787975, 3.795155, 3.817306, 3.770554.

## Read

The experiment falsifies ring depth as the missing variable in this schedule. Quadrupling the contiguous issue window does not approach control, even after decode evaluations fall from V3's 8,192/K16 to the control's 2,048/K16. The remaining structural cost is the band-opposed software pipeline itself: each band must still execute both phases, payload and decoded-B publication remain explicit, warp-private A doubles activation traffic across bands, and the compiler emits both uniform branch bodies. More buffering cannot close a >2x gap when D=2 -> D=4 changes only a few percent.

If Sean continues the line, the next variant must change ownership, not depth: use a producer/consumer warp-specialized schedule with split arrive/wait barriers and shared, CTA-cooperative A staging, so producers never carry row-MMA work and consumers never carry decode work. That requires a different output geometry or additional consumer coverage; merely extending this ring to D=8 is not supported by V4.

No model binary ran, no `ggml/` or loader/runtime source changed, and no commit was made. Raw manifest, compiler diagnostics, SASS, resource dump, binary, and benchmark JSON are in `v4-raw/`.
