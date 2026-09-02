# A4 MMA sidecar V2 — collective-consumer Slice 1 gate

Date: 2026-09-02  
Scope: `blk.0.ffn_gate`, K2 5120 -> 17408, direct op M=2048 on NVIDIA GeForce RTX 5090 (sm_120a; runtime CC 12.0).

## Verdict

**CONFIRM-REJECT**. V2 proves that the collective architecture fixes A4's duplication, register pressure, residency, and correctness failures, but it does not beat the packed shared-B control. The pre-resolved descriptor/record consumer has more decode/address ALU and is 1.51x slower at M=2048. Per the V2 terminal gate, the exact-packed sidecar line closes; use the 23/23I hybrid delivery path.

## Mechanism gate

| row | measured | threshold | result |
| --- | ---: | ---: | --- |
| representation growth | 6.275% | <=25% | PASS |
| correctness bit-compare | byte-exact reverse; 0 / 35,651,584 mismatches | exact; zero | PASS |
| registers / spills | FP16 96, FP32 128; stack/local/spills 0 | <=104 / <=128; zero | PASS |
| decode/address ALU | FP16 93 -> 137 (47.3% increase); FP32 87 -> 132 (51.7% increase) | >=30% reduction | FAIL |
| direct op M=2048 | 1.591085 -> 2.408634 ms (1.51x time; -33.9% throughput) | >=15% faster | FAIL |
| residency | FP16 2, FP32 2 CTA/SM; 45,056 B/CTA | >=2 CTA/SM | PASS |

## Mechanism result

The collective coverage is exact: each four-warp band executes 128 threads x 4 descriptors x 2 values = 1024 unique decodes, publishes one `[64][16]` tile, and all four row warps consume the same bytes via `ldmatrix`. The CTA therefore performs 2,048 codebook evaluations per K16 tile versus A4 V1's 8,192. The fully unrolled MMA seam avoids EXP-10 accumulator homing: FP16/FP32 use 96/128 registers with no local traffic and both retain two-CTA residency.

That structural repair is insufficient. Against the same packed control symbol, static `IMAD+LOP3+IADD3` rises rather than falling, and the directly measured FP16 operator loses 33.9% throughput. The remaining cost is the descriptor-driven adjacent-pair access and publication path, not four-row-warp duplication. This is the previously untested collective consumer of the pre-resolved stream, and it fails the two gates that determine whether the sidecar can outrun current packed decode.

Raw repack, compiler, SASS, resource, and benchmark evidence is in `v2-raw/`. No llama model binary ran; no `ggml/` or loader source was modified.
