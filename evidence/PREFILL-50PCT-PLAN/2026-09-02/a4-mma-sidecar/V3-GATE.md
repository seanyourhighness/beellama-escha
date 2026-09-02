# V3-PIPE4 final packed-exact Escha prefill mechanism gate

Date: 2026-09-02  
Scope: `blk.0.ffn_gate`, K2 5120 -> 17408, direct op M=2048 on NVIDIA GeForce RTX 5090 (sm_120a; runtime CC 12.0).

## Verdict

**CONFIRM-REJECT**. V3 is byte-exact, descriptor-free in its repeated BK64 body, uses the full 45,056-byte A/payload staging class, preserves two-CTA residency, and removes the decoded-B shared round trip. It nevertheless takes 3.456275 ms and its normalized decode/address ALU exceeds control. The packed-exact line closes; delivery returns to the standard-GGML hybrid frontier.

## Mechanism gate

| row | measured | threshold | result |
| --- | --- | --- | --- |
| representation growth % | 0.001147%; byte-exact; 0 u16/fp16-per-weight; one canonical word copy | <=25%, byte-exact reverse, no u16/fp16-per-weight stream | PASS |
| hot-loop descriptor traffic | 0-entry descriptor table; descriptor LDG/LDS/LDC = 0/0/0 in both repeated bodies | zero descriptor-table LDG/LDS in repeated superstage | PASS |
| pipeline shape | FP16/FP32 64/64 HMMA; 1/1 CTA rendezvous; 9/9 in-region LDGSTS; 32,768 B A + 12,288 B payload addressed | 64 FP16 HMMA per 4-K16 body; <=1 CTA rendezvous; future-A async before/interleaved | PASS |
| B path | decoded-B STS 0; B LDSM 0 (8 A-only LDSM); decoded-B publication barriers 0 | all zero | PASS |
| resources | FP16 94, FP32 127 regs; stack/local/spills 0; 2/2 CTAs/SM; 45,056 B/CTA | <=104 / <=128; zero; >=2 CTAs/SM | PASS |
| normalized decode/address ALU | FP16 410/4=102.5 vs 27; FP32 410/4=102.5 vs 27; SHF 130/4=32.5 vs 8 separately | <= control per-K16 in both modes | FAIL |
| correctness | 0 / 35,651,584 bit mismatches | 0 / 35,651,584 | PASS |
| direct operator | 3.456275 ms (2.172x fixed control; -53.97% throughput) | <=1.352422 ms | FAIL |

Diagnostics: `<=1.591085 ms` (beats shared-B): **FAIL**. `<=1.40 ms` (official-class): **FAIL**. The fresh control median was 1.576435 ms; the frozen comparison baseline remains 1.591085 ms.

## Attribution ablation

The one permitted descriptor-restored ablation was bit-exact and measured 3.919443 ms versus 3.451283 ms for the paired fixed-mapping schedule: +0.468160 ms (+13.56%). No descriptor-placement variants ran.

## Program decision

**close packed-exact line, hybrid delivery**. V3's fixed mapping saves descriptor cost, but the remaining duplicated direct codebook reconstruction is still well beyond the shared-B control. No `ggml/` or loader source was modified, and no model runtime was integrated.
