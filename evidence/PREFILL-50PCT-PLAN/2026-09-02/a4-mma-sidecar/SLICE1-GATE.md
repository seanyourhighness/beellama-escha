# A4 / EXP-11 Attempt 3 — Slice 1 mechanism gate

Date: 2026-09-02  
Scope: `blk.0.ffn_gate`, K2 5120 -> 17408, direct op M=2048 on NVIDIA GeForce RTX 5090 (sm_120a build; runtime CC 12.0).

## Verdict

**CONFIRM-REJECT**.  The representation is exact and compact, but the direct-fragment consumer fails four of five frozen mechanism rows.  Per the dispatch brief, stop before full-model work.  Attempt 3 and the packed-exact +50% line close here.

## Gate

| row | measured | threshold | result |
| --- | ---: | ---: | --- |
| representation growth | 6.275% | <=25% | PASS |
| registers / spills | FP16 140, FP32 168; stack/local/spills 0 | <=104 / <=128; zero spills | FAIL |
| decode/address ALU | FP16 93 -> 220 (-136.6% reduction); FP32 87 -> 211 (-142.5%) | >=30% reduction | FAIL |
| direct op M=2048 | 1.576154 -> 13.956268 ms (8.85x slower; -88.7% throughput) | >=15% faster | FAIL |
| residency | FP16 1, FP32 1 CTA/SM | >=2 CTA/SM | FAIL |

Correctness passed: the overlay reverse transform is byte-exact and the FP16 direct-op comparison has 0 bit mismatches across 35,651,584 outputs.

Two-CTA register math: FP16 requires `140*256*2 = 71,680` and FP32 requires `168*256*2 = 86,016` registers, both above the 65,536-register SM pool.  Shared memory would require only `16,384` bytes for two CTAs, so registers are binding.  CUDA occupancy independently reports one active CTA/SM in both modes.

The ALU counts are static occurrences in the actual sm_120a SASS emitted for the matched standalone control and sidecar symbols.  Raw SASS, ptxas diagnostics, cuobjdump resources, repack manifest, and benchmark JSON are retained under `slice1-raw/`.

## Mechanism finding

Pre-resolving `dw0/dw1/dsh` is insufficient under the 25% representation cap.  Four row warps still independently load and evaluate every fragment weight.  The compiler exposes that duplication directly: candidate ALU rises rather than falls, live state reaches 140/168 registers, occupancy drops to one CTA/SM, and the candidate is 8.85x slower.  This reproduces EXP-09's fundamental four-row-warp cost even though its index chain has been removed.

No model binary was run, no loader/model source was changed, and no full-model work was started.
