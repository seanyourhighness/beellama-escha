# V4-PIPE — next overlap test (Sean directive: continue the packed path)

Date: 2026-09-02 · Sean: "V2 was 1.5x, V3 2.17x — this is still the largest wall
time opportunity, not a close. We need more overlap: double or quadruple the
stage overlay. We have enough data to figure this out."

## The accumulated data (6 experiments, all on blk.0 ffn_gate K2 M=2048)

| version | mechanism | direct-op ms | vs 1.591 control | key numbers |
|---|---:|---:|---:|---|
| shared-B control | cooperative decode → shared → LDSM | 1.591 | — | 27 decode ALU/K16, 16 HMMA/K16 |
| V1 | per-warp decode, descriptors, shallow | ~13.96 | 8.85× | 4× decode (8192/CTA), 140 regs |
| V2 | decode-once collective, descriptors, 1-K16 phases | 2.409 | 1.51× | 96 regs, 2 CTA; serial phase latency |
| V3-PIPE4 | 4-K16 superstage, no descriptors, no shared-B, 64 HMMA | 3.456 | 2.17× | 94 regs, 2 CTA, 45 KiB; decode ALU 102.5/K16 (4× dup) |
| V3 ablation | descriptor restore | 3.919 | +13.6% | proves descriptors ~not the wall |

## What V3's attribution ablation proved (critical for V4)

Descriptor gathers were only +13.6%. The wall is the **4× duplicated direct
codebook reconstruction** (each of 4 row-warps per band decodes the same B
weights: 102.5 ALU/K16 vs control's cooperative 27). Removing the shared-B
round trip forced per-warp decode, which duplicated ALU 4× — and that
duplication costs more than the shared-B traffic it saved.

## V4 hypothesis (Sean's overlap thesis, made precise)

V2 lost to serial one-K16 phase latency. V3 removed that but paid 4× decode
duplication. The two prior designs each fixed one axis and lost on the other.
V4 must deliver **decode-once (cooperative, control-class 27 ALU/K16) AND deep
overlap** — i.e., the shared-B control's decode economics WITHOUT its serial
decode→barrier→reload→MMA phase structure. The mechanism is a
cooperative-decode, multi-stage, double/quadruple-buffered pipeline where
decode of superstage N+2/N+4 is overlapped under MMA of superstage N.

Concretely, V4 explores the official-class deep staging that V3 reserved but
did not truly exploit, on TOP of a cooperative decode:
- decode each 64-col band's B cooperatively ONCE per superstage into a ring of
  shared B slots (2× or 4× the K depth = BK64/BK128-class B ring);
- all warps consume via LDSM from the ring while the NEXT decode superstage
  runs — no warp re-decodes;
- A/payload staging at 45,056 B class or deeper (BK128 = 8 K16);
- zero descriptors (V3's fixed mapping retained);
- accumulator fully unrolled (no EXP-10 stack homing).

## Required V4 analysis before implementation (Sol)

1. From the SASS/bench of V2 (decode-once) and V3 (deep, no-shared-B): what is
   the actual per-superstage critical path? Is the shared-B round trip
   (STS+barrier+LDSM) or the decode ALU the binding resource at 2+ CTA/SM?
2. Does a cooperative-decode B-ring with depth D overlap decode of stage N+D
   under MMA of stage N well enough that decode ALU (27/K16 cooperative) hides
   under 64-128 HMMAs? Show the latency/occupancy math.
3. What is the smallest experiment that tests "decode-once + deep ring"
   without a full loader integration? (Standalone kernel, same M=2048 gate.)

## Mechanism gate (frozen, same protocol as V3)

- correctness: 0 bit mismatches over 35,651,584 outputs
- representation: <=25% growth, byte-exact (V3's fixed mapping reused)
- resources: FP16 <=104, FP32 <=128, stack/local/spills 0, >=2 CTA/SM
- B path: cooperative decode-once per band; no per-warp duplication
  (decoded evals/CTA/superstage ~= control class, NOT 4x)
- direct op: <1.591085 ms (beat shared-B) required to continue; <=1.352422 ms
  (>=15% faster) is the breakthrough row
- diagnostics: record 2x and 4x B-ring depth variants separately

## Decision after V4

- <1.591: mechanism real — iterate depth/geometry toward <=1.352 then families
- >=1.591: bank the data, report to Sean with the hybrid delivery option ready
  (do NOT unilaterally close — Sean owns that call)
