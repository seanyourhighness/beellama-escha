# V5-PIPE — warp-specialized producer/consumer (Sean directive: run now)

Date: 2026-09-02 · Sean: "Let's run V5 pipe now, make sure sol reviews it and
does final gate." Sol plans, implements, reviews, and issues the FINAL gate.

## Why V5 (the accumulated data set)

| exp | mechanism | ms vs 1.591 control | what it eliminated |
|---|---|---:|---|
| V1 | per-warp decode + descriptors | 8.85× | (baseline failure) |
| V2 | decode-once collective, 1-K16 | 1.51× | 4× decode dup, regs |
| V3 | 4-K16 deep, no descriptors, no shared-B | 2.17× | descriptors (+13.6% only), shared-B |
| V4 | decode-once + 2×/4× deep B-ring | 2.38–2.41× | ring depth (D2→D4 = 1.15%) |

V4 proved: with decode evals back at control-class (2,048/K16) AND deep
overlap, the pipeline is still 2.4× slower. The remaining structural cost is
the band-opposed schedule — each warp carries BOTH decode and MMA ownership,
publication stays explicit, warp-private A doubles activation traffic, and
both uniform branch bodies are emitted.

## V5 hypothesis (Sol's V4 read — change OWNERSHIP, not depth)

Warp-specialized producer/consumer: some warps decode ONLY (producers), other
warps MMA ONLY (consumers), with split arrive/wait barriers and
CTA-cooperative (shared) A staging so producers never carry row-MMA work and
consumers never carry decode work. This is the one structural architecture not
yet tested — and it matches the external sm120 lesson (jetha A4Q: dedicated
work partitioning beats every warp doing everything).

## Design tension to resolve (be honest)

Sol's V2 review ranked dedicated producer warps LOW because removing 2 of 8
MMA warps cuts nominal tensor ownership ~25%. V5 must therefore either:
(a) justify that decode-bound (not MMA-bound) phases make producer warps a net
win at this geometry, or (b) use a wider CTA (e.g. 12 warps = 8 consumers + 4
producers, or 16 warps = 8+8) or a different grid geometry so consumer MMA
coverage is NOT reduced. Show the occupancy/ownership math explicitly before
coding.

## Required V5 analysis (write into plan before code)

1. Producer/consumer split: how many decode warps vs MMA warps; what each
   carries; the arrive/wait barrier scheme per superstage (split barriers so
   producers and consumers advance independently).
2. CTA-cooperative A staging (shared A, single copy, both bands consume) to
   remove V4's warp-private-A doubling.
3. Output coverage proof: total MMA ownership across consumer warps still
   covers the full 128×128 CTA tile at >=2 CTAs/SM and <=104/128 regs.
4. B handoff: producers decode into the shared ring; consumers LDSM from it —
   quantify whether this reintroduces V2's publication cost or whether deeper
   ring + split barriers hide it.

## Mechanism gate (frozen, same protocol)

- correctness: 0 bit mismatches over 35,651,584 outputs (blk.0 ffn_gate K2,
  M=2048)
- representation: <=25% growth, byte-exact (V3/V4 fixed mapping reused)
- resources: FP16 <=104, FP32 <=128, stack/local/spills 0, >=2 CTA/SM
- ownership: consumer MMA coverage full-tile; producers carry zero MMA work
- barriers: split arrive/wait present; record count per superstage
- direct op: <1.591085 ms (beat shared-B) required to continue;
  <=1.352422 ms (>=15% faster) is the breakthrough row
- diagnostics: record producer/consumer ratio tried, A-traffic delta vs V4

## Decision after V5 (Sol issues FINAL gate per Sean)

- <1.591 ms: mechanism real — iterate toward <=1.352 then family expansion
- >=1.591 ms: Sol issues the final-gate verdict with the complete
  seven-experiment data set and recommends the delivery path; do NOT silently
  continue — report to Sean with the banked hybrid option (P-ARCH-23
  8.599GB/3023 or 23G 8.808GB/3228) ready.
