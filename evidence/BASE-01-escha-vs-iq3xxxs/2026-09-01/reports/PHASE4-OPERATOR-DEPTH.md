# BASE-01 Phase 4 — Direct Matched-Shape Operator Comparison (2026-09-01)

Method: per-depth prompts = prefixes of the canonical shared-2048 IDs (128/512/
1024/2048; 4096 = doubled prefix), same frozen/profile binaries, graphs OFF,
one profile+total run per arm per depth. Whole-run totals from the control
binary; per-family from symmetric CUDA-event hooks. Clocks recorded healthy
(1942–2887 MHz) at every depth.

## Whole-run graphs-off totals vs M (the cleanest signal)

| M | ESCHA ms | IQ3 ms | gap ms | ratio A/B |
|---|---|---|---|---|
| 128 | 142.2 | 63.2 | +79.0 | 2.25× |
| 512 | 258.9 | 155.2 | +103.7 | 1.67× |
| 1024 | 449.4 | 299.2 | +150.1 | 1.50× |
| 2048 | 801.6 | 590.1 | +211.5 | 1.36× |
| 4096 | 1628.0 | 1188.6 | +439.4 | 1.37× |

Interpretation: the ESCHA deficit is LARGEST at small M (2.25× at 128 rows) and
stabilizes at ~1.36× by M=2048–4096. This is consistent with:
- fixed launch/transform overhead (ESCHA issues rotate + GEMM + finalize +
  partial-buffer passes per projection; IQ3 issues ONE MMQ kernel per projection)
- the relative weight of that fixed overhead shrinks as M grows, while the
  packed-GEMM mainloop (which stays slower per FLOP than MMQ) dominates at
  large M — hence the ratio bottoms out above 1.0 rather than converging to 1.

## Per-call medians at M=2048 (real model activations)

| family | shape | ESCHA med ms | IQ3 med ms | ratio A/B |
|---|---|---|---|---|
| ffn_up | 5120→17408 K3 | 1.9556 | 1.2645 | 1.55× |
| ffn_down | 17408→5120 K3 | 2.4025 | 1.3278 | 1.81× |
| ffn_gate | 5120→17408 K2 | 1.8880 | 1.5516 | 1.22× |
| attn_qkv | 5120→10240 K2 | 1.1756 | 0.9665 | 1.22× |
| attn_q | 5120→12288 K2 | 1.4373 | 1.1528 | 1.25× |
| attn_gate | 5120→6144 K2 | 0.7841 | 0.7208 | 1.09× |

The FFN block dominates the per-call gap (up 1.55×, down 1.81×, gate 1.22×).
ffn_down shows the largest per-call penalty — the 17408→5120 K3 packed shape.

## Per-call median trend vs M (ffn_down, ms; verified from DEPTH-MATRIX.json)

| M | ESCHA | IQ3 | ratio |
|---|---|---|---|
| 128 | 0.2213 | 0.1333 | 1.66× |
| 512 | 0.6808 | 0.3216 | 2.12× |
| 1024 | 1.2276 | 0.6402 | 1.92× |
| 2048 | 2.4025 | 1.3278 | 1.81× |
| 4096 | 2.4066 | 1.3263 | 1.81× |

The per-call ratio is roughly flat ~1.7–2.1× across M, so the large-M
advantage is not a tiling crossover — it is a broad mainloop deficit.

## Conclusion for the depth question

IQ3's advantage does NOT come from better large-row tiling alone (per-call
ratios are roughly M-flat); it comes from (a) lower fixed overhead per
projection at small M (fewer kernels, no rotate/finalize passes) and (b) a
faster mainloop per FLOP at all M (standard MMQ decode-once structure vs
ESCHA's packed code + dep-table decode + shared-B + partial buffers).

Files: `operators/DEPTH-MATRIX.json`, `operators/*-M<M>.*`, `scripts/analyze-operator-depth.py`.
Caveat: profile totals include per-op sync overhead and do not close the
whole-run gap by construction; whole-run totals are the clean signal, per-call
medians are the shape-level evidence.
