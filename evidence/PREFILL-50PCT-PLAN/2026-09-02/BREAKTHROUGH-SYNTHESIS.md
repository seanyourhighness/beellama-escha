# Escha prefill breakthrough — merged synthesis (Sol review + multi-stream research)

Date: 2026-09-02

## The question

How do we make the PACKED Escha path faster on prefill — the exact-packed
route toward ~3030 (SGLang class) / ~3339 (native) tok/s at ~8.5 GB with all
BeeLlama native features? V1/V2 sidecars fixed correctness and occupancy but
are 8.85x/1.51x slower than the shared-B control. Five decode-restructuring
negatives are on the board (EXP-07/09/10, A4 V1/V2).

## Sol's V2 review — the central correction

The official `escham_code_gemm` does NOT use "decode once into shared then
broadcast" (that's Bee's control), and it does NOT use the 40+40
WARPSYNC/ENDCOLLECTIVE for B publication (those are its fused-Hadamard
epilogue shuffles). Its hot mainloop:

- warp-local direct decode (duplicated across row-warps — the official does
  NOT solve duplication; it makes duplication cheap and overlapped)
- zero decoded-B shared store / B LDSM / decoded-B publication barrier
- a **four-K16-class unrolled body: 64 HMMAs per repeated region** between one
  mainloop barrier and its backedge, with deep 45,056 B genuinely-addressed
  A/payload staging, packed-word extraction interleaved with A LDS and HMMA

V1 (per-warp decode, shallow, descriptor gathers) = 140 regs, 220 ALU.
V2 (decode-once collective, still descriptor gathers, one-K16 serial phases)
= 96 regs but 1.51x slower than control because each K16 runs as a serial
stage→decode→publish→reload→MMA transaction.

**Missing piece (Sol):** an official-shaped, four-K16 software pipeline with
direct register-fragment consumption, no runtime descriptor gathers, no
decoded-B shared round trip — the only untried exact-packed direction with a
credible mechanism. Recommended experiment: **V3-PIPE4** (frozen in
`V2-REVIEW-AND-BREAKTHROUGH.md`; hard gate <=1.352422 ms direct-op at M=2048,
<=25% representation growth, zero hot-loop descriptor traffic, 64 HMMA per
4-K16 body, regs <=104/128 no spills, bit-exact).

## External research (multi-stream: web/Reddit/X/repos)

### The exact problem class is known and named (X, jetha's A4Q campaign) [E1/E2/E9/E12]

On sm120/121, packed low-bit data (fp4 KV) was being "unpacked through an fp16
convert chain that burns ~9 instructions per tensor-core op" — reading a
quarter of the bytes and taking nearly twice as long. The fix (A4Q): feed the
same shared-memory bytes **directly into a block-scaled MMA**
(`mma.sync` with `mxf4nvf4.block_scale`) instead of expanding to fp16 first.
Corroborated by llama.cpp's `mmq-config-blackwell.cuh`: MXFP4/NVFP4 use a
dedicated `GGML_CUDA_MMQ_SRAM_LAYOUT_FP4` + `MMQ_ITER_K_FP4` path (hardware
block-scale MMA) rather than the generic dequant→fp16 chain.
**Translation to Escha:** the convert-chain/expansion cost is exactly the
decode ALU + shared-B round trip we keep measuring. The community solution for
formats WITH hardware block-scale support is direct packed→MMA. Escha's
codebook/trellis has NO hardware MMA for its codebook, so we cannot take the
A4Q path literally — but the architectural lesson (deep pipeline, avoid
per-op expansion serialization) matches Sol's V3-PIPE4.

### Independent Escha ports confirm the shared-B ceiling (X/repos) [E4/E5/E7]

ItsmeAjayKV (`Ajay9o9/llama.cpp-escha`, `escha-w2-dense` branch) decodes the
native 2-bit format in-kernel; his dense-branch kernel structure is the same
shared-B decode→shared→ldmatrix design we already have (upstream of our
control, no faster mainloop). ~30 t/s decode-class on 3090. Not a breakthrough
source — validates that nobody public has beaten the shared-B decode yet.

### The official runtime has a Blackwell route knob (X) [E1-new]

Escha's official runtime exposes `ESCHA_ROUTE=blackwell` worth ~1.5x on
Ampere decode. Confirms the official codebase has Blackwell-specific kernel
selection; the escham_code_gemm structure we audited is that route.

### 2-bit inference landscape (Reddit) [E19-new]

r/LocalLLaMA's 1/2-bit model tracking thread confirms active community work on
Bonsai/ternary/2-bit formats; no public kernel beats llama.cpp standard
dequant+MMA on prefill for these formats yet — the field is open.

## Merged conclusion

1. **Do NOT abandon the packed line.** V2's reject was real but not terminal;
   the official kernel's actual structure (deep direct-fragment pipeline) was
   never reproduced. V3-PIPE4 is the one remaining mechanism with a credible
   path and it is precisely specified with a hard gate.
2. The external world independently confirms: (a) nobody has a faster Escha
   mainloop in public; (b) on sm120 the winning pattern for packed data is
   deep pipelining + direct-to-MMA, not shared-B expansion; (c) the official
   runtime's own speed comes from this kernel shape.
3. If V3-PIPE4 misses (>=1.591085 ms or any resource/growth failure), the
   exact-packed line closes for good and the delivery is the hybrid frontier
   (P-ARCH-23 8.599 GB/3023 tok/s or 23G 8.808 GB/3228 tok/s) — already
   SGLang-class at the target size.

## Next step

Dispatch Sol to implement **V3-PIPE4** (frozen spec + mechanism gate in
`V2-REVIEW-AND-BREAKTHROUGH.md` §4): standalone blk.0 ffn_gate K2 M=2048,
four-K16 superstage, direct register-fragment B, zero descriptor gathers in
the hot loop, 64 HMMA per repeated body, deep A/payload staging, no shared-B.
Evidence: research run `~/.hermes/research-runs/escha-prefill-breakthrough-research/`
(packet.md, evidence.jsonl, MANIFEST.md).
