#!/usr/bin/env python3
"""A2: Which GGML quant decodes the hot FFN shapes fastest on SM120 stock MUL_MAT?

Compares Q2_K vs IQ3_XXS vs IQ3_XS vs IQ4_XS vs Q4_K on the dominant shapes:
  K2 gate/up:  M=2048 rows x [IC=5120, OC=17408]
  K3 down:     M=2048 rows x [IC=17408, OC=5120]
Method: build tiny single-layer GGUF models per quant? Too heavy.
Instead: measure the REAL models' FFN-only cost via ESCHA_PROFILE-style route
split is not possible on standard MUL_MAT (no per-op profile). Use llama-bench
per-model aggregate as the proxy and diff by quant inventory.

Pragmatic proxy: llama-bench pp512/pp2048 per model already isolates the body.
The A2 decision input is the cross-model table (control/23I/LowGPU) plus per-
tensor quant inventory from /tmp/gguf_compare_out.txt:
  - 23I FFN: 53 Q2_K + 20 IQ2_XS + 28 IQ3_S + 27 IQ3_XS + 6 IQ3_XXS + ... 
  - LowGPU FFN: IDENTICAL per-type counts (donor bytes!) - so FFN quant mix is
    the SAME. The 2.5% gap must come from elsewhere: attention/gate/head/embed.
Re-examine: 23I attention = 53 Q2_K + 18 type23(IQ4_XS) + 10 IQ3_XS + 7 Q4_K...
LowGPU attention = mixed I-quants (20 Q2_K, 19 IQ3_XS, 14 IQ4_XS...). And 23I
gate = uniform Q2_K vs LowGPU mixed. Embedding Q4_K (23I) vs IQ4_XS (LowGPU).

Hypothesis to test in A2b: 23I's UNIFORM Q2_K on gate+attention+linear families
decodes SLOWER per byte than LowGPU's mixed IQ3/IQ4 on those families, even
though FFN bytes are identical. Test = build a 23I-variant with gate/attention/
linear re-quantized to the LowGPU-style mix and bench.
"""
print("A2 analysis note: FFN inventories are byte-identical between 23I and")
print("LowGPU (donor copy). The 2.5% gap is in gate/attention/linear/embed/head")
print("quant mix. See /tmp/gguf_compare_out.txt family x type histograms.")
