# A2 — precise diff map: 23I vs LowGPU beyond the byte-identical FFN

Date: 2026-09-02 · Method: per-tensor GGUF type diff on 850 common names.

## Non-FFN differences (the 2.5% same-bin gap suspects)

| family | count | 23I | LowGPU | decode-speed hypothesis |
|---|---:|---|---|---|
| ssm_alpha/ssm_beta (`other`) | 96 | **F16** | **Q8_0** | F16 5120→48 matmuls may miss quantized fast kernels; Q8_0 likely faster |
| linear_attn (attn_qkv/ssm_out) | 67 differing | uniform Q2_K | per-layer mixed IQ (IQ1_S..IQ3_XXS) | Q2_K block-decode slower than IQ3_XXS/IQ3_S class on these shapes |
| GDN_gate | 25 differing | uniform Q2_K | mixed IQ | same |
| embedding | 1 | Q4_K | IQ4_XS | GET_ROWS only; minor |
| output head | — | packed lowgpu (3 sidecars) | standard Q4_K output.weight | 23I pays LOWGPU_MUL_MAT; LowGPU pays standard MUL_MAT — prefill applies head once per token? verify |

Note: FFN gate/up/down are byte-identical between artifacts (donor copy), so
the FFN cannot explain the 2.5%. The gap is concentrated in the 48 recurrent
layers' GDN/SSM projection stack (96 F16 + 67 linear + 25 gate = the dominant
non-FFN matmul surface) and possibly the head.

## A2 variant (next experiment)

Build `23J` = 23I body with:
- ssm_alpha/ssm_beta re-quantized Q8_0 (match LowGPU; expect stock-kernel win)
- linear_attn + GDN_gate re-quantized per-layer to the LowGPU mixed-IQ recipe
  (preserve 23I's reconstructed VALUES — only change the GGML quant type, not
  the math source; byte-identity to LowGPU is NOT expected or required since
  23I's gate/linear are reconstructed from Escha sidecars, not donor bytes)
- keep Q4_K embedding (or test IQ4_XS) and packed head initially
- bench graphs-ON vs 23I baseline 3242.8; target ≥3339 (LowGPU parity) then
  ≥3479 (+50%)

Gate: ≥+2% over 23I with quality unchanged; size stays ≤9.6 GB.

Evidence: A1-BASELINE.md, /tmp/diff_23i_lowgpu.py output, /tmp/gguf_compare_out.txt.
