# BASE-01 — Provenance Correlation Summary (Phase 1)

Method: dequantize corresponding projections from Arm A (ESCHA; reconstructed
from `escha_code`/`escha_rin`/`escha_rout` via `escham_cpu.reconstruct_deploy_weight`
with the same orientation as `convert_escha_to_gguf.py`) and Arm B (IQ3 LowGPU;
standard `gguf.quants.dequantize`). Correlation on flattened float32 over the full
projection. Empirical alignment validated by the bit-width ladder: correlation
rises with quantization bits, as expected for common-ancestry weights under
quantization noise.

## Results (layer 0 linear-attn, layer 3 full-attn; 2026-09-01)

| Projection | Family | Arm B type | Corr | Classification |
|---|---|---|---|---|
| blk.0.ffn_up | FFN K3 | IQ1_S (19) | 0.8729 | **same weights** (1.56-bpw noise) |
| blk.0.ffn_down | FFN K3 | IQ1_M (29) | 0.8727 | **same weights** (1.5-bpw noise) |
| blk.0.ffn_gate | FFN K2 | IQ1_S-class | 0.8350 (K=2, corrected) | **same weights** (1.5-bpw noise) |
| blk.3.attn_q | full-attn Q K2 | IQ2_S (22) | 0.9270 | **same weights** (2.1-bpw noise) |
| blk.3.attn_k | full-attn K K2 | IQ4_XS (23) | 0.9551 | **same weights** (4.25-bpw noise) |
| blk.3.attn_v | full-attn V K2 | IQ4_XS (23) | 0.9548 | **same weights** (4.25-bpw noise) |
| blk.3.attn_output | full-attn out K2 | (mixed) | 0.9398 (dims corrected 6144→5120) | **same weights** |
| blk.0.attn_qkv q+k block (rows 0..4096) | linear-attn QKV K2 | IQ3_XXS (18) | 0.9311 | **same weights** (3.4-bpw noise) |
| blk.0.attn_qkv v block (rows 4096..10240) | linear-attn QKV V | IQ3_XXS (18) | 0.0002 (best window) | **DIFFERENT weights** |
| blk.0.attn_gate | linear-attn gate K2 | IQ3_XXS (18) | 0.0431 | **DIFFERENT weights** (matches P-ARCH-21C: LowGPU attn_gate != checkpoint in_proj_z) |
| blk.0.ssm_out | linear-attn out K2 | IQ3_S (21) | 0.0396 | **DIFFERENT weights** |

Update 2026-09-01 (rev 2): all 10 projection families now classified. `ffn_gate`
was K=2 (code shape [32,1088,320]); the earlier K=3 guess was wrong. `attn_output`
is 6144→5120 (not 5120→5120). Both are same-weights after correction.

## Interpretation

1. **Common ancestry is proven for FFN (up/down) and full-attention Q/K/V**, and
   for the fused linear-attention QKV *q+k* portion. Correlations (0.87–0.96)
   track the Arm B bit width (IQ1_S < IQ2_S < IQ3_XXS < IQ4_XS), the signature of
   quantization noise on identical weights.
2. **The linear-attention `attn_gate`, `ssm_out`, and the QKV *v*-block are NOT
   the same weights** between artifacts (corr < 0.05, no reordering window
   recovers them). This is consistent with the P-ARCH-21C finding that the
   LowGPU GGUF's `attn_gate.weight` is a different projection than the Escha
   checkpoint's `in_proj_z` (corr 0.043 there too).
3. **Consequence for the comparison:** the two artifacts are NOT "identical
   weights in two quantizations." They share the same architecture, dimensions,
   layout (64 layers, 48 GDN + 16 full-attn, same tokenizer/vocab) and the same
   FFN + full-attn QKV weights, but differ in linear-attention gate/ssm_out/v
   projections. The runtime comparison remains valid as a same-architecture,
   same-shape comparison; the weight-difference families are the GDN gate/out
   path and the fused-QKV v branch, which are part of the ESCHA path we are
   attributing.
4. Storage difference: Arm A uses 400 packed `escha_code` tensors + LowGPU
   vocab sidecars; Arm B uses 851 standard GGML quantized tensors. Tensor-count
   difference (2058 vs 851) is sidecar/storage-driven, not semantic.
5. Vocab/head: both use the LowGPU 3-bit vocab family at the same shape
   [1920, 248320]; Arm A packs it as I8 codes, Arm B quantizes `token_embd`
   (IQ4_XS) and `output` (Q4_K). Same vocabulary semantics, different storage.

## Files
- `manifests/CORRELATION.json` — full-table correlation (script 1, validated orientation)
- `manifests/CORRELATION-QKVSPLIT.json` — attn_qkv q/k/v block dissection
- `manifests/CORRELATION-PERMUTE.json` — order/window probes (no recovery for v/gate/ssm_out)
- `manifests/CORRELATION-PROBES2.json`, `CORRELATION-FIX2.json` — layout sanity probes
- `scripts/correlate_artifacts.py`, `correlate_qkv_split.py`, `correlate_permute.py`
