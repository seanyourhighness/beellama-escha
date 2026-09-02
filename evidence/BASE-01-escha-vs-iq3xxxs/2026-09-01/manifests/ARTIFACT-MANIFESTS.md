# BASE-01 — Artifact Manifests (Phase 1, 2026-09-01)

## ARM A — ESCHA W2 (canonical full-ESCHA control)

| Field | Value |
|---|---|
| Absolute path | `/mnt/d/CODEX WORKSPACE/escha-w2-lowgpu/weights/escha-w2-lowgpu-mono-parity.gguf` |
| Filename | `escha-w2-lowgpu-mono-parity.gguf` |
| SHA256 (full) | `e307007f4a7489777c70f724e14d807d403959b1dc1bf6857c44ca1b6954778d` |
| Expected SHA prefix | `e307007f…` — **MATCHES certification exactly** |
| Byte size | 8,619,127,360 B (8.02 GiB / 8.62 GB) |
| Expected size objective | ~8.65 GB — consistent (8.62 GB decimal) |
| GGUF architecture | `qwen35` |
| File type | `MOSTLY_F16` (body is Escha packed codes; storage is F16/I8/I16 sidecars) |
| ESCHA-coded projections | **400** `blk.*.escha_code` tensors (I16), all 10 projection families, K2/K3 |
| Standard-projection substitutions | **None** (canonical full-Escha artifact) |
| Vocab | LowGPU packed (I8 codes/scales/zps for `token_embd` and `output`) |
| Tensor count | 2058 |
| Quality certification | club-3090 medium 5-pack, gated runtime: **65/75 (130/150 equiv)** — `escha-w2-lowgpu/evidence/club3090-medium/2026-08-31-gated/` |
| Historical canonical speed | 2355.9 tok/s (promoted Stage 2, paired noise protocol, graphs on) |
| Provenance | EschaLabs/Qwen3.8-27B-Escha-W2 (rev f0eadefa) × LowGPU vocab; converted via `convert_escha_to_gguf.py` |

### Arm A family inventory (tensor-name families, from GGUFReader)
```
attn_gate 192 · attn_k 64 · attn_k_norm 16 · attn_norm 64 · attn_output 64 · attn_q 64 ·
attn_q_norm 16 · attn_qkv 192 · attn_v 64 · escha_dep_k2 1 · escha_dep_k3 1 · escha_lut 1 ·
ffn_down 256 · ffn_gate 256 · ffn_up 256 · output.lowgpu_* 3 · output_norm 1 ·
post_attention_norm 64 · ssm_a 48 · ssm_alpha 48 · ssm_beta 48 · ssm_conv1d 48 · ssm_dt 48 ·
ssm_norm 48 · ssm_out 192 · token_embd.lowgpu_* 3
```
Note: family counts include sidecar tensors; the 400 `escha_code` tensors are the coded projections.

---

## ARM B — IQ3_XXS LowGPU (original LowGPU GGUF)

| Field | Value |
|---|---|
| Absolute path | `/mnt/d/CODEX WORKSPACE/beellama-release/models/Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf` |
| Filename | `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS.gguf` |
| SHA256 (full) | `ad85e40a28aafd907eebb6ff6b21786b897dd750b0918427f1243d6d84ebcc72` |
| SHA256SUMS.txt match | YES (release package `SHA256SUMS.txt`) |
| Byte size | 9,570,663,040 B (8.91 GiB / 9.57 GB) |
| GGUF architecture | `qwen35` |
| GGUF file type | 24 → `MOSTLY_IQ1_S` per repo enum (mixed quant; model reports as IQ3_XXS-family LowGPU artifact) |
| Quantization mix | IQ3_XXS 67, IQ3_S 27, IQ4_XS 48, IQ2_XXS 27, IQ2_XS 12, IQ2_S 50, IQ1_S 33, IQ1_M 14, Q2_K 112, Q4_K 10, Q5_K 2, Q8_0 96, F32 353 |
| ESCHA-coded tensors | **0** (no escha codes; standard GGML quantized path) |
| Token embd | `token_embd.weight` IQ4_XS (standard get_rows) |
| LM head | `output.weight` Q4_K |
| Tensor count | 851 |
| Base model | Qwen3.8-27B (TheWegemann `Qwen3.8-27B-LowGPU-NoMTP-IQ3XXXS-GGUF`; NoMTP = MTP head removed; derived from Qwen/Qwen3.8-27B) |
| Existing quality result | original-lowgpu **66/75 (132/150 equiv)** — gated 5-pack certification |
| Existing benchmark evidence | `escha-w2-lowgpu/evidence/P-ARCH-23/2026-08-30/orig-lowgpu-target/bench.json`: avg 614.736 ms / 3338.76 tok/s (samples 593.039/595.284/655.886 ms), build `0b035b3a2`, `-p 2048 -n 0 -b 2048 -ub 2048 -ctk f16 -ctv f16 -fa on -ngl -1` |

### Candidate-disambiguation note
The file `/home/sean/Models/cpu-gguf/Qwen3.8-27B-UD-IQ3_XXS.gguf` (SHA `c0b7c303…`, 10,934,860,704 B, 65 blocks, includes MTP `nextn` tensors, 866 tensors) is a DIFFERENT model variant (has MTP; different tokenizer pad id 248055 vs 248044; block_count 65 vs 64). It is NOT the artifact that produced the recorded 3,339 tok/s LowGPU reference; it is excluded from the matched campaign. The recorded ~3,600 tok/s figure has **no measured benchmark on any artifact in GBrain/docs/sessions**; it appears only as a stretch target in `docs/escha-w2-prefill-next-plan.md` §2. The canonical Arm B reference is the 3,338.76 tok/s measurement above.

---

## Architecture / provenance comparison (same base weights?)

| Property | Arm A (ESCHA) | Arm B (IQ3 LowGPU) | Verdict |
|---|---|---|---|
| architecture | qwen35 | qwen35 | same |
| block_count | 64 | 64 | same |
| embedding_length | 5120 | 5120 | same |
| feed_forward_length | 17408 | 17408 | same |
| head_count / kv | 24 / 4 | 24 / 4 | same |
| key/value_length | 256 / 256 | 256 / 256 | same |
| ssm.state_size | 128 | 128 | same |
| ssm.conv_kernel | 4 | 4 | same |
| ssm.inner_size | 6144 | 6144 | same |
| rope.dimension_count | 64 | 64 | same |
| rope.freq_base | 1e7 | 1e7 | same |
| tokenizer model | gpt2 | gpt2 | same |
| vocab size | 248,320 (from tensor shapes) | 248,320 (from tensor shapes) | same |
| bos/eos/pad ids | 248044 / 248046 / 248044 | 248044 / 248046 / 248044 | same |
| projection representation | 400 escha_code (packed) | standard GGML quantized weights | **quantization-only difference** |
| token_embd | LowGPU packed (I8) | IQ4_XS standard | storage/quantization |
| output/head | LowGPU packed | Q4_K standard | storage/quantization |
| tensor count | 2058 | 851 | storage-sidecar difference |
| 48 linear-attention (GDN) layers | yes (ssm_* 48) | yes (ssm_* 48) | same semantics |
| full-attention layers (every 4th, 16 total) | yes | yes | same layout |

Pending: dequantized projection-block correlation samples (see `manifests/CORRELATION.md`).
